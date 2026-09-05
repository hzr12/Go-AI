"""
WebUI：19 路围棋人机对弈 + MCTS 实时搜索可视化。

零第三方依赖（Python 标准库 http.server），复用 src.inference.GoAI 与
src.search.mcts.MCTS（含 2026-09 的树复用与增量展开优化）。

启动:
    python scripts/webui.py --model models/sft_19x19_v3.pth --device cuda
    # 浏览器打开 http://127.0.0.1:7860

特性:
  - 19×19 棋盘（人执黑 / AI 执白），点击落子，支持 pass
  - 悔棋：连 AI 应手一并回退（MCTS 树复用路径同步回退），终局后可复活
  - 中国规则（数子法 + 贴目 7.5），终局自动计分
  - AI 每步 MCTS 实时搜索：top 候选着法叠加在棋盘上（访问次数热力）
  - 每次模拟实时渲染：搜索期间前端轮询 /api/search_progress，visits 热力逐模拟演化
  -   侧栏显示各候选 visits / prior / AI 胜率 / 根价值 / 搜索耗时
  - MCTS 树复用（path_moves）：整局搜索树跨步继承，AI 越下越快
  - 可选纯策略模式（无 MCTS）：单次网络前向即落子，秒级响应
"""

import sys
import os
import json
import time
import argparse
import threading

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.inference import GoAI
from src.search.mcts import MCTS
from src.game.go_rules import GoBoard

PASS = -1


class Session:
    """服务端单局对弈状态（人执黑 / AI 执白）。"""

    def __init__(self, ai, board_size=19, num_threads=1, default_mode="mcts",
                 expand_topk=32, expand_chunk=8, solver_thresh=0.9,
                 spec_prefetch=True, leaf_ab_depth=0, leaf_ab_width=4,
                 priors_leaf=False, hybrid_sims=32, hybrid_blend=0.5):
        self.ai = ai
        self.size = board_size
        self.lock = threading.Lock()
        self.mcts = MCTS(ai, board_size=board_size, num_threads=num_threads,
                         expand_topk=expand_topk, expand_chunk=expand_chunk,
                         solver_thresh=solver_thresh, spec_prefetch=spec_prefetch,
                         leaf_ab_depth=leaf_ab_depth, leaf_ab_width=leaf_ab_width,
                         priors_leaf=priors_leaf)
        self.default_mode = default_mode
        self.hybrid_sims = hybrid_sims
        self.hybrid_blend = hybrid_blend
        self._progress_lock = threading.Lock()  # 保护 search_progress（搜索线程写/轮询线程读）
        self.search_progress = None             # 最近一次搜索进度快照（实时渲染用）
        self.reset()

    def reset(self):
        self.board = GoBoard(self.size)
        self.hist = [[-1, -1, -1], [-1, -1, -1]]  # [黑方最近3手, 白方最近3手]
        self.path_moves = []                       # MCTS 树复用：自上次搜索根的着法序列
        self.mcts._prev_root = None                # 丢弃旧局搜索树（防错误复用子树）
        self.move_count = 0
        self.last_move = None                      # (r, c) 或 ('pass',)
        self.candidates = []                       # AI 最近一次搜索的 top 候选
        self.ai_info = None                        # 最近一次 AI 步的搜索信息
        self.log = []                              # AI 步日志
        self.game_over = False
        self.final_score = None
        self.search_progress = None

    def _progress_cb(self, sims_done, root):
        """MCTS 每次模拟后的进度回调：快照根 children 的 visits/胜率供前端轮询。"""
        bs = self.size
        cands = []
        for mv, child in root.children.items():
            q_child = child.q()  # child.to_play（=人类）视角
            cands.append({
                "mv": mv, "r": mv // bs, "c": mv % bs,
                "visits": int(child.visit),
                "prior": round(float(child.prior), 4),
                "ai_winrate": round(0.5 - q_child / 2.0, 3),
            })
        cands.sort(key=lambda x: -x["visits"])
        snap = {"sims": sims_done, "candidates": cands[:10]}
        with self._progress_lock:
            self.search_progress = snap

    def progress(self):
        with self._progress_lock:
            return self.search_progress

    # ---- 视图 ---------------------------------------------------------------

    def state(self):
        over = self.board.passes >= 2
        score = None
        if over and self.final_score is None:
            self.final_score = self.board.score()
        if self.final_score is not None:
            score = self.final_score
        return {
            "size": self.size,
            "board": self.board.board.reshape(-1).tolist(),
            "to_play": self.board.current_player,
            "move_count": self.move_count,
            "last_move": self.last_move,
            "over": over,
            "score": score,
            "candidates": self.candidates,
            "ai_info": self.ai_info,
            "log": self.log[-40:],
        }

    def _apply_move(self, mv, mover):
        """mv: 扁平坐标或 -1(pass)。mover: 'human'/'ai'。"""
        if self.board.passes >= 2:
            return
        ok = self.board.play(mv)
        if not ok:
            return
        self.path_moves.append(mv)
        to_play_before = -self.board.current_player  # 落子方
        if mv >= 0:
            hist = self.hist[0] if to_play_before == 1 else self.hist[1]
            hist.pop(0)
            hist.append(mv)
            self.last_move = (mv // self.size, mv % self.size)
        else:
            self.last_move = "pass"
        self.move_count += 1
        if self.board.passes >= 2 and self.final_score is None:
            self.final_score = self.board.score()
            self.game_over = True

    def _rebuild_after_undo(self):
        """从 board.move_history 重建悔棋后的派生状态（hist/last_move/终局判定等）。"""
        n = self.size
        hist = [[-1, -1, -1], [-1, -1, -1]]
        last_move = None
        for idx, mv in enumerate(self.board.move_history):
            color = 1 if idx % 2 == 0 else -1  # 黑先
            if mv >= 0:
                h = hist[0] if color == 1 else hist[1]
                h.pop(0)
                h.append(mv)
                last_move = (mv // n, mv % n)
            else:
                last_move = "pass"
        self.hist = hist
        self.last_move = last_move
        self.move_count = len(self.board.move_history)
        self.candidates = []
        self.ai_info = None
        if self.board.passes >= 2:
            self.final_score = self.board.score()
            self.game_over = True
        else:
            self.final_score = None
            self.game_over = False

    # ---- 动作 ---------------------------------------------------------------

    def human_move(self, mv):
        with self.lock:
            if self.game_over:
                return {"error": "对局已结束"}
            if self.board.current_player != 1:
                return {"error": "当前轮到 AI"}
            legal = self.board.get_legal_moves()
            if mv != PASS and (mv < 0 or mv >= len(legal) or not legal[mv]):
                return {"error": "非法着法"}
            self._apply_move(mv, "human")
            return self.state()

    def human_pass(self):
        return self.human_move(PASS)

    def undo(self):
        """悔棋：撤销到上一次人类落子之前。

        - AI 已应手（轮到人类）：连撤 AI 一手 + 人类一手，共两手
        - AI 未应手（轮到 AI，人类刚落子）：只撤人类一手
        - 终局后也可悔棋复活对局
        path_moves（MCTS 树复用路径）同步回退，保证搜索树位置一致。
        """
        with self.lock:
            if not self.board._undo_stack:
                return {"error": "没有可撤销的着法"}
            steps = 2 if self.board.current_player == 1 else 1
            steps = min(steps, len(self.board._undo_stack))
            for _ in range(steps):
                if not self.board.undo():
                    break
                if self.path_moves:
                    self.path_moves.pop()
            self._rebuild_after_undo()
            return self.state()

    def _ai_move_policy(self):
        """纯策略模式：单次网络前向选点，完全不做 MCTS 搜索（极速，棋力=策略网络本身）。"""
        to_play = self.board.current_player
        legal = self.board.get_legal_moves()
        bs = self.size
        n_actions = bs * bs + 1
        t0 = time.perf_counter()
        policy, value = self.ai.predict(self.board, self.hist[0], self.hist[1], to_play)
        elapsed = time.perf_counter() - t0

        masked = policy.copy()
        for m in range(n_actions - 1):
            if not legal[m]:
                masked[m] = 0.0
        move_int = int(np.argmax(masked))

        # top 候选（policy 概率；visits 字段放概率×1000 供棋盘热力渲染）
        ranked = sorted(
            ((m, float(masked[m])) for m in range(n_actions - 1) if masked[m] > 0),
            key=lambda t: -t[1])[:10]
        self.candidates = [{
            "mv": m, "r": m // bs, "c": m % bs,
            "visits": max(1, int(round(p * 1000))), "prior": round(p, 4),
            "ai_winrate": round((value + 1) / 2.0, 3),
        } for m, p in ranked]

        is_pass = (move_int == n_actions - 1)
        self._apply_move(PASS if is_pass else move_int, "ai")
        ai_winrate = round((value + 1) / 2.0, 3)
        mv_str = "pass" if is_pass else f"{chr(ord('a') + self.last_move[1])}{self.last_move[0] + 1}"
        info = {
            "move": mv_str,
            "visits": 1,
            "simulations": 1,
            "ai_winrate": ai_winrate,
            "elapsed": round(elapsed, 3),
            "sps": int(1 / max(elapsed, 1e-6)),
            "mode": "policy",
        }
        self.ai_info = info
        self.log.append(info)
        return self.state()

    def _ai_move_hybrid(self):
        """策略+少量MCTS：策略网络主导选点，小规模搜索做价值修正。

        score = blend * policy(合法归一) + (1-blend) * visits(归一)。
        少量模拟下 visits 分布噪声大，策略分量保证下限；搜索分量提供
        value 头/子节点评估对局部战术的修正。建议配合 --priors-leaf。
        """
        to_play = self.board.current_player
        legal = self.board.get_legal_moves()
        bs = self.size
        n_actions = bs * bs + 1
        sims = self.hybrid_sims
        t0 = time.perf_counter()
        with self._progress_lock:
            self.search_progress = None
        policy, _value = self.ai.predict(self.board, self.hist[0], self.hist[1], to_play)
        visits, _probs, root_value = self.mcts.search(
            self.board, self.hist[0], self.hist[1], to_play,
            simulations=sims, path_moves=self.path_moves,
            progress_cb=self._progress_cb)
        elapsed = time.perf_counter() - t0

        pol = np.asarray(policy).reshape(-1).astype(np.float64).copy()
        for m in range(n_actions - 1):
            if not legal[m]:
                pol[m] = 0.0
        ps = pol.sum()
        pol_n = pol / ps if ps > 0 else np.zeros_like(pol)
        vs = visits.astype(np.float64)
        vs_n = vs / vs.sum() if vs.sum() > 0 else np.zeros_like(vs)
        score = self.hybrid_blend * pol_n + (1.0 - self.hybrid_blend) * vs_n
        move_int = int(np.argmax(score))

        # top 候选：MCTS 根子节点，按混合得分排序（prior 列显示策略分量）
        root = getattr(self.mcts, "_prev_root", None)
        cands = []
        if root is not None:
            for mv, child in root.children.items():
                cands.append({
                    "mv": mv, "r": mv // bs, "c": mv % bs,
                    "visits": int(child.visit),
                    "prior": round(float(pol_n[mv]), 4),
                    "ai_winrate": round(0.5 - child.q() / 2.0, 3),
                    "score": round(float(score[mv]), 4),
                })
            cands.sort(key=lambda x: -x["score"])
        self.candidates = cands[:10]

        is_pass = (move_int == n_actions - 1)
        self._apply_move(PASS if is_pass else move_int, "ai")
        ai_winrate = round((root_value + 1) / 2.0, 3)
        mv_str = "pass" if is_pass else f"{chr(ord('a') + self.last_move[1])}{self.last_move[0] + 1}"
        info = {
            "move": mv_str,
            "visits": int(visits.sum()),
            "simulations": sims,
            "ai_winrate": ai_winrate,
            "elapsed": round(elapsed, 2),
            "sps": int(sims / max(elapsed, 1e-6)),
            "mode": "hybrid",
        }
        self.ai_info = info
        self.log.append(info)
        return self.state()

    def ai_move(self, simulations, temperature=0.0, mode=None):
        with self.lock:
            if self.game_over:
                return {"error": "对局已结束"}
            if self.board.current_player == 1:
                return {"error": "当前轮到人类"}
            mode = mode or self.default_mode
            if mode == "policy":
                return self._ai_move_policy()
            if mode == "hybrid":
                return self._ai_move_hybrid()
            to_play = self.board.current_player
            legal = self.board.get_legal_moves()
            t0 = time.perf_counter()
            with self._progress_lock:
                self.search_progress = None
            visits, probs, root_value = self.mcts.search(
                self.board, self.hist[0], self.hist[1], to_play,
                simulations=simulations, path_moves=self.path_moves,
                progress_cb=self._progress_cb)
            elapsed = time.perf_counter() - t0

            move_int = int(np.argmax(visits)) if visits.sum() > 0 else len(visits) - 1
            if move_int != len(visits) - 1 and not legal[move_int]:
                # 极端情况回退：取合法着中访问最高者
                legal_visits = [(int(m), int(visits[m])) for m in range(len(visits) - 1) if legal[m]]
                legal_visits.append((len(visits) - 1, int(visits[-1])))
                move_int = max(legal_visits, key=lambda x: x[1])[0]

            # top 候选（来自本次搜索树根的 children：visits/prior/Q）
            root = getattr(self.mcts, "_prev_root", None)
            cands = []
            if root is not None:
                for mv, child in root.children.items():
                    r, c = divmod(mv, self.size)
                    q_child = child.q()          # child.to_play（=人类）视角
                    ai_winrate = 0.5 - q_child / 2.0
                    cands.append({
                        "mv": mv, "r": r, "c": c,
                        "visits": int(child.visit),
                        "prior": round(float(child.prior), 4),
                        "ai_winrate": round(ai_winrate, 3),
                    })
                cands.sort(key=lambda x: -x["visits"])
            self.candidates = cands[:10]

            is_pass = (move_int == len(visits) - 1)
            mv_play = PASS if is_pass else move_int
            self._apply_move(mv_play, "ai")
            # AI（白）视角胜率
            ai_winrate = round((root_value + 1) / 2.0, 3)
            mv_str = "pass" if is_pass else f"{chr(ord('a') + self.last_move[1])}{self.last_move[0] + 1}"
            info = {
                "move": mv_str,
                "visits": int(visits.sum()),
                "simulations": simulations,
                "ai_winrate": ai_winrate,
                "elapsed": round(elapsed, 2),
                "sps": int(simulations / max(elapsed, 1e-6)),
                "mode": "mcts",
            }
            self.ai_info = info
            self.log.append(info)
            return self.state()


def build_handler(session, html_page=None):
    page = HTML_PAGE if html_page is None else html_page

    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html):
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send_html(page)
            elif self.path == "/api/state":
                self._send(session.state())
            elif self.path == "/api/search_progress":
                self._send({"progress": session.progress()})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            body = self._body()
            try:
                if self.path == "/api/human_move":
                    mv = int(body.get("move", PASS))
                    self._send(session.human_move(mv))
                elif self.path == "/api/human_pass":
                    self._send(session.human_pass())
                elif self.path == "/api/ai_move":
                    self._send(session.ai_move(
                        simulations=int(body.get("simulations", 50)),
                        mode=body.get("mode")))
                elif self.path == "/api/undo":
                    self._send(session.undo())
                elif self.path == "/api/reset":
                    session.reset()
                    self._send(session.state())
                else:
                    self._send({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                self._send({"error": f"{type(e).__name__}: {e}"}, 500)

        def log_message(self, *a):  # 静默默认访问日志
            pass

    return Handler


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Go-AI MCTS WebUI</title>
<style>
  body { background:#1c1f26; color:#d7dae0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
         margin:0; padding:16px; display:flex; gap:20px; }
  #boardWrap { background:#2a2e37; padding:12px; border-radius:10px; }
  canvas { cursor:pointer; display:block; }
  #panel { width:380px; display:flex; flex-direction:column; gap:12px; }
  .card { background:#2a2e37; border-radius:10px; padding:12px 14px; }
  h2 { margin:0 0 8px; font-size:15px; color:#8ab4f8; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  td, th { padding:3px 6px; text-align:left; border-bottom:1px solid #3a3f4a; }
  th { color:#9aa0a6; font-weight:normal; }
  .mv { font-family:monospace; color:#8ab4f8; }
  .bar { height:16px; background:#444; border-radius:8px; overflow:hidden; margin-top:6px; }
  .bar > div { height:100%; background:#5bb974; text-align:center; font-size:11px;
               color:#111; line-height:16px; transition:width .4s; }
  select, button { background:#3a3f4a; color:#d7dae0; border:1px solid #4a5060;
                   border-radius:6px; padding:6px 12px; font-size:13px; cursor:pointer; }
  button:hover { background:#4a5160; }
  button.primary { background:#8ab4f8; color:#111; border:none; font-weight:bold; }
  button:disabled { opacity:.4; cursor:default; }
  #thinking { color:#fbbc64; font-size:13px; min-height:18px; }
  #log { font-size:12px; color:#9aa0a6; max-height:220px; overflow-y:auto;
         font-family:monospace; white-space:pre-wrap; }
  .stat { display:flex; justify-content:space-between; font-size:13px; padding:2px 0; }
  .stat b { color:#8ab4f8; }
  #err { color:#f28b82; font-size:13px; min-height:16px; }
</style>
</head>
<body>
<div id="boardWrap"><canvas id="bd" width="660" height="660"></canvas></div>
<div id="panel">
  <div class="card">
    <h2>控制</h2>
    <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
      <label>模式 <select id="mode">
        <option value="hybrid" __SEL_HYBRID__>策略+少量MCTS</option>
        <option value="mcts" __SEL_MCTS__>MCTS 搜索</option>
        <option value="policy" __SEL_POLICY__>纯策略（无搜索）</option>
      </select></label>
      <label>模拟数 <select id="sims">
        <option selected>50</option><option>400</option><option>800</option><option>1600</option>
      </select></label>
      <button class="primary" id="btnAI">AI 落子</button>
      <button id="btnPass">停一手</button>
      <button id="btnUndo">悔棋</button>
      <button id="btnReset">新对局</button>
    </div>
    <div id="thinking"></div>
    <div id="err"></div>
  </div>
  <div class="card">
    <h2>AI 评估（最近一步）</h2>
    <div class="stat"><span>着法</span><b id="stMove">-</b></div>
    <div class="stat"><span>AI 胜率</span><b id="stWin">-</b></div>
    <div class="bar"><div id="winBar" style="width:50%">50%</div></div>
    <div class="stat" style="margin-top:6px"><span>总访问 / 模拟数</span><b id="stVisits">-</b></div>
    <div class="stat"><span>搜索耗时</span><b id="stTime">-</b></div>
    <div class="stat"><span>吞吐 (sims/s)</span><b id="stSps">-</b></div>
  </div>
  <div class="card">
    <h2>MCTS Top 候选</h2>
    <table><thead><tr><th>着法</th><th>visits</th><th>prior</th><th>AI 胜率</th></tr></thead>
    <tbody id="cand"></tbody></table>
  </div>
  <div class="card"><h2>日志</h2><div id="log"></div></div>
</div>
<script>
const N = 19, PAD = 34, CS = (660 - PAD * 2) / (N - 1);
const cv = document.getElementById('bd'), ctx = cv.getContext('2d');
let S = null, thinking = false;

const STARS = [[3,3],[3,9],[3,15],[9,3],[9,9],[9,15],[15,3],[15,9],[15,9],[15,15]].filter(
  (v,i,a) => a.findIndex(x => x[0]===v[0]&&x[1]===v[1]) === i);
const cands = {};  // "r,c" -> candidate

function xy(rc){ return {x: PAD + rc*CS, y: PAD + rc*CS}; }

function draw(){
  ctx.fillStyle = "#dcb35c"; ctx.fillRect(0,0,660,660);
  ctx.strokeStyle = "#7a5c28"; ctx.lineWidth = 1;
  for (let i=0;i<N;i++){
    const p = xy(i);
    ctx.beginPath(); ctx.moveTo(PAD, p.y); ctx.lineTo(660-PAD, p.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p.x, PAD); ctx.lineTo(p.x, 660-PAD); ctx.stroke();
  }
  ctx.fillStyle = "#3a2a10";
  for (const [r,c] of STARS){ const p = xy(c), q = xy(r);
    ctx.beginPath(); ctx.arc(p.x, q.y, 3.4, 0, 2*Math.PI); ctx.fill(); }
  ctx.fillStyle = "#5a4318"; ctx.font = "12px monospace";
  for (let i=0;i<N;i++){ const p = xy(i);
    ctx.fillText(String.fromCharCode(65+i), p.x-4, 16);
    ctx.fillText(String(N-i), 8, p.y+4); }
  if (!S) return;
  const B = S.board;
  for (let i=0;i<N*N;i++){
    const v = B[i]; if (v===0) continue;
    const r = Math.floor(i/N), c = i%N, p = xy(c), q = xy(r);
    ctx.beginPath(); ctx.arc(p.x, q.y, CS*0.46, 0, 2*Math.PI);
    ctx.fillStyle = v===1 ? "#111" : "#f5f5f5"; ctx.fill();
    ctx.strokeStyle = v===1 ? "#000" : "#888"; ctx.stroke();
  }
  if (S.last_move && S.last_move !== "pass"){
    const p = xy(S.last_move[1]), q = xy(S.last_move[0]);
    ctx.beginPath(); ctx.arc(p.x, q.y, CS*0.16, 0, 2*Math.PI);
    ctx.fillStyle = "#e5443c"; ctx.fill();
  }
  for (const cd of (S.candidates||[])){
    if (B[cd.r*N+cd.c] !== 0) continue;
    const p = xy(cd.c), q = xy(cd.r);
    const mx = Math.max(...(S.candidates.map(x=>x.visits)));
    const rad = CS*0.22 + CS*0.3*(cd.visits/mx);
    ctx.beginPath(); ctx.arc(p.x, q.y, rad, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(91,185,116,.55)"; ctx.fill();
    ctx.strokeStyle = "#2e7d4f"; ctx.stroke();
    ctx.fillStyle = "#12331d"; ctx.font = "bold 11px monospace"; ctx.textAlign="center";
    ctx.fillText(cd.visits, p.x, q.y+4); ctx.textAlign="left";
  }
}

function render(){
  if (!S) return;
  document.getElementById('stMove').textContent = S.ai_info ? S.ai_info.move : '-';
  document.getElementById('stWin').textContent = S.ai_info ? (S.ai_info.ai_winrate*100).toFixed(1)+'%' : '-';
  const wr = S.ai_info ? S.ai_info.ai_winrate : 0.5;
  const bar = document.getElementById('winBar');
  bar.style.width = (wr*100).toFixed(0)+'%'; bar.textContent = (wr*100).toFixed(0)+'%';
  document.getElementById('stVisits').textContent = S.ai_info ? `${S.ai_info.visits} / ${S.ai_info.simulations}` : '-';
  document.getElementById('stTime').textContent = S.ai_info ? S.ai_info.elapsed+' s' : '-';
  document.getElementById('stSps').textContent = S.ai_info ? S.ai_info.sps : '-';
  const tb = document.getElementById('cand'); tb.innerHTML = '';
  for (const cd of (S.candidates||[])){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td class="mv">${String.fromCharCode(65+cd.c)}${N-cd.r}</td>` +
      `<td>${cd.visits}</td><td>${(cd.prior*100).toFixed(1)}%</td>` +
      `<td>${(cd.ai_winrate*100).toFixed(0)}%</td>`;
    tb.appendChild(tr);
  }
  document.getElementById('log').textContent = (S.log||[]).map(l =>
    `[${l.move}] visits ${l.visits}/${l.simulations} win ${(l.ai_winrate*100).toFixed(0)}% ${l.elapsed}s ${l.sps}/s`).join('\n');
  document.getElementById('btnUndo').disabled = thinking || !(S.move_count > 0);
  draw();
}

function setBusy(b){
  thinking = b;
  const mode = document.getElementById('mode').value;
  const tips = {policy: 'AI 单次前向推理…', hybrid: 'AI 策略+少量MCTS…'};
  document.getElementById('thinking').textContent =
    b ? (tips[mode] || 'AI 正在 MCTS 搜索…') : '';
  document.getElementById('btnAI').disabled = b;
  document.getElementById('btnPass').disabled = b;
  document.getElementById('btnUndo').disabled = b || !(S && S.move_count > 0);
}

async function post(url, body){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
                              body: JSON.stringify(body||{})});
  const j = await r.json();
  if (j.error){ document.getElementById('err').textContent = j.error; return null; }
  document.getElementById('err').textContent = '';
  S = j; render(); return j;
}

cv.addEventListener('click', async (e) => {
  if (thinking || !S || S.over || S.to_play !== 1) return;
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const c = Math.round((mx - PAD)/CS), r = Math.round((my - PAD)/CS);
  if (r<0||r>=N||c<0||c>=N) return;
  const p = xy(c), q = xy(r);
  if (Math.hypot(mx-p.x, my-q.y) > CS*0.45) return;
  if (S.board[r*N+c] !== 0) return;
  setBusy(true);
  const st = await post('/api/human_move', {move: r*N+c});
  if (st && !st.over && st.to_play === -1) await aiTurn();
  setBusy(false);
});

let pollTimer = null;
function stopProgressPoll(){
  if (pollTimer){ clearInterval(pollTimer); pollTimer = null; }
}
function startProgressPoll(){
  stopProgressPoll();
  pollTimer = setInterval(async () => {
    try {
      const j = await fetch('/api/search_progress').then(r => r.json());
      if (j.progress && S && thinking){
        S.candidates = j.progress.candidates;   // 实时 visits 热力直接替换候选
        document.getElementById('thinking').textContent =
          `AI 正在 MCTS 搜索… ${j.progress.sims} 模拟`;
        draw();
      }
    } catch(e){}
  }, 120);
}

async function aiTurn(){
  const mode = document.getElementById('mode').value;
  const sims = parseInt(document.getElementById('sims').value);
  if (mode !== 'policy') startProgressPoll();
  const st = await post('/api/ai_move', {simulations: sims, mode: mode});
  stopProgressPoll();
  if (st && st.over && st.score !== null){
    document.getElementById('thinking').textContent =
      `对局结束：黑${st.score>0?'胜':'负'}（黑-白 = ${st.score>0?'+':''}${st.score.toFixed(1)}）`;
  }
}

document.getElementById('btnAI').onclick = async () => {
  if (thinking) return; setBusy(true);
  await aiTurn(); setBusy(false);
};
document.getElementById('btnPass').onclick = async () => {
  if (thinking) return; setBusy(true);
  const st = await post('/api/human_pass', {});
  if (st && !st.over && st.to_play === -1) await aiTurn();
  setBusy(false);
};
document.getElementById('btnUndo').onclick = async () => {
  if (thinking) return; setBusy(true);
  await post('/api/undo', {});
  setBusy(false);
};
document.getElementById('btnReset').onclick = async () => {
  if (thinking) return; setBusy(true);
  await post('/api/reset', {}); setBusy(false);
};
document.getElementById('mode').addEventListener('change', function(){
  document.getElementById('sims').disabled = (this.value !== 'mcts');
});
document.getElementById('sims').disabled =
  (document.getElementById('mode').value !== 'mcts');

(async () => { const s = await fetch('/api/state').then(r=>r.json()); S = s; render(); })();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Go-AI WebUI（19 路人机对弈 + MCTS 可视化）")
    ap.add_argument("--model", default="models/sft_19x19_v3.pth")
    ap.add_argument("--board-size", type=int, default=19)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--num-threads", type=int, default=8, help="MCTS 选路径线程数")
    ap.add_argument("--mode", choices=["hybrid", "mcts", "policy"], default="hybrid",
                    help="默认 AI 落子模式：hybrid=策略+少量MCTS（默认）, mcts=MCTS 搜索, "
                         "policy=纯策略单次前向（无搜索）")
    ap.add_argument("--hybrid-sims", type=int, default=32,
                    help="hybrid 模式每次落子的 MCTS 模拟数（少量）")
    ap.add_argument("--hybrid-blend", type=float, default=0.5,
                    help="hybrid 模式策略分量权重：score=blend*policy+(1-blend)*visits（1=纯策略）")
    ap.add_argument("--priors-leaf", action="store_true",
                    help="子节点先验改用叶子自身 policy（标准 AlphaZero 方案；少量模拟时"
                         "搜索更贴近策略网络，hybrid 模式推荐开启）")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile 算子融合（MCTS 网络评估提速 20-40%%；与量化/ONNX 互斥）")
    ap.add_argument("--tf32", action="store_true",
                    help="CUDA tf32 matmul（fp32 matmul 约 2-4x 提速）")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="torch CPU 线程数（0=自动取 CPU 核数；与 --num-threads 无关）")
    ap.add_argument("--expand-topk", type=int, default=32,
                    help="MCTS 展开候选截断为 policy top-K（0=全部 ~N²+1 个，GPU 可关；CPU 建议 16-32）")
    ap.add_argument("--expand-chunk", type=int, default=8,
                    help="展开期 α-β 界截断块大小：按 prior 降序分块评估，发现必胜着即截断（0=一块评完，GPU 建议 0）")
    ap.add_argument("--solver-thresh", type=float, default=0.9,
                    help="MCTS-Solver 必胜/必败判定阈值（价值域 [-1,1]）")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="关闭 worker 推测性预评估（默认开启，与主线程前向重叠）")
    ap.add_argument("--leaf-ab-depth", type=int, default=0,
                    help="叶子内浅层 α-β 深度（0=关闭；2-4 = speculative 深化，棋力换时间）")
    ap.add_argument("--leaf-ab-width", type=int, default=4,
                    help="叶内 α-β 每节点 policy top-W 走法排序宽度")
    ap.add_argument("--quantize", action="store_true",
                    help="CPU int8 动态量化 Linear 层（与 --compile 互斥）")
    ap.add_argument("--onnx", nargs="?", const="models/goai_cpu.onnx", default=None,
                    help="导出 ONNX 并用 onnxruntime 推理（CPU 提速 1.5-3x，需 pip install onnx onnxruntime）")
    ap.add_argument("--use-rollout", action="store_true", help="启用 LightPLS 价值融合")
    ap.add_argument("--rollout-lambda", type=float, default=0.25)
    args = ap.parse_args()

    model_path = args.model
    if not os.path.isfile(model_path):
        print(f"[warn] 模型 {model_path} 不存在，使用随机权重（仅可验证流程）")
        model_path = None

    # CPU 线程配置（先于推理；interop=1 避免双层层级争抢）
    n_threads = args.cpu_threads if args.cpu_threads > 0 else (os.cpu_count() or 1)
    try:
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(1)
        print(f"[webui] torch CPU 线程数 = {n_threads}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 线程配置失败（可忽略）: {e}")

    # 量化/ONNX 与 torch.compile 互斥（量化编译后模型无意义）
    use_compile = args.compile and not (args.quantize or args.onnx)
    ai = GoAI(model_path=model_path, board_size=args.board_size, device=args.device,
              use_amp=True, attn_mode="window", attn_window=7,
              compile=use_compile, tf32=args.tf32,
              channels_last=(args.device.split(":")[0] == "cuda"))
    if args.quantize:
        ai.quantize_dynamic()
    if args.onnx:
        ai.export_onnx(args.onnx)
    session = Session(ai, board_size=args.board_size, num_threads=1,
                      default_mode=args.mode, expand_topk=args.expand_topk,
                      expand_chunk=args.expand_chunk,
                      solver_thresh=args.solver_thresh,
                      spec_prefetch=not args.no_prefetch,
                      leaf_ab_depth=args.leaf_ab_depth,
                      leaf_ab_width=args.leaf_ab_width,
                      priors_leaf=args.priors_leaf,
                      hybrid_sims=args.hybrid_sims,
                      hybrid_blend=args.hybrid_blend)
    session.mcts.use_rollout = args.use_rollout
    session.mcts.rollout_lambda = args.rollout_lambda
    if args.use_rollout:
        session.mcts._fast_policy = __import__(
            "src.search.light_rollout", fromlist=["FastPolicy"]).FastPolicy(args.board_size)

    html_page = (HTML_PAGE
                 .replace("__SEL_HYBRID__", "selected" if args.mode == "hybrid" else "")
                 .replace("__SEL_MCTS__", "selected" if args.mode == "mcts" else "")
                 .replace("__SEL_POLICY__", "selected" if args.mode == "policy" else ""))

    handler = build_handler(session, html_page)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"WebUI: http://127.0.0.1:{args.port}  (模型={args.model if model_path else '随机权重'}, "
          f"设备={ai.device}, 线程={args.num_threads})")
    print("Ctrl+C 退出")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
