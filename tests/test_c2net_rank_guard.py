"""C2NET 在 DDP 下的 rank 行为测试。

背景：`c2net.context.prepare()` 与 `--data` / `--model` 的覆盖**必须**在所有 rank
上执行——每个 rank 都要独立加载数据集/权重，而 `--data` 是 required=True，
非 rank 0 若拿不到 c2net 给的 dataset_path 会直接 argparse 报错退出。

真正需要按 rank 过滤的是**日志打印**：否则 N 卡会刷出 N 份重复的 [c2net] 行。

若将来有人简单地给 prepare() 加上 `if is_main:`，会引入一个只在多卡下暴露的
崩溃（rank 1+ 拿不到 --data）。本测试把该不变量锁住。
"""
import inspect
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_MODS = {
    'train_sft': 'scripts.train_sft',
    'selfplay_train': 'scripts.selfplay_train',
}


def _strip_comments(text):
    """去掉整行注释与行尾注释，避免把注释里的关键字当成代码。"""
    out = []
    for line in text.splitlines():
        s = re.sub(r'(?<!["\'])#.*$', '', line) if '#' in line else line
        # 字符串里的 # 不算注释（本文件用不到，但保守处理）
        out.append(s)
    return '\n'.join(out)


def _main_src(name):
    import importlib
    mod = importlib.import_module(_MODS[name])
    return inspect.getsource(mod.main)


def _c2net_block(name):
    """取出 main() 里 c2net 初始化块的源码（已去注释）。

    先在**原始**源码里用注释锚点定位范围，再对块内容去注释——否则作为锚点的
    那行注释会先被删掉，导致找不到块。
    """
    raw = _main_src(name)
    i = raw.find('C2NET 支持')
    assert i != -1, f'{name} 未找到 C2NET 块'
    j = raw.find('except ImportError', i)
    assert j != -1, f'{name} C2NET 块缺少 ImportError 分支'
    return _strip_comments(raw[i:j])


def test_prepare_runs_on_all_ranks_not_main_only():
    """prepare() 不能被 is_main 守卫（否则非 rank0 拿不到数据/权重路径）。"""
    for name in _MODS:
        blk = _c2net_block(name)
        pm = re.search(r'(?m)^\s*_c2net_ctx = prepare\(\)', blk)
        assert pm, f'{name} 未找到 prepare() 调用'
        # 守卫若写成 `if is_main:` 会独占一行、位于 prepare() 的**上一行**，
        # 故必须回看前若干行，只看当前行前缀会漏判（曾因此放过一株变异体）。
        before = blk[:pm.start()]
        window = '\n'.join(before.rsplit('\n', 4)[-4:])
        assert 'if is_main' not in window, \
            f'{name}: prepare() 被 is_main 守卫了——多卡下非 rank0 会拿不到路径而崩溃'


def test_data_and_model_override_run_on_all_ranks():
    """--data / --model 的覆盖必须在所有 rank 生效。"""
    assert 'args.data = resolve_c2net_data(' in _c2net_block('train_sft'), \
        'train_sft 缺少 --data 覆盖'
    assert 'resolve_c2net_model(' in _c2net_block('selfplay_train'), \
        'selfplay_train 缺少 --model 覆盖'


def test_c2net_logging_is_main_only():
    """[c2net] 的日志打印必须受 is_main 过滤，避免多卡重复刷屏。"""
    for name in _MODS:
        blk = _c2net_block(name)
        for m in re.finditer(r'(?:logger\.info|print)\(', blk):
            head = blk[:m.start()]
            # 该语句所处的最近一层 if：必须是 is_main
            last_if = head.rfind('if is_main')
            # 若最近出现的是同缩进的 else / 其它 if，说明不在 is_main 分支内
            last_other_if = max(head.rfind('\n            if '),
                                head.rfind('\n        if '))
            assert last_if > last_other_if, \
                f'{name}: 存在未被 is_main 守卫的 [c2net] 日志（多卡会重复刷屏）'


def test_c2net_upload_and_output_redirect_stay_main_only():
    """输出重定向与 upload_output 必须仍是 rank 0 专属（不能被本次改动放宽）。"""
    for name in _MODS:
        src = _strip_comments(_main_src(name))
        assert '_c2net_orig_out' in src, f'{name} 缺少输出路径重定向记录'
        # upload_output 调用必须在 is_main 守卫内
        for m in re.finditer(r'upload_output\(\)', src):
            head = src[:m.start()]
            assert 'if is_main' in head, \
                f'{name}: upload_output 调用缺少 is_main 守卫'


def test_shell_scripts_document_c2net_rank_semantics():
    """多卡 SFT 脚本需说明 c2net 的 rank 语义（避免后人误加 is_main 守卫）。"""
    shell_dir = os.path.join(ROOT, 'shell')
    if not os.path.isdir(shell_dir):
        pytest.skip('shell/ 不存在')
    checked = 0
    for f in sorted(os.listdir(shell_dir)):
        if not f.endswith('.sh') or not f.startswith('train_sft'):
            continue
        txt = open(os.path.join(shell_dir, f), encoding='utf-8').read()
        m = re.search(r'(?m)^WORLD_SIZE=(\d+)', txt)
        if not m or int(m.group(1)) < 2:
            continue
        comments = '\n'.join(l for l in txt.splitlines()
                             if l.strip().startswith('#'))
        assert 'required=True' in comments or '所有 rank' in comments, \
            f'{f} 是多卡脚本，需注释说明 c2net 的 dataset_path 为何要全 rank 生效'
        checked += 1
    assert checked >= 2, f'应至少校验 2 个多卡 SFT 脚本，实得 {checked}'
