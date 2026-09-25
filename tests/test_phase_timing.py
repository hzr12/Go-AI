"""分段计时测试。

动机：4 卡 910A 实测 4.25 s/step。逐区间核对日志后确认速率极稳定
（8 个无 eval 区间全在 4.22~4.30 s/step，30 分钟零漂移），而 eval
每次仅约 1.7 秒、占比 0.2% —— 即约 96% 的时间是黑盒，无法判断瓶颈
在取数 / 算子 / 通信 / 保存。

另外：SwanLab 的 NPU Utilization 曲线无法用于归因。npu-smi 采样间隔
10~14s，远粗于振荡周期，2.5 秒与 60 秒的停顿在图上都渲染成同样宽的
缺口，密集折线多为混叠；且 `Aicore Usage Rate` 是内核执行时间占比、
不是效率（实测该值 85~100% 而实际仅 659 samples/s/卡）。

关键不变量：
1. 绝不插 synchronize() —— 会打断预取与双缓冲流水，反而更慢；
2. 累加器必须在打点处统一重置，否则跨区间累加导致数值虚高；
3. save/eval 发生在其打点之后，因此计入下一区间（墙钟口径正确）。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()

# 去掉整行注释与 docstring 内的说明文字，避免把「解释旧公式为何错」的注释
# 误判成「旧公式仍在」，也避免把「不要插 synchronize」的注释当成调用。
CODE = '\n'.join(
    ln for ln in SRC.splitlines() if not ln.lstrip().startswith('#')
)


def test_no_synchronize_in_training_path():
    """训练路径（main 及其之后）不得出现 synchronize。

    计时必须纯 CPU 侧。插 synchronize 会把预取与双缓冲流水打断，
    测出来的数还比真实更慢。
    注意 _cuda_free/_npu_free 里的 synchronize 属设备选择、在 main 之前，
    是合法且必须保留的，故只检查 main 之后的区域。
    """
    m = re.search(r'^def main\(', CODE, re.M)
    assert m, '未找到 main()'
    body = CODE[m.start():]
    for bad in ('npu.synchronize', 'cuda.synchronize'):
        assert bad not in body, \
            f'main() 内出现 {bad}：会打断预取/双缓冲流水'
    # 设备选择辅助函数在 main 之前，允许存在
    assert 'cuda.synchronize' in CODE[:m.start()], \
        '设备选择的 _cuda_free 应保留 synchronize（用于探测空闲显存）'


def test_all_four_accumulators_initialised():
    for name in ('_t_data', '_t_comp', '_t_save', '_t_eval'):
        assert name in SRC, f'缺少计时累加器 {name}'
    assert re.search(
        r'_t_data = _t_comp = _t_save = _t_eval = 0\.0\s*\n'
        r'\s*_t_data_max = 0\.0\s*\n\s*_n_timed = 0',
        SRC,
    ), '循环入口未初始化四个累加器'


def test_every_segment_is_measured():
    """四段都要有 perf_counter 起止并累加。

    t_data 走中间变量 _dt（因为还要拿它更新 _t_data_max 尖峰），
    其余三段直接内联 perf_counter，两种写法都算合法。
    """
    for tag in ('_t_data0', '_t_comp0', '_t_save0', '_t_eval0'):
        assert f'{tag} = time.perf_counter()' in SRC, f'缺少 {tag} 起点'
    for acc in ('_t_comp', '_t_save', '_t_eval'):
        assert re.search(re.escape(acc) + r' \+= time\.perf_counter\(\) - ', SRC), \
            f'{acc} 未被累加'
    assert re.search(r'_t_data \+= _dt', SRC) or re.search(
        r'_t_data \+= time\.perf_counter\(\) - ', SRC), '_t_data 未被累加'


def test_accumulators_reset_together_at_logging():
    """重置必须在打点处统一进行，且三行齐全。"""
    assert re.search(
        r'_t_data = _t_comp = _t_save = _t_eval = 0\.0\s*\n'
        r'\s*_t_data_max = 0\.0\s*\n\s*_n_timed = 0',
        SRC,
    ), '累加器未在打点处统一重置（会跨区间累加，数值虚高）'
    # 重置必须在打印之前、且在打点块内。
    # 两个坑：
    #  1) 不能用 SRC.index('if _do_stdout:')：它会匹配到
    #     'if _do_stdout or _do_swanlab:' 的前缀，取到错误位置；
    #  2) 重置语句出现两次（循环入口初始化 + 打点处重置），
    #     必须从 blk 之后开始找，否则会拿到入口那处。
    blk = SRC.index('if _do_stdout or _do_swanlab:')
    out = SRC.index('\n            if _do_stdout:\n')
    reset = SRC.index('_t_data = _t_comp = _t_save = _t_eval = 0.0', blk)
    assert blk < reset < out, '重置位置应在打点块内、stdout 打印之前'
    # 入口处也必须初始化一次（否则首区间用到未定义变量）
    init = SRC.index('_t_data = _t_comp = _t_save = _t_eval = 0.0')
    assert init < blk, '循环入口未初始化累加器'


def test_data_wait_peak_is_tracked():
    """取数尖峰最能暴露数据管道饥饿，必须单独记 max。"""
    assert 'if _dt > _t_data_max:' in SRC
    assert '_dt = time.perf_counter() - _t_data0' in SRC
    assert '_dmax = _t_data_max * 1000.0' in SRC


def test_segments_logged_and_uploaded():
    assert re.search(r'd=%.0f c=%.0f s=%.0f e=%.0f dmax=%.0f ms', SRC), \
        '日志行缺少分段耗时字段'
    for key in ('"t_data_ms": _dms', '"t_comp_ms": _cms',
                '"t_save_ms": _sms', '"t_eval_ms": _ems'):
        assert key in SRC, f'swanlab 未上报 {key}'


def test_eval_timing_wraps_the_dominant_cost():
    """eval 计时只包住 evaluate_metrics（主导开销），避免大段重排缩进。"""
    m = re.search(
        r'_t_eval0 = time\.perf_counter\(\)\s*\n'
        r'\s*metrics = evaluate_metrics\(.*?\)\s*\n'
        r'\s*_t_eval \+= time\.perf_counter\(\) - _t_eval0',
        SRC, re.S,
    )
    assert m, 'eval 计时未包住 evaluate_metrics'
