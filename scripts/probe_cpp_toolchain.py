#!/usr/bin/env python3
"""检测本机**能不能用 C++ 加速**这个仓库的特征计算（梯子搜索 / Benson 面积迭代）。

为什么不是「有没有 g++」
------------------------
本项目要 C++ 的目的只有一个：把 `src/data/feature_v7_ladders.py`（梯子搜索）
与 `feature_v7.py::calculate_area`（Benson 面积迭代）这两块**逐盘面的 Python
对象计算**换成编译代码。⇒ 真正的判据是「**能不能构建并 import 一个 Python
扩展模块**」，而不是「PATH 上有没有编译器」：

  · 有 g++ 但没有 Python.h ⇒ 编不出扩展（纯 C++ 可用，Python 扩展不可用）
  · 有 Python.h 但编译器链接失败 / 缺 C++ 标准库 ⇒ 同样不可用
  · 编译器在 PATH 上但实际不能链接（常见于只装了 runtime 没装 dev 包）⇒ 不可用

所以下面分三层测，每层都能独立判死：
  L1 编译器：存在吗？版本？**能编并链接一个 C++ 程序吗**？
  L2 Python 侧：有 Python.h 吗？numpy 头文件吗？ABI 匹配吗？
  L3 端到端：**真的构建并 import 一个最小扩展**吗？（这才是最终判据）

用法
----
    python scripts/probe_cpp_toolchain.py            # 只探测，不留文件
    python scripts/probe_cpp_toolchain.py --keep     # 保留临时目录（看构建日志）

退出码：0 = L3 通过（可以用 C++）；1 = 不可用（看报告里哪一层挂的）
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile

OK, BAD, WARN = 'PASS', 'FAIL', 'WARN'
_lines = []
_fail = 0


def emit(level, name, detail=''):
    global _fail
    if level == BAD:
        _fail += 1
    mark = {OK: '[ OK ]', BAD: '[FAIL]', WARN: '[WARN]'}[level]
    _lines.append('%s %-34s %s' % (mark, name, detail))
    print('%s %-34s %s' % (mark, name, detail), flush=True)


def section(title):
    print('\n=== %s ===' % title, flush=True)


# --------------------------------------------------------------------------- #
# L1 编译器
# --------------------------------------------------------------------------- #
CXX_CANDIDATES = ['g++', 'clang++', 'c++', 'cl']


def _run(cmd, cwd=None, timeout=180):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def probe_compiler():
    section('L1  C++ 编译器')
    found = []
    for c in CXX_CANDIDATES:
        exe = shutil.which(c)
        if exe:
            found.append((c, exe))
    if not found:
        emit(BAD, '编译器', 'PATH 上找不到 g++/clang++/c++/cl')
        return None
    for c, exe in found:
        v = _run([exe, '--version'])
        ver = (v.stdout or v.stderr).splitlines()[0][:70] if (v.stdout or v.stderr) else '?'
        emit(OK, '编译器 %s' % c, ver)
    # 真正要用的那个：优先 g++（与 numpy/scipy wheels 的 ABI 习惯一致）
    c, exe = found[0]
    # **真编译 + 链接**一个最小 C++ 程序（用 <vector> 验证 C++ 标准库可用）
    tmp = tempfile.mkdtemp(prefix='cppprobe_')
    src = os.path.join(tmp, 't.cpp')
    out = os.path.join(tmp, 't.bin')
    with open(src, 'w', encoding='utf-8') as f:
        f.write('#include <vector>\n#include <string>\n'
                '#include <cstdio>\n'
                'int main(){ std::vector<std::string> v{"ok"}; '
                'std::printf("%s\\n", v[0].c_str()); return 0; }\n')
    if c == 'cl':
        cmd = ['cl', '/nologo', '/EHsc', src, '/Fe:' + out]
    else:
        cmd = [exe, '-O2', '-std=c++14', src, '-o', out]
    r = _run(cmd, cwd=tmp)
    if r.returncode != 0 or not os.path.exists(out):
        emit(BAD, '编译+链接最小 C++ 程序',
             (r.stderr or r.stdout).strip().splitlines()[-1][:110]
             if (r.stderr or r.stdout).strip() else 'rc=%d' % r.returncode)
        return None
    emit(OK, '编译+链接最小 C++ 程序', '含 <vector>/<string>，标准库可用')
    return c, exe


# --------------------------------------------------------------------------- #
# L2 Python 侧
# --------------------------------------------------------------------------- #
def probe_python_headers():
    section('L2  Python 开发头文件与扩展工具链')
    inc = sysconfig.get_paths().get('include')
    plat_inc = sysconfig.get_paths().get('platinclude')
    have_header = bool(inc) and os.path.isfile(os.path.join(inc, 'Python.h'))
    emit(OK if have_header else BAD, 'Python.h',
         (inc or '?') if have_header else
         '缺 Python.h（只装了 runtime、没装 dev 头 ⇒ 编不了扩展）')
    if plat_inc and plat_inc != inc:
        emit(WARN, 'platinclude', plat_inc)
    # numpy 头（C++ 扩展要 include numpy 的 C API 时需要）
    try:
        import numpy as _np
        np_inc = _np.get_include()
        emit(OK if os.path.isdir(np_inc) else WARN, 'numpy 头文件', np_inc)
    except Exception as e:
        emit(WARN, 'numpy 头文件', 'import 失败: %r' % e)
    # 可选：写扩展的两种口味
    for mod, why in (('Cython', '写 Cython 扩展（改动最小）'),
                     ('pybind11', '写 pybind11 扩展（真 C++）'),
                     ('nanobind', '写 nanobind 扩展（新）'),
                     ('setuptools', 'setuptools 构建后端'),
                     ('cmake', 'CMake（nanobind/scikit-build 需要）')):
        try:
            __import__(mod)
            emit(OK, mod, '已安装（%s）' % why)
        except Exception:
            emit(WARN, mod, '未安装（%s）' % why)
    return have_header


# --------------------------------------------------------------------------- #
# L3 端到端：真的构建并 import 一个最小扩展
# --------------------------------------------------------------------------- #
EXT_SRC = r'''
#include <Python.h>
#include <vector>
#include <string>

// 返回一个用 std::vector/std::string 算出来的串 —— 顺带证明 C++ 标准库
// 与 Python C API 能共存于同一个翻译单元（这正是要移植的那两处代码的形态）。
static PyObject* probe(PyObject* self, PyObject* args) {
    std::vector<std::string> v;
    for (int i = 0; i < 3; ++i) v.push_back("cpp_ok");
    std::string s = v[0] + "_" + v[1] + "_" + v[2];
    return PyUnicode_FromString(s.c_str());
}

static PyMethodDef Methods[] = {
    {"probe", probe, METH_NOARGS, "probe"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef Mod = {
    PyModuleDef_HEAD_INIT, "_cppprobe", NULL, -1, Methods
};
PyMODINIT_FUNC PyInit__cppprobe(void) { return PyModule_Create(&Mod); }
'''

SETUP_PY = '''
from setuptools import setup, Extension
setup(name='_cppprobe',
      ext_modules=[Extension('_cppprobe', sources=['_cppprobe.cpp'],
                             extra_compile_args=[])])
'''


def probe_build_ext(cxx, exe, keep=False):
    section('L3  端到端：构建并 import 一个最小 C++ 扩展')
    tmp = tempfile.mkdtemp(prefix='cppprobe_ext_')
    with open(os.path.join(tmp, '_cppprobe.cpp'), 'w', encoding='utf-8') as f:
        f.write(EXT_SRC)
    with open(os.path.join(tmp, 'setup.py'), 'w', encoding='utf-8') as f:
        f.write(SETUP_PY)
    # 让 setuptools 用我们选中的编译器
    env = dict(os.environ)
    if cxx == 'cl':
        env['CXX'] = exe
    else:
        env['CXX'] = exe
    r = _run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
             cwd=tmp, timeout=600, )
    if r.returncode != 0:
        tail = ((r.stderr or '') + (r.stdout or '')).strip().splitlines()
        emit(BAD, '构建扩展（setup.py build_ext）',
             (tail[-1][:110] if tail else 'rc=%d' % r.returncode))
        if keep:
            print('   构建目录保留在：%s' % tmp)
        return False
    r2 = _run([sys.executable, '-c',
               'import _cppprobe; print(_cppprobe.probe())'], cwd=tmp)
    if r2.returncode != 0 or 'cpp_ok' not in (r2.stdout or ''):
        emit(BAD, 'import 扩展并调用', (r2.stderr or '').strip()[:110])
        return False
    emit(OK, '构建 + import + 调用扩展', (r2.stdout or '').strip()[:60])
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print('   构建目录保留在：%s' % tmp)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--keep', action='store_true', help='保留临时构建目录')
    args = ap.parse_args()

    print('python : %s' % sys.version.split()[0])
    print('prefix : %s' % sys.prefix)
    print('platform: %s' % sys.platform)

    cxx = probe_compiler()
    probe_python_headers()
    if cxx:
        probe_build_ext(cxx[0], cxx[1], keep=args.keep)

    section('结论')
    if _fail == 0:
        print('✅ 可以用 C++ 加速：编译器、Python 头文件、扩展构建与 import 全部通过。')
        print('   下一步建议（按本仓实际需要）：')
        print('   1) 先跑 scripts/bench_v7_features.py 拿组件分解 —— 确认'
              '梯子/面积/gather 各占多少；')
        print('      若 gather（磁盘随机读）占大头，C++ 帮不上忙；')
        print('   2) 只把「梯子搜索 + Benson 面积迭代」这两块搬到 C++，'
              '并配一个')
        print('      「与现 Python 实现逐位对拍」的夹具（数值必须逐位一致，'
              '否则导出与官方对拍会漂）；')
        print('   3) 注意 A 段「预算特征」路线（stdata_to_npz.py）不需要 C++，'
              '收益更大、零数值风险 —— 值得先比。')
        return 0
    print('❌ 不能用（或不完整）：见上面 FAIL 的那几层。')
    print('   常见修法：')
    print('   · Debian/Ubuntu: apt-get install -y build-essential python3-dev')
    print('   · CentOS/RHEL  : yum groupinstall "Development Tools" && '
          'yum install python3-devel')
    print('   · 只想写 Cython 扩展：pip install cython（仍需编译器 + Python.h）')
    print('   · 完全不想碰工具链：走「预算特征」路线（不需要 C++）。')
    return 1


if __name__ == '__main__':
    sys.exit(main())
