# -*- coding: utf-8 -*-
"""C++ extension loader for spatial_channels_v7.

Build-on-first-import: if the compiled extension is missing, build it via
``setup.py build_ext --inplace`` in a subprocess.  On failure, log a warning
and fall back to the Python reference implementation (imported from
``feature_v7``).

The module-level ``_HAVE_CPP`` flag is set once at import time; all callers
below use it to decide which implementation to dispatch to — no per-call
try/except.
"""
import os
import sys
import warnings
import subprocess
import pathlib
import importlib

# Platform suffix for the compiled extension
EXT_SUFFIX = '.pyd' if os.name == 'nt' else '.so'

# Extension module name (what we'll compile to)
EXT_NAME = '_feature_v7_cpp'

# Path to the extension .so/.pyd file (next to this module)
_EXT_PATH = pathlib.Path(__file__).with_suffix(EXT_SUFFIX)


def _maybe_build() -> None:
    """Compile the C++ extension in a subprocess.

    Returns None on success, raises on failure.
    CXX env var controls the compiler; defaults to 'g++' on Linux, cl on Windows.
    """
    cmd = [sys.executable, 'setup.py', 'build_ext', '--inplace']
    env = dict(os.environ)
    if os.name == 'nt':
        env['CXX'] = 'cl'
    else:
        env['CXX'] = env.get('CXX', 'g++')
    r = subprocess.run(cmd, cwd=str(pathlib.Path(__file__).parent.parent.parent),
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'build_ext failed (rc={r.returncode}):\n{r.stderr or r.stdout}')


def _load_cpp():
    """Try to import the C++ extension. Build it first if not present."""
    if _EXT_PATH.exists():
        try:
            mod = importlib.import_module(f'src.data.{EXT_NAME}')
            return hasattr(mod, 'spatial_spatial')
        except Exception:
            pass
    # Not built yet — try building
    try:
        _maybe_build()
        mod = importlib.import_module(f'src.data.{EXT_NAME}')
        return hasattr(mod, 'spatial_spatial')
    except Exception as e:
        warnings.warn(f'feature_v7 C++ 扩展不可用，退回 Python：{e}')
        return False


_HAVE_CPP = _load_cpp()

if _HAVE_CPP:
    _cpp_mod = importlib.import_module(f'src.data.{EXT_NAME}')
    spatial_channels_v7 = _cpp_mod.spatial_spatial
else:
    # Fallback: import from feature_v7.py
    from src.data.feature_v7 import spatial_channels_v7  # noqa: E402, F401

