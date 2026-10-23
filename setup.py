from setuptools import setup, Extension
import pybind11
import sys

ext = Extension(
    'src.data._feature_v7_cpp',
    sources=['src/data/_feature_v7_cpp.cpp'],
    include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
    language='c++',
)
if sys.platform != 'win32':
    ext.extra_compile_args = ['-std=c++17', '-O2']
else:
    ext.extra_compile_args = ['/std:c++17']

setup(
    name='feature_v7_cpp',
    ext_modules=[ext],
    pybind11=True,
)
