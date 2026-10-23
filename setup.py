from setuptools import setup, Extension
import pybind11

ext = Extension(
    'src.data._feature_v7_cpp',
    sources=['src/data/_feature_v7_cpp.cpp'],
    include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
    extra_compile_args=['-std=c++17'],
    language='c++',
)

setup(
    name='feature_v7_cpp',
    ext_modules=[ext],
    pybind11=True,
)
