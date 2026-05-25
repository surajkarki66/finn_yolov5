from setuptools import setup, Extension
import pybind11
import numpy as np

ext_modules = [
    Extension(
        'cpp_detect',
        ['detect.cpp', 'motion_cpp.cpp'],
        include_dirs=[pybind11.get_include(), np.get_include()],
        language='c++',
    )
]

setup(
    name='cpp_detect',
    version='0.1',
    author='auto-generated',
    description='C++ optimized Detect forward for TinyYOLOv3',
    ext_modules=ext_modules,
    zip_safe=False,
)
