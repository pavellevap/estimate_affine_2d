from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np

extension = Extension('affine_estimation', sources=['ransac.pyx'], language='c++', 
                      extra_compile_args=['--std=c++11'], extra_link_args=['--std=c++11'])
setup(name='affine_estimation', ext_modules=cythonize(extension, language_level="3"))
