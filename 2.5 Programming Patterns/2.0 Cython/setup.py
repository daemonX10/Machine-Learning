from setuptools import setup
from Cython.Build import cythonize  # Note the lowercase 'c'

setup(
    ext_modules = cythonize('cython_cy.pyx')  # Note the lowercase 'c'
)