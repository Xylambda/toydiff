from setuptools import setup, find_packages
import versioneer

setup(
    name='Toydiff',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A simple package to perform automatic differentiation on \
        scalar variables',
    author='Alejandro Pérez',
    install_requires=[
        "numpy",
        "jupyter",
        "matplotlib"
    ],
)