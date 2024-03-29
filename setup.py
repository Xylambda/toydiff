from setuptools import setup, find_packages
from os import path
import versioneer

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='toydiff',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Tensor automatic differentiation and neural networks library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Xylambda/toydiff',
    author='Alejandro Pérez-Sanjuán',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3 :: Only',
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=("tests*", "profiling*", )),
    python_requires='>=3.8, <4',
    install_requires=[
        "numpy",
        "scipy",
        #"pyfma",  # fused multiply-add
    ],
    extras_require={
        "test": [
            "torch",
            "pytest",
        ],
        "profile": [
            "snakeviz",
            "perfplot",
        ]
    }
)