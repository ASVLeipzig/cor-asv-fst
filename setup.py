# -*- coding: utf-8 -*-
"""
Installs:
    - cor-asv-fst-train
"""
import codecs

from setuptools import Extension, setup, find_packages
from distutils.sysconfig import get_config_vars
from Cython.Distutils import build_ext

cfg_vars = get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "-std=c++11")
install_requires = open('requirements.txt').read().split('\n')

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name='ocrd_cor_asv_fst',
    version='0.1.0',
    description='OCR post-correction with error/lexicon Finite State '
                'Transducers and character-level LSTMs',
    long_description=README,
    author='Maciej Sumalvico, Robert Sachunsky',
    author_email='sumalvico@informatik.uni-leipzig.de, '
                 'sachunsky@informatik.uni-leipzig.de',
    url='https://github.com/ASVLeipzig/cor-asv-fst',
    license='Apache License 2.0',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=install_requires,
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'cor-asv-fst-train=scripts.run:cli',
            'ocrd-cor-asv-fst-process=ocrd_cor_asv_fst.wrapper.cli:ocrd_cor_asv_fst',
        ]
    },
    ext_modules=[
        Extension(
            "ocrd_cor_asv_fst.lib.extensions.composition",
            sources=[
                "ocrd_cor_asv_fst/lib/extensions/composition.pyx",
                "ocrd_cor_asv_fst/lib/extensions/composition_cpp.cpp"],
            libraries=["fst", "dl"],
            language="c++")
        ],
    cmdclass = {'build_ext': build_ext},
)
