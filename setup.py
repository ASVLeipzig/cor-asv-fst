# -*- coding: utf-8 -*-
"""
Installs:
    - cor-asv-fst-train
    - cor-asv-fst-process
    - cor-asv-fst-evaluate
    - ocrd-cor-asv-fst-process
"""
import codecs

from setuptools import setup, find_packages

install_requires = open('requirements.txt').read().split('\n')

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name='ocrd_cor_asv_fst',
    version='0.2.0',
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
    test_suite='tests',
    entry_points={
        'console_scripts': [
            'cor-asv-fst-train=ocrd_cor_asv_fst.scripts.train:main',
            'cor-asv-fst-process=ocrd_cor_asv_fst.scripts.process:main',
            'cor-asv-fst-evaluate=ocrd_cor_asv_fst.scripts.evaluate:main',
            'ocrd-cor-asv-fst-process=ocrd_cor_asv_fst.wrapper.cli:ocrd_cor_asv_fst',
        ]
    }
)
