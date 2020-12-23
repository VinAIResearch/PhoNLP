#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import re
from setuptools import setup, find_packages

PACKAGE_NAME = "phonlp"

with io.open("%s/__init__.py" % PACKAGE_NAME, "rt", encoding="utf8") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',

]

setup(
    name='phonlp',
    version=version,
    description='PhoNLP: A joint multi-task learning model for Vietnamese part-of-speech tagging, named entity recognition and dependency parsing',
    long_description='',
    author='Linh The Nguyen and Dat Quoc Nguyen',
    author_email='v.linhnt140@vinai.io',
    maintainer='linhnt',
    maintainer_email='v.linhnt140@vinai.io',
    classifiers=classifiers,
    keyword='phonlp',
    packages=find_packages(),
    install_requires=['transformers', 'torch>=1.4.0', 'numpy', 'gdown', 'pytest', 'tqdm'],
    python_requires='>=3.6'
)
