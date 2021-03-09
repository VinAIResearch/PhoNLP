#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import re
from os import path

from setuptools import find_packages, setup


PACKAGE_NAME = "phonlp"
here = path.abspath(path.dirname(__file__))

with io.open("%s/__init__.py" % PACKAGE_NAME, "rt", encoding="utf8") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.7",
]

setup(
    name="phonlp",
    version=version,
    description="PhoNLP: A joint multi-task learning model for Vietnamese part-of-speech tagging, named entity recognition and dependency parsing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/VinAIResearch/PhoNLP',
    author="Linh The Nguyen and Dat Quoc Nguyen",
    author_email="v.linhnt140@vinai.io",
    maintainer="linhnt",
    maintainer_email="v.linhnt140@vinai.io",
    classifiers=classifiers,
    keyword="phonlp",
    packages=find_packages(),
    install_requires=["transformers>=3.2.0", "torch>=1.4.0", "numpy", "gdown>=3.12.2", "pytest", "tqdm"],
    python_requires=">=3.6",
)
