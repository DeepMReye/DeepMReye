#!/usr/bin/env python
"""
DeepMReye Toolbox
© Markus Frey, Matthias Nau
https://github.com/DeepMReye/DeepMReye
Licensed under LGPL-3.0 License
"""
import sys

from setuptools import setup


SETUP_REQUIRES = ['setuptools >= 30.3.0']
SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []

if __name__ == "__main__":
    setup(setup_requires=SETUP_REQUIRES)
