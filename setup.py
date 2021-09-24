"""
DeepMReye Toolbox
© Markus Frey, Matthias Nau
https://github.com/DeepMReye/DeepMReye
Licensed under LGPL-3.0 License
"""
from setuptools import setup

long_description = open('README.md').read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='deepmreye',
    version='0.1',
    install_requires=requirements,
    author='Markus Frey',
    author_email='markus.frey1@gmail.com',
    description="MR-based eye tracker without eye tracking",
    long_description=long_description,
    url='https://github.com/DeepMReye/DeepMReye/',
    license='LGPL-3.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
)