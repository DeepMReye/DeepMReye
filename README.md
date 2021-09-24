[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
![py36 status](https://img.shields.io/badge/python3.6-supported-green.svg)
![Build Status](https://github.com/DeepMReye/DeepMReye/actions/workflows/main.yml/badge.svg)

![Logo](media/deepmreye_logo.png)

# DeepMReye: Magnetic resonance-based eye tracking using deep neural networks
[Click here for a walkthrough of the code](./notebooks/deepmreye_example_usage.ipynb), including eyeball coregistration and voxel extraction, model training and test and basic performance measures.

[Click here for data](https://osf.io/mrhk9/), including exemplary data for model training and test, source data of all paper figures as well as pre-trained model weights.

[Click here for online documentation](https://deepmreye.slite.com/p/channel/MUgmvViEbaATSrqt3susLZ), including user recommendations and Frequently-Asked-Questions (FAQ).

# Installation

## Pip installation
Install DeepMReye with the following command:
```
pip install git+https://github.com/DeepMReye/DeepMReye.git
```
Note that this installs a CPU version of tensorflow. See below for the GPU install. 

## Anaconda / Miniconda installation

Install Anaconda or miniconda and clone this repository:
```
git clone https://github.com/DeepMReye/DeepMReye.git
cd DeepMReye
```

Install a virtual environment for DeepMReye with the following commands:
```
conda create --name deepmreye python=3.7
conda install --file requirements.txt
conda activate deepmreye
```
If installation of [ANTsPy](https://github.com/ANTsX/ANTsPy) fails try to manually install it via:
```
git clone https://github.com/ANTsX/ANTsPy
cd ANTsPy
python3 setup.py install
```

## GPU Install
By default the CPU version of [tensorflow](https://www.tensorflow.org/install/) is installed, if you want to train on GPU (recommended) install tensorflow via:
```
conda install tensorflow-gpu
```
Note that you might need to install cudnn first (conda install -c conda-forge cudnn).

## System Requirements

### Hardware requirements

### Software requirements
The following python dependencies are being automatically installed when installing DeepMReye (specified in requirements.txt):
```
tensorflow-gpu (2.2.0)
numpy (1.19.1)
pandas (1.0.5)
matplotlib (3.2.2)
scipy (1.5.0)
ipython (7.13.0)
plotly (4.14.3)
```
Version in parentheses indicate the ones used for testing the framework. Its extensively tested on Linux 16.04 but should run on all OS (Windows, Mac, Linux) supporting a Python version >3.6 and pip. It is recommended to install the framework and dependencies in a virtual environment (e.g. conda). 