[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
![py36 status](https://img.shields.io/badge/python3.6-supported-green.svg)
<!-- ![Build Status](https://github.com/DeepMReye/DeepMReye/workflows/build/badge.svg) -->

![Logo](media/deepmreye_logo.png)

# Example Usage
[Click here for a walkthrough of the code](./notebooks/deepmreye_example_usage.ipynb), including eyeball coregistration and voxel extraction, model training and test and basic performance measures.

[Click here for user recommendations and Frequently-Asked-Questions (FAQ)](https://deepmreye.slite.com/p/channel/MUgmvViEbaATSrqt3susLZ).

# Installation
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
For training and evaluating models you need [tensorflow](https://www.tensorflow.org/install/) which you can install via:
```
conda install tensorflow-gpu
or
conda install tensorflow
```
Use the first command if your system has access to a GPU. Note that you might need to install cudnn first (conda install -c conda-forge cudnn).

To finally import the DeepMReye module use this at the top of your script / notebook:
```python
import sys
sys.path.insert(0, "/your/path/to/DeepMReye")
import deepmreye
```

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