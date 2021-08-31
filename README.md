[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
![py36 status](https://img.shields.io/badge/python3.6-supported-green.svg)
<!-- ![Build Status](https://github.com/DeepMReye/DeepMReye/workflows/build/badge.svg) -->

![Logo](media/deepmreye_logo.png)

## Example Usage
See [here](./notebooks/deepmreye_example_usage.ipynb) for a full walkthrough of how to use DeepMReye to preprocess your data, run the model training and obtain gaze labels.

## Installation
# Pip install
Currently unsopported until Ants pip install is fixed.

# Anaconda / Miniconda installation

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
pip install antspyx
```
If installation of ANTs fails try to manually install it via:
```
git clone https://github.com/ANTsX/ANTsPy
cd ANTsPy
python3 setup.py install
```

To import the DeepMReye module use this at the top of your script / notebook:
```python
import sys
sys.path.insert(0, "/your/path/to/DeepMReye")
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