[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
![py38 status](https://img.shields.io/badge/python3.8-supported-green.svg)
![Build Status](https://github.com/DeepMReye/DeepMReye/actions/workflows/main.yml/badge.svg)
[![NatNeuro Paper](https://img.shields.io/badge/DOI-10.XXXX%2FsXXXXX--XXX--XXXX--X-blue)](https://doi.org/XXX/XXX)

![Logo](media/deepmreye_logo.png)

# DeepMReye: magnetic resonance-based eye tracking using deep neural networks
This [Jupyter Notebook](./notebooks/deepmreye_example_usage.ipynb) provides a step-by-step walkthrough of the code. It includes eyeball coregistration, voxel extraction, model training and test as well as basic performance measures. Alternatively, here is a [Colab Notebook](https://colab.research.google.com/drive/1kYVyierbKdNZ3RY4_pbACtdWEw7PKQuz?usp=sharing).

This [Data Repository](https://osf.io/mrhk9/) includes exemplary data for model training and test, source data of all paper figures as well as pre-trained model weights.

Moreover, here are additional [User Recommendations](https://deepmreye.slite.com/p/channel/MUgmvViEbaATSrqt3susLZ/notes/kKdOXmLqe) as well as a [Frequently-Asked-Questions (FAQ)](https://deepmreye.slite.com/p/channel/MUgmvViEbaATSrqt3susLZ/notes/sargIAQ6t) page. If you have other questions, please reach out to us.

![deepMReye video](media/deepMReye_video.gif)

## Installation - Option 1: CPU version

### Pip installation
Install DeepMReye with a CPU version of [TensorFlow](https://www.tensorflow.org/install/) using the following command.
```
pip install git+https://github.com/DeepMReye/DeepMReye.git
```

### Anaconda / Miniconda installation

Install Anaconda or miniconda and clone this repository:
```
git clone https://github.com/DeepMReye/DeepMReye.git
cd DeepMReye
```

Create a virtual environment for DeepMReye with the following commands:
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

This CPU version runs on Windows, Mac and Linux, but it takes substantially more time to compute than the GPU version (see below). 

## Installation - Option 2: GPU version (recommended)
Install DeepMReye with a GPU version of [TensorFlow](https://www.tensorflow.org/install/) using following command. This version is substantially faster than the CPU version, but it requires CUDA and a NVIDIA GPU (not supported by Mac). The GPU version runs on Windows and Linux.
```
conda install tensorflow-gpu
```
Note that you might need to install cudnn first (conda install -c conda-forge cudnn).

## Installation - Option 3: Colab

We provide a Colab notebook showcasing model training and evaluation on a GPU provided by Google Colab. To use your own data, preprocess your data locally and upload only the extracted eyeball voxels. This saves space and avoids data privacy issues. See the [Jupyter notebook](./notebooks/deepmreye_example_usage.ipynb) for the preprocessing and eyeball-extraction code.

[![Model Training & Evaluation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kYVyierbKdNZ3RY4_pbACtdWEw7PKQuz?usp=sharing)

![Colab Walkthrough](media/colab_walkthrough.gif)

## System Requirements

## Hardware requirements

The GPU version of DeepMReye requires a NVIDIA GPU.

## Software requirements
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