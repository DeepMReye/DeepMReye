[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
![py38 status](https://img.shields.io/badge/python3.8-supported-green.svg)
![Build Status](https://github.com/DeepMReye/DeepMReye/actions/workflows/main.yml/badge.svg)
[![NatNeuro Paper](https://img.shields.io/badge/DOI-10.1038%2Fs41593--021--00947--w-blue)](https://doi.org/10.1038/s41593-021-00947-w)
![Docker Pulls](https://img.shields.io/docker/pulls/deepmreye/deepmreye)
![Logo](media/deepmreye_logo.png)

# DeepMReye: magnetic resonance-based eye tracking using deep neural networks
This [Jupyter Notebook](./notebooks/deepmreye_example_usage.ipynb) provides a step-by-step walkthrough of the code. It includes eyeball coregistration, voxel extraction, model training and test as well as basic performance measures. Alternatively, here is a [Colab Notebook](https://colab.research.google.com/drive/1kYVyierbKdNZ3RY4_pbACtdWEw7PKQuz?usp=sharing).

This [Data Repository](https://osf.io/mrhk9/) includes exemplary data for model training and test, source data of all paper figures as well as pre-trained model weights.

Moreover, here are additional [User Recommendations](https://deepmreye.slite.com/p/channel/MUgmvViEbaATSrqt3susLZ/notes/kKdOXmLqe) as well as a [Frequently-Asked-Questions (FAQ)](https://deepmreye.slite.com/p/channel/MUgmvViEbaATSrqt3susLZ/notes/sargIAQ6t) page. If you have other questions, please reach out to us.

![deepMReye video](media/deepMReye_video.gif)

## Installation

### Option 1: Pip install

#### Pip installation
Install DeepMReye with a CPU/GPU version of [TensorFlow](https://www.tensorflow.org/install/) using the following command.
```
pip install deepmreye
```

#### Anaconda / Miniconda installation

To encapsulate DeepMReye in a virtual environment install with the following commands:
```
conda create --name deepmreye python=3.9
conda activate deepmreye
pip install deepmreye
```
For GPU support, follow tensorflow [install](https://www.tensorflow.org/install/pip) instruction, e.g.:
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
If installation of [ANTsPy](https://github.com/ANTsX/ANTsPy) fails try to manually install it via:
```
git clone https://github.com/ANTsX/ANTsPy
cd ANTsPy
pip install CMake
python3 setup.py install
```

### Option 2: Colab

We provide a [Colab Notebook](https://colab.research.google.com/drive/1kYVyierbKdNZ3RY4_pbACtdWEw7PKQuz?usp=sharing) showcasing model training and evaluation on a GPU provided by Google Colab. To use your own data, preprocess your data locally and upload only the extracted eyeball voxels. This saves space and avoids data privacy issues. See the [Jupyter Notebook](./notebooks/deepmreye_example_usage.ipynb) for the preprocessing and eyeball-extraction code.

[![Model Training & Evaluation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kYVyierbKdNZ3RY4_pbACtdWEw7PKQuz?usp=sharing)

![Colab Walkthrough](media/colab_walkthrough.gif)

### Option 3: Docker

Pull the image from docker hub.

```bash
docker pull deepmreye/deepmreye
```

Use deepMReye in a docker container via jupyterlab:

```bash
mkdir -p $PWD/notebooks
docker run -it --rm \
    --publish 8888:8888 \
    --volume $PWD/notebooks:/home/neuro/notebooks \
    deepmreye/deepmreye:latest \
        jupyter-lab --no-browser --ip 0.0.0.0
```

### Option 4: Streamlit (Browser version)

If you would just like to try it out and decode gaze coordinates in your data using a pretrained model, the easiest way is using our streamlit app. 

Running the following commands will open a browser window that allows you to upload your data (.nii or .nii.gz) and then download the corresponding gaze coordinates shortly after. 
Please read the [User Recommendations](https://deepmreye.slite.com/p/channel/MUgmvViEbaATSrqt3susLZ/notes/kKdOXmLqe) before using the pretrained models.

```bash
git clone https://github.com/DeepMReye/DeepMReye.git
cd DeepMReye
pip install .
pip install streamlit
streamlit run streamlit/streamlit.py
```

### Data formats
The <u>**fMRI data**</u> should be organized in 4D NIFTI files (.nii), containing the realigned 3D images acquired over time. The pipeline then extracts the eyeball voxels automatically and saves them as Python Pickle files, which serve as model input. For model training, you additionally need <u>**training labels**</u>, a numpy array containing 10 gaze coordinates per functional volume. These gaze coordinates can either be camera-based eye-tracking labels or the coordinates of a fixation target, and many file formats can be easily read (e.g. .npy, .npz, .mat, .csv etc.).

Please see our [FAQ](https://deepmreye.slite.com/p/channel/MUgmvViEbaATSrqt3susLZ/notes/sargIAQ6t) page for more details on data formats and preprocessing.

## Hardware requirements

The GPU version of DeepMReye requires a NVIDIA GPU.

## Software requirements
The following python dependencies are being automatically installed when installing DeepMReye (specified in setup.cfg):
```
tensorflow-gpu (2.2.0)
numpy (1.19.1)
pandas (1.0.5)
matplotlib (3.2.2)
scipy (1.5.0)
ipython (7.13.0)
plotly (4.14.3)
```
Version in parentheses indicate the ones used for testing the framework. Its extensively tested on Linux 16.04 but should run on all OS (Windows, Mac, Linux) supporting a Python version >3.8 and pip. It is recommended to install the framework and dependencies in a virtual environment (e.g. conda). 

## BIDS app
If you would like to run a pretrained version of DeepMReye on datasets that follow the Brain Imaging Data Structure (BIDS) format, check out [bidsMReye](https://pypi.org/project/bidsmreye/): a wrapper for DeepMReye for BIDS datasets (incl. [fMRIprep](https://fmriprep.org/en/stable/) outputs) developed by [Remi Gau](https://remi-gau.github.io/). Please read the paper and [User Recommendations](https://deepmreye.slite.com/p/channel/MUgmvViEbaATSrqt3susLZ/notes/kKdOXmLqe) before you use it.

## Correspondence
If you have questions, comments or inquiries, please check out the online [User documention](https://deepmreye.slite.com/api/s/channel/MUgmvViEbaATSrqt3susLZ/DeepMReye%3A%20Documentation) and reach out to us: markus.frey[at]ntnu.no & matthias.nau[at]ntnu.no
