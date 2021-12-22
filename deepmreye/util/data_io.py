import os
import urllib
import numpy as np
from scipy.io import loadmat


# --------------------------------------------------------------------------------
# --------------------------IO-NAU------------------------------------------------
# --------------------------------------------------------------------------------
def get_subject_labels(subject_string, mat_data):
    for subject in mat_data:
        if str(subject['subID']) == subject_string:
            all_runs = subject['TR_xy'].tolist()
            return all_runs
    return -1


def get_all_subject_labels(subject_string, mat_data, num_downsampled=10, use_real=False):
    """
    For models with multiple outputs we want to estimate the sub-TR XY 
    Inputs: 
        - subject_string : Subject identified
        - mat_data : Data to subject logs in mat format
        - num_downsampled : How many sub-TR XY are left in the output
    """
    real_et = None
    for subject in mat_data:
        if str(subject['subID']) == subject_string:
            try:
                if use_real:
                    all_runs = subject['TR_xy_samples_ET'].tolist()
                    real_et = True
                else:
                    raise ValueError('Hack')
            except ValueError:
                print('Subject {} has no real eye tracking, use XY of moving dot.'.format(subject_string))
                all_runs = subject['TR_xy_samples'].tolist()
                real_et = False
            subject_xy = list()
            for run in all_runs:
                run_xy = list()
                for ft in run:
                    if len(ft.shape) > 1:
                        ft_xy = ft[:, np.linspace(0, ft.shape[1] - 1, num_downsampled, dtype=int)]
                        run_xy.append(ft_xy.transpose())
                    else:
                        run_xy.append(np.full([num_downsampled, 2], np.nan))
                subject_xy.append(np.array(run_xy))
            subject_xy = np.array(subject_xy)
            return (subject_xy, real_et)
    return (-1, -1)

# --------------------------------------------------------------------------------
# --------------------------IO-IGN------------------------------------------------
# --------------------------------------------------------------------------------


def get_all_subject_labels_ign(subject_string, num_downsampled=10):
    """
    For models with multiple outputs we want to estimate the sub-TR XY 
    Inputs: 
        - subject_string : Subject identified
        - mat_data : Data to subject logs in mat format
        - num_downsampled : How many sub-TR XY are left in the output
    """
    subj_data = np.load(subject_string)
    np.testing.assert_(np.unique(np.diff(np.where(subj_data[:, 0] == 1)[0]))[0] == 510)

    all_labels, all_runs = [], []
    for idx, l in enumerate(subj_data):
        if l[1] == 0:
            if all_labels:
                all_runs.append(np.array(all_labels))
            all_labels = []
        bln_tr = int(l[0])
        if bln_tr:
            if (idx + 510) >= subj_data.shape[0]:
                this_label = subj_data[np.linspace(idx, subj_data.shape[0] - 1, num_downsampled, dtype=int), 2:4]
            else:
                this_label = subj_data[np.linspace(idx, idx + 510, num_downsampled, dtype=int), 2:4]
            all_labels.append(this_label)
    all_runs.append(np.array(all_labels))
    all_runs = np.array(all_runs)

    return all_runs

# --------------------------------------------------------------------------------
# --------------------------IO-BMD------------------------------------------------
# --------------------------------------------------------------------------------


def get_all_subject_labels_bmd(subject_string, run_idx, num_downsampled=10, real_et=False):
    """
    For models with multiple outputs we want to estimate the sub-TR XY 
    Inputs: 
        - subject_string : Subject identified
        - run_idx : Index for run
        - num_downsampled : How many sub-TR XY are left in the output
    """
    mat_data = loadmat(subject_string, mat_dtype=True)
    if real_et:
        mat_data = mat_data['XY']['samples_ET'][0, 0]
    else:
        mat_data = mat_data['XY']['samples'][0, 0]
    if run_idx < mat_data.shape[1]:
        if mat_data[0, run_idx].shape[0] == 0:
            return np.array([])
        this_run = mat_data[0, run_idx][0, :]
        all_subtr = np.array([x[np.linspace(0, len(x) - 1, num_downsampled, dtype=int), :]
                              if x.size > 0 else np.zeros((num_downsampled, 2)) * np.nan for x in this_run])
        return all_subtr
    return np.array([])

# --------------------------------------------------------------------------------
# --------------------------IO-MMD------------------------------------------------
# --------------------------------------------------------------------------------


def get_all_subject_labels_mmd(subject_string, run_idx, num_downsampled=10):
    """
    For models with multiple outputs we want to estimate the sub-TR XY 
    Inputs: 
        - subject_string : Subject identified
        - run_idx : Index for run
        - num_downsampled : How many sub-TR XY are left in the output
    """
    mat_data = loadmat(subject_string, mat_dtype=True)
    mat_data = mat_data['XY']['samples_ET'][0, 0]
    if run_idx < mat_data.shape[1]:
        this_run = mat_data[0, run_idx][0, :]
        all_subtr = np.array([x[np.linspace(0, len(x) - 1, num_downsampled, dtype=int), :]
                              if x.size > 0 else np.zeros((num_downsampled, 2)) * np.nan for x in this_run])
        return all_subtr
    return np.array([])


# --------------------------------------------------------------------------------
# --------------------------IO-MASKS----------------------------------------------
# --------------------------------------------------------------------------------


def download_mask(data_path, remote_path='https://github.com/DeepMReye/DeepMReye/blob/main/deepmreye/masks/'):
    mask_name = os.path.basename(data_path)
    mask_remote = remote_path + '{}?raw=true'.format(mask_name)
    try:
        (f, m) = urllib.request.urlretrieve(mask_remote, data_path)
    except urllib.error.URLError as e:
        raise RuntimeError("Failed to download '{}'. '{}'".format(mask_remote, e.reason))