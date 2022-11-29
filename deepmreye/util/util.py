""" Additional methods which did not earn its own space in the main methods. Maybe because they are more general and made for higher purposes. """
import numpy as np
import pandas as pd
from math import atan2
from scipy import ndimage
from sklearn.metrics import r2_score
import warnings
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf


def augment_input(X, rotation=0, shift=0, zoom=0):
    """ Augments 3D images
    Inputs: 
    - X : Batch of 3D images
    - rotation : Rotation in degree
    - shift : Shift in pixels
    - zoom : Zoom in factor

    Outputs:
    - X : Augmented batch of 3D images
    """
    def scaleit(image, factor):
        with warnings.catch_warnings():  # Weird bug, should be fixed in next scipy version
            warnings.simplefilter("ignore")
            height, width, tiefe, depth = image.shape
            zheight = int(np.round(factor * height))
            zwidth = int(np.round(factor * width))
            ztiefe = int(np.round(factor * tiefe))
            zdepth = depth

            if factor < 1.0:
                newimg = np.zeros_like(image)
                row = (height - zheight) // 2
                col = (width - zwidth) // 2
                layer = (tiefe - ztiefe) // 2
                newimg[row:row+zheight, col:col+zwidth, layer:layer+ztiefe, :] = ndimage.zoom(image, (float(factor), float(
                    factor), float(factor), 1.0), order=0, mode='constant', cval=0, prefilter=False)[0:zheight, 0:zwidth, 0:ztiefe, :]

                return newimg

            elif factor > 1.0:
                row = (height - zheight) // 2
                col = (width - zwidth) // 2
                layer = (tiefe - ztiefe) // 2

                newimg = ndimage.zoom(image, (float(factor), float(factor), float(
                    factor), 1.0),  order=0, mode='constant', cval=0, prefilter=False)

                extrah = (newimg.shape[0] - height) // 2
                extraw = (newimg.shape[1] - width) // 2
                extrad = (newimg.shape[2] - tiefe) // 2
                newimg = newimg[extrah:extrah+height,
                                extraw:extraw+width, extrad:extrad+tiefe, :]

                return newimg

            else:
                return image

    def rotate_img(image1, max_angle, axis=[0, 1, 2]):
        if 0 in axis:
            # rotate along x-axis
            angle = np.random.uniform(-max_angle[0], max_angle[0])
            image1 = ndimage.rotate(image1, angle, mode='constant', cval=0, axes=(
                1, 2), reshape=False, prefilter=False, order=0)

        if 1 in axis:
            # rotate along y-axis
            angle = np.random.uniform(-max_angle[1], max_angle[1])
            image1 = ndimage.rotate(image1, angle, mode='constant', cval=0, axes=(
                0, 2), reshape=False, prefilter=False, order=0)

        if 2 in axis:
            # rotate along z-axis
            angle = np.random.uniform(-max_angle[2], max_angle[2])
            image1 = ndimage.rotate(image1, angle, mode='constant', cval=0, axes=(
                0, 1), reshape=False, prefilter=False, order=0)
        return image1
    # Rotate
    X = np.array([rotate_img(x, rotation, axis=[0, 1, 2]) for x in X])

    # Then shift
    X = np.array([ndimage.shift(x, shift=(np.random.randint(-shift, shift), np.random.randint(-shift, shift),
                                          np.random.randint(-shift, shift), 0), order=0, mode='constant', cval=0, prefilter=False) for x in X])
    # Then zoom
    X = np.array([scaleit(x, np.random.uniform(1-zoom, 1+zoom)) for x in X])

    return X


def mish(x):
    return x * K.tanh(K.softplus(x))


def step_decay_schedule(initial_lr=1e-4, decay_factor=0.9, num_epochs=50):
    def schedule(epoch):
        return initial_lr * (1 - epoch / num_epochs) ** decay_factor
    return LearningRateScheduler(schedule)


def euclidean_distance(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_true - y_pred), axis=-1))


def angle_between_points(y_true, y_pred):
    dx = y_pred[:, 0] - y_true[:, 0]
    dy = y_pred[:, 1] - y_true[:, 1]
    rads = np.array([atan2(dyy, dxx) for dyy, dxx in zip(dy, dx)])
    rads += np.pi
    return rads


def get_model_scores(real_y, pred_y, euc_pred, **args):
    try:
        (agg_scores, subtr_scores) = quantify_predictions(real_y, pred_y, euc_pred, percentile_cut=None)
        (agg_scores_pct, subtr_scores_pct) = quantify_predictions(real_y, pred_y, euc_pred, **args)
    except ValueError:
        # Participant has only NaNs or no data, return empty dataframes
        agg_scores, subtr_scores, agg_scores_pct, subtr_scores_pct = np.random.rand(9)*np.NaN, np.random.rand(9)*np.NaN, np.random.rand(9)*np.NaN, np.random.rand(9)*np.NaN # 9 return parameters
    df_scores = pd.DataFrame([agg_scores, subtr_scores, agg_scores_pct, subtr_scores_pct], index=['Default', 'Default subTR', 'Refined', 'Refined subTR'])
    df_scores.columns = pd.MultiIndex.from_tuples((('Pearson', 'X'), ('Pearson', 'Y'), ('Pearson', 'Mean'), ('R^2-Score', 'X'), ('R^2-Score', 'Y'), ('R^2-Score', 'Mean'),
                                                   ('Eucl. Error', 'Mean'), ('Eucl. Error', 'Median'), ('Eucl. Error', 'Std')))
    return df_scores


def quantify_predictions(y_true, y_pred, euc_pred, subtr_functor=np.median, percentile_cut=None):
    # Take care of NaN in XYs
    nan_indices = np.any(np.isnan(y_true), axis=(1, 2))
    y_true = y_true[~nan_indices, ...]
    y_pred = y_pred[~nan_indices, ...]
    euc_pred = euc_pred[~nan_indices, ...]

    if y_true.size < 1:
        raise ValueError('Participant has no ground truth data, nothing to quantify')

    # Aggregate across subTR values
    y_true_agg = subtr_functor(y_true, axis=1)
    y_pred_agg = subtr_functor(y_pred, axis=1)
    euc_pred_agg = subtr_functor(euc_pred, axis=1)

    # Use flattened array for subTR comparison
    y_true_flat = np.reshape(y_true, (y_true.shape[0] * y_true.shape[1], -1))
    y_pred_flat = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1], -1))
    euc_pred_flat = np.reshape(euc_pred, (euc_pred.shape[0] * euc_pred.shape[1], -1))

    agg_scores = calculate_scores(y_true_agg, y_pred_agg, euc_pred_agg, percentile_cut=percentile_cut)
    flat_scores = calculate_scores(y_true_flat, y_pred_flat, euc_pred_flat[..., 0], percentile_cut=percentile_cut)

    return (agg_scores, flat_scores)


def calculate_scores(y_true, y_pred, euc_pred, percentile_cut=None):
    if percentile_cut is not None:
        bad_indices = euc_pred > np.percentile(euc_pred, percentile_cut)
        y_true = y_true[~bad_indices, ...]
        y_pred = y_pred[~bad_indices, ...]
        euc_pred = euc_pred[~bad_indices, ...]

    pearson_x = np.corrcoef(y_true[:, 0], y_pred[:, 0])[0, 1]
    pearson_y = np.corrcoef(y_true[:, 1], y_pred[:, 1])[0, 1]
    pearson_mean = (pearson_x + pearson_y) / 2
    r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_true[:, 1], y_pred[:, 1])
    r2_mean = (r2_x + r2_y) / 2

    euc = euclidean_distance(y_true, y_pred)

    return (pearson_x, pearson_y, pearson_mean, r2_x, r2_y, r2_mean, np.mean(euc), np.median(euc), np.std(euc))


def smooth_signal(signal, N):
    """
    Simple smoothing by convolving a filter with 1/N.

    Parameters
    ----------
    signal : array_like
        Signal to be smoothed
    N : int
        smoothing_factor

    Returns
    -------
    signal : array_like
            Smoothed signal
    """
    # Preprocess edges
    signal = np.concatenate([signal[0:N], signal, signal[-N:]])
    # Convolve
    signal = np.convolve(signal, np.ones((N,))/N, mode='same')
    # Postprocess edges
    signal = signal[N:-N]

    return signal


class color:
    # From https://stackoverflow.com/questions/8924173/how-do-i-print-bold-text-in-python
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# --------------------------------------------------------------------------------
# ------------------------Command Line Options------------------------------------
# --------------------------------------------------------------------------------
class Arg:
    def __init__(self, *args, **kwargs):
        self.cli = args
        self.kwargs = kwargs

CLI_OPTIONS = {
    "verbosity": Arg(
        '-v', '--verbose',
        help='Verbosity level',
        default=0,
    ),
    "gpu_id": Arg(
        '--gpu_id',
        help="Which GPU to use for training",
        metavar='gpu_id',
        default=''
    ), 
    "dataset_path": Arg(
        '--dataset_path',
        help="Path to dataset. Base folder is also dataset name.",
        metavar='dataset_path',
        default='./'
    ), 
    "weights_path": Arg(
        '--weights_path',
        help="Path to where weights should be stored.",
        metavar='weights_path',
        default='./weights/'
    ), 
    "datasets": Arg(
        '--datasets',
        help="If given only train subset of all datasets.",
        metavar='datasets',
        default=None)}
