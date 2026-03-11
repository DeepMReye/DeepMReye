try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from deepmreye.config import DeepMReyeConfig
from deepmreye.preprocess import run_participant, download_mask, get_masks
