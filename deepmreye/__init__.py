"""Decode eye gaze from fMRI without an eye tracker."""
import os

# h5py takes an exclusive lock by default, which fails on the shared filesystems
# this corpus lives on and blocks the labeling UI from reading a file an
# extraction worker has open.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
