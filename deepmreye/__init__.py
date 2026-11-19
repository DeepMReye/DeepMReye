import os

# Keras 3 (default since tensorflow 2.16) is not supported yet:
# use legacy Keras 2 provided by the tf-keras package.
# This must be set before tensorflow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from . import analyse, architecture, preprocess, train, util  # noqa: E402
