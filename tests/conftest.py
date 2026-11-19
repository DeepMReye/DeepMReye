import urllib.request
from pathlib import Path

import pytest

PRETRAINED_WEIGHTS_URL = "https://osf.io/download/mr87v/"


@pytest.fixture
def path_to_masks():
    path = Path(__file__).parents[1] / "deepmreye" / "masks"
    return path


@pytest.fixture
def path_to_testdata():
    path = Path(__file__).parents[0] / "data"
    return path


@pytest.fixture(scope="session")
def path_to_pretrained_weights():
    """Pretrained weights (datasets 1 to 6) in the legacy (keras 2) h5 format.

    Reuse the file downloaded for the notebooks if it exists, otherwise download it.
    """
    root = Path(__file__).parents[1]
    notebook_weights = root / "notebooks" / "model_weights" / "datasets_1to6.h5"
    if notebook_weights.exists():
        return notebook_weights
    weights = root / "tests" / "data" / "weights" / "datasets_1to6.h5"
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        tmp = weights.with_suffix(".tmp")
        urllib.request.urlretrieve(PRETRAINED_WEIGHTS_URL, tmp)
        tmp.rename(weights)
    return weights
