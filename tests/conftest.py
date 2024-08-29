from pathlib import Path

import pytest


@pytest.fixture
def path_to_masks():
    path = Path(__file__).parents[1] / "deepmreye" / "masks"
    return path


@pytest.fixture
def path_to_testdata():
    path = Path(__file__).parents[0] / "data"
    return path
