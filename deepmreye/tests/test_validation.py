import pytest
from deepmreye.validation import validate_and_extract_tr, MissingTRError
import ants
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

def create_mock_nifti(spacing, path):
    # Create a small dummy image for successful extraction
    img = ants.from_numpy(np.zeros((2, 2, 2, 2) if len(spacing) >= 4 else (2, 2, 2)), spacing=spacing)
    ants.image_write(img, path)

def test_validate_and_extract_tr_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.nii.gz")
        create_mock_nifti(spacing=(2.0, 2.0, 2.0, 1.5), path=path)
        tr = validate_and_extract_tr(path)
        assert tr == 1.5

def test_validate_and_extract_tr_missing_dimension():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.nii.gz")
        # Only 3 spatial dimensions
        create_mock_nifti(spacing=(2.0, 2.0, 2.0), path=path)
        with pytest.raises(MissingTRError, match="Missing TR"):
            validate_and_extract_tr(path)

@patch("deepmreye.validation.ants.image_read")
def test_validate_and_extract_tr_invalid_value(mock_image_read):
    # Mock ANTs image to return zero TR
    mock_img_zero = MagicMock()
    mock_img_zero.spacing = (2.0, 2.0, 2.0, 0.0)
    mock_image_read.return_value = mock_img_zero
    
    with pytest.raises(MissingTRError, match="Invalid TR"):
        validate_and_extract_tr("dummy_path")
        
    # Mock ANTs image to return negative TR
    mock_img_neg = MagicMock()
    mock_img_neg.spacing = (2.0, 2.0, 2.0, -1.5)
    mock_image_read.return_value = mock_img_neg
    
    with pytest.raises(MissingTRError, match="Invalid TR"):
        validate_and_extract_tr("dummy_path")
