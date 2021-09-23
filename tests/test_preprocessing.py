import pytest
import numpy as np
from deepmreye import preprocess

def test_masks(path_to_masks):
    (eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = preprocess.get_masks(path_to_masks)
    for m in [eyemask_small, eyemask_big, dme_template, mask]:
        np.testing.assert_equal(m.shape, (91, 109, 91))
    np.testing.assert_equal(x_edges, (41, 18, 73, 49))

def test_example_participant(path_to_masks, path_to_participant):
    (eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = preprocess.get_masks(path_to_masks)
    (masked_eye, transformation_statistics) = preprocess.run_participant(path_to_participant, dme_template, eyemask_big, eyemask_small, x_edges, y_edges, z_edges)
    np.testing.assert_equal(masked_eye.shape, (47, 29, 18, 2))
    # Ants transform is not fully deterministic so check for rough match of transformation
    assert(transformation_statistics[0] < -2)
    assert(transformation_statistics[1] < 0.5)
    assert(transformation_statistics[2] < 0)