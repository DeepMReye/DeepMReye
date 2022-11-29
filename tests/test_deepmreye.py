import os
import pytest
import numpy as np
from deepmreye import preprocess, train
from deepmreye.util import model_opts, data_generator, util

# --------------------------------------------------------------------------------
# --------------------------PREPROCESSING-----------------------------------------
# --------------------------------------------------------------------------------
def test_download(path_to_masks):
    # To only test this command python -m pytest -k 'download', all commands python -m pytest
    # Delete files if already in folder to see if download works
    for m in ['eyemask_small.nii', 'eyemask_big.nii', 'dme_template.nii']:
        if os.path.exists(path_to_masks + m):
            os.remove(path_to_masks + m)
    test_masks(path_to_masks)

def test_masks(path_to_masks):
    (eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = preprocess.get_masks(path_to_masks)
    for m in [eyemask_small, eyemask_big, dme_template, mask]:
        np.testing.assert_equal(m.shape, (91, 109, 91))
    np.testing.assert_equal(x_edges, (41, 18, 73, 49))

def test_example_participant(path_to_masks, path_to_testdata):
    # Create eye mask for example participant
    path_to_participant = path_to_testdata + 'test_participant.nii'
    (eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = preprocess.get_masks(path_to_masks)
    (masked_eye, transformation_statistics) = preprocess.run_participant(path_to_participant, dme_template, eyemask_big, eyemask_small, x_edges, y_edges, z_edges)
    np.testing.assert_equal(masked_eye.shape, (47, 29, 18, 2))
    
    # Ants transform is not fully deterministic so check for rough match of transformation
    assert(transformation_statistics[0] < -2)
    assert(transformation_statistics[1] < 0.5)
    assert(transformation_statistics[2] < 0.5)
    
    # Combine with labels
    this_mask = preprocess.normalize_img(masked_eye)
    this_label = preprocess.load_label(path_to_testdata, label_type='calibration_run')
    this_label = this_label[0: this_mask.shape[3], ...] # Adjust for testing
    this_id = ['test_participant']*this_label.shape[0], [0]*this_label.shape[0]

    preprocess.save_data('test_participant', [this_mask], [this_label], [this_id], path_to_testdata + 'processed/', center_labels=False)

def test_load_label(path_to_testdata):
    this_label = preprocess.load_label(path_to_testdata, label_type='calibration_run')
    np.testing.assert_equal(this_label.shape, (135, 10, 2))

# --------------------------------------------------------------------------------
# --------------------------MODEL TRAINING----------------------------------------
# --------------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_model_training(path_to_testdata):
    # Force test on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "" 

    # Define test options. Use small values to speed up CI
    opts = model_opts.get_opts()
    opts['epochs'] = 2
    opts['steps_per_epoch'] = 1
    opts['validation_steps'] = 1

    # For test method use same participant in training and testset
    generators = data_generator.create_leaveoneout_generators([path_to_testdata + 'processed/', path_to_testdata + 'processed/'], batch_size=opts['batch_size'], augment_list=((opts['rotation_x'], opts['rotation_y'], opts['rotation_z']), opts['shift'], opts['zoom']), mixed_batches=True)
    
    # Train model
    (model, model_inference) = train.train_model(dataset='example_data', generators=generators[0], opts=opts, use_multiprocessing=True,
                                                return_untrained=False, verbose=1, save=False)

    # Evaluate model
    (evaluation, scores) = train.evaluate_model(dataset='example_data', model=model_inference, generators=generators[0], save=False, model_description='', verbose=2, percentile_cut=80)
    assert(isinstance(evaluation, dict))

def test_model_evaluation():
    # Generate artificial data
    num_points = 1000
    real_y = np.random.rand(num_points, 10, 2)
    pred_y = np.random.rand(num_points, 10, 2)
    euc_pred = np.random.rand(num_points, 10, 1)
    df_scores = util.get_model_scores(real_y, pred_y, euc_pred)
    
    # For all NaN participant
    real_y = np.random.rand(num_points, 10, 2) * np.NaN
    pred_y = np.random.rand(num_points, 10, 2) * np.NaN
    euc_pred = np.random.rand(num_points, 10, 1) * np.NaN
    df_scores = util.get_model_scores(real_y, pred_y, euc_pred)
