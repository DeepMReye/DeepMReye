from . import util


def get_opts():
    opts = dict()

    opts['kernel'] = 3
    opts['lr'] = 0.00002
    opts['filters'] = 32
    opts['multiplier'] = 2
    opts['depth'] = 4
    opts['dropout_rate'] = 0.1
    opts['num_dense'] = 2
    opts['num_fc'] = 1024
    opts['gaussian_noise'] = 0
    opts['activation'] = util.mish
    opts['groups'] = 8
    opts['inner_timesteps'] = 10

    # Loss weights
    opts['loss_euclidean'] = 1
    opts['loss_confidence'] = 0.1

    # Training
    opts['epochs'] = 125
    opts['steps_per_epoch'] = 1500
    opts['validation_steps'] = 1500
    opts['train_test_split'] = 0.6
    opts['batch_size'] = 8
    opts['balance_datasets'] = False
    opts['mixed_batches'] = True
    opts['balanced_batches'] = False
    opts['mc_dropout'] = False
    opts['rotation_x'] = 5
    opts['rotation_y'] = 5
    opts['rotation_z'] = 5
    opts['shift'] = 4
    opts['zoom'] = 0.15

    return opts
