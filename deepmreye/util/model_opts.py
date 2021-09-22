from . import util


def get_opts():
    opts = dict()

    opts['kernel'] = 3 # Kernel size for all convolutional layers
    opts['lr'] = 0.00002 # Learning rate
    opts['filters'] = 32 # Number of filters in convolutional layers
    opts['multiplier'] = 2 # How much does the number of filters increase in each layer
    opts['depth'] = 4 # Maximum number of layers
    opts['dropout_rate'] = 0.1 # Dropout ratio for fully connected layers
    opts['num_dense'] = 2 # Number of fully connected layers
    opts['num_fc'] = 1024 # Number of units in fully connected layer
    opts['gaussian_noise'] = 0 # How much gaussian noise is added (unit = standard deviation)
    opts['activation'] = util.mish # Activation function for all layers
    opts['groups'] = 8 # Number of groups to normalize across (see GroupNorm)
    opts['inner_timesteps'] = 10 # Default number of subTR samples which are being reconstructed

    # Loss weights
    opts['loss_euclidean'] = 1 # Loss weight for euclidean distance
    opts['loss_confidence'] = 0.1 # Loss weight for uncertainty measure

    # Training
    opts['epochs'] = 25 # Number of epochs
    opts['steps_per_epoch'] = 1500 # Number of steps per training epoch
    opts['validation_steps'] = 1500 # Number of steps per validation epoch
    opts['train_test_split'] = 0.6 # Default proportion of train (60%)-test(40%) split
    opts['batch_size'] = 8 # Batch size used for training the model
    opts['mixed_batches'] = True # If true, each batch contains samples across participants
    opts['mc_dropout'] = False # If true, monte carlo dropout is used
    opts['rotation_x'] = 5 # Augmentation parameter, rotation in x-axis
    opts['rotation_y'] = 5 # Augmentation parameter, rotation in y-axis
    opts['rotation_z'] = 5 # Augmentation parameter, rotation in z-axis
    opts['shift'] = 4 # Augmentation parameter, shift in all axes
    opts['zoom'] = 0.15 # Augmentation parameter, zoom in all axes

    return opts
