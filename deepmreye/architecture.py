from tensorflow.keras.layers import Conv3D, Activation, Add, UpSampling3D, Lambda, Dense, GaussianNoise, RepeatVector, AveragePooling3D, InputSpec, Input, Reshape, Flatten, Dropout, concatenate, Layer
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
import tensorflow as tf


def create_standard_model(input_shape, opts):
    """Creates convolutional model for training and inference

    Parameters
    ----------
    input_shape : list
        Input shape for each sample (X,Y,Z). Size given by shape of smallest eye mask
    opts : dict
        All model options used for creating and training the model. See util.model_opts for available options

    Returns
    -------
    model : Keras Model
        Full model instance, used for training uncertainty estimate
    model_inference : Keras Model
        Model instance used for inference, provides uncertainty estimate (unsupervised model)
    """
    # Create input layer and add optional noise
    input_layer = Input(input_shape)
    input_layer_noise = GaussianNoise(opts['gaussian_noise'])(input_layer)

    # Initial convolution + dropout
    x = conv3d_block(input_layer_noise, filters=opts['filters'],
                     kernel_size=opts['kernel'], strides=1, activation=opts['activation'])
    x = Dropout(opts['dropout_rate'])(x, training=opts['mc_dropout'])

    # Downsample to bottleneck layer, but keep skip layers
    x, skip_layers = downsample_block(x, filters=opts['filters'], depth=opts['depth'],
                                      multiplier=opts['multiplier'], groups=opts['groups'], activation=opts['activation'])

    # After this layer we split into segmentation and regression part
    bottleneck_layer = Flatten()(x)

    # Regression block
    out_regression = regression_block(bottleneck_layer, num_dense=opts['num_dense'], num_fc=opts['num_fc'],
                                      activation=opts['activation'], dropout_rate=opts['dropout_rate'], inner_timesteps=opts['inner_timesteps'], mc_dropout=opts['mc_dropout'])

    # Confidence for regression block
    out_confidence = confidence_block(
        bottleneck_layer, num_fc=opts['num_fc'], activation=opts['activation'], dropout_rate=opts['dropout_rate'], inner_timesteps=opts['inner_timesteps'], mc_dropout=opts['mc_dropout'])

    # Create model
    real_regression_shape = out_regression.shape.as_list()
    real_regression = Input(real_regression_shape[1::])
    model = Model(inputs=[input_layer, real_regression], outputs=[out_regression])
    model_inference = Model(inputs=input_layer, outputs=[out_regression, out_confidence])

    # Add losses
    loss_euclidean, loss_confidence = compute_standard_loss(out_confidence, real_regression, out_regression)
    model.add_loss(opts['loss_euclidean']*loss_euclidean)
    model.add_loss(opts['loss_confidence']*loss_confidence)

    # Compile it
    model.compile(optimizer=optimizers.Adam(opts['lr']))
    model.metrics.append(loss_euclidean)
    model.metrics_names.append("Euclidean loss")
    model.metrics.append(loss_confidence)
    model.metrics_names.append("Confidence loss")

    return model, model_inference

# --- adult blocks
def res_block(input_layer, filters, groups, activation):
    input_layer_res = conv3d_block(input_layer, filters=filters, kernel_size=1, strides=1, activation=activation)

    x = GroupNormalization(groups=groups, axis=-1)(input_layer)
    x = Activation(activation)(x)
    x = conv3d_block(x, filters=filters, kernel_size=3, strides=1, activation=activation)

    x = GroupNormalization(groups=groups, axis=-1)(x)
    x = Activation(activation)(x)
    x = conv3d_block(x, filters=filters, kernel_size=3, strides=1, activation=activation)

    out = Add()([x, input_layer_res])
    return out

def downsample_block(input_layer, filters, depth, multiplier, groups, activation):
    x = input_layer
    skip_layers = []
    for level_number in range(depth):
        n_level_filters = int(multiplier**level_number) * filters

        for level in range(0, level_number):
            x = res_block(x, filters=n_level_filters, groups=groups, activation=activation)
        # For segmentation save layer after res_blocks
        skip_layers.append(x)
        if level_number < (depth - 1):
            x = conv3d_block(x, filters=n_level_filters, kernel_size=3, strides=2, activation=activation)
    x = GroupNormalization(groups=groups, axis=-1)(x)
    x = Activation(activation)(x)

    return x, skip_layers

def regression_block(input_layer, num_dense, num_fc, activation, dropout_rate, inner_timesteps, mc_dropout, dense_out=2):
    x = RepeatVector(inner_timesteps)(input_layer)
    # Split timesteps so each gets its own weight
    all_xs = list()
    for i in range(0, inner_timesteps):
        x0 = Lambda(lambda x: x[:, i, :])(x)
        for d in range(0, num_dense):
            x0 = Dense(num_fc, activation=activation)(x0)
            x0 = Dropout(dropout_rate)(x0, training=mc_dropout)
        x0 = Dense(dense_out, activation='linear')(x0)
        x0 = Reshape((1, -1))(x0)
        all_xs.append(x0)

    out = concatenate(all_xs, axis=1)
    return out

def confidence_block(input_layer, num_fc, activation, dropout_rate, inner_timesteps, mc_dropout):
    out_conf = Dense(num_fc, activation=activation)(input_layer)
    out_conf = Dropout(dropout_rate)(out_conf, training=mc_dropout)
    out_conf = Dense(inner_timesteps, activation=activation)(out_conf)
    return out_conf

# --- baby blocks
def conv3d_block(input_layer, filters, kernel_size, strides, activation):
    if strides > 1:
        x = Conv3D(filters=filters, kernel_size=(kernel_size, kernel_size, kernel_size),
                   strides=(1, 1, 1), padding='same', activation=activation)(input_layer)
        x = AveragePooling3D()(x)
    else:
        x = Conv3D(filters=filters, kernel_size=(kernel_size, kernel_size, kernel_size),
                   strides=strides, padding='same', activation=activation)(input_layer)
    return x

def upsampling_block(input_layer, size=2):
    x = UpSampling3D(size=size)(input_layer)
    return x

# --- loss blocks
def compute_standard_loss(out_confidence, real_reg, pred_reg):
    loss_euclidean = euclidean_distance(real_reg, pred_reg)
    loss_confidence = mean_squared_error(loss_euclidean, out_confidence)

    return K.mean(loss_euclidean), K.mean(loss_confidence)

def euclidean_distance(y_true, y_pred):
    return tf.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))

def mean_squared_error(y_true, y_pred):
    return K.square(y_true - y_pred)

# --- Custom Layers
# Group Norm --- from https://raw.githubusercontent.com/titu1994/Keras-Group-Normalization/master/group_norm.py
class GroupNormalization(Layer):
    """Group normalization layer

    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes

    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
