import tensorflow as tf
import numpy as np
from stable_baselines.a2c.utils import linear, conv_to_fc
from stable_baselines.common.policies import register_policy
from stable_baselines.td3.policies import FeedForwardPolicy


def ortho_init(scale=1.0):
    """
    Orthogonal initialization for the policy weights
    :param scale: (float) Scaling factor for the weights.
    :return: (function) an initialization function for the weights
    """

    # _ortho_init(shape, dtype, partition_info=None)
    def _ortho_init(shape, *_, **_kwargs):
        """Intialize weights as Orthogonal matrix.
        Orthogonal matrix initialization [1]_. For n-dimensional shapes where
        n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
        corresponds to the fan-in, so this makes the initialization usable for
        both dense and convolutional layers.
        References
        ----------
        .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
               "Exact solutions to the nonlinear dynamics of learning in deep
               linear
        """
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        elif len(shape) == 5:  # assumes NDHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
        weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
        weights = weights.reshape(shape)
        return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def conv3d(input_tensor, scope, *, n_filters, filter_size, stride,
         pad='VALID', init_scale=1.0, data_format='NDHWC', one_dim_bias=False):
    """
    Creates a 3d convolutional layer for TensorFlow
    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
    or the height and width of kernel filter if the input is a list or tuple
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 3d convolutional layer
    """
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width  = filter_size[1]
        filter_depth  = filter_size[2]
    else:
        filter_height = filter_size
        filter_width  = filter_size
        filter_depth  = filter_size

    if data_format == 'NDHWC':
        channel_ax = 4
        strides = [1, stride, stride, stride, 1]
        bshape = [1, 1, 1, 1, n_filters]
    elif data_format == 'NHWDC':
        channel_ax = 1
        strides = [1, 1, stride, stride, stride]
        bshape = [1, n_filters, 1, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, filter_depth, n_input, n_filters]
    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NDHWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv3d(input_tensor, weight, strides=strides, padding=pad, data_format=data_format)

def cnn_3d(scaled_voxels, n_hidden, filters, filter_sizes, strides, **kwargs):
    """
    CNN in 3D.
    :param scaled_voxels: (TensorFlow Tensor) Voxel input placeholder
    :param n_hidden: (int) Number of nodes in the last linear layer
    :param filters: (array) Filter numbers for the convolutional layers of the CNN
    :param filter_sizes: (array) Filter sizes for the convolutional layers of the CNN
    :param strides: (array) Strides for the convolutional layers of the CNN
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.tanh

    layers = []

    for i, (n_filter, filter_size, stride) in enumerate(zip(filters, filter_sizes, strides)):

        input_layer = scaled_voxels if i == 0 else layers[-1]
        label = 'c%d' % (i + 1)
        layer = activ(conv3d(input_layer, label, n_filters=n_filter, filter_size=filter_size, stride=stride, init_scale=np.sqrt(2), **kwargs))
        layers.append(layer)
        print('layer_%d' % (i + 1), layer.shape)

    layer = conv_to_fc(layers[-1])
    
    return tf.tanh(linear(layer, 'fc1', n_hidden=n_hidden, init_scale=np.sqrt(2)))

class LnCnn3dPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, n_hidden=128, filters=[128, 256], filter_sizes=[3, 3], strides=[2, 1], **_kwargs):
        super(LnCnn3dPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="cnn", cnn_extractor=cnn_3d, layer_norm=True, n_hidden=n_hidden, filters=filters, filter_sizes=filter_sizes, strides=strides, **_kwargs)

# register_policy("LnCnn3dPolicy", LnCnn3dPolicy)
