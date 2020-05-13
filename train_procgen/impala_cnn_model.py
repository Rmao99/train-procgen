#from https://github.com/openai/baselines/blob/master/baselines/common/models.py to use for editing layers

import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk

#Layer norm code from here:
#https://gist.github.com/vitchyr/bd2dfc7946c95c5291fd9416baebc051
LAYER_NORM_BIAS_DEFAULT_NAME = "ln_bias"
LAYER_NORM_GAIN_DEFAULT_NAME = "ln_gain"
LAYER_NORMALIZATION_DEFAULT_NAME = "layer_normalization"

def layer_normalize(
        input_pre_nonlinear_activations,
        input_shape,
        epsilon=1e-5,
        name=LAYER_NORMALIZATION_DEFAULT_NAME,
):
    """
    Layer normalizes a 2D tensor along its second axis, which corresponds to
    normalizing within a layer.
    :param input_pre_nonlinear_activations:
    :param input_shape:
    :param name: Name for the variables in this layer.
    :param epsilon: The actual normalized value is
    ```
        norm = (x - mean) / sqrt(variance + epsilon)
    ```
    for numerical stability.
    :return: Layer-normalized pre-non-linear activations
    """
    mean, variance = tf.nn.moments(input_pre_nonlinear_activations, [1],
                                   keep_dims=True)
    normalised_input = (input_pre_nonlinear_activations - mean) / tf.sqrt(
        variance + epsilon)
    with tf.variable_scope(name):
        gains = tf.get_variable(
            LAYER_NORM_GAIN_DEFAULT_NAME,
            input_shape,
            initializer=tf.constant_initializer(1.),
        )
        biases = tf.get_variable(
            LAYER_NORM_BIAS_DEFAULT_NAME,
            input_shape,
            initializer=tf.constant_initializer(0.),
        )
    return normalised_input * gains + biases


def build_impala_cnn(unscaled_images, is_train = True, depths=[16,32,32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    print("building custom impala cnn")
    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        #out = tf.compat.v1.layers.batch_normalization(out,training=is_train)
        out = layer_normalize(out,out.shape)
        out = tf.nn.relu(out)
        #if is_train:
        #out = tf.nn.dropout(out,0.7)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out


@register("impala_cnn")
def impala_cnn(**conv_kwargs):
    def network_fn(X):
        return build_impala_cnn_2(X)
    return network_fn
