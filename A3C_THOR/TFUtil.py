""" Tensorflow building blocks
"""

import tensorflow as tf


def conv_layer(name, input, shape, stride, activation=tf.nn.relu, variable_dict=None):
    with tf.variable_scope(name):
        conv_weights = tf.get_variable('conv_weights', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        conv_bias = tf.get_variable('conv_bias', shape=[shape[3]], initializer=tf.constant_initializer(0.0))
        if variable_dict is not None:
            variable_dict[name + '/conv_weights'] = conv_weights
            variable_dict[name + '/conv_bias'] = conv_bias
    return activation(tf.nn.conv2d(input, conv_weights, strides=(1, stride, stride, 1), padding='VALID') + conv_bias)


def fc_layer(name, input, input_size, num_neron, activation=tf.nn.relu, variable_dict=None):
    with tf.variable_scope(name):
        weights = tf.get_variable('fc_weights', shape=[input_size, num_neron], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('fc_bias', shape=[num_neron], initializer=tf.constant_initializer(0.0))
        if variable_dict is not None:
            variable_dict[name + '/fc_weights'] = weights
            variable_dict[name + '/fc_bias'] = bias
    if activation is not None:
        return activation(tf.matmul(input, weights) + bias)
    else:
        return tf.matmul(input, weights) + bias


def flatten(input, feature_length):
    return tf.reshape(input, (-1, feature_length))


def huber_loss(x):
    # quadratic for |delta| < 1 and linear for |delta| >= 1
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


def get_device_str(use_gpu, gpu_id):
    if use_gpu:
        device_id = gpu_id if gpu_id is not None else 0
        device_str = '/gpu:' + str(device_id)
    else:
        device_str = '/cpu:0'
    return device_str