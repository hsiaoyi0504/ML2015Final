import numpy as np
import tensorflow as tf

def normalize(x):
    return (x - np.mean(x, 0)) / (np.std(x, 0) + 1e-9)

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

def pool_layer(x, w_shape, padding):
    return tf.nn.max_pool(x, w_shape, [1, 1, 1, 1], padding)

def conv_layer(x, w_shape, padding):
    return tf.nn.conv2d(x, weight(w_shape), [1, 1, 1, 1], padding) + bias(w_shape[-1:])

def full_layer(x, w_shape):
    return tf.matmul(x, weight(w_shape)) + bias(w_shape[-1:])

def linear_nn(current_layer, dropout, arch):
    for i in range(len(arch) - 1):
        current_layer = full_layer(current_layer, arch[i:i + 2])
        current_layer = tf.nn.dropout(tf.nn.relu6(current_layer), dropout)
    return current_layer

