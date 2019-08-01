from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import gin
layers = tf.keras.layers


def transform_diff(transform):
    return (
        tf.matmul(transform, transform, transpose_b=True) -
        tf.eye(transform.shape[-1]))


@gin.configurable(blacklist=['x', 'units', 'training'])
def mlp(x, units, training=None, use_batch_norm=True,
        batch_norm_momentum=0.99, dropout_rate=0, activation='relu'):
    use_bias = not use_batch_norm
    for u in units:
        tf.keras.layers.Dense
        x = layers.Dense(u, use_bias=use_bias)(x)
        if use_batch_norm:
            x = layers.BatchNormalization(momentum=batch_norm_momentum)(
                x, training=training)
        x = layers.Activation(activation)(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x, training=training)
    return x



@gin.configurable(blacklist=['features', 'training'])
def feature_transform_net(
        features, num_dims, training=None, bn=True, bn_momentum=0.99,
        local_activation='relu', global_activation='relu',
        local_units=(64, 128, 1024), global_units=(512, 256),
        reduction=tf.reduce_max):
    """
    Feature transform network.

    Args:
        inputs: (B, N, f_in) inputs features
        training: flag used in batch norm
        bn_momentum: batch norm momentum
        num_dims: output dimension

    Retturns:
        (B, num_dims, num_dims) transformation matrix,
    """
    x = mlp(
        features, local_units, training=training, activation=local_activation)
    x = reduction(x, axis=-2)
    x = mlp(x, global_units, training=training, activation=global_activation)

    delta = layers.Dense(num_dims**2)(x)
    delta = layers.Reshape((num_dims,)*2)(delta)

    delta = delta + tf.eye(num_dims, dtype=delta.dtype)
    return delta
