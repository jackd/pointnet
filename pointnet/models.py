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


# @gin.configurable(blacklist=['x', 'units', 'training'])
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


# @gin.configurable(blacklist=['features', 'training'])
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



def pointnet_classifier(
        inputs, training, output_spec, dropout_rate=0.3,
        reduction=tf.reduce_max, units0=(64, 64), units1=(64, 128, 1024),
        global_units=(512, 256), transform_reg_weight=0.0005):
    """
    Get a pointnet classifier.

    Args:
        inputs: `tf.keras.layers.Input` representing cloud coordinates.
        training: bool indicating training mode.
        num_classes: number of classes in the classification problem.
        dropout_rate: rate used in Dropout for global mlp.
        reduction: reduction function accepting (., axis) arguments.
        units0: units in initial local mlp network.
        units1: units in second local mlp network.
        global_units: units in global mlp network.
        transform_reg_weight: weight used in l2 regularizer. Note this should
            be half the weight of the corresponding code in the original
            implementation since we used keras, which does not half the squared
            norm.

    Returns:
        keras model with logits as outputs.
    """
    num_classes = output_spec.shape[-1]
    cloud = inputs
    transform0 = feature_transform_net(cloud, 3, training=training)
    cloud = tf.matmul(cloud, transform0)
    cloud = mlp(cloud, units0, training=training)

    transform1 = feature_transform_net(cloud, units0[-1])
    cloud = tf.matmul(cloud, transform1)

    cloud = mlp(cloud, units1, training=training)

    features = reduction(cloud, axis=-2)
    features = mlp(
        features, global_units, training=training, dropout_rate=dropout_rate)
    logits = tf.keras.layers.Dense(num_classes)(features)

    model = tf.keras.models.Model(
        inputs=tf.nest.flatten(inputs), outputs=logits)

    if transform_reg_weight:
        regularizer = tf.keras.regularizers.l2(transform_reg_weight)
        for transform in (transform1,):
            model.add_loss(regularizer(transform_diff(transform)))

    return model


def deserialize(
        inputs, training, output_spec, name='pointnet_classifier', **kwargs):
    return {
        'pointnet_classifier': pointnet_classifier
    }[name](inputs, training, output_spec, **kwargs)