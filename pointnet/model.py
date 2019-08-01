from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
from pointnet import util


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
    transform0 = util.feature_transform_net(cloud, 3, training=training)
    cloud = tf.matmul(cloud, transform0)
    cloud = util.mlp(cloud, units0, training=training)

    transform1 = util.feature_transform_net(cloud, units0[-1])
    cloud = tf.matmul(cloud, transform1)

    cloud = util.mlp(cloud, units1, training=training)

    features = reduction(cloud, axis=-2)
    features = util.mlp(
        features, global_units, training=training, dropout_rate=dropout_rate)
    logits = tf.keras.layers.Dense(num_classes)(features)

    model = tf.keras.models.Model(
        inputs=tf.nest.flatten(inputs), outputs=logits)

    if transform_reg_weight:
        regularizer = tf.keras.regularizers.l2(transform_reg_weight)
        for transform in (transform1,):
            model.add_loss(regularizer(util.transform_diff(transform)))

    return model
