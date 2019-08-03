"""Stuff necessary for tf version < 1.14."""

import tensorflow as tf
major, minor, patch = tf.__version__.split('.')  # pylint: disable=no-member
major = int(major)
minor = int(minor)

if major == 1:
    if not hasattr(tf.keras.losses, 'SparseCategoricalCrossentropy'):
        tf.keras.losses.SparseCategoricalCrossentropy = \
            tf.keras.losses.CategoricalCrossentropy
