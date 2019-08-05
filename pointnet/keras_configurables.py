"""Supplies a default set of configurables from tensorflow.compat.v1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gin import config

import tensorflow as tf
from pointnet import tf_compat


def _register_callables(package, module, blacklist):
    for k in dir(package):
        if k not in blacklist:
            v = getattr(package, k)
            if callable(v):
                config.external_configurable(v, name=k, module=module)


blacklist = set(('serialize', 'deserialize', 'get'))
for package, module in (
        (tf.keras.losses, 'tf.keras.losses'),
        (tf.keras.metrics, 'tf.keras.metrics'),
        (tf.keras.optimizers, 'tf.keras.optimizers'),
        (tf.keras.regularizers, 'tf.keras.regularizers')
        ):
    _register_callables(package, module, blacklist)

if tf_compat.major == 1:
    for package, module in (
            (tf.keras.callbacks, 'tf.keras.callbacks'),
            (tf.keras.constraints, 'tf.keras.constraints'),
            (tf.keras.layers, 'tf.keras.layers'),
            ):
        _register_callables(package, module, blacklist)

# clean up namespace
del package, module, blacklist
