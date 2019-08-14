from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import gin


def not_implemented_fn(*args, **kwargs):
    raise NotImplementedError()


@gin.configurable
def problem(value=None):
    if value is None:
        from pointnet.problems import ModelnetProblem
        value = ModelnetProblem()
    return value


@gin.configurable
def optimizer(value=None):
    if value is None:
        value = tf.keras.optimizers.Adam()
    return value


@gin.configurable
def batch_size(value=32):
    return value


@gin.configurable
def model_fn(value=not_implemented_fn):
    return value


@gin.configurable
def callbacks(value=[]):
    return list(value)


@gin.configurable
def chkpt_callback(value=None):
    return value


@gin.configurable
def epochs(value=None):
    return value


@gin.configurable
def num_batches_in_examples(num_examples, batch_size):
    return num_examples / batch_size


@gin.configurable
def num_epochs_in_examples(num_examples, problem):
    """Get the number of epochs in the given number of steps."""
    return num_examples / problem.examples_per_epoch(split='train')
