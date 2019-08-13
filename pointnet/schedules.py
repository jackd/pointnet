from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
# import tensorflow as tf
import numpy as np
import functools

# @gin.configurable(module='pointnet.schedules')
# class ClippedExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):

#     def __init__(self,
#                  initial_learning_rate,
#                  decay_steps,
#                  decay_rate,
#                  min_value,
#                  staircase=False,
#                  name=None):
#         super(ClippedExponentialDecay, self).__init__(
#             initial_learning_rate=initial_learning_rate,
#             decay_steps=decay_steps,
#             decay_rate=decay_rate,
#             staircase=staircase,
#             name=name,
#         )
#         self.min_value = min_value

#     def __call__(self, step):
#         base = super(ClippedExponentialDecay, self).__call__(step)
#         return tf.maximum(base, self.min_value)

#     def get_config(self):
#         config = super(ClippedExponentialDecay, self).get_config()
#         config['min_value'] = self.min_value
#         return config


@gin.configurable(module='pointnet.schedules', blacklist=['step'])
def exponential_decay(step,
                      initial_value,
                      decay_steps,
                      decay_rate,
                      min_value=None,
                      staircase=False):
    exponent = step / decay_steps
    if staircase:
        exponent = np.floor(exponent)
    value = initial_value * decay_rate**exponent
    if min_value is not None:
        value = max(value, min_value)
    return value


@gin.configurable(module='pointnet.schedules', blacklist=['step'])
def complementary_exponential_decay(step,
                                    initial_value,
                                    decay_steps,
                                    decay_rate,
                                    max_value=0.99,
                                    staircase=False):
    return 1 - exponential_decay(step,
                                 1 - initial_value,
                                 decay_steps,
                                 decay_rate,
                                 None if max_value is None else 1 - max_value,
                                 staircase=staircase)


# better caching via gin - maybe check out singletons?


@gin.configurable(module='pointnet.schedules')
def exponential_decay_fn(initial_value,
                         decay_steps,
                         decay_rate,
                         min_value=None,
                         staircase=False):
    return functools.partial(
        exponential_decay,
        initial_value=initial_value,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        min_value=min_value,
    )


@gin.configurable(module='pointnet.schedules')
def complementary_exponential_decay_fn(initial_value,
                                       decay_steps,
                                       decay_rate,
                                       max_value=0.99,
                                       staircase=False):
    return functools.partial(
        complementary_exponential_decay,
        initial_value=initial_value,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        max_value=max_value,
        staircase=staircase,
    )
