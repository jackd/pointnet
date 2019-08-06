from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
import tensorflow as tf


@gin.configurable(module='util')
def gpu_options(
        allow_growth=None,
        per_process_gpu_memory_fraction=None,
        cuda_visible_devices=None):
    if cuda_visible_devices is not None:
        if isinstance(cuda_visible_devices, int):
            cuda_visible_devices = str(cuda_visible_devices)
        elif hasattr(cuda_visible_devices, '__iter__'):
            cuda_visible_devices = ','.join(
                str(d) for d in cuda_visible_devices)
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    reset = False
    config = tf.compat.v1.ConfigProto()
    options = config.gpu_options  # pylint: disable=no-member
    if allow_growth is not None:
        options.allow_growth = True  # pylint: disable=no-member
        reset = True
    if per_process_gpu_memory_fraction is not None:
        options.per_process_gpu_memory_fraction = (  # pylint: disable=no-member
            per_process_gpu_memory_fraction)
        reset = True
    if reset:
        session = tf.compat.v1.Session(config=config)
        tf.keras.backend.set_session(session)
