"""
Perlin noise generator

https://en.wikipedia.org/wiki/Perlin_noise
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pointnet.util import interp
import gin


def scale_to_grid(coords, grid_shape, eps=1e-5):
    # eps ensures we don't end up interpolating on the upper boundary.
    # causes issues with calculating corners according to floor and floor + 1
    shift = tf.reduce_min(coords, axis=0)
    coords = coords - shift
    scale = tf.reduce_max(coords) / (tf.cast(grid_shape, tf.float32) - 1)
    scale = scale + eps
    coords = coords / scale
    def rescale(c):
        return c * scale + shift
    return coords, rescale


# @gin.configurable(blacklist=['coords'])
def add_perlin_noise(
        coords, grid_shape=(4, 4, 4), stddev=0.25, eps=1e-5, rescale=True):
    if isinstance(grid_shape, int):
        num_dims = coords.shape[-1]
        grid_shape = (grid_shape,)*num_dims
    else:
        num_dims = len(grid_shape)
    shifts = []
    if rescale:
        coords, rescale_fn = scale_to_grid(coords, grid_shape, eps=eps)

    # TODO: vectorize?
    for _ in range(num_dims):
        gradients = tf.random.normal(
            shape=grid_shape + (num_dims,), stddev=stddev)
        shifts.append(interp.perlin_interp(gradients, coords))
    coords = coords + tf.stack(shifts, axis=-1)
    if rescale:
        coords = rescale_fn(coords)
    return coords