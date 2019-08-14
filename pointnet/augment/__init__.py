from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import functools
import tensorflow as tf
from pointnet.augment.ffd import random_ffd
from pointnet.augment.jitter import jitter_positions
from pointnet.augment.jitter import jitter_normals
from pointnet.augment.perlin import add_perlin_noise
from pointnet.augment.rigid import random_rigid_transform
from pointnet.augment.rigid import random_scale
from pointnet.augment.rigid import rotate_by_scheme
from pointnet.augment.rigid import maybe_reflect

# @gin.configurable
# def get_augment_cloud_fn(
#         jitter_sttdev=None,
#         scale_stddev=None,
#         rigid_transform_stddev=None,
#         maybe_reflect_x=None,
#         perlin_grid_shape=None,
#         perlin_stddev=None,
#         rotate_scheme='none'):
#     return functools.partial(
#         augment_cloud,
#         jitter_sttdev=jitter_sttdev,
#         scale_stddev=scale_stddev,
#         rigid_transform_stddev=rigid_transform_stddev,
#         maybe_reflect_x=maybe_reflect_x,
#         perlin_grid_shape=perlin_grid_shape,
#         perlin_stddev=perlin_stddev,
#         rotate_scheme=rotate_scheme,
#         )


@gin.configurable(blacklist=['inputs', 'labels'], module='pointnet.augment')
def augment_cloud(
        inputs,
        labels,
        # jitter_positions_transform=jitter_positions,
        # jitter_normals=None,
        jitter_stddev=None,
        jitter_clip=None,
        scale_stddev=None,
        rigid_transform_stddev=None,
        maybe_reflect_x=False,
        perlin_grid_shape=None,
        perlin_stddev=None,
        rotate_scheme='none',
        rotation_axis=2,
):
    if isinstance(inputs, dict):
        positions = inputs['positions']
        normals = inputs['normals']
        positions_only = False
    else:
        positions = inputs
        positions_only = True
        normals = None

    if not positions_only:
        raise NotImplementedError()

    if jitter_stddev is not None:
        positions = jitter_positions(positions,
                                     stddev=jitter_stddev,
                                     clip=jitter_clip)

    if scale_stddev is not None:
        positions = random_scale(positions, stddev=scale_stddev)

    if rigid_transform_stddev is not None:
        positions = random_rigid_transform(positions,
                                           stddev=rigid_transform_stddev)

    if maybe_reflect_x:
        positions = maybe_reflect(positions)

    if perlin_grid_shape is not None:
        positions = add_perlin_noise(positions,
                                     perlin_grid_shape,
                                     stddev=perlin_stddev)

    if rotate_scheme:
        positions = rotate_by_scheme(positions,
                                     scheme=rotate_scheme,
                                     rotation_axis=rotation_axis)

    if positions_only:
        inputs = positions
    else:
        inputs = dict(positions=positions, normals=normals)
    return inputs, labels
