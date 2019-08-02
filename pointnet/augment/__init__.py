from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
from pointnet.augment.ffd import random_ffd
from pointnet.augment.jitter import jitter_positions
from pointnet.augment.jitter import jitter_normals
from pointnet.augment.perlin import add_perlin_noise
from pointnet.augment.rigid import random_rigid_transform
from pointnet.augment.rigid import random_scale
from pointnet.augment.rigid import rotate_by_scheme
from pointnet.augment.rigid import maybe_reflect
from pointnet.augment.transforms import deserialize


# @gin.configurable(blacklist=['inputs', 'labels'])
def augment_cloud(
        inputs, labels, positions_only=True,
        num_points_sampled=1024,
        jitter_positions_transform=jitter_positions,
        # jitter_normals=None,
        scale_transform=None,
        rigid_transform=None,
        reflect_transform=None,
        perlin_transform=None,
        ffd_transform=None,
        rotate_transform=rotate_by_scheme,
    ):
    if not positions_only:
        raise NotImplementedError()
    if isinstance(inputs, dict):
        positions = inputs['positions']
        normals = inputs['normals']
    else:
        positions = inputs
        normals = inputs

    if num_points_sampled is not None:
        if positions_only:
            positions = tf.random.shuffle(positions)[:num_points_sampled]
        else:
            indices = tf.range(tf.shape(positions)[0])
            indices = tf.random.shuffle(indices)[:num_points_sampled]
            positions = tf.gather(positions, indices)
            normals = tf.gather(positions, indices)

    # if jitter_positions is not None:
    #     positions = jitter_positions(positions, **jitter_positions)
    # if jitter_normals is not None and not positions_only:
    #     normals = jitter_normals(normals, **jitter_normals)

    for transform in (
            jitter_positions_transform,
            scale_transform,
            rigid_transform,
            reflect_transform,
            perlin_transform,
            ffd_transform,
            ):
        if transform is not None:
            if not callable(transform):
                transform = deserialize(transform)
            positions = transform(positions)

    if rotate_transform is not None:
        if not callable(rotate_transform):
            rotate_transform = deserialize(rotate_transform)
        if positions_only:
            positions = rotate_transform(positions)
        else:
            positions, normals = rotate_transform(positions, normals)

    if positions_only:
        inputs = positions
    else:
        inputs = dict(positions=positions, normals=normals)
    return inputs, labels


# @gin.configurable
def pack_augmentation_parameters(
        positions_only=True,
        num_points_sampled=1024,
        jitter_positions_stddev=0, jitter_positions_clip=0.05,
        scale_min=None, scale_max=None,
        rigid_transform_stddev=None,
        maybe_reflect_x=False,
        perlin_grid_shape=None, perlin_stddev=0.25,
        ffd_grid_shape=None, ffd_stddev=0.2,
        rotate_scheme='random'
    ):
    """pack flat arguments into kwargs for augment_cloud."""
    import functools
    kwargs = dict(
        positions_only=positions_only,
        num_points_sampled=num_points_sampled,
    )

    if jitter_positions_stddev is not None:
        kwargs['jitter_positions_transform'] =  functools.partial(
            jitter_positions,
            stddev=jitter_positions_stddev,
            clip=jitter_positions_clip)

    if scale_min is not None:
        kwargs['scale_transform'] = functools.partial(
            random_scale, scale_range=(scale_min, scale_max))

    if rigid_transform_stddev is not None:
        kwargs['rigid_transform'] = functools.partial(
            random_rigid_transform, stddev=rigid_transform_stddev)

    if maybe_reflect_x:
        kwargs['reflect_transform'] = maybe_reflect

    if perlin_grid_shape is not None:
        kwargs['perlin_transform'] = functools.partial(
            add_perlin_noise,
            grid_shape=perlin_grid_shape, stddev=perlin_stddev)

    if ffd_grid_shape is not None:
        kwargs['ffd_transform'] = functools.partial(
            random_ffd, grid_shape=ffd_grid_shape, stddev=ffd_stddev)

    kwargs['rotate_transform'] = functools.partial(
        rotate_by_scheme, scheme=rotate_scheme)

    return kwargs


def flat_augment_cloud(inputs, labels, **kwargs):
    return augment_cloud(
        inputs, labels, **(pack_augmentation_parameters(**kwargs)))
