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
# from pointnet.augment import transforms as trans


@gin.configurable(blacklist=['inputs', 'labels'])
def augment_cloud(
        inputs, labels,
        jitter_positions_transform=jitter_positions,
        # jitter_normals=None,
        scale_transform=None,
        rigid_transform=None,
        reflect_transform=None,
        perlin_transform=None,
        rotate_transform=rotate_by_scheme,
    ):
    if isinstance(inputs, dict):
        positions = inputs['positions']
        normals = inputs['normals']
        positions_only = False
    else:
        positions = inputs
        positions_only = True

    if not positions_only:
        raise NotImplementedError()

    for transform in (
            jitter_positions_transform,
            scale_transform,
            rigid_transform,
            reflect_transform,
            perlin_transform,
            ):
        if transform is not None:
            # if not callable(transform):
            #     transform = trans.deserialize(transform)
            positions = transform(positions)

    if rotate_transform is not None:
        # if not callable(rotate_transform):
        #     rotate_transform = trans.deserialize(rotate_transform)
        if positions_only:
            positions = rotate_transform(positions)
        else:
            positions, normals = rotate_transform(positions, normals)

    if positions_only:
        inputs = positions
    else:
        inputs = dict(positions=positions, normals=normals)
    return inputs, labels


@gin.configurable
def pack_augmentation_parameters(
        jitter_positions_stddev=0, jitter_positions_clip=0.05,
        scale_stddev=None,
        rigid_transform_stddev=None,
        maybe_reflect_x=False,
        perlin_grid_shape=None, perlin_stddev=0.25,
        ffd_grid_shape=None, ffd_stddev=0.2,
        rotate_scheme='random'
    ):
    """pack flat arguments into kwargs for augment_cloud."""
    import functools
    kwargs = {}

    if jitter_positions_stddev is not None:
        kwargs['jitter_positions_transform'] =  functools.partial(
            jitter_positions,
            stddev=jitter_positions_stddev,
            clip=jitter_positions_clip)

    if scale_stddev is not None:
        kwargs['scale_transform'] = functools.partial(
            random_scale, stddev=scale_stddev)

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


@gin.configurable
def flat_augment_cloud(inputs, labels, **kwargs):
    return augment_cloud(
        inputs, labels, **(pack_augmentation_parameters(**kwargs)))


@gin.configurable
def deserialize(name='flat_augment_cloud', **kwargs):
    fn = {
        'flat_augment_cloud': flat_augment_cloud
    }[name]
    return functools.partial(fn, **kwargs)
