from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import gin


def _rotate(positions, normals=None, angle=None, impl=tf):
    """
    Randomly rotate the point cloud about the z-axis.

    Args:
        positions: (n, 3) float array
        normals (optional): (n, 3) float array
        angle: float scalar. If None, a uniform random angle in [0, 2pi) is
            used.
        impl:

    Returns:
        rotated (`positions`, `normals`). `normals` will be None if not
        provided. shape and dtype is the same as provided.
    """
    dtype = positions.dtype
    if angle is None:
        angle = tf.random.uniform((), dtype=dtype) * (2 * np.pi)

    if normals is not None:
        assert (normals.dtype == dtype)
    c = impl.cos(angle)
    s = impl.sin(angle)
    # multiply on right, use non-standard rotation matrix (-s and s swapped)
    rotation_matrix = impl.reshape(
        impl.stack([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]), (3, 3))

    positions = impl.matmul(positions, rotation_matrix)
    if normals is None:
        return positions, None
    else:
        return positions, impl.matmul(normals, rotation_matrix)


def rotate(positions, normals=None, angle=None):
    """See _rotate. `angle` may also be 'pca-xy'."""
    if angle != 0:
        positions, normals = _rotate(positions, normals, angle, tf)
    if normals is None:
        return positions
    else:
        return positions, normals


def rotate_np(positions, normals=None, angle=None):
    return _rotate(positions, normals, angle, np)


def reflect(xyz, dim=0, axis=-1):
    values = tf.unstack(xyz, axis=axis)
    values[dim] *= -1
    return tf.stack(values, axis=axis)


def random_rigid_transform_matrix(stddev=0.02, clip=None, dim=3):
    dim = getattr(dim, 'value', dim)  # TF-COMPAT
    offset = tf.random.normal(shape=(dim, dim), stddev=stddev)
    if clip:
        offset = tf.clip_by_value(offset, -clip, clip)  # pylint: disable=invalid-unary-operand-type
    return tf.eye(dim) + offset


@gin.configurable(blacklist=['positions', 'normals'])
def rotate_by_scheme(positions, normals=None, scheme='random'):
    """scheme should be in ("random", "pca-xy", "none")."""
    if scheme == 'none':
        angle = 0
    elif scheme == 'pca-xy':
        from pointnet.augment import pca
        angle = pca.get_pca_xy_angle(positions)
    elif scheme == 'random':
        angle = tf.random.uniform(shape=(), dtype=positions.dtype) * (2 * np.pi)
    else:
        raise ValueError('Unrecognized scheme "%s"' % scheme)
    return rotate(positions, normals, angle)


@gin.configurable(blacklist=['points'])
def random_rigid_transform(points, normals=None, stddev=0.02, clip=None):
    transform = random_rigid_transform_matrix(stddev, clip, points.shape[-1])
    points = tf.matmul(points, transform)
    if normals is None:
        return points
    else:
        raise NotImplementedError('Normal rigid transform not implemented')


@gin.configurable(blacklist=['positions', 'axis'])
def maybe_reflect(positions, axis=-1, dim=0, prob=0.5):
    should_reflect = tf.random.uniform(shape=(), dtype=tf.float32) > prob
    return tf.cond(should_reflect,
                   lambda: reflect(positions, dim=dim, axis=axis),
                   lambda: positions)


@gin.configurable(blacklist=['positions'])
def random_scale(positions, stddev):
    scale = tf.random.truncated_normal(shape=(), mean=1.0, stddev=stddev)
    return positions * scale
