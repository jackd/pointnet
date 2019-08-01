from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import gin


def get_pca_xy_angle(positions):
    from sklearn.decomposition import PCA

    def get_pca(xy):
        pca = PCA(n_components=1)
        xy = xy.numpy()
        pca.fit_transform(xy)
        pca_vec = tf.squeeze(pca.components_, axis=0)
        return pca_vec

    xy, _ = tf.split(positions, [2, 1], axis=-1)
    pca_vec = tf.py_function(get_pca, [xy], positions.dtype)
    pca_vec.set_shape((2,))
    x, y = tf.unstack(pca_vec, axis=0)
    return tf.atan2(y, x)


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
        assert(normals.dtype == dtype)
    c = impl.cos(angle)
    s = impl.sin(angle)
    # multiply on right, use non-standard rotation matrix (-s and s swapped)
    rotation_matrix = impl.reshape(
        impl.stack([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]),
        (3, 3))

    positions = impl.matmul(positions, rotation_matrix)
    if normals is None:
        return positions, None
    else:
        return positions, impl.matmul(normals, rotation_matrix)


@gin.configurable(blacklist=['positions', 'normals'])
def rotate(positions, normals=None, angle=None):
    """See _rotate. `angle` may also be 'pca-xy'."""
    if angle == 0:
        if normals is None:
            return positions
        else:
            return positions, normals
    return _rotate(positions, normals, angle, tf)


def rotate_np(positions, normals=None, angle=None):
    return _rotate(positions, normals, angle, np)


@gin.configurable
def rotate_by_scheme(positions, normals=None, scheme='random'):
    """scheme should be in ("random", "pca-xy")."""
    if scheme == 'pca-xy':
        angle = get_pca_xy_angle(positions)
    elif angle == 'random':
        angle = tf.random.uniform(0, 2*np.pi, dtype=positions.dtype)
    else:
        raise ValueError('Unrecognized scheme "%s"' % scheme)
    return _rotate(positions, normals, angle, tf)


@gin.configurable(blacklist=['positions'])
def jitter_positions(positions, stddev=0.02, clip=None):
    """
    Randomly jitter points independantly by normally distributed noise.

    Args:
        positions: float array, any shape
        stddev: standard deviation of jitter
        clip: if not None, jittering is clipped to this
    """
    jitter = tf.random.normal(shape=tf.shape(positions), stddev=stddev)
    if clip is not None:
        jitter = tf.clip_by_norm(jitter, clip, axes=[-1])
    # scale by max norm
    max_val = tf.reduce_max(tf.linalg.norm(positions, axis=-1))
    jitter = jitter * max_val
    return positions + jitter


@gin.configurable(blacklist=['normals'])
def jitter_normals(
        normals, stddev=0.02, clip=None, some_normals_invalid=False):
    if stddev == 0:
        return normals
    normals = jitter_positions(normals, stddev, clip)
    norms = tf.linalg.norm(normals, axis=-1, keepdims=True)
    if some_normals_invalid:
        # some normals might be invalid, in which case they'll initially be 0.
        thresh = 0.1 if clip is None else 1 - clip
        return tf.where(
            tf.less(norms, thresh), tf.zeros_like(normals), normals / norms)
    else:
        return normals / norms


def reflect_x(xyz):
    x, y, z = tf.unstack(xyz, axis=-1)
    return tf.stack([-x, y, z], axis=-1)


@gin.configurable(blacklist=['inputs', 'labels'])
def augment_cloud(
        inputs, labels,
        num_to_sample=1024,
        rotate_scheme='random',  # one of ('random', 'pca-xy', 'none')
        jitter_positions=jitter_positions,
        jitter_normals=None,
        scale_range=None,
        maybe_reflect_x=False,
        positions_only=True):
    if isinstance(inputs, dict):
        positions = inputs['positions']
        normals = inputs['normals']
        if positions_only:
            normals = None
    else:
        positions = inputs
        if not positions_only:
            raise ValueError('Cannot return normals when inputs is tensor')

    if num_to_sample is not None:
        if positions_only:
            positions = tf.random.shuffle(positions)[:num_to_sample]
        else:
            indices = tf.range(tf.shape(positions)[0])
            indices = tf.random.shuffle(indices)[:num_to_sample]
            positions = tf.gather(positions, indices)
            normals = tf.gather(positions, indices)

    if rotate_scheme is not None and rotate_scheme != 'none':
        if positions_only:
            positions = rotate_by_scheme(positions, normals)
        else:
            positions, normals = rotate(positions, normals)

    if scale_range is not None:
        min_scale, max_scale = scale_range
        scale = min_scale + (max_scale - min_scale)*tf.random.uniform(shape=())
        positions = positions * scale

    if maybe_reflect_x:
        should_reflect = tf.random.uniform(shape=(), dtype=tf.float32) > 0.5
        if positions_only:
            positions = tf.cond(
                should_reflect,
                lambda: reflect_x(positions),
                lambda: positions)
        else:
            positions, normals = tf.cond(
                should_reflect,
                lambda: (reflect_x(positions), reflect_x(normals)),
                lambda: (positions, normals))

    if jitter_positions is not None:
        positions = jitter_positions(positions)
    if jitter_normals is not None and not positions_only:
        normals = jitter_normals(normals)

    if positions_only:
        inputs = positions
    else:
        inputs = dict(positions=positions, normals=normals)
    return inputs, labels
