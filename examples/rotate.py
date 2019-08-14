"""rotation_axis == 1 not quite working..."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from pointnet.augment import rigid

num_points = 100
xyz = tf.random_normal(shape=(num_points, 3))
xyz0 = rigid.rotate_by_scheme(xyz, scheme='pca-xy', rotation_axis=2)
x, y, z = tf.unstack(xyz, axis=-1)

# this guy fails :S
yzx = tf.stack([y, z, x], axis=-1)
yzx1 = rigid.rotate_by_scheme(yzx, scheme='pca-xy', rotation_axis=1)
y1, z1, x1 = tf.unstack(yzx, axis=-1)
xyz1 = tf.stack((x1, y1, z1), axis=-1)

zxy = tf.stack([z, x, y], axis=-1)
zxy2 = rigid.rotate_by_scheme(zxy, scheme='pca-xy', rotation_axis=0)
z2, x2, y2 = tf.unstack(zxy2, axis=-1)
xyz2 = tf.stack((x2, y2, z2), axis=-1)

print(tf.reduce_max(tf.linalg.norm(xyz0 - xyz1, axis=-1).numpy()))
print(tf.reduce_max(tf.linalg.norm(xyz0 - xyz2, axis=-1).numpy()))
