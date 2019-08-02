from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pointnet.util import perlin
tf.compat.v1.enable_eager_execution()

num_points = 10000
grid_shape= (4, 4, 4)
coords = tf.random.normal(shape=(num_points, 3))
coords /= tf.linalg.norm(coords, axis=-1, keepdims=True)

modified_coords = perlin.add_perlin_noise(coords, grid_shape, stddev=0.25)

from trimesh import PointCloud
p0 = PointCloud(coords.numpy(), color=(255, 0, 0))
p1 = PointCloud(modified_coords.numpy(), color=(0, 255, 0))
scene = p0.scene()
scene.add_geometry(p1)

scene.show()
