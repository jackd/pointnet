from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from pointnet.problems import FfdModelnetConfig
from shape_tfds.shape import modelnet

grid_shape = 4
num_points = 2048

config = FfdModelnetConfig(num_points, grid_shape)
builder = modelnet.Modelnet40(config=config)

builder.download_and_prepare()
