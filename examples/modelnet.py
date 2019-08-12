from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from mayavi import mlab
from shape_tfds.shape import modelnet
from pointnet.augment import augment_cloud
import functools
tf.compat.v1.enable_eager_execution()

builder = modelnet.Modelnet40(config=modelnet.CloudConfig(num_points=2048))
dataset = builder.as_dataset(split='train', as_supervised=True).map(
    functools.partial(augment_cloud, rotate_scheme='pca-xy'))
for positions, label in dataset:
    mlab.points3d(*positions.numpy().T)
    mlab.show()
