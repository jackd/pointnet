from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow_datasets as tfds
from shape_tfds.shape.modelnet.pointnet import Pointnet
from pointnet.augment import augment_cloud
# import trimesh
from mayavi import mlab

builder = Pointnet()
builder.download_and_prepare()

for positions, label in builder.as_dataset(split='test', as_supervised=True):
    if label.numpy() == 8:
        # positions = tf.unstack(positions, axis=-1)
        # positions = tf.stack([positions[i] for i in [0, 2, 1]], axis=-1)

        p2, _ = augment_cloud(positions,
                              label,
                              rotate_scheme='pca-xy',
                              rotation_axis=1)

        # pc = trimesh.PointCloud(positions.numpy(), color=(0, 0, 255))
        # scene = pc.scene()
        # scene.add_geometry(trimesh.PointCloud(p2.numpy(), color=(255, 0, 0)))
        # scene.show()
        p2 = positions
        mlab.points3d(*(p2.numpy().T))
        mlab.show()
