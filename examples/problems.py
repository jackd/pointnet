from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds
import functools
from pointnet.problems import ModelnetProblem
from pointnet.builders import pointnet_builder
from pointnet.builders import trimesh_builder
from pointnet.augment import augment_cloud
from mayavi import mlab
rotate_scheme = 'pca-xy'

# builder = trimesh_builder()
builder = pointnet_builder(version=2)
builder.download_and_prepare()

map_fn = functools.partial(augment_cloud, rotate_scheme=rotate_scheme)
problem = ModelnetProblem(builder, map_fn=map_fn, num_points=1024)

for positions, label in tfds.as_numpy(problem.get_dataset(split='train')):
    mlab.points3d(*(positions.T))
    mlab.show()
