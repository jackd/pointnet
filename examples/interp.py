from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pointnet.util import interp

tf.compat.v1.enable_eager_execution()

nx = 100
ny = 100
x = tf.linspace(-1., 1., nx)
y = tf.linspace(-1., 1., ny)
xy = tf.stack(tf.meshgrid(x, y, indexing='ij'), axis=-1)

grid_values = tf.exp(-tf.reduce_sum(xy**2, axis=-1))

coords = tf.random.uniform(
    shape=(5000, 2), dtype=tf.float32)*tf.cast(tf.shape(
        grid_values)-1, tf.float32)
interped_values = interp.linear_interp(grid_values, coords)


import matplotlib.pyplot as plt
_, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(grid_values.numpy())
xv, yv = coords.numpy().T
ax1.scatter(xv, yv, s=1, c=interped_values.numpy())
ax1.set(xlim=(0, nx), ylim=(0, ny))
ax1.axis('square')
plt.show()
