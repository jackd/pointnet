from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from shape_tfds.shape import modelnet
import gin


class FfdModelnetConfig(modelnet.CloudConfig):

    def __init__(self, num_points, grid_shape=4, name=None, **kwargs):
        if name is None:
            if not isinstance(grid_shape, int):
                assert (len(grid_shape) == 3)
                if all(g == grid_shape[0] for g in grid_shape[1:]):
                    grid_shape = grid_shape[0]

            grid_shape_str = ('%d' % grid_shape if isinstance(grid_shape, int)
                              else 'x'.join(str(g) for g in grid_shape))
            name = 'ffd-%s-%d' % (grid_shape_str, num_points)

        if isinstance(grid_shape, int):
            grid_shape = (grid_shape,) * 3
        self._grid_shape = grid_shape
        super(FfdModelnetConfig, self).__init__(num_points=num_points,
                                                name=name,
                                                **kwargs)
        if tf.executing_eagerly():

            def f(points):
                from pointnet.augment import ffd
                b, p = ffd.get_ffd(points, grid_shape)
                return dict(b=b, p=p)

            self._f = f
        else:
            raise NotImplementedError(
                'Please generate data in a separate script using separately '
                'tf.compat.v1.enable_eager_execution')

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def feature_item(self):
        from tensorflow_datasets.core import features
        import numpy as np
        grid_size = np.prod(self.grid_shape)
        return 'ffd', features.FeaturesDict({
            'b':
                features.Tensor(shape=(self.num_points, grid_size),
                                dtype=tf.float32),
            'p':
                features.Tensor(shape=(grid_size, 3), dtype=tf.float32),
        })

    def load_example(self, off_path):
        points = super(FfdModelnetConfig, self).load_example(off_path)
        return self._f(points)


@gin.configurable(module='pointnet.builders')
def pointnet_builder(version=1, num_classes=40):
    """Builder based on pointnet v1/2 data."""
    if version == 1:
        if num_classes == 10:
            raise NotImplementedError(
                'Only 40-class pointnet v1 builder available')
        return modelnet.Pointnet()
    elif version == 2:
        return modelnet.Pointnet2(
            config=modelnet.get_pointnet2_config(num_classes))
    else:
        raise ValueError('version must be 1 or 2, got {}'.format(version))


@gin.configurable(module='pointnet.builders')
def trimesh_builder(num_points_base=2048, num_classes=40):
    """Builder based on sampling via trimesh."""
    config = modelnet.CloudConfig(num_points_base)
    return {
        10: modelnet.Modelnet10,
        40: modelnet.Modelnet40,
    }[num_classes](config=config)

    # self._source = source
    #     dim_order = None
    #     if source == 'pointnet':
    #         builder = modelnet.Pointnet()
    #         dim_order = (0, 2, 1)
    #         assert (num_classes == 40)
    #     elif source == 'pointnet2':
    #         builder = modelnet.sampled.ModelnetSampled(
    #             config=modelnet.sampled.get_config(num_classes))
    #     else:
    #         if source == 'trimesh':
    #             num_points_base = 2048
    #         elif isinstance(source, tuple):
    #             key, num_points_base = source
    #             assert (key == 'trimesh')
    #             config = modelnet.CloudConfig(num_points=num_points_base)
    #             builder = {
    #                 10: modelnet.Modelnet10,
    #                 40: modelnet.Modelnet40,
    #             }[num_classes](config=config)
    #         else:
    #             raise ValueError('Invalid source {}'.format(source))
