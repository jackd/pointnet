from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from pointnet.tune_model import GinTuneModel
import tensorflow as tf

config = {
    # 'batch_size': 2,  # SMOKE-TEST
    'batch_size':
        32,
    'problem': {
        'name': 'modelnet40',
    },
    'augmentation_spec': {
        'train': {
            'jitter_positions_stddev': 1e-2,
            'scale_min': 0.95,
            'scale_max': 1.05,
            'rigid_transform_stddev': 1e-2,
            'maybe_reflect_x': True,
            'perlin_grid_shape': 4,
            'perlin_stddev': 0.25,
            # 'ffd_grid_shape': 4,
            # 'ffd_stddev': 0.2,
            'rotate_scheme': 'pca-xy',
        },
        'validation': {
            'rotate_scheme': 'pca-xy',
        },
    },
    'model_fn': {
        'name': 'pointnet_classifier'
    },
    'optimizer':
        tf.keras.utils.serialize_keras_object(
            tf.keras.optimizers.SGD(momentum=0.9, lr=1e-3, nesterov=True))
}

tune_model = GinTuneModel(config=config)
checkpoint_dir = '/tmp/test_tune_model'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
tune_model._save(checkpoint_dir)
print('done')
