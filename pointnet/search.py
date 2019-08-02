from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
import copy
import numpy as np


def trunc_norm(loc, scale):
    x = np.clip(np.random.normal(), -2, 2)
    return loc + scale*x


def explore(config):
    '''Custom explore function.

    Args:
        config: dictionary containing ray config params.

    Returns:
        Copy of config with modified augmentation policy.
    '''
    def default_factor(loc=1, scale=0.1):
        return trunc_norm(loc=loc, scale=scale)

    def up_or_down():
        return 1 if np.random.uniform() > 0.5 else -1

    config = copy.deepcopy(config)
    params = config['problem']['augmentation_spec']['train']

    params['jitter_positions_stddev'] *= default_factor()
    scale_factor = default_factor()
    params['scale_min'] *= scale_factor
    params['scale_max'] *= scale_factor
    params['rigid_transform_stddev'] *= default_factor()
    if np.random.uniform() < 0.2:
        params['maybe_reflect_x'] = not params['maybe_reflect_x']
    if np.random.uniform() < 0.2:
        params['perlin_grid_shape'] = max(
            2, params['perlin_grid_shape'] + up_or_down())
    params['perlin_stddev'] *= default_factor()
    if np.random.uniform() < 0.2:
        params['ffd_grid_shape'] = max(
            2, params['ffd_grid_shape'] + up_or_down())
    params['ffd_stddev'] *= default_factor()
    return config


# quick hack
class Flags(object):
    pass


FLAGS = Flags()

FLAGS.name = 'test'
FLAGS.local_dir = '/tmp/ray_test'
FLAGS.checkpoint_freq = 5
FLAGS.cpu = 4
FLAGS.gpu = 0
FLAGS.num_samples = 100
FLAGS.max_epochs = 100


def main(_):
    import ray
    from pointnet.tune_model import TuneModel
    from ray.tune.schedulers import PopulationBasedTraining
    from ray.tune import run_experiments
    import tensorflow as tf

    pbt = PopulationBasedTraining(
        time_attr='training_iteration',
        reward_attr='val_sparse_categorical_accuracy',
        perturbation_interval=5,
        custom_explore_fn=explore,
        log_config=True)

    train_spec = {
        'run': TuneModel,
        'resources_per_trial': {
            'cpu': FLAGS.cpu,
            'gpu': FLAGS.gpu
        },
        'stop': {
            'training_iteration': FLAGS.max_epochs,
        },
        'config': {
            'batch_size': 32,
            'problem': {
                'name': 'modelnet40',
                'augmentation_spec': {
                    'train': {
                        'jitter_positions_stddev': 1e-2,
                        'scale_min': 0.95,
                        'scale_max': 0.95,
                        'rigid_transform_stddev': 1e-2,
                        'maybe_reflect_x': True,
                        'perlin_grid_shape': 4,
                        'perlin_stddev': 0.25,
                        'ffd_grid_shape': 4,
                        'ffd_stddev': 0.2,
                        'rotate_scheme': 'pca-xy'
                    },
                    'validation': {'rotate_scheme': 'pca-xy'}
            }},
            'model_fn': {
                'name': 'pointnet_classifier'
            },
            'optimizer': tf.keras.utils.serialize_keras_object(
                tf.keras.optimizers.SGD(momentum=0.9, lr=1e-3, nesterov=True))
        },
        # 'local_dir': FLAGS.local_dir,
        # 'checkpoint_freq': FLAGS.checkpoint_freq,
        'num_samples': FLAGS.num_samples
    }

    ray.init()
    run_experiments(
        {
            FLAGS.name: train_spec
        },
        scheduler=pbt,
        reuse_actors=True,
        verbose=True)


if __name__ == '__main__':
    app.run(main)
