from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import logging
import os
import copy
import numpy as np
import gin
from pointnet.cli.actions import actions
from pointnet.tune_model import TuneModel


def trunc_norm(loc, scale):
    x = np.clip(np.random.normal(), -2, 2)
    return loc + scale*x


@gin.configurable
def mutable_bindings(
        jitter_positions_stddev=1e-2,
        random_scale_stddev=0.05,
        random_rigid_transform_stddev=0.05,
        perlin_stddev=None):
    """Get initial mutable bindings."""
    return {
        'jitter_positions.stddev': jitter_positions_stddev,
        'random_scale.stddev': random_scale_stddev,
        'random_rigid_transform.stddev': random_rigid_transform_stddev,
        'add_perlin_noise.stddev': perlin_stddev
    }


def explore(config):
    def default_factor(loc=1, scale=0.1):
        return trunc_norm(loc=loc, scale=scale)

    config = copy.deepcopy(config)
    mutable_bindings = config['mutable_bindings']
    for k, scale in (
            ('jitter_positions.stddev', 0.1),
            ('jitter_positions.stddev', 0.1),
            ('random_scale.stddev', 0.05),
            ('add_perlin_noise.stddev', 0.1),
        ):
        if k in mutable_bindings:
            mutable_bindings[k] *= default_factor(scale=scale)
            config['reset_generators'] = True

    return config


@gin.configurable
def outer_config(
        config_dir=None,
        inner_gin_file='aug_opt/models/base.gin',
        inner_bindings=[],
        inner_mutable_bindings=mutable_bindings,
        batch_size=32,
        initial_weights_path=None):
    if config_dir is None:
        from pointnet.cli.config import get_config_dir
        config_dir = get_config_dir()
    if callable(inner_mutable_bindings):
        inner_mutable_bindings = inner_mutable_bindings()
    return dict(
        config_dir=config_dir,
        gin_file=inner_gin_file,
        bindings=inner_bindings,
        mutable_bindings=inner_mutable_bindings,
        batch_size=batch_size,
    )


@gin.configurable
def train_spec(
        cpus_per_trial=4, gpus_per_trial=1, max_epochs=1000,
        outer_config=outer_config, num_samples=4, local_dir=None):
    if local_dir is not None:
        local_dir = os.path.expandvars(os.path.expanduser(local_dir))
    return {
        'run': TuneModel,
        'resources_per_trial': {
            'cpu': cpus_per_trial,
            'gpu': gpus_per_trial,
        },
        'stop': {
            'training_iteration': max_epochs,
        },
        'config': (
            outer_config() if callable(outer_config) else outer_config),
        'num_samples': num_samples,
        'local_dir': local_dir,
    }


@actions.register
@gin.configurable
def search(
        name='aug_base',
        train_spec=train_spec,
        objective='val_sparse_categorical_accuracy',
        mode='max',
        perturbation_interval=5,
        custom_explore_fn=explore,
        log_config=True,
        time_attr='training_iteration',
        max_failures=1,
        resume=True,
        clean=False,
        ):
    import ray
    from ray.tune.schedulers import PopulationBasedTraining
    from ray.tune import run
    from ray.tune import Experiment

    if callable(train_spec):
        train_spec = train_spec()

    pbt = PopulationBasedTraining(
        time_attr=time_attr,
        metric=objective,
        mode=mode,
        perturbation_interval=perturbation_interval,
        custom_explore_fn=custom_explore_fn,
        log_config=log_config)

    experiment = Experiment.from_json(name=name, spec=train_spec)
    if clean and os.path.exists(experiment.local_dir):
        import shutil
        shutil.rmtree(experiment.local_dir)

    ray.init()
    run(
        experiment,
        name=name,
        scheduler=pbt,
        reuse_actors=True,
        verbose=True,
        resume=resume,
        max_failures=max_failures,
    )


# def explore(config):
#     '''Custom explore function.

#     Args:
#         config: dictionary containing ray config params.

#     Returns:
#         Copy of config with modified augmentation policy.
#     '''
#     def default_factor(loc=1, scale=0.1):
#         return trunc_norm(loc=loc, scale=scale)

#     def up_or_down():
#         return 1 if np.random.uniform() > 0.5 else -1

#     config = copy.deepcopy(config)
#     aug_spec = config['augmentation_spec']['train']

#     aug_spec['jitter_positions_stddev'] *= default_factor()
#     diff = aug_spec['scale_max'] - 1
#     diff *= default_factor()
#     aug_spec['scale_min'] = 1 - diff
#     aug_spec['scale_max'] = 1 + diff
#     aug_spec['rigid_transform_stddev'] *= default_factor()
#     if np.random.uniform() < 0.2:
#         aug_spec['maybe_reflect_x'] = not aug_spec['maybe_reflect_x']
#     if np.random.uniform() < 0.2:
#         aug_spec['perlin_grid_shape'] = max(
#             2, aug_spec['perlin_grid_shape'] + up_or_down())
#     aug_spec['perlin_stddev'] *= default_factor()
#     # if np.random.uniform() < 0.2:
#     #     aug_spec['ffd_grid_shape'] = max(
#     #         2, aug_spec['ffd_grid_shape'] + up_or_down())
#     # aug_spec['ffd_stddev'] *= default_factor()
#     return config


# # quick hack
# class Flags(object):
#     pass


# FLAGS = Flags()

# FLAGS.name = 'test'
# FLAGS.local_dir = '/tmp/ray_test'
# FLAGS.checkpoint_freq = 5
# FLAGS.cpu = 4
# FLAGS.gpu = 1

# FLAGS.num_samples = 4
# FLAGS.perturbation_interval = 5
# FLAGS.max_epochs = 100
# FLAGS.max_failures = 1
# FLAGS.rotate_scheme = 'random'
# # FLAGS.rotate_scheme = 'pca-xy'

# # # SMOKE-TEST
# # FLAGS.num_samples = 2
# # FLAGS.perturbation_interval = 1
# # FLAGS.checkpoint_freq = 1


# def main(_):
#     import ray
#     from ray.tune.schedulers import PopulationBasedTraining
#     from ray.tune import run
#     from ray.tune import Experiment
#     # from ray.tune import run_experiments
#     import tensorflow as tf
#     from pointnet import problems
#     from pointnet.tune_model import TuneModel

#     logging.set_verbosity(logging.INFO)

#     train_spec = {
#         'run': TuneModel,
#         'resources_per_trial': {
#             'cpu': FLAGS.cpu,
#             'gpu': FLAGS.gpu,
#         },
#         'stop': {
#             'training_iteration': FLAGS.max_epochs,
#         },
#         'config': {
#             # 'batch_size': 2,  # SMOKE-TEST
#             'batch_size': 32,
#             'problem': {
#                 'name': 'modelnet40',
#             },
#             'augmentation_spec': {
#                 'train': {
#                     'jitter_positions_stddev': 1e-2,
#                     'scale_min': 0.95,
#                     'scale_max': 1.05,
#                     'rigid_transform_stddev': 1e-2,
#                     'maybe_reflect_x': True,
#                     'perlin_grid_shape': 4,
#                     'perlin_stddev': 0.25,
#                     # 'ffd_grid_shape': 4,
#                     # 'ffd_stddev': 0.2,
#                     'rotate_scheme': FLAGS.rotate_scheme,
#                 },
#                 'validation': {
#                     'rotate_scheme': FLAGS.rotate_scheme,
#                 },
#             },
#             'model_fn': {
#                 'name': 'pointnet_classifier'
#             },
#             'optimizer': tf.keras.utils.serialize_keras_object(
#                 tf.keras.optimizers.SGD(momentum=0.9, lr=1e-3, nesterov=True))
#         },
#         # 'local_dir': FLAGS.local_dir,
#         # 'checkpoint_freq': FLAGS.checkpoint_freq,
#         'num_samples': FLAGS.num_samples,
#         # 'max_failures': FLAGS.max_failures,
#     }

#     problem = problems.deserialize(**train_spec['config']['problem'])
#     objective = problem.objective

#     pbt = PopulationBasedTraining(
#         time_attr='training_iteration',
#         metric=objective.name,
#         mode=objective.mode,
#         perturbation_interval=FLAGS.perturbation_interval,
#         custom_explore_fn=explore,
#         log_config=True)

#     ray.init()
#     run(
#         Experiment.from_json(name=FLAGS.name, spec=train_spec),
#         name=FLAGS.name,
#         scheduler=pbt,
#         reuse_actors=True,
#         verbose=True,
#         resume=True,
#         max_failures=FLAGS.max_failures,
#     )


# if __name__ == '__main__':
#     app.run(main)
# # main(None)
