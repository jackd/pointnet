from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import logging
import os
import copy
import numpy as np
import gin
from pointnet.tune_model import TuneModel


def trunc_norm(loc, scale):
    x = np.clip(np.random.normal(), -2, 2)
    return loc + scale*x


@gin.configurable
def mutable_bindings(
        jitter_positions_stddev=1e-2,
        random_scale_stddev=0.05,
        random_rigid_transform_stddev=0.05,
        perlin_grid_shape=None,
        perlin_stddev=None):
    """Get initial mutable bindings."""
    bindings = {
        'train/augment_cloud.jitter_stddev': jitter_positions_stddev,
        'train/augment_cloud.scale_stddev': random_scale_stddev,
        'train/augment_cloud.rigid_transform_stddev': random_rigid_transform_stddev,
        'train/augment_cloud.perlin_grid_shape': perlin_grid_shape,
        'train/augment_cloud.perlin_stddev': perlin_stddev,
    }
    return {k: v for k, v in bindings.items() if v is not None}


@gin.configurable
def mutate_int_value(
        perlin_grid_shape, prob=0.2, min_value=3, max_value=10):
    if np.random.uniform() < prob:
        if np.random.uniform() < 0.5:
            perlin_grid_shape = max(perlin_grid_shape-1, min_value)
        else:
            perlin_grid_shape = min(perlin_grid_shape + 1, max_value)
    return perlin_grid_shape


def explore(config):
    def default_factor(loc=1, scale=0.1):
        return trunc_norm(loc=loc, scale=scale)

    config = copy.deepcopy(config)
    mutable_bindings = config['mutable_bindings']
    for k, scale in (
            ('train/augment_cloud.jitter_stddev', 0.1),
            ('train/augment_cloud.scale_stddev', 0.1),
            ('train/augment_cloud.scale_stddev', 0.05),
            ('train/augment_cloud.perlin_stddev', 0.1),
        ):
        if k in mutable_bindings:
            mutable_bindings[k] *= default_factor(scale=scale)
            config['reset_generators'] = True

    if 'train/perlin_grid_shape' in mutable_bindings:
        config['train/augment_cloud.perlin_grid_shape'] = mutate_int_value(
            config['train/augment_cloud.perlin_grid_shape'])
        config['reset_generators'] = True

    return config


@gin.configurable
def outer_config(
        inner_config_files,
        inner_bindings=[],
        inner_mutable_bindings=mutable_bindings,
        initial_weights_path=None,
        config_dir=None,
        verbosity=logging.INFO):
    if callable(inner_mutable_bindings):
        inner_mutable_bindings = inner_mutable_bindings()
    return dict(
        config_dir=config_dir,
        config_files=inner_config_files,
        bindings=inner_bindings,
        mutable_bindings=inner_mutable_bindings,
        verbosity=verbosity,
    )


@gin.configurable
def train_spec(
        outer_config, cpus_per_trial=4, gpus_per_trial=1, max_epochs=1000,
        num_samples=4, checkpoint_freq=5, max_failures=1,
        local_dir=None):
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
        'checkpoint_freq': checkpoint_freq,
        'local_dir': local_dir,
        'max_failures': max_failures,
    }


@gin.configurable
def aug_opt(
        name,
        train_spec,
        objective='val_sparse_categorical_accuracy',
        mode='max',
        perturbation_interval=5,
        custom_explore_fn=explore,
        log_config=True,
        time_attr='training_iteration',
        resume=True,
        clean=False,
        num_cpus=None,
        num_gpus=None,
        local_mode=False,  # switch to true for debugging
        ):
    import ray
    from ray.tune.schedulers import PopulationBasedTraining
    from ray.tune import run
    from ray.tune import Experiment

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

    ray.init(num_gpus=num_gpus, num_cpus=num_cpus, local_mode=local_mode)
    run(
        experiment,
        name=name,
        scheduler=pbt,
        reuse_actors=True,
        verbose=True,
        resume=resume,
    )
