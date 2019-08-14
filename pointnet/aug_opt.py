from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import logging
import os
import copy
import numpy as np
import gin
from pointnet.tune_model import GinTuneModel
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray import tune
from ray.tune import schedulers
from pointnet.bin.util import DEFAULT


def trunc_norm(loc, scale):
    x = np.clip(np.random.normal(), -2, 2)
    return loc + scale * x


@gin.configurable
def mutable_bindings(jitter_positions_stddev=1e-2,
                     random_scale_stddev=0.05,
                     random_rigid_transform_stddev=0.05,
                     perlin_grid_shape=4,
                     perlin_stddev=0.25):
    """Get initial mutable bindings."""
    bindings = {
        'train/augment_cloud.jitter_stddev':
            jitter_positions_stddev,
        'train/augment_cloud.scale_stddev':
            random_scale_stddev,
        'train/augment_cloud.rigid_transform_stddev':
            random_rigid_transform_stddev,
        'train/augment_cloud.perlin_grid_shape':
            perlin_grid_shape,
        'train/augment_cloud.perlin_stddev':
            perlin_stddev,
    }
    return {k: v for k, v in bindings.items() if v is not None}


@gin.configurable
def mutate_int_value(perlin_grid_shape, prob=0.2, min_value=3, max_value=10):
    if np.random.uniform() < prob:
        if np.random.uniform() < 0.5:
            perlin_grid_shape = max(perlin_grid_shape - 1, min_value)
        else:
            perlin_grid_shape = min(perlin_grid_shape + 1, max_value)
    return perlin_grid_shape


@gin.configurable
def custom_explore(config):

    def default_factor(loc=1, scale=0.1):
        return trunc_norm(loc=loc, scale=scale)

    config = copy.deepcopy(config)
    mutable_bindings = config['mutable_bindings']
    for k, scale in (
        ('train/pointnet.augment.augment_cloud.jitter_stddev', 0.1),
        ('train/pointnet.augment.augment_cloud.scale_stddev', 0.1),
        ('train/pointnet.augment.augment_cloud.scale_stddev', 0.05),
        ('train/pointnet.augment.augment_cloud.perlin_stddev', 0.1),
    ):
        if k in mutable_bindings:
            mutable_bindings[k] *= default_factor(scale=scale)
            config['reset_generators'] = True

    if 'train/pointnet.augment.perlin_grid_shape' in mutable_bindings:
        config[
            'train/pointnet.augment.augment_cloud.perlin_grid_shape'] = mutate_int_value(
                config['train/pointnet.augment.augment_cloud.perlin_grid_shape']
            )
        config['reset_generators'] = True

    return config


def clipped_normal(mean=0, stddev=1, clip=2):
    return np.clip(np.random.normal(), -clip, clip) * stddev + mean


@gin.configurable
def random_bindings():
    import random
    return {
        'augment_cloud.rotate_scheme':
            tune.choice(['random', 'pca-xy']),
        'train/pointnet.augment.augment_cloud.jitter_stddev':
            tune.sample_from(lambda spec: 10**(-(random.random() * 4 + 1))),
        'train/pointnet.augment.augment_cloud.scale_stddev':
            tune.sample_from(lambda spec: 10**(-(random.random() * 5))),
        'train/pointnet.augment.augment_cloud.rigid_transform_stddev':
            tune.sample_from(lambda spec: 10**(-(random.random() * 4 + 1))),
        'train/pointnet.augment.augment_cloud.perlin_grid_shape':
            tune.randint(2, 11),
        'train/pointnet.augment.augment_cloud.perlin_stddev':
            tune.sample_from(lambda spec: 5 * 10**(-(random.random() * 2 + 1))),
        'train/pointnet.augment.augment_cloud.maybe_reflect_x':
            tune.choice([False, True])
    }


@gin.configurable
def search_space(rotate_scheme=('random', 'pca-xy'),
                 maybe_reflect_x=(False, True),
                 jitter=True,
                 scale=True,
                 rigid_transform=True,
                 perlin=True):
    from hyperopt import hp
    bindings = {}
    out = {'mutable_bindings': bindings}
    if isinstance(rotate_scheme, (list, tuple)):
        rotate_scheme = hp.choice('rotate_scheme', rotate_scheme)
    # macro below doesn't seem to work?
    # bindings['rotate_scheme'] = rotate_scheme
    bindings['pointnet.augment.augment_cloud.rotate_scheme'] = rotate_scheme

    if isinstance(maybe_reflect_x, (list, tuple)):
        maybe_reflect_x = hp.choice('maybe_reflect_x', maybe_reflect_x)
    bindings['train/augment_cloud.maybe_reflect_x'] = maybe_reflect_x

    if jitter:
        bindings['train/augment_cloud.jitter_stddev'] = hp.choice(
            'jitter', (
                None,
                hp.loguniform('jitter_stddev', np.log(1e-5), np.log(1e-1)),
            ))

    if scale:
        bindings['train/augment_cloud.scale_stddev'] = hp.choice(
            'scale', (
                None,
                hp.loguniform('scale_stddev', np.log(1e-5), np.log(2e-1)),
            ))

    if rigid_transform:
        bindings['train/augment_cloud.rigid_transform_stddev'] = hp.choice(
            'rigid_transform', (
                None,
                hp.loguniform('train/augment_cloud.rigid_transform_stddev',
                              np.log(1e-5), np.log(1e-1)),
            ))

    if perlin:
        bindings['perlin'] = hp.choice('perlin', (None, {
            'train/augment_cloud.perlin_grid_shape':
                hp.quniform('perlin_grid_shape', 2, 5, 1),
            'train/augment_cloud.perlin_stddev':
                hp.loguniform('perlin_stddev', np.log(1e-3), np.log(0.5))
        }))

    return out


@gin.configurable
def tune_config(inner_config_files,
                inner_mutable_bindings,
                inner_bindings=[],
                initial_weights_path=None,
                verbosity=logging.INFO,
                allow_growth=True):
    out = dict(
        config_files=inner_config_files,
        bindings=inner_bindings,
        verbosity=verbosity,
        initial_weights_path=initial_weights_path,
        allow_growth=allow_growth,
    )
    if inner_mutable_bindings is not None:
        out['mutable_bindings'] = inner_mutable_bindings
    return out


def trial_str_creator(trial):
    return "{}_{}_123".format(trial.trainable_name, trial.trial_id)


@gin.configurable
def train_spec(tune_config,
               cpus_per_trial=4,
               gpus_per_trial=1,
               max_epochs=1000,
               num_samples=4,
               checkpoint_freq=5,
               max_failures=1,
               local_dir=None):
    return {
        'run': GinTuneModel,
        'resources_per_trial': {
            'cpu': cpus_per_trial,
            'gpu': gpus_per_trial,
        },
        'stop': {
            'training_iteration': max_epochs,
        },
        'config': tune_config,
        'num_samples': num_samples,
        'checkpoint_freq': checkpoint_freq,
        'local_dir': local_dir,
        'max_failures': max_failures,
    }


PopulationBasedTraining = gin.external_configurable(
    schedulers.PopulationBasedTraining)
AsyncHyperBandScheduler = gin.external_configurable(
    schedulers.AsyncHyperBandScheduler)
MedianStoppingRule = gin.external_configurable(schedulers.MedianStoppingRule)
HyperOptSearch = gin.external_configurable(HyperOptSearch)


@gin.configurable
def scheduler(cls,
              metric='val_sparse_categorical_accuracy',
              mode='max',
              time_attr='training_iteration',
              **kwargs):

    return cls(
        metric=metric,
        mode=mode,
        time_attr=time_attr,
        **kwargs,
    )


@gin.configurable
def aug_opt(
        name,
        train_spec,
        scheduler,
        inner_config_dir,
        resume=None,
        fresh=False,
        search_alg=None,
):
    from ray.tune import run
    from ray.tune import Experiment
    train_spec['config']['config_dir'] = inner_config_dir
    if resume is None:
        resume = not fresh

    experiment = Experiment.from_json(name=name, spec=train_spec)
    if fresh and os.path.exists(experiment.local_dir):
        import shutil
        shutil.rmtree(experiment.local_dir)

    run(experiment,
        name=name,
        scheduler=scheduler,
        reuse_actors=True,
        verbose=True,
        resume=resume,
        search_alg=search_alg,
        trial_name_creator=tune.function(trial_str_creator))
