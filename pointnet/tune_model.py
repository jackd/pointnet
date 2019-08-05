from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import tensorflow as tf
import os
import gin
import functools
import six
import tempfile

from pointnet import problems
from pointnet import models
from pointnet import augment as aug
import gin
from pointnet.cli.config import parse_config


@gin.configurable
def problem(arg=None):
    return arg


@gin.configurable
def optimizer(arg=None):
    return arg


@gin.configurable
def batch_size(arg=None):
    return arg


@gin.configurable
def model_fn(arg=lambda inputs, training, output_spec: None):
    return arg


@gin.configurable
def train_map_fn(arg=None):
    return arg


@gin.configurable
def validation_map_fn(arg=None):
    return arg


@gin.configurable
def initial_weights_path(arg=None):
    return arg


class TuneModel(tune.Trainable):
    """
    Experiments are configured almost entirely via gin.

    gin configs should have the following macros defined:
        train_map_fn:         (inputs, labels) -> (inputs, labels)
        validation_map_fn:    (inputs, labels) -> (inputs, labels)
        optimizer:            () -> optimizer
        model:                (inputs, training, output_spec) -> model
        problem:              () -> problem
        batch_size:           int
        initial_weights_path: optional str, path

    config dictionary passed to constructor should have
        str             config_dir
        List<str>       gin_file
        List<str>       bindings
        Dict<string, ?> mutable_bindings
        int             verbosity (defaults to logging.INFO)

    Mutations are expected in mutable_bindings, and should also provide the
    flags (assumed `False` if absent)
        bool reset_optimizer
        bool reset_problem
        bool reset_generators
        bool reset_model
        bool reset_session
    """

    def _setup(self, *args):
        config = self.config
        logging.set_verbosity(config.get('verbosity', logging.INFO))

        logging.info("calling setup")
        mutable_bindings = [
            '{} = {}'.format(left, right)
            for left, right in config['mutable_bindings'].items()]
        bindings = config['bindings'] + mutable_bindings
        with gin.unlock_config():
            parse_config(
                config['config_dir'], config['gin_file'], bindings)

        self._generators = None
        self._problem = None
        self._optimizer = None
        self._model = None

        wp = initial_weights_path()
        if wp is not None:
            self.model.load_weights(wp)

        self._save_operative_config()

    def reset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = self._save(tmp_dir)
            tf.keras.backend.clear_session()
            self._generators = None
            self._problem = None
            self._optimizers = None
            self._model = None
            self._restore(checkpoint)

    @property
    def problem(self):
        if self._problem is None:
            # self._problem = get_scoped_value('problem')
            self._problem = problem()
        return self._problem

    @property
    def generators(self):
        if self._generators is None:
            map_fns = {
                'train': train_map_fn(),
                'validation': validation_map_fn(),
            }
            self._generators = {k: self.problem.get_generator(
                split=k,
                batch_size=self.batch_size,
                map_fn=map_fns[k]) for k in ('train', 'validation')}
        return self._generators

    @property
    def model(self):
        if self._model is None:
            inputs = tf.nest.map_structure(
                lambda x: tf.keras.layers.Input(shape=x.shape, dtype=x.dtype),
                self.problem.input_spec)

            # create/compile model and checkpoint
            # note: we use training=True even during evaluation
            # this ensure spurious errors caused by batch norm state being out of
            # sync do not unfairly punish potentially good performing models
            self._model = model_fn()(    # pylint: disable=not-callable
                inputs, training=True, output_spec=self.problem.output_spec)
            self._model.compile(
                loss=self.problem.loss,
                metrics=self.problem.metrics,
                optimizer=self.optimizer)
        return self._model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = optimizer()
        return self._optimizer

    @property
    def batch_size(self):
        return batch_size()

    def steps_per_epoch(self, split):
        return self.problem.examples_per_epoch(split) // self.batch_size

    def _train(self):
        """Runs one epoch of training, and returns current epoch accuracies."""
        logging.info("training for iteration: {}".format(self._iteration))
        epoch = self._iteration
        generators = self.generators
        history = self.model.fit(
            generators['train'],
            epochs=epoch+1,
            initial_epoch=epoch,
            validation_data=generators['validation'],
            steps_per_epoch=self.steps_per_epoch('train'),
            validation_steps=self.steps_per_epoch('validation'),
            verbose=False
        )
        vals = {k: v[-1] for k, v in history.history.items()}
        vals_str = '\n'.join('%-35s: %f' % (k, vals[k]) for k in sorted(vals))
        logging.info(
            "finished iteration %d\n%s" % (self._iteration, vals_str))
        return vals

    def _save(self, checkpoint_dir):
        """Uses tf trainer object to checkpoint."""
        save_name = os.path.join(
            checkpoint_dir, 'model-%05d.h5' % self._iteration)
        self.model.save_weights(save_name)
        return save_name
        # manager = tf.train.CheckpointManager(
        #     self.checkpoint, directory=checkpoint_dir, max_to_keep=5)
        # manager.save(save_name)
        # save_path = manager.latest_checkpoint
        # logging.info("saved model {}".format(save_path))
        # return save_path

    def _restore(self, save_path):
        """Restores model from checkpoint."""
        logging.info("RESTORING: {}".format(save_path))
        self.model.load_weights(save_path)

        # self.model = tf.keras.models.load_model(save_path)
        # self.model.load_weights(save_path)
        # self.model = tf.keras.models.load_model(save_path)
        # status = self.checkpoint.restore(save_path)
        # status.assert_consumed()

    def _save_operative_config(self):
        # calling these setters ensures the operative config is complete
        self.generators
        self.model
        path = os.path.join(
            self.logdir, 'operative_config{}.gin'.format(self._iteration))
        with open(path, 'w') as fp:
            fp.write(gin.operative_config_str())

    def reset_config(self, new_config):
        """Resets trainer config for fast PBT implementation."""
        with gin.unlock_config():
            for k, v in new_config['mutable_bindings'].items():
                gin.bind_parameter(k, v)

        if new_config.get('reset_session'):
            self.reset()
        else:
            if new_config.get('reset_optimizer'):
                self._optimizer = None
                self._model = None
            if new_config.get('reset_problem'):
                self._problem = None
                self._generators = None
            if new_config.get('reset_generators'):
                self._generators = None
            if new_config.get('reset_model'):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    checkpoint = self._save(tmp_dir)
                    self._model = None
                    self._restore(checkpoint)
        self.config = new_config
        self._save_operative_config()
        return True





# class TuneModel(tune.Trainable):

#     def _setup(self, *args):
#         logging.set_verbosity(logging.INFO)
#         logging.info("calling setup")
#         self._reset()

#     def _stop(self):
#         tf.keras.backend.clear_session()

#     def _reset(self, save_path=None):
#         tf.keras.backend.clear_session()
#         self.problem = problems.deserialize(**self.config['problem'])
#         self.datasets = {
#             k: self.problem.get_dataset(
#                 split=k,
#                 batch_size=self.batch_size,
#                 map_fn=functools.partial(
#                     aug.flat_augment_cloud,
#                     **self.config['augmentation_spec'][k]))
#                 for k in ('train', 'validation')}
#         inputs = tf.nest.map_structure(
#                 lambda x: tf.keras.layers.Input(shape=x.shape, dtype=x.dtype),
#                 self.problem.input_spec)

#         optimizer = tf.keras.optimizers.deserialize(self.config['optimizer'])

#         # create/compile model and checkpoint
#         # note: we use training=True even during evaluation
#         # this ensure spurious errors caused by batch norm state being out of
#         # sync do not unfairly punish potentially good performing models

#         self.model = models.deserialize(
#             inputs, training=True, output_spec=self.problem.output_spec,
#             **self.config['model_fn'])
#         self.model.compile(
#             loss=self.problem.loss,
#             metrics=self.problem.metrics,
#             optimizer=optimizer,
#         )
#         if save_path is not None:
#             self.model.load_weights(save_path)

#     @property
#     def batch_size(self):
#         return self.config['batch_size']

#     def steps_per_epoch(self, split):
#         return self.problem.examples_per_epoch(split) // self.batch_size

#     def _train(self):
#         """Runs one epoch of training, and returns current epoch accuracies."""
#         logging.info("training for iteration: {}".format(self._iteration))
#         epoch = self._iteration
#         datasets = self.datasets
#         history = self.model.fit(
#             datasets['train'],
#             epochs=epoch+1,
#             initial_epoch=epoch,
#             validation_data=datasets['validation'],
#             steps_per_epoch=self.steps_per_epoch('train'),
#             validation_steps=self.steps_per_epoch('validation'),
#             verbose=False
#         )
#         vals = {k: v[-1] for k, v in history.history.items()}
#         vals_str = '\n'.join('%-35s: %f' % (k, vals[k]) for k in sorted(vals))
#         logging.info(
#             "finished iteration %d\n%s" % (self._iteration, vals_str))
#         return vals

#     def _save(self, checkpoint_dir):
#         """Uses tf trainer object to checkpoint."""
#         save_name = os.path.join(
#             checkpoint_dir, 'model-%05d.h5' % self._iteration)
#         self.model.save_weights(save_name)
#         return save_name
#         # manager = tf.train.CheckpointManager(
#         #     self.checkpoint, directory=checkpoint_dir, max_to_keep=5)
#         # manager.save(save_name)
#         # save_path = manager.latest_checkpoint
#         # logging.info("saved model {}".format(save_path))
#         # return save_path

#     def _restore(self, save_path):
#         """Restores model from checkpoint."""
#         logging.info("RESTORING: {}".format(save_path))
#         self.model.load_weights(save_path)

#         # self.model = tf.keras.models.load_model(save_path)
#         # self.model.load_weights(save_path)
#         # self.model = tf.keras.models.load_model(save_path)
#         # status = self.checkpoint.restore(save_path)
#         # status.assert_consumed()


#     # def different(self, new_config, keys):
#     #     if isinstance(keys, six.string_types):
#     #         keys = keys,
#     #     new_spec = new_config
#     #     old_spec = self.config
#     #     for k in keys:
#     #         new_spec = new_spec[k]
#     #         old_spec = old_spec[k]
#     #     return new_spec != old_spec

#     def reset_config(self, new_config):
#         """Resets trainer config for fast PBT implementation."""
#         # return False
#         self.config = new_config
#         path = '/tmp/tune_model_weights.h5'
#         self.model.save_weights(path)
#         self._reset(save_path=path)
#         os.remove(path)
#         return True


#         # # if self.different(new_config, 'problem'):
#         # #     self._reset_problem(new_config['problem'])
#         # # if (self.different(new_config, 'batch_size') or
#         # #     self.different(new_config, 'augmentation_spec')):
#         # #     self.datasets = None
#         # # if not hasattr(self, 'datasets') or self.datasets is None:
#         # #     self._reset_datasets(rebuild_model=True)
#         # return True
