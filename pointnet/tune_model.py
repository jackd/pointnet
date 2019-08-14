from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from ray.tune import Trainable
import tensorflow as tf
import os
import gin
import functools
import six
import tempfile
import time

from pointnet import problems
from pointnet import models
from pointnet import augment as aug
from pointnet.bin import util
from pointnet import callbacks as cb
from pointnet import blocks


def get_tensorboard_hparams(rotate_scheme=('random', 'pca-xy'),
                            maybe_reflect_x=(False, True),
                            jitter=True,
                            scale=True,
                            rigid_transform=True,
                            perlin=True):
    from tensorboard.plugins.hparams import api as hp
    hparams = {}
    if isinstance(rotate_scheme, (list, tuple)):
        hparams['augment_cloud.rotate_scheme'] = hp.HParam(
            'rotate_scheme', hp.Discrete(rotate_scheme))
    if isinstance(maybe_reflect_x, (list, tuple)):
        hparams['train/augment_cloud.maybe_reflect_x'] = hp.HParam(
            'maybe_reflect_x', hp.Discrete(maybe_reflect_x))
    if jitter:
        hparams['train/augment_cloud.jitter_stddev'] = hp.HParam(
            'jitter_stddev', hp.RealInterval(1e-5, 1e-1))
    if scale:
        hparams['train/augment_cloud.scale_stddev'] = hp.HParam(
            'scale', hp.RealInterval(1e-5, 2e-1))
    if rigid_transform:
        hparams['train/augment_cloud.rigid_transform_stddev'] = hp.HParam(
            'rigid_transform', hp.RealInterval(1e-5, 1e-1))
    if perlin:
        hparams['train/augment_cloud.perlin_grid_shape'] = hp.HParam(
            'perlin_grid_shape', hp.IntInterval(2, 5))
        hparams['train/augment_cloud.perlin_stddev'] = hp.HParam(
            'perlin_stddev', hp.RealInterval(1e-3, 0.5))
    return hparams


def get_tensorboard_hparams_callback(value_dict, log_dir):
    from tensorboard.plugins.hparams import api as hp
    key_dict = get_tensorboard_hparams()
    hparams = {key_dict[k]: v for k, v in value_dict.items()}
    return hp.KerasCallback(log_dir, hparams)


def operative_config_path(log_dir, iteration=0):
    return os.path.join(log_dir, 'operative_config{}.gin'.format(iteration))


def parse_operative_config_path(path):
    log_dir, filename = os.path.split(path)
    iteration = int(filename[len('operative_config'):-4])
    return log_dir, iteration


def _parse_config_item(key, value):
    if isinstance(value, dict):
        for k, v in value.items():
            _parse_config_item(k, v)
        return
    elif isinstance(value, (list, tuple)):
        for k, v in value:
            _parse_config_item(k, v)
    elif value is None:
        return
    else:
        assert (key is not None)
        # if isinstance(value, six.string_types):
        #     gin.bind_parameter(key, '"{}"'.format(value))
        # else:
        gin.bind_parameter(key, value)


class GinTuneModel(Trainable):
    """
    Experiments are configured entirely via gin.

    see `TUNE_MODEL_CONFIG` for macros to override.

    config dictionary passed to constructor should have
        str             config_dir
        List<str>       config_files
        List<str>       bindings (variations from config_files)
        Dict<string, ?> mutable_bindings (presumably changed each mutation)
        int             verbosity

    Mutations are expected in `mutable_bindings`, and should also provide the
    flags (assumed `False` if absent)
        bool reset_optimizer
        bool reset_problem
        bool reset_generators
        bool reset_model
        bool reset_session
    to flag whether or not the mutations require resetting the relevant
    objects.
    """

    def _setup(self, config):
        util.tf_init(gpus=None,
                     allow_growth=config.get('allow_growth', True),
                     eager=False)
        logging.set_verbosity(config.get('verbosity', logging.INFO))

        logging.info("calling setup")
        config_files = config['config_files']
        if isinstance(config_files, six.string_types):
            config_files = [config_files]

        with gin.unlock_config():
            config_dir = config.get('config_dir')
            if config_dir is None:
                config_dir = util.get_config_dir()
            util.parse_config(config_dir,
                              config_files,
                              config['bindings'],
                              finalize_config=False)
            _parse_config_item(None, config['mutable_bindings'])
            gin.finalize()

        self._generators = None
        self._problem = None
        self._optimizer = None
        self._model = None
        self._reset_callbacks()

        wp = config.get('initial_weights_path', None)
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
            self.model  # force creation via getter
            self._restore(checkpoint)

    @property
    def problem(self):
        if self._problem is None:
            self._problem = blocks.problem()
        return self._problem

    @property
    def generators(self):
        if self._generators is None:
            self._generators = {
                k: self.problem.get_generator(split=k,
                                              batch_size=self.batch_size,
                                              repeats=None)
                for k in ('train', 'validation')
            }
        return self._generators

    @property
    def model(self):
        if self._model is None:
            inputs = tf.nest.map_structure(
                lambda x: tf.keras.layers.Input(shape=x.shape, dtype=x.dtype),
                self.problem.input_spec)

            # create/compile model and checkpoint
            # note: we use training=True even during evaluation
            # this ensure spurious errors caused by batch norm state being out
            # of sync do not unfairly punish potentially good performing models
            self._model = blocks.model_fn()(  # pylint: disable=not-callable
                inputs,
                training=True,
                output_spec=self.problem.output_spec)
            self._model.compile(loss=self.problem.loss,
                                metrics=self.problem.metrics,
                                optimizer=self.optimizer)
        return self._model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = blocks.optimizer()
        return self._optimizer

    @property
    def batch_size(self):
        return blocks.batch_size()

    def steps_per_epoch(self, split):
        return self.problem.examples_per_epoch(split) // self.batch_size

    def _train(self):
        """Runs one epoch of training, and returns current epoch accuracies."""
        logging.info("training for iteration: {}".format(self._iteration))
        epoch = self._iteration

        generators = self.generators
        t = time.time()
        history = self.model.fit(
            generators['train'],
            epochs=epoch + 1,
            initial_epoch=epoch,
            validation_data=generators['validation'],
            steps_per_epoch=self.steps_per_epoch('train'),
            validation_steps=self.steps_per_epoch('validation'),
            verbose=False,
            callbacks=self._callbacks)
        dt = time.time() - t
        time_str = 'Step took {:.2f}s'.format(dt)
        vals = {k: v[-1] for k, v in history.history.items()}
        vals_str = '\n'.join(
            '{:35s}: {:f}'.format(k, vals[k]) for k in sorted(vals))
        logging.info("finished iteration {}\n{}\n{}".format(
            self._iteration, vals_str, time_str))
        return vals

    def _save(self, checkpoint_dir):
        """Uses tf trainer object to checkpoint."""
        if not tf.io.gfile.isdir(checkpoint_dir):
            tf.io.gfile.makedirs(checkpoint_dir)
        save_name = os.path.join(
            checkpoint_dir,
            'model-{epoch:05d}.h5'.format(epoch=self._iteration))
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
        if not tf.io.gfile.isdir(self.logdir):
            tf.io.gfile.makedirs(self.logdir)
        path = operative_config_path(self.logdir, self._iteration)
        with open(path, 'w') as fp:
            fp.write(gin.operative_config_str())

    def _reset_callbacks(self):
        self._callbacks = [
            # get_tensorboard_hparams_callback(self.config['mutable_bindings'],
            #                                  self.logdir)
        ]

    def reset_config(self, new_config):
        """Resets trainer config for fast PBT implementation."""
        with gin.unlock_config():
            _parse_config_item(None, new_config['mutable_bindings'])

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
        self._reset_callbacks()
        self._save_operative_config()
        return True


@gin.configurable
def evaluate_tune(log_dir,
                  split='validation',
                  verbose=True,
                  eval_batch_size=None):
    from pointnet import train
    log_dir = os.path.realpath(os.path.expanduser(os.path.expandvars(log_dir)))
    paths = [os.path.join(log_dir, d) for d in tf.io.gfile.listdir(log_dir)]
    op_configs = [d for d in paths if d.endswith('.gin')]
    op_config = max(op_configs, key=lambda x: parse_operative_config_path(x)[1])
    dirs = [d for d in paths if tf.io.gfile.isdir(d)]
    final_dir = max(dirs, key=lambda x: int(x.split('_')[-1]))
    logging.info('Using operative_config {}, checkpoint {}'.format(
        op_config, final_dir))

    with gin.unlock_config():
        gin.parse_config_file(op_config)
    chkpt_callback = cb.ModelCheckpoint(final_dir)
    if eval_batch_size is None:
        eval_batch_size = blocks.batch_size()

    train.evaluate(blocks.problem(),
                   blocks.model_fn(),
                   blocks.optimizer(),
                   eval_batch_size,
                   chkpt_callback,
                   split=split,
                   verbose=verbose)
