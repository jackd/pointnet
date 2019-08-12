from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
import six
import tensorflow as tf
import contextlib

# @contextlib.contextmanager
# def temp_attrs(obj, **attrs):
#     keys = sorted(attrs)
#     backups = tuple((k, getattr(obj, k)) for k in keys)

#     def restore():
#         try:
#             for k, v in backups:
#                 setattr(obj, k, v)
#         except Exception:
#             pass

#     try:
#         for k in keys:
#             setattr(obj, k, attrs[k])
#     except Exception:
#         restore()
#         raise
#     yield obj
#     restore()


@gin.configurable(module='pointnet.callbacks')
class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint with slightly extended interface."""

    def __init__(self,
                 directory,
                 save_freq='epoch',
                 load_weights_on_restart=False,
                 max_to_keep=5,
                 **kwargs):
        directory = os.path.expandvars(os.path.expanduser(directory))
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self._directory = directory
        filepath = os.path.join(directory, 'model-{epoch:05d}.h5')
        self._max_to_keep = max_to_keep
        super(ModelCheckpoint,
              self).__init__(filepath=filepath,
                             save_freq=save_freq,
                             load_weights_on_restart=load_weights_on_restart,
                             **kwargs)

    @property
    def directory(self):
        return self._directory

    @property
    def latest_checkpoint(self):
        return self._get_most_recently_modified_file_matching_pattern(
            self.filepath)

    @property
    def latest_epoch(self):
        chkpt = self.latest_checkpoint
        return None if chkpt is None else self.epoch(chkpt)

    def epoch(self, checkpoint):
        return int(checkpoint[-7:-3])

    def restore(self, checkpoint):
        self.model.load_weights(checkpoint)

    def restore_latest(self):
        checkpoint = self.latest_checkpoint
        if checkpoint is not None:
            self.restore(checkpoint)
        return checkpoint

    def _save_model(self, epoch, logs):
        super(ModelCheckpoint, self)._save_model(epoch, logs)
        if self._max_to_keep is not None:
            directory = self.directory
            checkpoints = [
                fn for fn in os.listdir(self.directory)
                if fn.startswith('model') and fn.endswith('.h5')
            ]
            if len(checkpoints) > self._max_to_keep:
                checkpoints = sorted(checkpoints)
                for checkpoint in checkpoints[:-self._max_to_keep]:
                    os.remove(os.path.join(directory, checkpoint))

    def save_model(self, epoch):
        self._save_model(epoch, logs=None)

    # def on_predict_start(self, logs=None):
    #     self.on_test_begin(logs)

    # def on_test_begin(self, logs=None):
    #     self.restore_latest()


# @gin.configurable(module='pointnet.callbacks')
# class CheckpointCallback(tf.keras.callbacks.Callback):
#     def __init__(
#             self, directory, save_freq=1, save_on_end=True,
#             save_optimizer=True, max_to_keep=5,
#             load_weights_on_restart=True,
#             **manager_kwargs):
#         if save_freq == 'epoch':
#             save_freq = 1
#         manager_kwargs['max_to_keep'] = max_to_keep
#         self._manager_kwargs = manager_kwargs
#         self._directory = directory
#         self._save_freq = save_freq
#         self._save_on_end = save_on_end
#         self._save_optimizer = save_optimizer
#         self.load_weights_on_restart = load_weights_on_restart
#         self._last_save = None
#         self._last_restore = None
#         self.epochs = None

#     def set_model(self, model):
#         super(CheckpointCallback, self).set_model(model)

#         with tf.compat.v1.keras.backend.get_session().as_default():
#             self._checkpoint = tf.train.Checkpoint(model=self.model)
#             self._manager = tf.train.CheckpointManager(
#                 self._checkpoint, directory=self._directory,
#                 **self._manager_kwargs)

#     @property
#     def directory(self):
#         return self._directory

#     def save(self):
#         if self._last_save == self.epochs:
#             return self.latest_checkpoint
#         else:
#             self._last_save = self.epochs
#             with tf.compat.v1.keras.backend.get_session().as_default():
#                 return self._manager.save(self.epochs)

#     def on_epoch_end(self, epoch, logs=None):
#         super(CheckpointCallback, self).on_epoch_end(epoch, logs)
#         if self._last_save is None or (
#                 self.epochs - self._last_save >= self._save_freq):
#             self.save()

#     def on_train_end(self, logs=None):
#         super(CheckpointCallback, self).on_train_end(logs)
#         self.epochs += 1
#         if self.epochs is not None and self._save_on_end:
#             self.save()

#     @property
#     def latest_checkpoint(self):
#         return self._manager.latest_checkpoint

#     @property
#     def latest_epoch(self):
#         return self.epoch(self.latest_checkpoint)

#     @property
#     def status(self):
#         return self._status

#     def epoch(self, checkpoint):
#         return (
#             None if checkpoint is None else int(checkpoint.split('-')[-1]) + 1)

#     def restore(self, checkpoint):
#         epoch = self.epoch(checkpoint)
#         if epoch != self._last_restore:
#             with tf.compat.v1.keras.backend.get_session().as_default():
#                 self._status = self._checkpoint.restore(checkpoint)
#                 self._status.initialize_or_restore()
#                 self._status.assert_consumed()
#             self._last_restore = epoch
#             self.epochs = epoch

#     def on_train_begin(self, logs=None):
#         super(CheckpointCallback, self).on_train_begin(logs)
#         self.epochs = self.params['epochs']
#         if self.load_weights_on_restart:
#             self.restore_latest()

#     def on_test_begin(self, logs=None):
#         super(CheckpointCallback, self).on_test_begin(logs)
#         self.restore_latest()

#     def on_predict_begin(self, logs=None):
#         super(CheckpointCallback, self).on_predict_begin(logs)
#         self.restore_latest()

#     def restore_latest(self):
#         if not hasattr(self, '_manager'):
#             raise RuntimeError('Cannot get manager before calling `set_model`')
#         checkpoint = self._manager.latest_checkpoint
#         if checkpoint is None:
#             return None
#         else:
#             self.restore(checkpoint)
#             return checkpoint


@gin.configurable(module='pointnet.callbacks')
class GinConfigWriterCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir):
        if not isinstance(log_dir, six.string_types):
            raise ValueError(
                '`log_dir` must be a string, got {}'.format(log_dir))
        self._log_dir = log_dir

    def on_train_begin(self, logs=None):
        super(GinConfigWriterCallback, self).on_train_begin(logs)
        epochs = self.params['epochs']
        path = os.path.join(self._log_dir, 'operative-config%d.gin' % epochs)
        with tf.io.gfile.GFile(path, 'w') as fp:
            fp.write(gin.operative_config_str())


@gin.configurable(blacklist=['epoch'])
def original_lr_schedule(
        epoch,
        lr0=1e-3,
        # 16881 * 20 is a magic number from original
        # 9843 is number of examples in official modelnet train split
        decay_epochs=(16881 * 20) // 9843,
        decay_ray=0.5,
        min_lr=1e-5,
):
    return max(lr0 * decay_ray**(epoch // decay_epochs), min_lr)


@gin.configurable(module='pointnet.callbacks')
def get_additional_callbacks(terminate_on_nan=True,
                             log_dir=None,
                             lr_schedule=None,
                             save_config=False):
    callbacks = []
    if terminate_on_nan:
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    if log_dir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
    if lr_schedule is not None:
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(schedule=lr_schedule))
    if save_config:
        callbacks.append(GinConfigWriterCallback(log_dir))
    return callbacks
