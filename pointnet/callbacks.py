from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import gin


@gin.configurable(module='pointnet.callbacks')
class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint with slightly extended interface."""
    def __init__(
            self, directory, save_freq='epoch',
            load_weights_on_restart=False,
            max_to_keep=5, **kwargs):
        directory = os.path.expandvars(os.path.expanduser(directory))
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self._directory = directory
        filepath = os.path.join(directory, 'model-{epoch:05d}.h5')
        self._max_to_keep = max_to_keep
        super(ModelCheckpoint, self).__init__(
            filepath=filepath, save_freq=save_freq,
            load_weights_on_restart=load_weights_on_restart, **kwargs)

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
                fn for fn in os.listdir(self.directory) if
                fn.startswith('model') and fn.endswith('.h5')]
            if len(checkpoints) > self._max_to_keep:
                checkpoints = sorted(checkpoints)
                for checkpoint in checkpoints[:-self._max_to_keep]:
                    os.remove(os.path.join(directory, checkpoint))

    def save_model(self, epoch):
        self._save_model(epoch, logs=None)



# @gin.configurable(module='pointnet.callbacks')
# class CheckpointCallback(tf.keras.callbacks.Callback):
#     def __init__(
#             self, directory, save_freq=1, save_on_end=True,
#             save_optimizer=True, max_to_keep=5,
#             restore_on_train_begin=True,
#             **manager_kwargs):
#         if save_freq == 'epoch':
#             save_freq = 1
#         manager_kwargs['max_to_keep'] = max_to_keep
#         self._manager_kwargs = manager_kwargs
#         self._directory = directory
#         self._save_freq = save_freq
#         self._save_on_end = save_on_end
#         self._save_optimizer = save_optimizer
#         self._restore_on_train_begin = restore_on_train_begin
#         self._last_save = None
#         self._epoch = None
#         self._status = None

#     def set_model(self, model):
#         super(CheckpointCallback, self).set_model(model)
#         checkpoint_kwargs = dict(model=self.model)
#         if self._save_optimizer:
#             checkpoint_kwargs['optimizer'] = self.model.optimizer

#         with tf.compat.v1.keras.backend.get_session().as_default():
#             self._checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
#             self._manager = tf.train.CheckpointManager(
#                 self._checkpoint, directory=self._directory,
#                 **self._manager_kwargs)

#     @property
#     def directory(self):
#         return self._directory

#     def _save(self):
#         self._last_save = self._epoch
#         with tf.compat.v1.keras.backend.get_session().as_default():
#             self._manager.save(self._epoch)

#     def on_epoch_end(self, epoch, logs=None):
#         self._epoch = epoch
#         if self._last_save is None or (
#                 epoch - self._last_save >= self._save_freq):
#             self._save()

#     def on_train_end(self, logs=None):
#         if self._epoch is not None and self._save_on_end:
#             self._save()

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
#         return int(checkpoint.split('-')[-1]) + 1

#     def restore(self, checkpoint):
#         with tf.compat.v1.keras.backend.get_session().as_default():
#             self._status = self._checkpoint.restore(checkpoint)

#     def on_train_begin(self, logs=None):
#         if self._restore_on_train_begin:
#             self.restore_latest()
#         if self._status is not None:
#             self._status.assert_consumed()
#             self._status = None

#     def restore_latest(self):
#         checkpoint = self._manager.latest_checkpoint
#         if checkpoint is None:
#             return None
#         else:
#             self.restore(checkpoint)
#             return checkpoint


@gin.configurable(blacklist=['epoch'])
def original_lr_schedule(epoch, lr0=1e-3):
    return lr0 * 2 ** (-epoch // 20)


@gin.configurable(module='pointnet.callbacks')
def get_additional_callbacks(
        terminate_on_nan=True, log_dir=None, lr_schedule=None):
    callbacks = []
    if terminate_on_nan:
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    if log_dir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
    if lr_schedule is not None:
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(schedule=lr_schedule))
    return callbacks
