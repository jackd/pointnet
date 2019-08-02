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

from pointnet import problems
from pointnet import models


class TuneModel(tune.Trainable):

    def _setup(self, *args):
        logging.set_verbosity(logging.INFO)
        logging.info("calling setup")
        self.problem = problems.deserialize(**self.config['problem'])
        # construct problem/datasets/inputs
        batch_size = self.config['batch_size']
        self.steps_per_epoch = {
            k: self.problem.examples_per_epoch(k) // batch_size
            for k in ('train', 'validation')}
        self._reset_datasets()
        inputs = tf.nest.map_structure(
            lambda x: tf.keras.layers.Input(shape=x.shape, dtype=x.dtype),
            self.problem.input_spec)

        # create/compile model and checkpoint
        self.model = models.deserialize(
            inputs, training=None, output_spec=self.problem.output_spec,
            **self.config['model_fn'])
        self.model.compile(
            loss=self.problem.loss,
            metrics=self.problem.metrics,
            optimizer=tf.keras.optimizers.deserialize(self.config['optimizer'])
        )
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.model.optimizer, model=self.model)
        # self.log_dir = self.config['log_dir']

    def _train(self):
        """Runs one epoch of training, and returns current epoch accuracies."""
        logging.info("training for iteration: {}".format(self._iteration))
        epoch = self._iteration
        history = self.model.fit(
            self.datasets['train'],
            epochs=epoch+1,
            initial_epoch=epoch,
            validation_data=self.datasets['validation'],
            steps_per_epoch=self.steps_per_epoch['train'],
            validation_steps=self.steps_per_epoch['validation'],
            # callbacks=[self.tb_callback],
        )
        return {k: v[-1] for k, v in history.history.items()}

    def _save(self, checkpoint_dir):
        """Uses tf trainer object to checkpoint."""
        save_name = os.path.join(
            checkpoint_dir, 'model-%05d' % self._iteration)
        save_path = self.checkpoint.save(save_name)
        logging.info("saved model {}".format(save_path))
        return save_path

    def _restore(self, save_path):
        """Restores model from checkpoint."""
        logging.info("RESTORING: {}".format(save_path))
        status = self.checkpoint.restore(save_path)
        status.assert_consumed()

    def _reset_datasets(self):
        self.datasets = {
            split: self.problem.get_dataset(
                split=split, batch_size=self.config['batch_size'])
            for split in ('train', 'validation')}

    # def _reset_callbacks(self):
    #     self.tb_callback = tf.keras.callbacks.TensorBoard(self.log_dir)

    def reset_config(self, new_config):
        """Resets trainer config for fast PBT implementation."""
        self.problem.reset_map_fn(new_config['augmentation_spec'])
        self.config = new_config
        self._reset_datasets()
        return True
