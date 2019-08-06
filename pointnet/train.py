from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import os
import gin
import tensorflow as tf
from pointnet import callbacks as cb
from pointnet.util.gpu_options import gpu_options


@gin.configurable
def train(
        problem, map_fn, model_fn, optimizer, batch_size=32, verbose=True,
        epochs=100, chkpt_callback=None, callbacks=None,
        save_config=True):
    gpu_options()
    train_steps, validation_steps = (
        problem.examples_per_epoch(k) // batch_size
        for k in ('train', 'validation'))

    train_ds, val_ds = (
        problem.get_dataset(split=k, batch_size=batch_size, map_fn=map_fn)
        for k in ('train', 'validation'))

    inputs = tf.nest.map_structure(
        lambda spec: tf.keras.layers.Input(shape=spec.shape, dtype=spec.dtype),
        problem.input_spec)

    model = model_fn(inputs, training=None, output_spec=problem.output_spec)

    model.compile(
        loss=problem.loss,
        metrics=problem.metrics,
        optimizer=optimizer)

    callbacks = [] if callbacks is None else list(callbacks)

    initial_epoch = 0
    if chkpt_callback is not None:
        callbacks.append(chkpt_callback)
        directory = chkpt_callback.directory
        if chkpt_callback.load_weights_on_restart:
            initial_epoch = chkpt_callback.latest_epoch

            if initial_epoch is None:
                initial_epoch = 0

    if initial_epoch >= epochs:
        logging.info('')

    config_path = os.path.join(
        directory, 'operative_config-{}.gin'.format(initial_epoch))
    with open(config_path, 'w') as fp:
        fp.write(gin.operative_config_str())

    history = model.fit(
        train_ds,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        initial_epoch=initial_epoch
        )

    return history


@gin.configurable
def evaluate(
        problem, map_fn, model_fn, batch_size=32,
        split='validation', chkpt_callback=None):
    gpu_options()
    if problem is None:
        from pointnet import problems
        problem = problems.deserialize()
    if model_fn is None:
        from pointnet import models
        model_fn = models.deserialize()

    inputs = tf.nest.map_structure(
        lambda spec: tf.keras.layers.Input(shape=spec.shape, dtype=spec.dtype),
        problem.input_spec)

    model = model_fn(inputs, training=None, output_spec=problem.output_spec)

    if chkpt_callback is not None:
        latest = chkpt_callback.restore_latest()
        logging.info('Loading models from {}'.format(latest))

    dataset = problem.get_dataset(
        split=split, map_fn=map_fn, batch_size=batch_size)

    return model.evaluate(dataset, steps=problem.examples_per_epoch(split))
