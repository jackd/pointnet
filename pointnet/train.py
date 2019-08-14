from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from pointnet import callbacks as cb
from pointnet import fns
from pointnet import problems


def get_model(problem, model_fn, optimizer, compiled=True, training=None):
    inputs = tf.nest.map_structure(
        lambda spec: tf.keras.layers.Input(shape=spec.shape, dtype=spec.dtype),
        problem.input_spec)

    model = model_fn(inputs, training=training, output_spec=problem.output_spec)
    if compiled:
        model.compile(loss=problem.loss,
                      metrics=problem.metrics,
                      optimizer=optimizer)
    return model


def train(
        problem,
        model_fn,
        optimizer,
        batch_size,
        chkpt_callback=None,
        callbacks=None,
        epochs=100,
        verbose=True,
        fresh=False,
        training=None,  # make True to avoid possible bn issues
):
    model = get_model(problem,
                      model_fn,
                      optimizer,
                      compiled=True,
                      training=training)

    initial_epoch = 0
    if chkpt_callback is not None:
        if fresh:
            if tf.io.gfile.isdir(chkpt_callback.directory):
                tf.io.gfile.rmtree(chkpt_callback.directory)
        elif chkpt_callback.load_weights_on_restart:
            chkpt_callback.set_model(model)
            chkpt = chkpt_callback.latest_checkpoint
            if chkpt is not None:
                initial_epoch = chkpt_callback.epoch(chkpt)
                chkpt_callback.restore(chkpt)

    if callbacks is None:
        callbacks = []
    callbacks = [chkpt_callback] + callbacks

    train_ds, val_ds = problem.get_dataset(split=('train', 'validation'),
                                           batch_size=batch_size,
                                           repeats=None)
    train_steps, val_steps = problem.examples_per_epoch(('train', 'validation'),
                                                        batch_size=batch_size)
    history = model.fit(
        train_ds,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        initial_epoch=initial_epoch,
    )
    return history


def evaluate(problem,
             model_fn,
             optimizer,
             batch_size,
             chkpt_callback,
             split='validation',
             verbose=True,
             training=None):
    model = get_model(problem,
                      model_fn,
                      optimizer,
                      compiled=True,
                      training=training)
    chkpt_callback.set_model(model)
    chkpt_callback.restore_latest()
    return model.evaluate(problem.get_dataset(split,
                                              batch_size=batch_size,
                                              repeats=False),
                          steps=problem.examples_per_epoch(
                              split, batch_size=batch_size),
                          verbose=verbose)
