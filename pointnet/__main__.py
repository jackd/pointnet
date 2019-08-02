from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from pointnet.model import pointnet_classifier
from pointnet.problem import ModelnetProblem
from pointnet import augment as aug

batch_size = 32
model_dir = '/tmp/pointnet_default'
lr0 = 1e-3
epochs = 100
verbose = True

optimizer = tf.keras.optimizers.Adam(lr=lr0)

if tf.io.gfile.isdir(model_dir):
    tf.io.gfile.rmtree(model_dir)

tf.io.gfile.makedirs(model_dir)
initial_epoch = 0

shared_kwargs = dict(
    rotate_scheme='pca-xy',
    positions_only=True,
)

problem = ModelnetProblem()
train_steps, validation_steps = (
    problem.examples_per_epoch(k) // batch_size
    for k in ('train', 'validation'))
train_ds, val_ds = (
    problem.get_dataset(split=k, batch_size=batch_size)
    for k in ('train', 'validation'))

inputs = tf.nest.map_structure(
    lambda shape, dtype: tf.keras.layers.Input(shape=shape[1:], dtype=dtype),
    train_ds.output_shapes[0], train_ds.output_types[0])

model = pointnet_classifier(
    inputs, training=None, output_spec=problem.output_spec)


def schedule(t):
    return lr0 * 2 ** (-t // 20)


ckpt_path = os.path.join(model_dir, 'cp-{epoch:04d}.ckpt')
callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=model_dir),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.LearningRateScheduler(schedule),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path),
]

model.compile(
    loss=problem.loss,
    metrics=problem.metrics,
    optimizer=optimizer,
)

model.fit(
    train_ds,
    epochs=epochs,
    verbose=verbose,
    callbacks=callbacks,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=validation_steps,
    initial_epoch=initial_epoch)
