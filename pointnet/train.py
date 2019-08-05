from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import gin
from pointnet import callbacks as cb
from pointnet.cli.actions import actions


@actions.register
@gin.configurable
def train(
        problem=None, map_fn=None, model_fn=None,
        optimizer=None, batch_size=32, verbose=True, epochs=100,
        chkpt_callback=None, callbacks=None, resume=True,
        save_config=True):
    if problem is None:
        from pointnet import problems
        problem = problems.deserialize()
    if model_fn is None:
        from pointnet import models
        model_fn = models.deserialize()
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)

    train_steps, validation_steps = (
        problem.examples_per_epoch(k) // batch_size
        for k in ('train', 'validation'))

    train_ds, val_ds = (
        problem.get_dataset(split=k, batch_size=batch_size, map_fn=map_fn)
        for k in ('train', 'validation'))

    inputs = tf.nest.map_structure(
        lambda shape, dtype: tf.keras.layers.Input(
            shape=shape[1:], dtype=dtype),
        train_ds.output_shapes[0], train_ds.output_types[0])

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
        initial_epoch = chkpt_callback.latest_epoch

        if initial_epoch is None:
            initial_epoch = 0

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


if __name__ == '__main__':
    gin_config = '''
    import pointnet.models
    import pointnet.problems
    import pointnet.augment
    import pointnet.keras_configurables
    import pointnet.callbacks

    model_dir = '/tmp/pointnet_default'

    train.problem = @ModelnetProblem()
    train.map_fn = {
        'train': @train/augment_cloud,
        'validation': @validation/augment_cloud,
    }
    train.model_fn = @pointnet_classifier
    # train.batch_size = 2                            # SMOKE-TEST
    train.batch_size = 32                           # SMOKE-TEST
    train.optimizer = @tf.keras.optimizers.SGD()
    tf.keras.optimizers.SGD.lr = 1e-3
    tf.keras.optimizers.SGD.momentum = 0.99

    # train.chkpt_callback = @pointnet.callbacks.ModelCheckpoint()
    # pointnet.callbacks.ModelCheckpoint.directory = %model_dir
    # train.chkpt_callback = @pointnet.callbacks.CheckpointCallback()
    # pointnet.callbacks.CheckpointCallback.directory = %model_dir
    train.chkpt_callback = @pointnet.callbacks.ModelCheckpoint()
    pointnet.callbacks.ModelCheckpoint.load_weights_on_restart = True
    pointnet.callbacks.ModelCheckpoint.directory = %model_dir

    # rotate_by_scheme.scheme = 'pca-xy'
    # jitter_positions.stddev = None
    rotate_by_scheme.scheme = 'random'
    # train/jitter_positions.stddev = 1e-2

    tf.keras.callbacks.TensorBoard.log_dir = %model_dir

    tf.keras.callbacks.LearningRateScheduler.schedule = @original_lr_schedule

    train.misc_callbacks = [
        @tf.keras.callbacks.TerminateOnNaN(),
        # @tf.keras.callbacks.LearningRateScheduler(),
        @tf.keras.callbacks.TensorBoard(),
    ]

    '''
    gin.parse_config(gin_config)
    train()
