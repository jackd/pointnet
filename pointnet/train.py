from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
import six
import tensorflow as tf
from pointnet import callbacks as cb
from pointnet import fns
from pointnet import problems
from pointnet.util.gpu_options import gpu_options

# @gin.configurable
# class ModelBuilder(object):
#     def __init__(
#             self, problem=None, model_fn=None, optimizer=None, batch_size=32):
#         self.problem = problems.get(problem)
#         self.model_fn = fns.get(model_fn)
#         self.optimizer = tf.keras.optimizers.get(optimizer)
#         self.batch_size = batch_size

#     def get_model(self, compiled=True, training=None):
#         problem = self.problem
#         inputs = tf.nest.map_structure(
#             lambda spec: tf.keras.layers.Input(
#                 shape=spec.shape, dtype=spec.dtype),
#             problem.input_spec)

#         model = self.model_fn(
#             inputs, training=None, output_spec=problem.output_spec)
#         if compiled:
#             model.compile(
#             loss=problem.loss,
#             metrics=problem.metrics,
#             optimizer=self.optimizer)
#         return model

#     @classmethod
#     def from_config(cls, config):
#         return ModelBuilder(
#             config.get('problem'),
#             config.get('model_fn'),
#             config.get('optimizer'),
#             config.get('batch_size'))

#     def get_config(self):
#         return dict(
#             problem=tf.keras.utils.serialize_keras_object(self.problem),
#             model_fn=fns.get_config(self.model_fn),
#             optimizer=tf.keras.utils.serialize_keras_object(self.optimizer),
#             batch_size=self.batch_size,
#         )


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


@gin.configurable
def train(problem,
          model_fn,
          optimizer,
          batch_size,
          chkpt_callback=None,
          callbacks=None,
          epochs=100,
          verbose=True,
          fresh=False):
    gpu_options()
    model = get_model(problem,
                      model_fn,
                      optimizer,
                      compiled=True,
                      training=True)  # less issues with val bn statistics

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


@gin.configurable
def evaluate(problem,
             model_fn,
             optimizer,
             batch_size,
             chkpt_callback,
             split='validation',
             verbose=True):
    gpu_options()
    model = get_model(problem,
                      model_fn,
                      optimizer,
                      compiled=True,
                      training=None)
    chkpt_callback.set_model(model)
    chkpt_callback.restore_latest()
    return model.evaluate(problem.get_dataset(split,
                                              batch_size=batch_size,
                                              repeats=False),
                          steps=problem.examples_per_epoch(
                              split, batch_size=batch_size),
                          verbose=verbose)


@gin.configurable
def evaluate_directory(model_dir,
                       split='validation',
                       verbose=True,
                       eval_batch_size=None):
    import os
    from absl import logging

    model_dir = os.path.realpath(
        os.path.expanduser(os.path.expandvars(model_dir)))
    paths = [os.path.join(model_dir, d) for d in tf.io.gfile.listdir(model_dir)]
    op_configs = [d for d in paths if d.endswith('.gin')]
    if len(op_configs) > 1:
        # max?
        # op_config = max(op_configs, key=lambda x: parse_operative_config_path(x)[1])
        raise NotImplementedError()
    op_config = op_configs[-1]

    # dirs = [d for d in paths if tf.io.gfile.isdir(d)]
    # final_dir = max(dirs, key=lambda x: int(x.split('_')[-1]))

    logging.info('Using operative_config {}, checkpoint {}'.format(
        op_config, model_dir))

    with gin.unlock_config():
        gin.parse_config('''
            evaluate.problem = %problem
            evaluate.model_fn = %model_fn
            evaluate.optimizer = %optimizer
            evaluate.batch_size = %batch_size
        ''')
        gin.parse_config_file(op_config)
    chkpt_callback = cb.ModelCheckpoint(model_dir)
    kwargs = {}
    if eval_batch_size is not None:
        kwargs['batch_size'] = eval_batch_size

    return evaluate(chkpt_callback=chkpt_callback,
                    split=split,
                    verbose=verbose,
                    **kwargs)
