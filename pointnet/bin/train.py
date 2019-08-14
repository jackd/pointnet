from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from pointnet.bin import util

flags.DEFINE_boolean('fresh',
                     default=False,
                     help='Delete relevant chkpt_callback directory if True')
flags.DEFINE_boolean('verbose', default=True, help='used in model.fit')
flags.DEFINE_boolean('clean_after',
                     default=False,
                     help='delete files afterwards')

FLAGS = flags.FLAGS
util.add_tf_flags()
util.add_gin_flags()


def train():
    util.tf_init()
    import os
    import tensorflow as tf
    from pointnet.train import train
    from pointnet import blocks
    util.set_proc_title()
    util.parse_config()

    chkpt_callback = blocks.chkpt_callback()
    history = train(
        problem=blocks.problem(),
        model_fn=blocks.model_fn(),
        optimizer=blocks.optimizer(),
        batch_size=blocks.batch_size(),
        chkpt_callback=chkpt_callback,
        callbacks=blocks.callbacks(),
        epochs=blocks.epochs(),
        fresh=FLAGS.fresh,
        verbose=FLAGS.verbose,
    )
    if FLAGS.clean_after:
        directory = chkpt_callback.directory
        logging.info('Cleaning directory {}'.format(directory))
        tf.io.gfile.rmtree(directory)
        directory = os.path.dirname(directory)
        while len(os.listdir(directory)) == 0:
            os.rmdir(directory)
            directory = os.path.dirname(directory)
            logging.info('Removing empty directory {}'.format(directory))
    return history


def main(argv):
    util.assert_clargs_parsed(argv)
    train()
