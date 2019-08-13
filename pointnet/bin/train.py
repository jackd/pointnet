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

FLAGS = flags.FLAGS


def train():
    from pointnet.train import train
    from pointnet import blocks
    util.set_proc_title()
    util.tf_init()
    util.parse_config()

    return train(
        problem=blocks.problem(),
        model_fn=blocks.model_fn(),
        optimizer=blocks.optimizer(),
        batch_size=blocks.batch_size(),
        chkpt_callback=blocks.chkpt_callback(),
        callbacks=blocks.callbacks(),
        epochs=blocks.epochs(),
        fresh=FLAGS.fresh,
        verbose=FLAGS.verbose,
    )


def main(argv):
    train()
