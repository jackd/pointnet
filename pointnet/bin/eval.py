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
flags.DEFINE_boolean(
    'training',
    default=None,
    help='Use True to avoid possible batch norm statistics issues')
flags.DEFINE_string(
    'split',
    default='validation',
    help='dataset split to use - one of "train", "validation", "test"')

flags.register_validator(
    'split',
    lambda split: split in ('train', 'validation', 'test'),
    message='split must one of "train", "validation", "test"')

util.add_tf_flags()
util.add_gin_flags()
FLAGS = flags.FLAGS


def evaluate():
    import os
    from pointnet.train import evaluate
    from pointnet import blocks
    util.set_proc_title()
    util.tf_init()
    util.parse_config()

    return evaluate(
        problem=blocks.problem(),
        model_fn=blocks.model_fn(),
        optimizer=blocks.optimizer(),
        batch_size=blocks.batch_size(),
        chkpt_callback=blocks.chkpt_callback(),
        split=FLAGS.split,
        verbose=FLAGS.verbose,
        training=FLAGS.training,
    )


def main(argv):
    util.assert_clargs_parsed(argv)
    evaluate()
