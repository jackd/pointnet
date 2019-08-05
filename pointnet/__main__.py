from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import gin
from pointnet.cli import config
from pointnet.cli.actions import actions

import pointnet.models
import pointnet.problems
import pointnet.augment
import pointnet.keras_configurables
import pointnet.callbacks
import pointnet.train
import pointnet.path


flags.DEFINE_string('action', default='log_config', help='function to run')
flags.DEFINE_string(
    'config_dir', None,
    'Root config directory. See `pointnet.clip.config.get_config_dir`')
flags.DEFINE_multi_string(
    'gin_file', None,
    'List of paths to the config files relative to `config_dir`.')
flags.DEFINE_multi_string(
    'gin_params', None, 'List/newline separated config params.')
FLAGS = flags.FLAGS


@gin.configurable
def f(x=None):
    print(x)


def main(args):
    gin_file = [
        gf if gf.endswith('.gin') else '{}.gin'.format(gf)
        for gf in FLAGS.gin_file]
    config.parse_config(FLAGS.config_dir, gin_file, FLAGS.gin_params)
    actions[FLAGS.action]()


if __name__ == '__main__':
    app.run(main)
