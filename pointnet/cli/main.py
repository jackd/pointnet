from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import os
import gin
from pointnet.cli import config
import tempfile

flags.DEFINE_string(
    'action',
    default=None,
    help='function to run. Must be in `pointnet.cli.config.ACTION_CONFIGS`')
flags.DEFINE_string(
    'config_dir',
    default=None,
    help='Root config directory. See `pointnet.cli.config.get_config_dir`')
flags.DEFINE_multi_string(
    'imports',
    default=None,
    help='list of modules to import before looking up action.')
flags.DEFINE_multi_string(
    'config_files',
    default=None,
    help='List of paths to the config files relative to `config_dir`.')
flags.DEFINE_multi_string(
    'bindings',
    default=None,
    help='List/newline separated config params.')
flags.DEFINE_string('proc_title', default=None, help='process title')
FLAGS = flags.FLAGS


def set_proc_title(title):
    if title is not None:
        try:
            import prctl
            prctl.set_proctitle(title)
        except ImportError:
            logging.warning('Failed to import prctl - cannot change title.')


@gin.configurable
def run(fn=None):
    if fn is None:
        logging.warning('No `run.fn` configured. Exiting')
        exit()
    fn()


def run_main(_):
    proc_title = FLAGS.proc_title
    if proc_title is None:
        proc_title = 'pointnet-{}'.format(FLAGS.action)
    set_proc_title(proc_title)
    config_files = FLAGS.config_files
    if config_files is not None:
        config_files = [
            gf if gf.endswith('.gin') else '{}.gin'.format(gf)
            for gf in config_files]
    config.parse_config(
        config_dir=FLAGS.config_dir,
        config_files=config_files,
        bindings=FLAGS.bindings,
        action=FLAGS.action,
        imports=FLAGS.imports,
    )
    run()


def main():
    app.run(run_main)
