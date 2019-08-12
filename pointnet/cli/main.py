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
    'config_dir',
    default=None,
    help='Root config directory. See `pointnet.cli.config.get_config_dir`')
flags.DEFINE_multi_string(
    'files',
    default=[],
    help='List of paths to the config files relative to `config_dir`.')
flags.DEFINE_multi_string('bindings',
                          default=[],
                          help='List/newline separated config params.')
FLAGS = flags.FLAGS

UNIVERSAL = '''
set_proc_title.title = %proc_title
run.fn = %main

proc_title = 'pointnet'
main = None
'''


@gin.configurable
def set_proc_title(title=None):
    if title is not None:
        try:
            import setproctitle
            setproctitle.setproctitle(title)
        except ImportError:
            logging.warning(
                'Failed to import setproctitle - cannot change title.')


@gin.configurable
def run(fn=None):
    if fn is None:
        logging.warning('No `run.fn` configured. Exiting')
    else:
        fn()


def run_main(args):
    set_proc_title()

    files = list(FLAGS.files)
    bindings = list(FLAGS.bindings)
    for arg in args[1:]:
        (bindings if '=' in arg else files).append(arg)
    gin.parse_config(UNIVERSAL)

    files = [gf if gf.endswith('.gin') else '{}.gin'.format(gf) for gf in files]
    config.parse_config(
        config_dir=FLAGS.config_dir,
        config_files=files,
        bindings=bindings,
    )
    run()


def main():
    app.run(run_main)
