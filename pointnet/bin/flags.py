from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import contextlib

import gin

flags.DEFINE_string(
    'config_dir', None,
    'Root config directory. See `pointnet.bin.flags.get_config_dir`')
flags.DEFINE_string(
    'gin_file', None, 'List of paths to the config files. '
    'Relative paths are relative to `config_dir`')
flags.DEFINE_multi_string('gin_param', None,
                          'Newline separated list of Gin parameter bindings.')


@contextlib.contextmanager
def change_dir_context(path):
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def get_config_dir():
    """
    Get root configurations directory.

    This will be `FLAGS.config_dir` or `os.environ.get('POINTNET_CONFIG_DIR')`.
    """
    config_dir = flags.FLAGS.config_dir
    if config_dir is None:
        config_dir = os.environ.get('POINTNET_CONFIG_DIR')
    if config_dir is None:
        return None
    config_dir = os.path.realpath(
        os.path.expandvars(os.path.expanduser(config_dir)))
    return config_dir


def parse_config():
    """Parse config from flags."""
    gin_file = flags.FLAGS.gin_file
    if not gin_file.endswith('.gin'):
        gin_file = '%s.gin' % gin_file
    gin.bind_parameter('model_id', gin_file[:-4])
    config_dir = get_config_dir()
    if not os.path.isdir(config_dir):
        raise IOError('No directory at config_dir %s' % config_dir)

    def parse():
        gin.parse_config_files_and_bindings([gin_file], flags.FLAGS.gin_param)

    if config_dir is not None:
        with change_dir_context(config_dir):
            parse()
    else:
        parse()
