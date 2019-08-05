from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import contextlib

import gin
from pointnet.cli.actions import actions


@contextlib.contextmanager
def change_dir_context(path):
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def get_config_dir(config_dir=None):
    """
    Get root configurations directory.

    This will be
        - supplied `config_dir`; or
        - os.environ.get('POINTNET_CONFIG_DIR')`; or
        - 'POINTNET/config', where POINTNET_DIR is the root directory of the
            git repository (i.e. above the pip installation directory); or
        - None if None of the above exist.
    """
    if config_dir is None:
        config_dir = os.environ.get('POINTNET_CONFIG_DIR')
    if config_dir is None:
        config_dir = os.path.realpath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'config'))
    config_dir = os.path.realpath(
        os.path.expandvars(os.path.expanduser(config_dir)))
    if not os.path.isdir(config_dir):
        config_dir = None
    return config_dir


def parse_config(config_dir, gin_file, bindings):
    """Parse config from flags."""
    config_dir = get_config_dir(config_dir)
    if isinstance(gin_file, six.string_types):
        gin_file = [gin_file]

    def parse():
        gin.parse_config_files_and_bindings(gin_file, bindings)

    if config_dir is not None:
        if not os.path.isdir(config_dir):
            raise IOError('No directory at config_dir %s' % config_dir)
        with change_dir_context(config_dir):
            parse()
    else:
        parse()


@actions.register
def log_config():
    from absl import logging
    logging.info(gin.operative_config_str())
