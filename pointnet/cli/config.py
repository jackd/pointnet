from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import gin
import contextlib

import collections


@contextlib.contextmanager
def change_dir_context(path):
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                                  'config')


def get_config_dir(config_dir=None):
    """
    Get root configurations directory.

    This will be
        - supplied `config_dir`; or
        - os.environ.get('POINTNET_CONFIG_DIR')`; or
        - `pointnet.cli.config.DEFAULT_CONFIG_DIR`; or
        - `None` if none of the above exist.
    """
    if config_dir is None:
        config_dir = os.environ.get('POINTNET_CONFIG_DIR')
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR
    config_dir = os.path.realpath(
        os.path.expandvars(os.path.expanduser(config_dir)))
    if not os.path.isdir(config_dir):
        config_dir = None
    return config_dir


def parse_config(config_dir, config_files, bindings, finalize_config=True):
    """Parse config from flags."""
    config_dir = get_config_dir(config_dir)
    if config_files is None:
        config_files = []
    elif isinstance(config_files, six.string_types):
        config_files = [config_files]
    if isinstance(bindings, six.string_types):
        bindings = [bindings]

    def parse():
        gin.parse_config_files_and_bindings(config_files,
                                            bindings,
                                            finalize_config=finalize_config)

    if config_dir is not None:
        if not os.path.isdir(config_dir):
            raise IOError('No directory at config_dir %s' % config_dir)
        with change_dir_context(config_dir):
            parse()
    else:
        parse()
