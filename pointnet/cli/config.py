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


DEFAULT_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'config')


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


def parse_config(
        config_dir, config_files, bindings, action=None, imports=None):
    """Parse config from flags."""
    # allow registration of actions in imported moduled before gin configs are
    # passes
    if imports is not None:
        import importlib
        for imp in imports:
            importlib.import_module(imp)

    config_dir = get_config_dir(config_dir)
    if config_files is None:
        config_files = []
    elif isinstance(config_files, six.string_types):
        config_files = [config_files]
    if action is not None:
        config_files = [ACTION_CONFIGS[action]] + config_files
    if isinstance(bindings, six.string_types):
        bindings = [bindings]

    def parse():
        gin.parse_config_files_and_bindings(config_files, bindings)

    if config_dir is not None:
        if not os.path.isdir(config_dir):
            raise IOError('No directory at config_dir %s' % config_dir)
        with change_dir_context(config_dir):
            parse()
    else:
        parse()


class _ActionConfigs(collections.Mapping):
    """
    Mapping interface that doesn't allow overwriting of items."""
    def __init__(self):
        self._map = {}

    def __getitem__(self, key):
        return self._map[key]

    def __setitem__(self, key, value):
        if key in self._map:
            raise KeyError('Cannot overwrite key entry "{}"'.format(key))
        if not isinstance(key, six.string_types):
            raise KeyError('only string keys accepted')
        if not isinstance(value, six.string_types):
            raise ValueError('only string values accepted')
        if not value.endswith('.gin'):
            raise ValueError('only paths to `.gin` files accepted')
        if not os.path.isfile(value):
            raise ValueError(
                'values must be existing paths, but "%s" does not exist'
                % value)
        self._map[key] = value

    def __iter__(self):
        return iter(self._map)

    def keys(self):
        return self._map.keys()

    def __len__(self):
        return len(self._map)

    def __contains__(self, key):
        return key in self._map


ACTION_CONFIGS_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), 'action_configs'))


ACTION_CONFIGS = _ActionConfigs()
for filename in os.listdir(ACTION_CONFIGS_DIR):
    ACTION_CONFIGS[filename[:-4]] = os.path.join(ACTION_CONFIGS_DIR, filename)
