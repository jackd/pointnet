from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import logging
import contextlib
import six
import gin

DEFAULT = '__default__'
flags.DEFINE_string('proctitle', default='pointnet', help='process title')

# gin params
flags.DEFINE_string(
    'config_dir', None,
    'Root config directory. See `pointnet.bin.flags.get_config_dir`')
flags.DEFINE_multi_string(
    'configs', [], 'List of paths to the config files. '
    'Relative paths are relative to `config_dir`')
flags.DEFINE_multi_string('bindings', [],
                          'Newline separated list of gin parameter bindings.')

# tensorflow params
flags.DEFINE_boolean('allow_growth', default=None, help='gpu_config options')
flags.DEFINE_string('gpus', default=None, help='CUDA_VISIBLE_DEVICES')
flags.DEFINE_boolean('eager', default=False, help='run in eager mode')

# ray params
flags.DEFINE_string('redis_address',
                    default=None,
                    help='address of redis server for usage with ray')
flags.DEFINE_integer(
    'num_cpus',
    default=None,
    help='number of cpus to use with ray. Leave as None to delegate to ray')
flags.DEFINE_integer(
    'num_gpus',
    default=None,
    help='number of cpus to use with ray. Leave as None to delegate to ray')

DEFAULT_CONFIG_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'config'))


def set_proc_title(title=DEFAULT):
    if title is DEFAULT:
        title = flags.FLAGS.proctitle
    if title is not None:
        try:
            import setproctitle
            setproctitle.setproctitle(title)
        except ImportError:
            logging.warning(
                'Failed to import setproctitle - cannot change title.')


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
        - `FLAGS.config_dir`; or
        - os.environ.get('POINTNET_CONFIG_DIR')`; or
        - `pointnet.cli.config.DEFAULT_CONFIG_DIR`; or
        - `None` if none of the above exist.
    """
    if config_dir is None:
        config_dir = flags.FLAGS.config_dir
    if config_dir is None:
        config_dir = os.environ.get('POINTNET_CONFIG_DIR')
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR
    config_dir = os.path.realpath(
        os.path.expandvars(os.path.expanduser(config_dir)))
    if not os.path.isdir(config_dir):
        raise ValueError('No config directory at {}'.format(config_dir))
    return config_dir


def parse_config(config_dir=None,
                 configs=None,
                 bindings=None,
                 finalize_config=True):
    """Parse config from flags."""
    config_dir = get_config_dir(config_dir)
    if configs is None:
        configs = flags.FLAGS.configs
    elif isinstance(configs, six.string_types):
        configs = [configs]
    configs = [c if c.endswith('.gin') else '{}.gin'.format(c) for c in configs]
    if bindings is None:
        bindings = flags.FLAGS.bindings
    elif isinstance(bindings, six.string_types):
        bindings = [bindings]

    with change_dir_context(config_dir):
        gin.parse_config_files_and_bindings(configs,
                                            bindings,
                                            finalize_config=finalize_config)


def ray_init(redis_address=None, num_cpus=None, num_gpus=None, **kwargs):
    import ray
    FLAGS = flags.FLAGS
    if redis_address is None:
        redis_address = FLAGS.redis_address
    if redis_address is None:
        redis_address = os.environ.get('REDIS_ADDRESS')
    return ray.init(redis_address or FLAGS.redis_address or
                    os.environ.get('REDIS_ADDRESS'),
                    num_cpus=FLAGS.num_cpus if num_cpus is None else num_cpus,
                    num_gpus=FLAGS.num_gpus if num_gpus is None else num_gpus,
                    **kwargs)


def tf_init(gpus=None, allow_growth=DEFAULT, eager=DEFAULT):
    """
    Initialize tensorflow.

    This will reset the

    Args:
        gpus: string/int/list of either denoting gpu ids to use in
            CUDA_VISIBLE_DEVICES. If None, looks at flags. If None there,
            does not override, so tf uses os.environ['CUDA_VISIBLE_DEVICES']
        allow_growth: gpu memory option. if DEFAULT, uses flag value
        eager: whether or not to start in eager mode.
    """
    FLAGS = flags.FLAGS
    if eager is DEFAULT:
        eager = FLAGS.eager
    if gpus is None:
        gpus = FLAGS.gpus
    elif isinstance(gpus, (list, tuple)):
        gpus = ','.join(str(gpu) for gpu in gpus)
    elif isinstance(gpus, int):
        gpus = str(gpus)
    if allow_growth is DEFAULT:
        allow_growth = FLAGS.allow_growth
    import os
    if gpus:
        logging.info('Setting CUDA_VISIBLE_DEVICES={}'.format(gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    import tensorflow as tf
    if allow_growth is not None:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
    else:
        config = None
    if eager:
        tf.compat.v1.enable_eager_execution(config=config)
    elif config is not None:
        tf.keras.backend.set_session(tf.compat.v1.Session(config=config))
