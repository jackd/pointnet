"""DEFAULT initial values are replaced with relevant FLAG values."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import logging
import contextlib
import six
import numpy as np

DEFAULT = '__default__'
flags.DEFINE_string('proctitle', default='pointnet', help='process title')


def set_proc_title(title=DEFAULT):
    if title == DEFAULT:
        title = flags.FLAGS.proctitle
    if title is not None:
        try:
            import setproctitle
            setproctitle.setproctitle(title)
        except ImportError:
            logging.warning(
                'Failed to import setproctitle - cannot change title.')


def add_gin_flags():
    # gin params
    flags.DEFINE_string(
        'config_dir', None,
        'Root config directory. See `pointnet.bin.flags.get_config_dir`')
    flags.DEFINE_multi_string(
        'configs', [], 'List of paths to the config files. '
        'Relative paths are relative to `config_dir`')
    flags.DEFINE_multi_string(
        'bindings', [], 'Newline separated list of gin parameter bindings.')


def add_tf_flags():
    # tensorflow params
    flags.DEFINE_boolean('allow_growth',
                         default=True,
                         help='gpu_config options')
    flags.DEFINE_string('gpus', default=None, help='CUDA_VISIBLE_DEVICES')
    flags.DEFINE_boolean('eager', default=False, help='run in eager mode')


def add_ray_flags():
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
    flags.DEFINE_boolean('local_mode',
                         help='ray init parameter - useful for debugging',
                         default=False)


DEFAULT_CONFIG_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'config'))


@contextlib.contextmanager
def change_dir_context(path):
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


@contextlib.contextmanager
def nullcontext(*args, **kwargs):
    yield


def get_config_dir():
    """
    Get root configurations directory.

    This will be
        - `FLAGS.config_dir`; or
        - os.environ.get('POINTNET_CONFIG_DIR')`; or
        - `pointnet.cli.config.DEFAULT_CONFIG_DIR`; or
        - `None` if none of the above exist.
    """
    config_dir = getattr(flags.FLAGS, 'config_dir', None)
    if config_dir is None:
        config_dir = os.environ.get('POINTNET_CONFIG_DIR')
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR
    config_dir = os.path.realpath(
        os.path.expandvars(os.path.expanduser(config_dir)))
    if not os.path.isdir(config_dir):
        raise ValueError('No config directory at {}'.format(config_dir))
    return config_dir


def parse_config(config_dir=DEFAULT,
                 configs=DEFAULT,
                 bindings=DEFAULT,
                 finalize_config=True):
    """Parse config from flags."""
    import gin
    FLAGS = flags.FLAGS
    if config_dir == DEFAULT:
        config_dir = get_config_dir()
    if configs == DEFAULT:
        configs = getattr(FLAGS, 'configs', [])
    elif isinstance(configs, six.string_types):
        configs = [configs]
    configs = np.concatenate([c.split(',') for c in configs])
    configs = [c if c.endswith('.gin') else '{}.gin'.format(c) for c in configs]
    if bindings == DEFAULT:
        bindings = getattr(FLAGS, 'bindings', [])
    elif isinstance(bindings, six.string_types):
        bindings = [bindings]

    # log
    log_strs = ['Parsing config', 'config_dir: {}'.format(config_dir)]
    if configs:
        log_strs.append('Files:')
        log_strs.extend(('  ' + c for c in configs))
    if bindings:
        log_strs.append('Bindings:')
        log_strs.extend(('  ' + b for b in bindings))
    logging.info('\n'.join(log_strs))

    context = nullcontext() if config_dir is None else change_dir_context(
        config_dir)
    with context:
        gin.parse_config_files_and_bindings(configs,
                                            bindings,
                                            finalize_config=finalize_config)


def ray_init(redis_address=DEFAULT,
             num_cpus=DEFAULT,
             num_gpus=DEFAULT,
             local_mode=DEFAULT,
             **kwargs):
    # likely requires add_ray_flags to be called
    import ray
    FLAGS = flags.FLAGS
    if redis_address == DEFAULT:
        redis_address = FLAGS.redis_address
        if redis_address is None:
            redis_address = os.environ.get('REDIS_ADDRESS')
    return ray.init(
        redis_address,
        num_cpus=FLAGS.num_cpus if num_cpus == DEFAULT else num_cpus,
        num_gpus=FLAGS.num_gpus if num_gpus == DEFAULT else num_gpus,
        local_mode=FLAGS.local_mode if local_mode == DEFAULT else local_mode,
        **kwargs)


def tf_init(gpus=DEFAULT, allow_growth=DEFAULT, eager=DEFAULT):
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
    # requires add_tf_flags to be called
    FLAGS = flags.FLAGS
    if eager == DEFAULT:
        eager = FLAGS.eager
    if gpus == DEFAULT:
        gpus = FLAGS.gpus
    elif isinstance(gpus, (list, tuple)):
        gpus = ','.join(str(gpu) for gpu in gpus)
    elif isinstance(gpus, int):
        gpus = str(gpus)
    if allow_growth == DEFAULT:
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


def assert_clargs_parsed(argv):
    if len(argv) > 1:
        raise ValueError('Unpassed command line args: {}'.format(' '.join(
            argv[1:])))
