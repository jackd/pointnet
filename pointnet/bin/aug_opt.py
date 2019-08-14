from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from pointnet.bin import util

util.add_gin_flags()
util.add_ray_flags()

flags.DEFINE_boolean('resume', default=None, help='resume from previous work')
flags.DEFINE_boolean('fresh',
                     default=False,
                     help='if True, delete directory on start')
flags.DEFINE_string(
    'inner_config_dir',
    default=None,
    help='root config dir used by TuneModel. Defaults to same as outer scope.')


def aug_opt():
    util.ray_init()
    util.parse_config()
    FLAGS = flags.FLAGS
    from pointnet.aug_opt import aug_opt
    return aug_opt(  # pylint: disable=no-value-for-parameter
        resume=FLAGS.resume,
        fresh=FLAGS.fresh,
        inner_config_dir=FLAGS.inner_config_dir)


def main(argv):
    util.assert_clargs_parsed(argv)
    aug_opt()
