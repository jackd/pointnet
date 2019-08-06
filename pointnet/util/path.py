from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import os


@gin.configurable(module='util')
def local_dir(base_dir='~/pointnet_models', subdir=None, name=None, run=0):
    """Convenient function for setting local directories via gin."""
    path = os.path.join(
        *(x for x in (base_dir, subdir, name) if x is not None))
    if isinstance(run, int):
        run = 'run%03d' % run
    path = os.path.join(path, run)
    return os.path.expandvars(os.path.expanduser(path))
