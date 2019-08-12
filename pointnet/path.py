from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import os


@gin.configurable(module='pointnet.path')
def local_dir(base_dir='~/pointnet_models', subdir=None, name=None):
    """Convenient function for setting local directories via gin."""
    path = os.path.join(*(x for x in (base_dir, subdir, name) if x is not None))
    return os.path.expandvars(os.path.expanduser(path))
