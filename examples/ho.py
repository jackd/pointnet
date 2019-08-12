from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pointnet.aug_opt import search_space
from hyperopt.pyll.stochastic import sample
import six

space = search_space()
for _ in range(10):
    print("---")
    print(
        isinstance(
            sample(space)['mutable_bindings']['rotate_scheme'],
            six.string_types))
