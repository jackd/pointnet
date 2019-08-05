from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin


@gin.configurable
def f():
    print('f')


@gin.configurable
def g(fn=lambda: print('default')):
    fn()


conf = '''
g.fn = %c
c = @f
'''


gin.parse_config(conf)
g()
