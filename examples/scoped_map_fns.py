from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
# from pointnet.augment import augment_cloud


@gin.configurable
def f(x='default', y='y_default'):
    print(x, y)


@gin.configurable
def g(fn=f):
    fn()


gin_config = '''
f.x = 'base'

train/f.x = 'x_train'
train/f.y = 'y_train'
g.fn = @train/f
diff/train/f.x = 'x_diff'
'''

gin.parse_config(gin_config)
gin.bind_parameter

# diff_config = '''
#
# '''

g()
with gin.config_scope('diff'):
    # gin.parse_config(diff_config)
    g()
g()
