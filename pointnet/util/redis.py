from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin


@gin.configurable
def redis(address=None, ip=None, port=None):
    if address is None:
        if ip is None:
            assert (port is None)
            return os.environ.get('REDIS_ADDRESS', None)
        assert (port is not None)
        return '{}:{}'.format(ip, port)
    else:
        return address
