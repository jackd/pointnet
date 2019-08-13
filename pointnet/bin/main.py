from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags


def main():
    import os
    import importlib
    import sys
    if len(sys.argv) < 2:
        raise ValueError('Usage: python -m pointnet PROG prog-specific args')

    module_name = 'pointnet.bin.{}'.format(sys.argv[1])
    del sys.argv[1]
    main = getattr(importlib.import_module(module_name), 'main')
    app.run(main)
