from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class _Actions(collections.Mapping):
    def __init__(self):
        self._fns = {}

    def register(self, name=None, module=None):
        def decorator(f):
            if name is None:
                used_name = f.__name__
            else:
                used_name = name
            if module is not None:
                used_name = '{}.{}'.format(module, name)
            if used_name in self._fns:
                raise RuntimeError(
                    'Cannot register with name {}: already exists'.format(
                        used_name))
            if not callable(f):
                raise ValueError(
                    'registered value for key {} must be callable, '
                    'got {}'.format(used_name, f))
            self._fns[used_name] = f
            return f

        if callable(name):
            f = name
            name = None
            if module is not None:
                raise ValueError(
                    'Cannot provide callable `name` and non-None `module`')
            return decorator(f)
        else:
            return decorator

    def __setitem__(self, key, value):
        self.register(key)(value)

    def __getitem__(self, key):
        if key not in self._fns:
            raise KeyError(
                'Invalid key "%s": must be one of:\n' +
                '\n'.join('* %s' % k for k in sorted(self.keys())) +
                '\nEnsure you have imported the correct files via gin')
        return self._fns[key]

    def __iter__(self):
        return iter(self._fns)

    def __contains__(self, key):
        return key in self._fns

    def __len__(self):
        return len(self._fns)

    def keys(self):
        return self._fns.keys()


actions = _Actions()
