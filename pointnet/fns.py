from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import importlib
import functools
import six

_ATOMIC_TYPES = (int, float, bool) + six.string_types


class ConfigurableFunction(object):

    def __init__(self, fn=None, name=None, module=None):
        if fn is None:
            fn = getattr(importlib.import_module(module), name)
        if name is None:
            name = fn.__name__
        else:
            assert (fn.__name__ == name)
        if module is None:
            module = fn.__module__
        else:
            assert (fn.__module__ == module)
        self._fn = fn
        self._name = name
        self._module = module

    @classmethod
    def from_config(cls, config):
        return ConfigurableFunction(name=config['name'],
                                    module=config['module'])

    def get_config(self):
        return dict(name=self._name, module=self._module)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class ConfigurablePartial(object):

    def __init__(self, func, **keywords):
        if isinstance(func, functools.partial):
            kwargs = func.keywords
            for k in keywords:
                if k in kwargs:
                    raise ValueError('Repeated keyword "{}"'.format(k))
            kwargs.update(keywords)
            keywords = kwargs
            func = func.func
        self._func = func
        self._keywords = keywords

    @classmethod
    def from_config(cls, config):
        return ConfigurablePartial(ConfigurableFunction(config['func']),
                                   **config['keywords'])

    def __call__(self, *args, **kwargs):
        for k in kwargs:
            if k in self._keywords:
                raise ValueError('Repeated keyword "{}"'.format(k))
        kwargs.update(self._keywords)
        return self._func(*args, **kwargs)


_module_objects = {
    'ConfigurablePartial': ConfigurablePartial,
    'ConfigurableFunction': ConfigurableFunction,
}


def as_configurable(fn):
    if hasattr(fn, 'get_config'):
        return fn
    elif isinstance(fn, functools.partial):
        return ConfigurablePartial(fn)
    else:
        return ConfigurableFunction(fn)


def get_config(fn):
    return as_configurable(fn).get_config()


def get(identifier):
    if callable(identifier):
        return identifier
    else:
        return tf.keras.utils.deserialize_keras_object(identifier,
                                                       _module_objects)
