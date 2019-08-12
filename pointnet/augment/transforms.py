from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import six

_transforms = {}  # pylint: disable=unreachable


def register_transform(name=None):

    def decorator(f):
        used_name = name or f.__name__
        if used_name in _transforms:
            raise KeyError(
                'transform "%s" already registered - cannot overwrite' %
                used_name)
        _transforms[used_name] = f
        return f

    return decorator


def deserialize(name_, **kwargs):
    fn = _transforms[name_]
    return functools.partial(fn, **kwargs)


def _assert_registered(name):
    if name not in _transforms:
        raise ValueError('name "%s" not registered' % name)


# def deserialize(obj):
#     # no nested dicts allowed
#     if obj is None:
#         return None
#     if callable(obj):
#         return obj
#     elif isinstance(obj, list):
#         return [deserialize(o) for o in obj]
#     elif isinstance(obj, tuple):
#         return tuple(deserialize(o) for o in obj)
#     elif isinstance(obj, dict):
#         return _deserialize(**obj)
#     elif isinstance(obj, six.string_types):
#         return _transforms[obj]
#     else:
#         raise ValueError('Cannot deserialize <%s> as transform' % str(obj))

# def serialize(obj):
#     if isinstance(obj, list):
#         return [serialize(o) for o in obj]
#     elif isinstance(obj, tuple):
#         return tuple(serialize(o) for o in obj)
#     elif isinstance(obj, dict):
#         return {k: serialize(v) for k, v in obj.items()}
#     elif isinstance(obj, functools.partial):
#         name = obj.func.__name__
#         _assert_registered(name)
#         return {'name': name, 'kwargs': obj.keywords}
#     elif hasattr(obj, '__name__') and callable(obj):
#         name = obj.__name__
#         _assert_registered(name)
#         return {'name': name}
#     else:
#         raise ValueError('Cannot serialize <%s> as transform' % str(obj))


def _register_default():
    from pointnet.augment import ffd
    from pointnet.augment import jitter
    from pointnet.augment import perlin
    from pointnet.augment import rigid

    register_transform()(ffd.random_ffd)
    register_transform()(jitter.jitter_positions)
    register_transform()(perlin.add_perlin_noise)
    register_transform()(rigid.random_rigid_transform)
    register_transform()(rigid.random_scale)
    register_transform()(rigid.rotate_by_scheme)
    register_transform()(rigid.maybe_reflect)


_register_default()

# def get_map_fn(transform=None, positions_only=True):
#     transform = deserialize(transform)

#     def fn(inputs, labels):
#         if isinstance(inputs, dict):
#             positions = inputs['positions']
#             _ = inputs['normals']
#         else:
#             positions = inputs

#         if not positions_only:
#             raise NotImplementedError

#         if callable(transform):
#             positions = transform(positions)
#         elif isinstance(transform, (list, tuple)):
#             for t in transform:
#                 positions = t(positions)
#         elif transform is None:
#             pass
#         else:
#             raise ValueError('Weird transform: %s' % transform)
#         return positions, labels

#     return fn

# if __name__ == '__main__':
#     # import trimesh
#     from mayavi import mlab
#     from pointnet.problem import ModelnetProblem
#     import tensorflow as tf
#     tf.compat.v1.enable_eager_execution()

#     rotate = {'name': 'rotate_by_scheme', 'kwargs': {'scheme': 'pca-xy'}}
#     # rotate = {'name': 'rotate_by_scheme', 'kwargs': {'scheme': 'random'}}

#     map_fn_spec = {
#         'train': [
#             {'name': 'random_ffd', 'kwargs': {'grid_shape': 4, 'stddev': 0.1}},
#             rotate,
#         ],
#         'validation': [
#             rotate,
#         ]
#     }
#     num_per_split = 5
#     map_fn = {k: get_map_fn(v) for k, v in map_fn_spec.items()}
#     problem = ModelnetProblem(map_fn=map_fn)
#     for split in ('train', 'validation'):
#         print(split)
#         dataset = problem.get_dataset(split=split).take(num_per_split)
#         for positions, labels in dataset:
#             mlab.points3d(*(positions.numpy().T))
#             mlab.show()
#             # trimesh.PointCloud(positions.numpy()).show()
