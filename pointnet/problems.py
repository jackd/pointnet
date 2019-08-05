from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
from pointnet.augment import flat_augment_cloud
from shape_tfds.shape import modelnet

Objective = collections.namedtuple('Objective', ['name', 'mode'])



class Problem(object):
    @abc.abstractmethod
    def get_dataset(self, split, batch_size=None, map_fn=None, ):
        raise NotImplementedError

    @abc.abstractmethod
    def examples_per_epoch(self, split=tfds.Split.TRAIN):
        raise NotImplementedError

    def get_generator(self, split, batch_size=None, map_fn=None, ):
        graph = tf.Graph()
        with graph.as_default():
            dataset = self.get_dataset(
                split, batch_size=batch_size, map_fn=map_fn, )
        return tfds.as_numpy(dataset, graph=graph)

    @abc.abstractproperty
    def input_spec(self):
        raise NotImplementedError

    @abc.abstractproperty
    def output_spec(self):
        """tf.keras.layers.InputSpec of the output any models."""
        raise NotImplementedError

    @abc.abstractproperty
    def loss(self):
        raise NotImplementedError

    @property
    def objective(self):
        """Objective, used in hyper-parameter optimization."""
        return None

    @property
    def metrics(self):
        return None

    @classmethod
    def rebuild(cls, **kwargs):
        raise NotImplementedError


def examples_per_epoch(builder, split):
    return int(builder.info.splits[split].num_examples)


# @gin.configurable
class TfdsProblem(Problem):
    def __init__(
            self, builder, input_spec, output_spec, loss, metrics=None,
            base_map_fn=None,
            as_supervised=True, shuffle_buffer=1024, objective=None):
        self._builder = builder
        self._output_spec = output_spec
        self._loss = loss
        self._metrics = tuple(metrics)
        self._base_map_fn = base_map_fn
        self._as_supervised = as_supervised
        self._shuffle_buffer = 1024
        self._objective = objective
        self._input_spec = input_spec

    @classmethod
    def rebuild(cls, **kwargs):
        return TfdsProblem(**kwargs)

    def _split(self, split):
        # allow for possible override
        return split

    def examples_per_epoch(self, split=tfds.Split.TRAIN):
        return examples_per_epoch(self._builder, self._split(split))

    def get_dataset(self, split, batch_size=None, map_fn=None):
        if isinstance(map_fn, dict):
            map_fn = map_fn[split]
        dataset = self._builder.as_dataset(
            batch_size=None, split=self._split(split),
            as_supervised=self._as_supervised,
            shuffle_files=split==tfds.Split.TRAIN)

        base_map_fn = self._base_map_fn
        if isinstance(base_map_fn, dict):
            base_map_fn = base_map_fn[split]

        dataset = dataset.repeat()

        if split == tfds.Split.TRAIN:
            dataset = dataset.shuffle(self._shuffle_buffer)

        if base_map_fn is not None:
            dataset = dataset.map(
                base_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if map_fn is not None:
            dataset = dataset.map(
                map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    @property
    def input_spec(self):
        return self._input_spec

    @property
    def output_spec(self):
        return self._output_spec

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return list(self._metrics)

    @property
    def objective(self):
        return self._objective


def base_modelnet_map(
        inputs, labels, positions_only=True, num_points_sampled=None):
    if not positions_only:
        raise NotImplementedError()
    if isinstance(inputs, dict):
        positions = inputs['positions']
        normals = inputs['normals']
    else:
        positions = inputs
        normals = inputs

    if num_points_sampled is not None:
        if positions_only:
            positions = tf.random.shuffle(positions)[:num_points_sampled]
        else:
            indices = tf.range(tf.shape(positions)[0])
            indices = tf.random.shuffle(indices)[:num_points_sampled]
            positions = tf.gather(positions, indices)
            normals = tf.gather(positions, indices)
    if positions_only:
        inputs = positions
    else:
        inputs = dict(positions=positions, normals=normals)
    return inputs, labels


class FfdModelnetConfig(modelnet.CloudConfig):
    def __init__(self, num_points, grid_shape=4, name=None, **kwargs):
        if name is None:
            if not isinstance(grid_shape, int):
                assert(len(grid_shape) == 3)
                if all(g == grid_shape[0] for g in grid_shape[1:]):
                    grid_shape = grid_shape[0]

            grid_shape_str = ('%d' % grid_shape if isinstance(grid_shape, int)
                              else 'x'.join(str(g) for g in grid_shape))
            name = 'ffd-%s-%d' % (grid_shape_str, num_points)

        if isinstance(grid_shape, int):
            grid_shape = (grid_shape,)*3
        self._grid_shape = grid_shape
        super(FfdModelnetConfig, self).__init__(
            num_points=num_points, name=name, **kwargs)
        if tf.executing_eagerly():
            def f(points):
                from pointnet.augment import ffd
                b, p = ffd.get_ffd(points, grid_shape)
                return dict(b=b, p=p)

            self._f = f
        else:
            raise NotImplementedError(
                'Please generate data in a separate script using separately '
                'tf.compat.v1.enable_eager_execution')

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def feature_item(self):
        from tensorflow_datasets.core import features
        import numpy as np
        grid_size = np.prod(self.grid_shape)
        return 'ffd', features.FeaturesDict({
            'b': features.Tensor(
                shape=(self.num_points, grid_size), dtype=tf.float32),
            'p': features.Tensor(shape=(grid_size, 3), dtype=tf.float32),
        })

    @abc.abstractmethod
    def load_example(self, off_path):
        points = super(FfdModelnetConfig, self).load_example(off_path)
        return self._f(points)



@gin.configurable
class ModelnetProblem(TfdsProblem):
    def __init__(
            self,
            num_classes=40,
            num_points_base=2048,
            num_points_sampled=1024,
            positions_only=True,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=(tf.keras.metrics.SparseCategoricalAccuracy(),),
            objective=None,
            use_train_test_split=False,
            train_percent=90,
            download_kwargs={},
            num_examples_override=None,
            **kwargs):
        import functools
        self._num_examples_override = num_examples_override
        if objective is None:
            objective = Objective('val_%s' % metrics[0].name, 'max')

        config = modelnet.CloudConfig(num_points=num_points_base)
        builder = {
            10: modelnet.Modelnet10,
            40: modelnet.Modelnet40,
        }[num_classes](config=config)
        builder.download_and_prepare(**download_kwargs)
        input_spec = tf.keras.layers.InputSpec(
            shape=(num_points_sampled, 3), dtype=tf.float32)
        output_spec = tf.keras.layers.InputSpec(
            shape=(num_classes,), dtype=tf.float32)
        self._num_points_sampled = num_points_sampled
        self._train_percent = train_percent
        self._use_train_test_split = use_train_test_split

        base_map_fn = functools.partial(
            base_modelnet_map,
            positions_only=positions_only,
            num_points_sampled=num_points_sampled)
        super(ModelnetProblem, self).__init__(
            builder=builder,
            input_spec=input_spec,
            output_spec=output_spec,
            loss=loss,
            metrics=metrics,
            objective=objective,
            base_map_fn=base_map_fn,
            **kwargs)

    def _split(self, split):
        if self._use_train_test_split:
            if split == tfds.Split.TRAIN:
                return split
            elif split in (tfds.Split.VALIDATION, tfds.Split.TEST):
                return tfds.Split.TEST
        else:
            ReadInstruction = tfds.core.tfrecords_reader.ReadInstruction
            if split == tfds.Split.TRAIN:
                return ReadInstruction(
                    'train', to=self._train_percent, unit='%')
            elif split == tfds.Split.VALIDATION:
                return ReadInstruction(
                    'train', from_=self._train_percent, unit='%')
            else:
                return split

    def examples_per_epoch(self, split):
        if self._num_examples_override is not None:
            return self._num_examples_override
        # return 10  # SMOKE-TEST
        if self._use_train_test_split:
            return examples_per_epoch(self._builder, self._split(split))
        if split == tfds.Split.TRAIN:
            return int((self._train_percent / 100) *
                       examples_per_epoch(self._builder, tfds.Split.TRAIN))
        elif split == tfds.Split.VALIDATION:
            return int((1 - self._train_percent / 100) *
                       examples_per_epoch(self._builder, tfds.Split.TRAIN))
        else:
            assert(split == tfds.Split.TEST)
            return super(ModelnetProblem, self).examples_per_epoch(split)


def deserialize(name='modelnet40', **kwargs):
    if name == 'modelnet40':
        return ModelnetProblem(num_classes=40, **kwargs)
    elif name == 'modelnet10':
        return ModelnetProblem(num_classes=10, **kwargs)
    else:
        raise ValueError('Unrecognized problem name "%s"' % name)
