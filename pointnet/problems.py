from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
from shape_tfds.shape import modelnet
import six
from pointnet import fns

Objective = collections.namedtuple('Objective', ['name', 'mode'])


def reconfigure(instance, **updates):
    if hasattr(instance, 'reconfigure'):
        instance.reconfigure(**updates)
    else:
        config = instance.get_config()
        config.update(updates)
        return type(instance).from_config(config)


class Problem(object):

    @abc.abstractmethod
    def get_dataset(self, split, batch_size=None, repeats=False):
        raise NotImplementedError

    @abc.abstractmethod
    def _examples_per_epoch(self, split=tfds.Split.TRAIN, batch_size=None):
        raise NotImplementedError

    def examples_per_epoch(self, split=tfds.Split.TRAIN, batch_size=None):
        if isinstance(split, six.string_types):
            return self._examples_per_epoch(split, batch_size)
        else:
            return tf.nest.map_structure(
                lambda s: self._examples_per_epoch(s, batch_size), split)

    def _get_generator(self, split, batch_size=None, repeats=False):
        graph = tf.Graph()
        with graph.as_default():  # pylint: disable=not-context-manager
            dataset = self.get_dataset(split,
                                       batch_size=batch_size,
                                       repeats=repeats)
        return tfds.as_numpy(dataset, graph=graph)

    def get_generator(self, split, batch_size=None, repeats=False):
        return tf.nest.map_structure(
            lambda s: self._get_generator(
                s, batch_size=batch_size, repeats=repeats), split)

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
    def from_config(cls, config):
        return cls(**config)

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError


def examples_per_epoch(builder, split, batch_size=None):
    value = int(builder.info.splits[split].num_examples)
    if batch_size is not None:
        value //= batch_size
    return value


def input_spec_config(spec):
    return dict(
        dtype=spec.dtype,
        shape=spec.shape,
    )


def as_type(cls, instance):
    if isinstance(instance, cls):
        return instance
    elif isinstance(instance, collections.Iterable):
        return cls(*instance)
    elif isinstance(instance, collections.Mapping):
        return cls(**instance)
    else:
        raise ValueError('Invalid instance {} of type {}'.format(instance, cls))


@gin.configurable
class TfdsProblem(Problem):

    def __init__(self,
                 builder,
                 input_spec,
                 output_spec,
                 loss,
                 metrics=None,
                 map_fn=None,
                 as_supervised=True,
                 shuffle_buffer=-1,
                 objective=None,
                 download_kwargs={}):
        self._builder = builder
        self._output_spec = output_spec
        self._loss = loss
        self._metrics = metrics
        self._as_supervised = as_supervised
        self._shuffle_buffer = shuffle_buffer
        self._objective = objective
        self._input_spec = input_spec
        self._map_fn = map_fn

        if download_kwargs is not None:
            self.builder.download_and_prepare(**download_kwargs)

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, builder):
        if isinstance(builder, six.string_types):
            builder = tfds.builder(builder)
        elif isinstance(builder, dict):
            builder = tfds.builder(**builder)
        self._builder = builder

    @property
    def loss(self):
        return self._loss

    # @loss.setter
    # def loss(self, loss):
    #     self._loss = tf.keras.losses.get(loss)

    @property
    def metrics(self):
        return list(self._metrics)

    # @metrics.setter
    # def metrics(self, metrics):
    #     self._metrics = tuple(tf.keras.metrics.get(m) for m in metrics)

    @property
    def input_spec(self):
        return self._input_spec

    # @input_spec.setter
    # def input_spec(self, input_spec):
    #     self._input_spec = as_type(tf.keras.layers.InputSpec, input_spec)

    @property
    def output_spec(self):
        return self._output_spec

    # @output_spec.setter
    # def output_spec(self, output_spec):
    #     self._output_spec = as_type(tf.keras.layers.InputSpec, output_spec)

    @property
    def objective(self):
        return self._objective

    # @objective.setter
    # def objective(self, objective):
    #     self._objective = as_type(Objective, objective)

    @property
    def map_fn(self):
        return self._map_fn

    # @map_fn.setter
    # def map_fn(self, map_fn):
    #     if isinstance(map_fn, dict) and not 'class_name' in map_fn:
    #         map_fn = {k: fns.get(v) for k, v in map_fn.items()}
    #     else:
    #         map_fn = fns.get(map_fn)
    #     self._map_fn = map_fn

    def get_config(self):
        return dict(builder=self._builder.name,
                    input_spec=input_spec_config(self._input_spec),
                    output_spec=input_spec_config(self._output_spec),
                    loss=self._loss.get_config(),
                    metrics=[m.get_config() for m in self._metrics],
                    map_fn=tf.nest.map_structure(fns.get_config, self._map_fn),
                    as_supervised=self._as_supervised,
                    shuffle_buffer=self._shuffle_buffer,
                    objective=dict(name=self._objective.name,
                                   mode=self._objective.name))

    # def reconfigure(self, **updates):
    #     for k, v in updates.items():
    #         setattr(self, k, v)

    def _split(self, split):
        # allow for possible override
        return split

    def _examples_per_epoch(self, split=tfds.Split.TRAIN, batch_size=None):
        return examples_per_epoch(self._builder, self._split(split), batch_size)

    def _get_base_dataset(self, split):
        return self._builder.as_dataset(batch_size=None,
                                        split=self._split(split),
                                        as_supervised=self._as_supervised,
                                        shuffle_files=split == tfds.Split.TRAIN)

    def _get_dataset(self, split, batch_size=None, repeats=False):
        dataset = self._get_base_dataset(split)

        if repeats is not False:
            dataset = dataset.repeat(repeats)

        if split == tfds.Split.TRAIN:
            shuffle_buffer = self._shuffle_buffer
            if shuffle_buffer == -1:
                shuffle_buffer = self._examples_per_epoch(split)

            dataset = dataset.shuffle(shuffle_buffer)

        map_fn = self._map_fn
        if isinstance(map_fn, dict):
            if split == 'test' and 'test' not in map_fn:
                map_fn = map_fn['validation']
            else:
                map_fn = map_fn[split]

        if map_fn is not None:
            dataset = dataset.map(
                map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def get_dataset(self, split, batch_size=None, repeats=False):
        return tf.nest.map_structure(
            lambda s: self._get_dataset(
                s, batch_size=batch_size, repeats=repeats), split)


def base_modelnet_map(inputs,
                      labels,
                      positions_only=True,
                      num_points_sampled=None):
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
                assert (len(grid_shape) == 3)
                if all(g == grid_shape[0] for g in grid_shape[1:]):
                    grid_shape = grid_shape[0]

            grid_shape_str = ('%d' % grid_shape if isinstance(grid_shape, int)
                              else 'x'.join(str(g) for g in grid_shape))
            name = 'ffd-%s-%d' % (grid_shape_str, num_points)

        if isinstance(grid_shape, int):
            grid_shape = (grid_shape,) * 3
        self._grid_shape = grid_shape
        super(FfdModelnetConfig, self).__init__(num_points=num_points,
                                                name=name,
                                                **kwargs)
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
            'b':
                features.Tensor(shape=(self.num_points, grid_size),
                                dtype=tf.float32),
            'p':
                features.Tensor(shape=(grid_size, 3), dtype=tf.float32),
        })

    @abc.abstractmethod
    def load_example(self, off_path):
        points = super(FfdModelnetConfig, self).load_example(off_path)
        return self._f(points)


@gin.configurable
class ModelnetProblem(TfdsProblem):

    def __init__(self,
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
                 num_examples_override=None,
                 map_fn=None,
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
        input_spec = tf.keras.layers.InputSpec(shape=(num_points_sampled, 3),
                                               dtype=tf.float32)
        output_spec = tf.keras.layers.InputSpec(shape=(num_classes,),
                                                dtype=tf.float32)
        self._num_points_sampled = num_points_sampled
        self._train_percent = train_percent
        self._use_train_test_split = use_train_test_split

        self._base_map_fn = functools.partial(
            base_modelnet_map,
            positions_only=positions_only,
            num_points_sampled=num_points_sampled)
        super(ModelnetProblem, self).__init__(builder=builder,
                                              input_spec=input_spec,
                                              output_spec=output_spec,
                                              loss=loss,
                                              metrics=metrics,
                                              objective=objective,
                                              map_fn=map_fn,
                                              **kwargs)

    def get_config(self):
        base = super(ModelnetProblem, self).get_config()
        for k in 'input_spec', 'output_spec':
            base.pop(k)
        updates = dict(
            num_points_samples=self._num_points_sampled,
            train_percent=self._train_percent,
            use_train_test_split=self._use_train_test_split,
            num_classes=self._builder.builder_configconfig.num_classes)
        base.update(updates)
        return base

    def _get_base_dataset(self, split):
        base = super(ModelnetProblem, self)._get_base_dataset(split)
        return base.map(self._base_map_fn)

    def _split(self, split):
        if self._use_train_test_split:
            if split == tfds.Split.TRAIN:
                return split
            elif split in (tfds.Split.VALIDATION, tfds.Split.TEST):
                return tfds.Split.TEST
        else:
            ReadInstruction = tfds.core.tfrecords_reader.ReadInstruction
            if split == tfds.Split.TRAIN:
                return ReadInstruction('train',
                                       to=self._train_percent,
                                       unit='%')
            elif split == tfds.Split.VALIDATION:
                return ReadInstruction('train',
                                       from_=self._train_percent,
                                       unit='%')
            else:
                return split

    def _examples_per_epoch(self, split, batch_size=None):
        if self._num_examples_override is not None:
            value = self._num_examples_override
        elif self._use_train_test_split:
            value = examples_per_epoch(self._builder, self._split(split))
        elif split == tfds.Split.TRAIN:
            value = int((self._train_percent / 100) *
                        examples_per_epoch(self._builder, tfds.Split.TRAIN))
        elif split == tfds.Split.VALIDATION:
            value = int((1 - self._train_percent / 100) *
                        examples_per_epoch(self._builder, tfds.Split.TRAIN))
        else:
            assert (split == tfds.Split.TEST)
            value = super(ModelnetProblem, self)._examples_per_epoch(split)
        if batch_size is not None:
            value //= batch_size
        return value


def deserialize(name='modelnet40', **kwargs):
    if name == 'modelnet40':
        return ModelnetProblem(num_classes=40, **kwargs)
    elif name == 'modelnet10':
        return ModelnetProblem(num_classes=10, **kwargs)
    else:
        raise ValueError('Unrecognized problem name "%s"' % name)


_problems = {'ModelnetProblem': ModelnetProblem}


def get(identifier):
    if isinstance(identifier, Problem):
        return identifier
    else:
        if isinstance(identifier, six.string_types):
            identifier = {'class_name': identifier, 'config': {}}
        return tf.keras.utils.deserialize_keras_object(identifier,
                                                       module_objects=_problems)
