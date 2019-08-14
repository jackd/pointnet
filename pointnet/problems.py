from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
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
                      num_points=None,
                      up_dim=2):
    if not positions_only:
        raise NotImplementedError()
    if isinstance(inputs, dict):
        positions = inputs['positions']
        normals = None if positions_only else inputs['normals']
    else:
        positions = inputs
        normals = None
    if up_dim != 2:
        shift = 2 - up_dim
        positions = tf.roll(positions, shift, axis=-1)
        if normals is not None:
            normals = tf.roll(normals, shift, axis=-1)

    if num_points is not None:
        if positions_only:
            positions = tf.random.shuffle(positions)[:num_points]
        else:
            indices = tf.range(tf.shape(positions)[0])
            indices = tf.random.shuffle(indices)[:num_points]
            positions = tf.gather(positions, indices)
            normals = tf.gather(positions, indices)
    if positions_only:
        inputs = positions
    else:
        inputs = dict(positions=positions, normals=normals)
    return inputs, labels


@gin.configurable(module='pointnet.problems')
class ModelnetProblem(TfdsProblem):

    def __init__(
            self,
            builder,
            num_points=1024,
            positions_only=True,
            loss=None,
            metrics=None,
            objective=None,
            train_split='full',  # 'full' or integer percent
            num_examples_override=None,
            map_fn=None,
            shuffle_buffer=-1,
            **kwargs):
        import functools
        if loss is None:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)
        if metrics is None:
            metrics = (
                tf.keras.metrics.SparseCategoricalAccuracy(),
                tf.keras.metrics.SparseCategoricalCrossentropy(
                    from_logits=True),
            )

        num_classes = builder.num_classes

        self._num_examples_override = num_examples_override
        if objective is None:
            if len(metrics) == 0:
                objective = None
            else:
                objective = Objective('val_%s' % metrics[0].name, 'max')

        input_spec = tf.keras.layers.InputSpec(shape=(num_points, 3),
                                               dtype=tf.float32)
        output_spec = tf.keras.layers.InputSpec(shape=(num_classes,),
                                                dtype=tf.float32)
        self._num_points = num_points
        self._train_split = train_split

        self._base_map_fn = functools.partial(base_modelnet_map,
                                              positions_only=positions_only,
                                              num_points=num_points,
                                              up_dim=builder.up_dim)
        super(ModelnetProblem, self).__init__(builder=builder,
                                              input_spec=input_spec,
                                              output_spec=output_spec,
                                              loss=loss,
                                              metrics=metrics,
                                              objective=objective,
                                              map_fn=map_fn,
                                              shuffle_buffer=shuffle_buffer,
                                              **kwargs)

    def get_config(self):
        raise NotImplementedError('builder config not implemented')
        # base = super(ModelnetProblem, self).get_config()
        # for k in 'input_spec', 'output_spec':
        #     base.pop(k)
        # updates = dict(builder=None,
        #                num_points=self._num_points,
        #                train_split=self._train_split,
        #                num_classes=self._builder.builder_config.num_classes)
        # base.update(updates)
        # return base

    def _get_base_dataset(self, split):
        base = super(ModelnetProblem, self)._get_base_dataset(split)
        return base.map(self._base_map_fn)

    def _split(self, split):
        if self._train_split in 'full':
            if split == tfds.Split.TRAIN:
                return split
            elif split in (tfds.Split.VALIDATION, tfds.Split.TEST):
                return tfds.Split.TEST
        else:
            ReadInstruction = tfds.core.tfrecords_reader.ReadInstruction
            if split == tfds.Split.TRAIN:
                return ReadInstruction('train', to=self._train_split, unit='%')
            elif split == tfds.Split.VALIDATION:
                return ReadInstruction('train',
                                       from_=self._train_split,
                                       unit='%')
            else:
                return split

    def _examples_per_epoch(self, split, batch_size=None):
        if self._num_examples_override is not None:
            value = self._num_examples_override
        elif self._train_split == 'full':
            value = examples_per_epoch(self._builder, self._split(split))
        elif split == tfds.Split.TRAIN:
            value = int((self._train_split / 100) *
                        examples_per_epoch(self._builder, tfds.Split.TRAIN))
        elif split == tfds.Split.VALIDATION:
            value = int((1 - self._train_split / 100) *
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
