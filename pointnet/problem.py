from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
from pointnet.augment import augment_cloud

Objective = collections.namedtuple('Objective', ['metric_name', 'mode'])


class Problem(object):
    @abc.abstractmethod
    def get_dataset(self, split, batch_size=None):
        raise NotImplementedError

    @abc.abstractmethod
    def examples_per_epoch(self, split=tfds.Split.TRAIN):
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


def examples_per_epoch(builder, split):
    return int(builder.info.splits[split].num_examples)



@gin.configurable
class TfdsProblem(Problem):
    def __init__(
            self, builder, output_spec, loss, metrics=None, map_fn=None,
            as_supervised=True, shuffle_buffer=1024, objective=None):
        self._builder = builder
        self._output_spec = output_spec
        self._loss = loss
        self._metrics = tuple(metrics)
        self._map_fn = map_fn
        self._as_supervised = as_supervised
        self._shuffle_buffer = 1024
        self._objective = objective

    def _split(self, split):
        # allow for possible override
        return split

    def examples_per_epoch(self, split=tfds.Split.TRAIN):
        return examples_per_epoch(self._builder, self._split(split))

    def get_dataset(self, split, batch_size=None):
        dataset = self._builder.as_dataset(
            batch_size=None, split=self._split(split),
            as_supervised=self._as_supervised,
            shuffle_files=split==tfds.Split.TRAIN)

        map_fn = self._map_fn
        if isinstance(map_fn, dict):
            map_fn = map_fn[split]

        if split == tfds.Split.TRAIN:
            dataset = dataset.repeat().shuffle(self._shuffle_buffer)

        if map_fn is not None:
            dataset = dataset.map(
                map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset.batch(batch_size).prefetch(
            tf.data.experimental.AUTOTUNE)

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


@gin.configurable
class ModelnetProblem(TfdsProblem):
    def __init__(
            self, num_classes=40, num_points_base=2048, map_fn=augment_cloud,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=(tf.keras.metrics.SparseCategoricalAccuracy(),),
            objective=Objective('SparseCategoricalCrossentropy', 'max'),
            train_percent=90,
            download_kwargs={},
            **kwargs):
        from shape_tfds.shape import modelnet
        config = modelnet.CloudConfig(num_points=num_points_base)
        builder = {
            10: modelnet.Modelnet10,
            40: modelnet.Modelnet40,
        }[num_classes](config=config)
        builder.download_and_prepare(**download_kwargs)
        output_spec = tf.keras.layers.InputSpec(
            shape=(num_classes,), dtype=tf.float32)
        self._train_percent = train_percent
        super(ModelnetProblem, self).__init__(
            builder=builder,
            output_spec=output_spec,
            loss=loss,
            metrics=metrics,
            objective=objective,
            **kwargs)

    def _split(self, split):
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
        if split == tfds.Split.TRAIN:
            return int((self._train_percent / 100) *
                       examples_per_epoch(self._builder, tfds.Split.TRAIN))
        elif split == tfds.Split.VALIDATION:
            return int((1 - self._train_percent / 100) *
                       examples_per_epoch(self._builder, tfds.Split.TRAIN))
        else:
            assert(split == tfds.Split.TEST)
            return super(ModelnetProblem, self).examples_per_epoch(split)
