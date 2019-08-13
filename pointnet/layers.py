from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf


@gin.configurable(module='pointnet.layers')
class VariableMomentumBatchNormalization(tf.keras.layers.BatchNormalization):

    def build(self, input_shape):
        if not self.built:
            super(VariableMomentumBatchNormalization, self).build(input_shape)
            self.momentum = self.add_weight(
                'momentum',
                initializer=tf.keras.initializers.constant(self.momentum),
                trainable=False)

    def get_config(self):
        config = super(VariableMomentumBatchNormalization, self).get_config()
        config['momentum'] = tf.keras.backend.get_value(config['momentum'])
        return config
