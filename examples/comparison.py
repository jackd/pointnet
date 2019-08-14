from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from pointnet.models import transform_diff


# original version
def get_loss(pred, label, transform, reg_weight):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,
                                                          labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


def our_loss(pred, label, transform, reg_weight):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
        label, pred)
    reg = tf.keras.regularizers.l2(reg_weight)
    loss = loss + reg(transform_diff(transform))
    return loss


batch_size = 10
num_classes = 20
K = 64
reg_weight = 0.001

labels = tf.cast(tf.random.uniform(shape=(batch_size,)) * num_classes, tf.int64)
preds = tf.random.normal(shape=(batch_size, num_classes), dtype=tf.float32)
transform = tf.random.normal(shape=(batch_size, K, K), dtype=tf.float32)
print(get_loss(preds, labels, transform, reg_weight).numpy())
print(our_loss(preds, labels, transform, reg_weight / 2).numpy())
