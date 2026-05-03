"""Loss helpers for weak clip supervision and masked window distillation."""

from __future__ import annotations

import tensorflow as tf


def masked_bce_with_logits(
    logits: tf.Tensor,
    targets: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
    mask = tf.cast(mask, losses.dtype)
    while mask.shape.rank < losses.shape.rank:
        mask = mask[..., tf.newaxis]
    weighted = losses * mask
    denom = tf.reduce_sum(mask) * tf.cast(tf.shape(losses)[-1], losses.dtype)
    return tf.reduce_sum(weighted) / tf.maximum(denom, tf.constant(1.0, dtype=losses.dtype))


def clip_bce_with_logits(
    logits: tf.Tensor,
    targets: tf.Tensor,
) -> tf.Tensor:
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
    return tf.reduce_mean(losses)


def masked_sigmoid_mse(
    student_logits: tf.Tensor,
    teacher_logits: tf.Tensor,
    mask: tf.Tensor,
    class_mask: tf.Tensor | None = None,
) -> tf.Tensor:
    student_prob = tf.math.sigmoid(student_logits)
    teacher_prob = tf.math.sigmoid(teacher_logits)
    losses = tf.math.squared_difference(student_prob, teacher_prob)

    mask = tf.cast(mask, losses.dtype)
    while mask.shape.rank < losses.shape.rank:
        mask = mask[..., tf.newaxis]
    weighted = losses * mask

    if class_mask is not None:
        class_weight = tf.cast(class_mask, losses.dtype)[tf.newaxis, tf.newaxis, :]
        weighted = weighted * class_weight
        class_count = tf.reduce_sum(class_weight)
    else:
        class_count = tf.cast(tf.shape(losses)[-1], losses.dtype)

    denom = tf.reduce_sum(mask) * tf.maximum(class_count, tf.constant(1.0, dtype=losses.dtype))
    return tf.reduce_sum(weighted) / tf.maximum(denom, tf.constant(1.0, dtype=losses.dtype))
