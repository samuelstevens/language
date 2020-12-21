import tensorflow.compat.v1 as tf


def zero(prefix):
    assignments = []

    for a in tf.trainable_variables(scope=prefix):
        assignments.append(a.assign(tf.zeros_like(a)))

    return assignments


def copy_params(prefix_a, prefix_b):
    assignments = []

    for a, b in zip(
        tf.trainable_variables(scope=prefix_a),
        tf.trainable_variables(scope=prefix_b),
    ):
        assert a.name[len(prefix_a) :] == b.name[len(prefix_b) :]
        assignments.append(b.assign(a))

    return assignments


def add_params(prefix_a, prefix_b):
    assignments = []

    for a, b in zip(
        tf.trainable_variables(scope=prefix_a),
        tf.trainable_variables(scope=prefix_b),
    ):
        assert a.name[len(prefix_a) :] == b.name[len(prefix_b) :]
        assignments.append(b.assign_add(a))

    return assignments


def mean_and_assign_params(prefix_a, prefix_b, k):
    assignments = []

    for a, b in zip(
        tf.trainable_variables(scope=prefix_a),
        tf.trainable_variables(scope=prefix_b),
    ):
        assert a.name[len(prefix_a) :] == b.name[len(prefix_b) :]
        assignments.append(b.assign(a / k))

    return assignments
