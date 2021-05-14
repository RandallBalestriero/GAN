import tensorflow as tf
import time


@tf.function
def clone_relu_v1(x):
    S = x.shape
    mask = x[: S[0] // 2] > 0
    mask = tf.tile(mask, [2] + [1] * (len(S) - 1))
    return tf.where(mask, x, 0)


@tf.function
def clone_relu_v2(x):
    S = x.shape
    x = tf.reshape(x, (2, S[0] // 2) + S[1:])
    mask = x[0] > 0
    out = tf.where(mask, x, 0)
    return tf.reshape(out, S)


@tf.function
def clone_relu_v3(x):
    xonly = x[: x.shape[0] // 2]
    xonly = tf.tile(xonly, [2] + [1] * (len(x.shape) - 1))
    with tf.autodiff.ForwardAccumulator(primals=xonly, tangents=x) as acc:
        output = tf.nn.relu(xonly)
    return acc.jvp(output)


@tf.function
def relu(x):
    return tf.nn.relu(x)
