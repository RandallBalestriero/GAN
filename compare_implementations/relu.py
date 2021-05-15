import tensorflow as tf
import time


def clone_v0(x):
    S = x.shape
    inp = tf.tile(x[:x.shape[0]//2], [2] + [1] * (len(S) - 1))
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape2.watch(inp)
            output = tf.nn.relu(inp)
        v = tf.ones(output.shape)
        tape1.watch(v)
        first_grad = tape2.gradient(output, inp,v)
    return tape1.gradient(first_grad,v,x)
    mask = tf.tile(mask, [2] + [1] * (len(S) - 1))
    return tf.where(mask, x, 0)

def clone_v1(x):
    S = x.shape
    mask = x[: S[0] // 2] > 0
    mask = tf.tile(mask, [2] + [1] * (len(S) - 1))
    return tf.where(mask, x, 0.2*x)

def clone_v2(x):
    S = x.shape
    x = tf.reshape(x, (2, S[0] // 2) + S[1:])
    mask = x[0] > 0
    out = tf.where(mask, x, 0)
    return tf.reshape(out, S)

def clone_v3(x):
    x, u = tf.split(x, 2)
    mask = x[0] > 0
    out = tf.where(mask, x, 0)
    return tf.concat([tf.nn.relu(x), out],0)

def base(x):
    return tf.nn.leaky_relu(x)
