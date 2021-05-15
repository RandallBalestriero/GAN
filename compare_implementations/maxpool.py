import tensorflow as tf
import numpy as np

def clone_v1(x, ksize=(3, 3), strid=3, pad="VALID"):
    x, u = tf.split(x, 2)
    ma_x, Jx = tf.nn.max_pool_with_argmax(
        x, ksize, strid, pad, include_batch_in_index=True
    )
    flat_u = tf.reshape(u, [-1])
    Jx_u = tf.gather(flat_u, Jx)
    out = tf.concat([ma_x, Jx_u], 0)
    return out

def clone_v2(x, ksize=(3, 3), strid=3, pad="VALID"):
    Jx = tf.nn.max_pool_with_argmax(
        x[:x.shape[0]//2], ksize, strid, pad, include_batch_in_index=True
    )[1]
    Jx = tf.tile(Jx, [2]+[1]*(len(x.shape)-1))
    Jx[x.shape[0]//2:] += np.prod(x.shape)
    flat = tf.reshape(x, [-1])
    return tf.gather(flat, Jx)

def base(x, ksize=(3, 3), strid=1, pad="VALID"):
    return tf.nn.max_pool(x, ksize, strid, pad)