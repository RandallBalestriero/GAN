import tensorflow as tf
import numpy as np
import time


@tf.function
def clone_relu1(x):
    S = x.shape
    mask = x[: S[0] // 2] > 0
    mask = tf.tile(mask, [2] + [1] * (len(S) - 1))
    return tf.where(mask, x, 0)


@tf.function
def clone_relu2(x):
    S = x.shape
    x = tf.reshape(x, (2, S[0] // 2) + S[1:])
    mask = x[0] > 0
    out = tf.where(mask, x, 0)
    return tf.reshape(out, S)


@tf.function
def clone_relu3(x):
    xonly = x[: x.shape[0] // 2]
    xonly = tf.tile(xonly, [2] + [1] * (len(x.shape) - 1))
    with tf.autodiff.ForwardAccumulator(primals=xonly, tangents=x) as acc:
        output = tf.nn.relu(xonly)
    return acc.jvp(output)


@tf.function
def relu(x):
    return tf.nn.relu(x)


@tf.function
def clone_sigmoid(x):
    S = tuple(x.shape)
    x = tf.reshape(x, (2, S[0] // 2) + S[1:])
    sigmoid_x = tf.sigmoid(x[0])
    sigmoid_x2 = tf.square(sigmoid_x)
    Jx = sigmoid_x - sigmoid_x2
    bx = sigmoid_x - Jx * x[0]
    return tf.reshape(Jx * x + bx, S)


@tf.function
def sigmoid(x):
    return tf.sigmoid(x)


@tf.function
def clone_maxpool(x, ksize=(3, 3), strid=1, pad="VALID"):
    x, u = tf.split(x, 2)
    ma_x, Jx = tf.nn.max_pool_with_argmax(
        x, ksize, strid, pad, include_batch_in_index=True
    )
    Jx = tf.cast(Jx, "int32")
    flat_u = tf.reshape(u, [-1])
    Jx_u = tf.gather(flat_u, Jx)
    out = tf.concat([ma_x, Jx_u], 0)
    return out


@tf.function
def maxpool(x, ksize=(3, 3), strid=1, pad="VALID"):
    return tf.nn.max_pool(x, ksize, strid, pad)


@tf.function
def clone_dropout(x, rate=0.5, train=True):
    if not train:
        return x
    S = tuple(x.shape)
    x = tf.reshape(x, (2, S[0] // 2) + S[1:])
    keep_prob = 1 - rate
    rand = tf.random.uniform(x[0].shape, seed=1)
    out = tf.where(rand > rate, x, 0)
    return out / keep_prob


@tf.function
def dropout(x, rate=0.5, train=True):
    if not train:
        return x
    keep_prob = 1 - rate
    S = x.shape
    uniform = tf.random.uniform(S, seed=1)
    out = tf.where(uniform > rate, x, 0)
    return out / keep_prob


def compare_times_forward():
    N = 10000
    x = tf.random.normal((256, 64, 64, 32))
    u = x
    v = tf.concat([x, u], 0)

    dropout(x)
    clone_dropout(v)

    t = time.time()
    for i in range(N):
        clone_dropout(v)
    clone_time = time.time() - t

    t = time.time()
    for i in range(N):
        dropout(x)
    print(clone_time / (time.time() - t))

    maxpool(x)
    clone_maxpool(v)

    t = time.time()
    for i in range(N):
        clone_maxpool(v)
    clone_time = time.time() - t

    t = time.time()
    for i in range(N):
        maxpool(x)
    print(clone_time / (time.time() - t))

    sigmoid(x)
    clone_sigmoid(v)

    t = time.time()
    for i in range(N):
        clone_sigmoid(v)
    clone_time = time.time() - t

    t = time.time()
    for i in range(N):
        sigmoid(x)
    print(clone_time / (time.time() - t))

    relu(x)
    clone_relu(v)

    t = time.time()
    for i in range(N):
        clone_relu(v)
    clone_time = time.time() - t

    t = time.time()
    for i in range(N):
        relu(x)
    print(clone_time / (time.time() - t))


def sanity_check():
    x = tf.random.normal((256, 64, 64, 32))
    u = tf.random.normal((256, 64, 64, 32))
    v = tf.concat([x, x - u], 0)

    @tf.function
    def relu_jvp(x, u):
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=u) as acc:
            output = relu(x)
        return acc.jvp(output)

    @tf.function
    def clone_relu_jvp(v):
        output = clone_relu(v)
        return output[:256] - output[256:]

    @tf.function
    def sigmoid_jvp(x, u):
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=u) as acc:
            output = sigmoid(x)
        return acc.jvp(output)

    @tf.function
    def clone_sigmoid_jvp(v):
        output = clone_sigmoid(v)
        return output[:256] - output[256:]

    @tf.function
    def sigmoid_jvp(x, u):
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=u) as acc:
            output = sigmoid(x)
        return acc.jvp(output)

    @tf.function
    def clone_sigmoid_jvp(v):
        output = clone_sigmoid(v)
        return output[:256] - output[256:]

    @tf.function
    def maxpool_jvp(x, u):
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=u) as acc:
            output = maxpool(x)
        return acc.jvp(output)

    @tf.function
    def clone_maxpool_jvp(v):
        output = clone_maxpool(v)
        return output[:256] - output[256:]

    print(np.allclose(relu_jvp(x, u).numpy(), clone_relu_jvp(v).numpy(), atol=1e-5))
    print(
        np.allclose(sigmoid_jvp(x, u).numpy(), clone_sigmoid_jvp(v).numpy(), atol=1e-5)
    )
    print(
        np.allclose(maxpool_jvp(x, u).numpy(), clone_maxpool_jvp(v).numpy(), atol=1e-5)
    )


def compare_depth_jvp():
    def create_model_jvp(D, option, model):
        init = tf.keras.initializers.GlorotUniform(seed=1)

        filters = [init((3, 3, 256, 256)) for l in range(D)]
        biases = [init((256,)) for l in range(D)]
        if option == "conditional":

            @tf.function
            def dn(v):
                output = v * 1
                for l in range(D):
                    if model == "linear_conv":
                        output = (
                            tf.nn.conv2d(output, filters[l], strides=1, padding="SAME")
                            + biases[l]
                        )
                    elif model == "nonlinear_conv":
                        output = (
                            tf.nn.conv2d(output, filters[l], strides=1, padding="SAME")
                            + biases[l]
                        )
                        output = clone_relu1(output)

                return output[: output.shape[0] // 2] - output[output.shape[0] // 2 :]

        else:

            @tf.function
            def dn(x, u):
                with tf.autodiff.ForwardAccumulator(primals=x, tangents=u) as acc:
                    output = x * 1
                    for l in range(D):
                        if model == "linear_conv":
                            output = (
                                tf.nn.conv2d(
                                    output, filters[l], strides=1, padding="SAME"
                                )
                                + biases[l]
                            )
                        elif model == "nonlinear_conv":
                            output = relu(
                                tf.nn.conv2d(
                                    output, filters[l], strides=1, padding="SAME"
                                )
                                + biases[l]
                            )
                return acc.jvp(output)

        return dn

    x = tf.random.normal((16, 128, 128, 256))
    u = tf.random.normal((16, 128, 128, 256))
    v = tf.concat([x, x - u], 0)

    for f in range(2, 140, 5):
        baseline = create_model_jvp(f, "baseline", "nonlinear_conv")
        conditional = create_model_jvp(f, "conditional", "nonlinear_conv")

        print(baseline(x, u)[0, 0, 0, :2], conditional(v)[0, 0, 0, :2])
        print(np.allclose(baseline(x, u).numpy(), conditional(v).numpy(), atol=1e-3))

        print("layers", f)
        t = time.time()
        for i in range(10):
            baseline(x, u)
        base_time = time.time() - t

        t = time.time()
        for i in range(10):
            conditional(v)
        print(base_time / (time.time() - t))


compare_depth_jvp()
# compare_times_forward()
