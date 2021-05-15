import tensorflow as tf
import numpy as np
import timeit
import time
import relu
import maxpool


def measure_time(fun, repeat, *args):
    times = np.empty(repeat)
    for i in range(repeat):
        t = time.time()
        fun(*args)
        times[i] = time.time()-t
    return times


if __name__ == "__main__":

    x = tf.random.normal((256, 64, 64, 32))
    u = tf.random.normal((256, 64, 64, 32))
    v = tf.concat([x, x - u], 0)

    relu.base(x)
    relu.clone_v1(v)
    relu.clone_v2(v)
    relu.clone_v3(v)
    maxpool.base(x)
    maxpool.clone_v1(v)

    print("testing times for ReLU implementations:")
    print("\t baseline:",np.min(measure_time(relu.base, 1000, x)))
    print("\t clone_v1:",np.min(measure_time(relu.clone_v1, 1000, v)))
    print("\t clone_v2:",np.min(measure_time(relu.clone_v2, 1000, v)))
    print("\t clone_v3:",np.min(measure_time(relu.clone_v3, 1000, v)))

    print("testing times for maxpooling implementations:")
    print("\t baseline:",np.min(measure_time(maxpool.base, 1000, x)))
    print("\t clone_v1:",np.min(measure_time(maxpool.clone_v1, 1000, v)))
    print("\t clone_v2:",np.min(measure_time(maxpool.clone_v2, 1000, v)))
    # print("\t clone_v3:",timeit.timeit('relu.clone_v3(v)', globals=globals(), number=100))