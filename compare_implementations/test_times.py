import tensorflow as tf
import numpy as np
import timeit
import relu



if __name__ == "__main__":

    x = tf.random.normal((256, 64, 64, 32))
    u = tf.random.normal((256, 64, 64, 32))
    v = tf.concat([x, x - u], 0)

    times = np.empty((len(functions), 1000))
    print(timeit.timeit('relu.relu(x)'.format(f.__name__), globals=globals(), number=10))
    print(timeit.timeit('relu.relu_clone_v1(v)'.format(f.__name__), globals=globals(), number=10))
    print(timeit.timeit('relu.relu_clone_v2(v)'.format(f.__name__), globals=globals(), number=10))
    print(timeit.timeit('relu.relu_clone_v3(v)'.format(f.__name__), globals=globals(), number=10))