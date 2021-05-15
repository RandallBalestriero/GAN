import tensorflow as tf
import relu, maxpool
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow import keras


n_channels=3
spatial=256
batch=4


def create_model_jvp(depth, model, option):

    if option == "conditional":
        activation = relu.clone_v1
        pooling = maxpool.clone_v1
    else:
        activation = relu.base
        pooling = maxpool.base
        

    tf.random.set_seed(1)
    input = keras.layers.Input((spatial, spatial, n_channels), batch_size=batch * (1+int(option=="conditional")))
    output = input * 1
    for l in range(depth):
        output = keras.layers.Conv2D(128, 3, padding="SAME")(output)
        output = keras.layers.BatchNormalization()(output)
        output_res = keras.layers.Conv2D(128, 1)(output)
        output = activation(output)
        output = keras.layers.BatchNormalization()(output + output_res)
        if l < 5:
            output = pooling(output,ksize=(2,2), strid=2)

    output = keras.layers.Conv2D(128, 1)(output)
    output = pooling(output,ksize=output.shape[1:3], strid=output.shape[1:3])
    output = keras.layers.Dense(10)(output)
    model = keras.Model(input, output)

    if option == "conditional":
        
        @tf.function
        def dn(x, u, v):
            output = model(v, training=False)
            return output[:output.shape[0]//2] - output[output.shape[0]//2:] 

    elif option =="double":

        @tf.function
        def dn(x, u, v):
            with tf.GradientTape() as tape1:
                with tf.GradientTape() as tape2:
                    tape2.watch(x)
                    output = model(x, training=False)
                v = tf.ones(output.shape)
                tape1.watch(v)
                first_grad = tape2.gradient(output, x, v)
            return tape1.gradient(first_grad,v,u)

    elif option == "jvp":

        @tf.function
        def dn(x, u, v):
            with tf.autodiff.ForwardAccumulator(primals=x, tangents=u) as acc:
                output = model(x, training=False)
            return acc.jvp(output)

    return dn

def get_times(depth, model, option, generate):
        baseline = create_model_jvp(depth, model, option)
        # baseline = create_model_jvp(depth, "baseline", "linear_conv")
        # conditional = create_model_jvp(depth, "conditional", "linear_conv")

        # # print(baseline(x, u)[0, 0, 0, :2], conditional(v)[0, 0, 0, :2])
        # print("Is the JVP correct?",np.allclose(baseline(x, u).numpy(), conditional(v).numpy(), atol=1e-4))
        # we call it once to compile
        args = generate()
        output = baseline(*args)
        times = np.empty(200)
        for i in range(len(times)):
            args = generate()
            t = time.time()
            baseline(*args)
            times[i] = time.time() - t

        return times, output

if __name__ == "__main__":

    def generate():
        x = tf.random.normal((batch, spatial, spatial, n_channels))
        u = tf.random.normal((batch, spatial, spatial, n_channels))
        v = tf.concat([x, x - u], 0)
        return x, u, v

    model = "resnet"
    times = np.zeros((500, 3, 100))
    for run in range(500):
        print("run", run)
        for depth in range(1,101):
            times1, output1 = get_times(depth, model, "double", generate)
            times2, output2  = get_times(depth, model, "jvp", generate)
            times3, output3 = get_times(depth, model, "conditional", generate)
            times[run, :,depth - 1] = np.array([times1.min(),times2.min(),times3.min()])

            # print("Is the JVP correct?",np.allclose(output1.numpy(), output2.numpy(), atol=1e-4))
            # print("Is the JVP correct?",np.allclose(output3.numpy(), output2.numpy(), atol=1e-4))
            print(times1.mean(),times2.mean(),times3.mean())
            print(times1.min(),times2.min(),times3.min())
            # plt.hist([times1, times2, times3], bins=20, density=True, color=["red", "green", "blue"])
            # plt.savefig("histogram_times_{}_{}.png".format(model,depth))
            # plt.close()
            np.savez("times_saving_deep.npz", times=times)

    # times = np.load("times_saving.npz")
    # times = times["times"]
    # print(times)
    # times = times[:3].mean(0)
    # plt.plot(times[0],color="r")
    # plt.plot(times[1],color="g")
    # plt.plot(times[2],color="b")

    # plt.savefig("times.png")