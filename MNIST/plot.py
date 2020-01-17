import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


data = np.load('mnist_sampling_100_100_0.000311_epoch119.npz')


x_samples, z_samples, det = data['samples'], data['z'], data['determinant']
print(z_samples.shape)
print(det)
w = det > 0
det = det[w]
x_samples = x_samples[w]
z_samples = z_samples[w]

proba = multivariate_normal(np.zeros(100)).pdf(z_samples) / det

args = np.argsort(proba)
x_samples = x_samples[args]

for i in range(20):
    plt.subplot(2, 20, 1 + i)
    plt.imshow(x_samples[i].reshape((28, 28)), aspect='auto')

    plt.subplot(2, 20, 21 + i)
    plt.imshow(x_samples[-i].reshape((28, 28)), aspect='auto')

plt.show(block=True)




