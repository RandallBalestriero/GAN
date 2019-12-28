import sys
sys.path.insert(0, "../../TheanoXLA")

import numpy as np
import theanoxla.tensor as T
from theanoxla import layers, losses, optimizers, function, gradients
from theanoxla.utils import batchify, vq_to_boundary
from sklearn import datasets
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


def get_stats(determinants):
    mean = np.mean(determinants)
    std = np.std(determinants)
    gap = np.abs(determinants.min() - determinants.max())
    return mean, std, gap

filename = 'determinant_comparison_{}_{}_{}_{}_{}_0.npz'
data = list()

D = [2, 4, 6, 8, 10, 12]
STD = [0.033, 0.066, 0.1, 0.133]
RADIUS = [0.5, 1, 1.5, 2]

for d in D:
    for std in STD:
        for radius in RADIUS:
            datum = np.load(filename.format(radius, std, d, 64, 4))
            data.append(get_stats(datum['determinant']))

data = np.array(data).reshape((len(D), len(STD), len(RADIUS), 3))
for k, d in enumerate(D):
    plt.subplot(3, len(D), 1+k)
    plt.imshow(data[k, :, :, 0], aspect='auto')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, len(D), 1+len(D)+k)
    plt.imshow(data[k, :, :, 1], aspect='auto')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, len(D), 1+2*len(D)+k)
    plt.imshow(data[k, :, :, 1], aspect='auto')
    plt.xticks([])
    plt.yticks([])


plt.show()
