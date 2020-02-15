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

import matplotlib as mpl
label_size = 18
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size


def get_stats(f1, f2):
    # remove the duplicates
    outputs = [[f1[0], f2[0]]]
    for i in range(1, len(f1)):
        if np.abs(f1[i] - outputs[-1][0]).min() > 1e-10:
            outputs.append([f1[i], f2[i]])
    outputs = np.asarray(outputs)
    print(outputs.shape)
    # compute histograms
    a, b = np.histogram(outputs[:, 0], MAXI, density=True)

    # now get distances
    c = list()
    for i in range(len(b)-1):
        which = (outputs[:, 0] > b[i]) & (outputs[:, 0] <= b[i+1])
        if which.sum() > 0:
            c.append(outputs[which, 1].mean())
        else:
            c.append(np.nan)
    return np.stack([a, np.array(c), b[1:]], 0)
#    determinants =  (determinants + 1e-12)
    mean = np.mean(determinants)
    std = np.std(determinants)
    gap = np.abs(determinants.min() - determinants.max())
    return np.log(gap)


def do_plot(samples, data, name, run):
    for j in range(samples.shape[0]):
        plt.figure(figsize=(4, 4))
        plt.plot(samples[j, 0, :, 0],
                 samples[j, 0, :, 1], 'xr', alpha=0.2)
        plt.plot(samples[j, 1, :, 0],
                 samples[j, 1, :, 1], 'xb', alpha=0.2)
        plt.ylim([min(samples[j, 0, :, 1].min(),
                      samples[j, 1, :, 1].min()),
                  max(samples[j, 0, :, 1].max(),
                      samples[j, 1, :, 1].max())])
        plt.xlim([min(samples[j, 0, :, 0].min(),
                      samples[j, 1, :, 0].min()),
                  max(samples[j, 0, :, 0].max(),
                      samples[j, 1, :, 0].max())])
        plt.tight_layout()
        plt.savefig('{}_data_2d_std{}_run{}.jpg'.format(name, j, run))
        plt.close()

        plt.figure(figsize=(4, 2))
        y, z, x = data[j]
        plt.plot(x, y, 'b', lw=4)
        plt.ylim([min(data[:, 0].min(), data[:, 1].min()),
                  max(data[:, 0].max(), data[:, 1].max())])
        plt.xlim([data[:, 2].min(), data[:, 2].max()])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        for pos in ['top', 'right', 'left']:
            ax.spines[pos].set_edgecolor('white')
        plt.savefig('{}_deter_2d_std{}_run{}.jpg'.format(name, j, run))
        plt.close()


filename = 'determinant_comparison_{}_{}_{}_{}_{}_{}_{}.npz'
MAXI = 30
RUN = range(10)

# STD


def do_std():
    STD = [0.01, 0.05, 0.1, 0.3, 1.]
    scale = 1.
    radius = 0.
    cpt = 0

    for run in RUN:
        data = list()
        samples = list()
        for std in STD:
            file = np.load(filename.format(radius, scale, std, 2, 64, 2, run))
            det, samp, dat = file['determinant'], file['samples'], file['data']
            w = det > 0
            det = det[w]
            distances = ((dat[:, None, :] - samp)**2).mean(2).min(0)
            distances = distances[w]
            data.append(get_stats(np.log(1/det), distances))
            samples.append([samp[:500], dat])
        data = np.array(data).reshape((len(STD), 3, MAXI))
        samples = np.array(samples).reshape((len(STD), 2, 500, 2))
        do_plot(samples, data, 'std', cpt)
        cpt += 1


# SCALE
def do_scale():
    SCALE = [0.002, 0.01, 0.05, 0.1, 0.3, 1., 2.]
    radius = 0.
    std = 0.3
    cpt = 0

    for run in RUN:
        data = list()
        samples = list()
        for scale in SCALE:
            file = np.load(filename.format(radius, scale, std, 2, 64, 2, run))
            det, samp, dat = file['determinant'], file['samples'], file['data']
            distances = ((dat[:, None, :] - samp)**2).mean(2).min(0)
            w = det > 0
            det = det[w]
            distances = distances[w]
            data.append(get_stats(np.log(1/det), distances))
            samples.append([samp[:500], dat])
        data = np.array(data).reshape((len(SCALE), 3, MAXI))
        samples = np.array(samples).reshape((len(SCALE), 2, 500, 2))
        do_plot(samples, data, 'scale', cpt)
        cpt += 1


# RADIUS
def do_radius():
    RADIUS = [0., 0.5, 1., 2., 4.]
    scale = 1.
    std = 0.3
    cpt = 0

    for run in RUN:
        data = list()
        samples = list()
        for radius in RADIUS:
            file = np.load(filename.format(radius, scale, std, 2, 64, 2, run))
            det, samp, dat = file['determinant'], file['samples'], file['data']
            w = det > 0
            det = det[w]
            distances = ((dat[:, None, :] - samp)**2).mean(2).min(0)
            distances = distances[w]
            data.append(get_stats(np.log(1/det), distances))
            samples.append([samp[:500], dat])
        data = np.array(data).reshape((len(RADIUS), 3, MAXI))
        samples = np.array(samples).reshape((len(RADIUS), 2, 500, 2))
        do_plot(samples, data, 'radius', cpt)
        cpt += 1


do_radius()

sadf
#data = data.transpose([0, 1, 3, 2, 4]).reshape((len(D), len(STD), 2, -1))

plt.figure(figsize=(14, 8))
cpt = 1

for i in range(len(D)):
    for j in range(len(STD)):
        plt.subplot(len(D), len(STD), cpt)
        y, z, x = get_stats(data[i, j, 0], data[i, j, 1])
        plt.plot(x, y, 'b')
        plt.plot(x, z, 'r')
        plt.ylim([0, 1.5])
        plt.xlim([-10, 20])
#        plt.ylim([data[:, :, :, 0].min(), data[:, :, :, 0].max()])
#        plt.xlim([data[:, :, :, 2].min(), data[:, :, :, 2].max()])
        cpt += 1
    plt.tight_layout()
    plt.savefig('deter_2d_std{}_run{}.jpg'.format(j, run))
    plt.close()


plt.figure(figsize=(8, 6))
cpt = 1
for i in range(len(D)):
    for j in range(len(STD)):
        plt.subplot(len(D), len(STD), cpt)
        for r in range(len(RUN)):
            plt.plot(data[i, j, r, 1])  # ,(data[i, j, BEST[i, j], 0]))
#        plt.xlim([xmin, xmax])
        plt.ylim([data[:, :, :, 1].min(), data[:, :, :, 1].max()])
#        plt.xticks([])
#        plt.yticks([])
#        ax = plt.gca()
#        ax2 = ax.twinx()
#        ax2.plot(data[i, j, BEST[i, j], 2], data[i, j, BEST[i, j], 1], 'r')
        cpt += 1
plt.tight_layout()
plt.savefig('test_plot3.jpg')
plt.close()


plt.figure(figsize=(8, 6))
cpt = 1
for i in range(len(D)):
    for j in range(len(STD)):
        plt.subplot(len(D), len(STD), cpt)
        try:
            w = data3[i, j, BEST[i, j], 1] < 250
            plt.plot(data3[i, j, BEST[i, j], 0, w], 'xb')
            w = data3[i, j, BEST[i, j], 1] > 250
            plt.plot(data3[i, j, BEST[i, j], 0, w], 'xr')
            plt.xticks([])
        except:
            print('skipping')
        cpt += 1


plt.savefig('test_plot2.jpg')
