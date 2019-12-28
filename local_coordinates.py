import time
import pickle
import jax
import numpy as np
import sys
sys.path.insert(0, "../TheanoXLA")
from scipy.io.wavfile import read
import glob
import theanoxla
import theanoxla.tensor as T
from theanoxla import layers, losses, optimizers, function, gradients, jacobians
from theanoxla.utils import batchify, vq_to_boundary
import matplotlib.pyplot as plt
from jax.lib import xla_client
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib
import argparse
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D

parse = argparse.ArgumentParser()
parse.add_argument('--dataset', type=str)
parse.add_argument('--WG', type=int)
args = parse.parse_args()



def generator(Z, out_dim, D=4):
    layer = [layers.Dense(Z, D)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], D))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], out_dim))
    return layer

def discriminator(X, D=32):
    layer = [layers.Dense(X, D)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], D))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 2))
    return layer

# some hyper parameters
BS = 100
lr = 0.001
Z = 2
X = 3
WG = args.WG
N = 1000

# create the data
DATA, _ = make_swiss_roll(N, 0.01)


# create the graph inputs
x = T.Placeholder([BS, X], 'float32')
z = T.Placeholder([BS, Z], 'float32')

# and now the computational graph
G = generator(z, X, WG)
D = discriminator(T.concatenate([G[-1], x]))

labels = T.concatenate([T.zeros(BS, dtype='int32'), T.ones(BS, dtype='int32')])

# the loss for G and D
D_loss = losses.sparse_crossentropy_logits(labels, D[-1]).mean()
G_loss = losses.sparse_crossentropy_logits(1 - labels[:BS], D[-1][:BS]).mean()

# gather their variables
D_vars = sum([l.variables() for l in D], [])
G_vars = sum([l.variables() for l in G], [])

# optimizers generating the updates
D_ups = optimizers.Adam(D_loss, D_vars, lr)
G_ups = optimizers.Adam(G_loss, G_vars, lr)
updates = {**D_ups, **G_ups}

# get the A vectors for the generator
masks = T.concatenate([G[1] > 0, G[3] > 0], 1)
G_A = (jacobians(G[-1], [z])[0]).squeeze()

# create the function that will compile the graph
f = function(z, x, outputs = [D_loss, G_loss], updates = updates)
g = function(z, outputs=[G[-1]])
get_mask = function(z, outputs=[masks])
get_A = function(z, outputs=[G_A])

# training
print('training')
for epoch in range(7000):
    for x in batchify(DATA, batch_size=BS, option='random_see_all'):
        z = np.random.rand(BS, Z) * 2 -1
        f(z, x)


# sampling final distribution
#points = np.random.rand(1000, Z) * 2 -1
#H = []
#for x in batchify(points, batch_size=BS, option='random_see_all'):
#    H.append(g(x)[0])
#H = np.concatenate(H)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(DATA[:,0], DATA[:,1], DATA[:, 2], 'bx')
#ax.scatter(H[:,0], H[:,1], H[:, 2], 'rx')


# GET INPUT SPACE PARTITION
print('get mask')
NN = 300
xx, yy = np.meshgrid(np.linspace(-1, 1, NN), np.linspace(-1, 1, NN))
XX = np.stack([xx.flatten(), yy.flatten()], 1)
masks = list()
for x in batchify(XX, batch_size=BS, option='continuous'):
    masks.append(get_mask(x)[0])
masks = np.concatenate(masks)

partition = vq_to_boundary(masks, NN, NN)
#k_partition = [vq_to_boundary(masks[:, [k]], NN, NN) for k in range(masks.shape[1])]
#partition = np.stack(k_partition).sum(0)
partition_location = XX[partition.reshape((-1,)) > 0]
#partition_location = XX[partition.reshape((-1,)) > 0]
print(partition_location)
partition_location_output = list()
for x in batchify(partition_location, batch_size=BS, option='continuous'):
    partition_location_output.append(g(x)[0])
partition_location_output = np.concatenate(partition_location_output)

ax.scatter(partition_location_output[:, 0], partition_location_output[:, 1],
                partition_location_output[:, 2])


#plt.subplot(241)
#plt.imshow(partition, aspect='auto', origin='lower', extent=(-2, 2, -2, 2))
#plt.title('z space partition')
#
#plt.subplot(242)
#plt.imshow(O2.reshape((NN, NN)), aspect='auto', origin='lower', extent=(-2, 2, -2, 2))
#plt.title('z space A determinant')
#
#plt.subplot(243)
#plt.imshow(p2, aspect='auto', origin='lower', extent=(-2, 2, -2, 2))
#plt.title('x space corresponding partition')
#
#plt.subplot(244)
#plt.plot(G[:, 0], G[:, 1], 'x')
#plt.xlim([-2, 2])
#plt.ylim([-2, 2])
#plt.title('generated points')
#
#plt.subplot(246)
#for i in range(6):
#    plt.plot(H1[i][:,0], H1[i][:,1])
#
#plt.subplot(248)
#for i in range(6):
#    plt.plot(H2[i][:,0], H2[i][:,1])
#

plt.show(block=True)

