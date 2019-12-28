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
from theanoxla import layers, losses, optimizers, function, gradients
from theanoxla.utils import batchify
import matplotlib.pyplot as plt
from jax.lib import xla_client
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib
import argparse
from itertools import product
import tqdm

cmap = matplotlib.cm.get_cmap('hsv')
np.random.seed(114)

parse = argparse.ArgumentParser()
parse.add_argument('--occlusion', type=int, default=1)
parse.add_argument('--WG', type=int, default=64)
parse.add_argument('--L', type=int, default=1)
args = parse.parse_args()



def generator(Z, out_dim, dropout, D=32, L=1):
    layer = [layers.Dense(Z, D)]
    for l in range(L):
        layer.append(layers.Activation(layer[-1], T.leaky_relu))
        layer.append(layers.Dense(layer[-1], D * 2))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1] * dropout, out_dim))
    return layer

def discriminator(X, D=64):
    layer = [layers.Dense(X, D)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], D))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 2))
    return layer

# some hyper parameters
BS = 300
lr = 0.0002
Z = 1
X = 2
WG = args.WG
L = args.L

# create the data
DATA = np.random.randn(1400, 2)
DATA /= np.abs(DATA).max(1, keepdims=True)
if args.occlusion:
    DATA = DATA[np.sqrt((DATA**2).sum(1))<1.32]

# add some noise
DATA += np.random.randn(DATA.shape[0], 2) * 0.01

# create the graph inputs
x = T.Placeholder([BS, X], 'float32')
z = T.Placeholder([BS, Z], 'float32')
dropout = T.Placeholder((2 * WG,), 'float32')

# and now the computational graph
G = generator(z, 2, dropout, WG, L)
D = discriminator(T.concatenate([G[-1], x]))

labels = T.concatenate([T.zeros(BS, dtype='int32'), T.ones(BS, dtype='int32')])

# the loss for G and D
D_loss = losses.sparse_crossentropy_logits(labels, D[-1]).mean()
G_loss = losses.sparse_crossentropy_logits(1 - labels[:BS], D[-1][:BS]).mean()

# gather their variables
D_vars = sum([l.variables() for l in D], [])
G_vars = sum([l.variables() for l in G], [])

# optimizers generating the updates
D_ups, dvars = optimizers.Adam(D_loss, D_vars, lr)
G_ups, gvars = optimizers.Adam(G_loss, G_vars, lr)
updates = {**D_ups, **G_ups}


# create the function that will compile the graph
f = function(z, x, dropout, outputs = [D_loss, G_loss], updates = updates)
g = function(z, dropout, outputs=[G[-1]])

allG = list()
for T in range(5):
    for var in D_vars:
        var.reset()
    for var in G_vars:
        var.reset()
    for var in dvars:
        var.reset()
    for var in gvars:
        var.reset()


    # training
    print('training')
    for epoch in range(12000):
        for x in batchify(DATA, batch_size=BS, option='random_see_all'):
            z = np.random.rand(BS, Z) * 2 -1
            dropout = np.ones(WG * 2)
            f(z, x, dropout)

    # sampling final distribution and As
    line = np.linspace(-1, 1, 10000).reshape((-1, 1))
    dropout = np.array(np.ones(WG * 2))
    G = list()
    for x in batchify(line, batch_size=BS, option='continuous'):
        G.append(g(x, dropout)[0])
    allG.append(np.concatenate(G))


fig = plt.figure(figsize=(4,4))
plt.plot(DATA[:, 0], DATA[:, 1], 'xb', alpha=0.5)
for G in allG:
    plt.plot(G[:, 0], G[:, 1], color=cmap(1), lw=3)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
ax = plt.gca()
ax.axis('off')
fig.patch.set_visible(False)
plt.show(block=True)
plt.savefig('interpolation_{}_{}_{}.jpg'.format(args.occlusion, WG, L))
plt.close()
