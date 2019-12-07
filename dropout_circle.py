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


parse = argparse.ArgumentParser()
parse.add_argument('--dataset', type=str)
parse.add_argument('--WG', type=int)
args = parse.parse_args()



def generator(Z, out_dim, dropout, D=32):
    layer = [layers.Dense(Z, D)]
#    layer.append(layers.Activation(layer[-1], T.relu))
#    layer.append(layers.Dense(layer[-1]*dropout[0], D))
    layer.append(layers.Activation(layer[-1], T.relu))
    layer.append(layers.Dense(layer[-1] * dropout, out_dim))
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
Z = 1
X = 2
WG = args.WG

# create the data
DATA = np.random.randn(1000, 2)
if args.dataset == 'circle':
    DATA /= np.sqrt((DATA**2).sum(1, keepdims=True))
elif args.dataset == 'square':
    DATA /= np.abs(DATA).sum(1, keepdims=True)

# add some noise
DATA += np.random.randn(1000, 2) * 0.01

# create the graph inputs
x = T.Placeholder([BS, X], 'float32')
z = T.Placeholder([BS, Z], 'float32')
dropout = T.Placeholder((WG,), 'float32')

# and now the computational graph
G = generator(z, 2, dropout, WG)
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
PP = [gradients(G[-1][:,0], [z]), gradients(G[-1][:, 1], [z])]
G_A = T.concatenate([PP[0][0], PP[1][0]], 1)

# create the function that will compile the graph
f = function(z, x, dropout, outputs = [D_loss, G_loss], updates = updates)
g = function(z, dropout, outputs=[G[-1]])
get_A = function(z, outputs=[G_A])

# training
print('training')
for epoch in range(3000):
    for x in batchify(DATA, batch_size=BS, option='random_see_all'):
        z = np.random.rand(BS, Z) * 2 -1
        dropout = (np.random.rand(WG) > 0.5).astype('int32')
        f(z, x, dropout)

# sampling final distribution and As
combinations = list(product([0, 1],repeat = 200))
for i in range(len(combinations)):
    p = np.permutation(WG)
    o = np.ones(WG)
    o[p[:200]] = combinations[i]
    combinations[i] = o + 0.


line = np.linspace(-1, 1, 10000).reshape((-1, 1))

GG = list()
for i in tqdm.tqdm(range(2**(WG))):
    dropout = np.array(combinations[i])
    G = list()
    for x in batchify(line, batch_size=BS, option='continuous'):
        G.append(g(x, dropout)[0])
    GG.append(np.concatenate(G))

dropout = np.array(np.ones(WG))
G = list()
for x in batchify(line, batch_size=BS, option='continuous'):
    G.append(g(x, dropout)[0])
HH = np.concatenate(G)


# sampling final distribution
H = list()
dropout = np.ones(WG)
for i in range(100):
    H.append(g(np.random.rand(BS, Z) * 2 - 1, dropout)[0])
H = np.concatenate(H)


for gg in GG:
    plt.plot(gg[:,0], gg[:,1], alpha=0.2)

plt.plot(HH[:, 0], HH[:, 1], 'r')
plt.show(block=True)

