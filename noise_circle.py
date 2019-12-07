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


parse = argparse.ArgumentParser()
parse.add_argument('--dataset', type=str)
parse.add_argument('--WG', type=int)
args = parse.parse_args()


def get_angles(points, get_A):
    A = list()
    for x in batchify(points, batch_size=BS, option='continuous'):
        A.append(get_A(x)[0])
    A = np.concatenate(A)
    angles = np.abs((A[:-1]*A[1:]).sum(1))
    angles /= np.sqrt((A[:-1]**2).sum(1) * (A[1:]**2).sum(1))
    angles = np.arccos(np.clip(angles, 0, 1))
    return angles



def generator(Z, out_dim, D=32):
    layer = [layers.Dense(Z, D)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
#    layer.append(layers.Dense(layer[-1], D))
#    layer.append(layers.Activation(layer[-1], T.leaky_relu))
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
DATA += np.random.randn(1000, 2) * 0.1

# create the graph inputs
x = T.Placeholder([BS, X], 'float32')
z = T.Placeholder([BS, Z], 'float32')

# and now the computational graph
G = generator(z, 2, WG)
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
f = function(z, x, outputs = [D_loss, G_loss], updates = updates)
g = function(z, outputs=[G[-1]])
get_A = function(z, outputs=[G_A])

# training
print('training')
for epoch in range(3000):
    for x in batchify(DATA, batch_size=BS, option='random_see_all'):
        z = np.random.rand(BS, Z) * 2 -1
        f(z, x)

# sampling final distribution and As
G = list()
print('training')
line = np.linspace(-1, 1, 10000).reshape((-1, 1))
for x in batchify(line, batch_size=BS, option='continuous'):
    G.append(g(x)[0])
G = np.concatenate(G)


angles = get_angles(line, get_A)
print(angles.min(), angles.max())


# sampling final distribution
H = list()
line = np.linspace(-1, 1, 10000)
for i in range(100):
    H.append(g(np.random.rand(BS, Z) * 2 - 1)[0])
H = np.concatenate(H)


#plt.plot(DATA[:,0], DATA[:,1], 'bx')
#plt.plot(H[:,0], H[:,1], 'rx')
#plt.plot(G[:,0], G[:,1], '-k')
plt.subplot(121)
cmap = matplotlib.cm.get_cmap('jet')
ANGLES = angles/angles.max()
colors = cmap(ANGLES)
colors[:, -1] = (ANGLES > 1e-4).astype('float32')
plt.scatter(G[1:,0], G[1:, 1], s=(ANGLES**2).reshape((-1,1))*40, c=colors)

plt.subplot(122)
plt.hist(np.log(angles[angles > 1e-8]), 200)
plt.show(block=True)

