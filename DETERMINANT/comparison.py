import sys
sys.path.insert(0, "../../TheanoXLA")

import numpy as np
import theanoxla.tensor as T
from theanoxla import layers, losses, optimizers, function, gradients
from theanoxla.utils import batchify, vq_to_boundary
from sklearn import datasets
import matplotlib.pyplot as plt
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--radius', type=float, default=1)
parse.add_argument('--std', type=float, default=1)
parse.add_argument('--D', type=int, default = 2)
parse.add_argument('--WG', type=int, default=64)
parse.add_argument('--n_modes', type=int, default=4)
args = parse.parse_args()



def generator(Z, out_dim, D=64):
    layer = [layers.Dense(Z, D)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], D))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], out_dim))
    return layer

def discriminator(X):
    layer = [layers.Dense(X, 64)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 64))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 2))
    return layer


BS = 100
lr = 0.00005

# create dataset
centroids = np.random.randn(args.n_modes, 2)
centroids /= np.sqrt((centroids**2).sum(1, keepdims=True))
DATA = np.random.randn(args.n_modes * 150, 2) * args.std
DATA += centroids.repeat(0, 150)

# create placeholders and predictions
X = T.Placeholder([BS, args.D], 'float32')
Z = T.Placeholder([BS, args.D], 'float32')
G_sample = generator(Z, args.D, args.WG)
logits = discriminator(T.concatenate([G_sample[-1], X]))
labels = T.concatenate([T.zeros(BS, dtype='int32'), T.ones(BS, dtype='int32')])

# compute the losses
disc_loss = losses.sparse_crossentropy_logits(labels, logits[-1]).mean()
gen_loss = losses.sparse_crossentropy_logits(1 - labels[:BS],
                                             logits[-1][:BS]).mean()

# create the vq mask
masks = T.concatenate([G_sample[1] > 0, G_sample[3] > 0], 1)

# compute the slope matrix for the poitns and its determinant
A = T.stack([gradients(G_sample[-1][:,0].sum(), [Z])[0],
             gradients(G_sample[-1][:,1].sum(), [Z])[0]], 1)
det = T.abs(T.det(A))

# variables
d_variables = sum([l.variables() for l in logits], [])
g_variables = sum([l.variables() for l in G_sample], [])

# updates
updates_d, _ = optimizers.Adam(disc_loss, d_variables, lr)
updates_g, _ = optimizers.Adam(gen_loss, g_variables, lr)
updates = {**updates_d, **updates_g}

# functions
f = function(Z, X, outputs = [disc_loss, gen_loss],
             updates = updates)
g = function(Z, outputs=[G_sample[-1]])
h = function(Z, outputs=[det])

# do training
for epoch in range(12000):
    for x in batchify(DATA, batch_size=BS, option='random_see_all'):
        z = np.random.rand(BS, args.D) * 2 -1
        f(z, x)

# get determinant
XX = np.random.rand(50000, args.D) * 2 -1
O = list()
for x in batchify(XX, batch_size=BS, option='continuous'):
    O.append(h(x)[0])
O = np.concatenate(O)

