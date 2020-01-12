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
from scipy.special import gamma


parse = argparse.ArgumentParser()
parse.add_argument('--radius', type=float, default=1)
parse.add_argument('--std', type=float, default=1)
parse.add_argument('--D', type=int, default = 2)
parse.add_argument('--WG', type=int, default=64)
parse.add_argument('--n_modes', type=int, default=2)

args = parse.parse_args()
Zdim = 2


def generator(Z, out_dim, D=32):
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
lr = 0.00001

# create dataset
def create_dataset(centroids=None):
#    centroids = np.random.randn(args.n_modes, args.D)
#    centroids[1] = -centroids[0]
#    centroids /= np.sqrt((centroids**2).sum(1, keepdims=True))
#    centroids *= args.radius

    DATA = np.random.randn(args.n_modes * 250, args.D) * args.std
#    projector = np.random.randn(args.n_modes, Zdim, args.D)
#    inv = np.linalg.inv(np.einsum('mkD,mKD->mKk', projector, projector))
#    projector = np.einsum('mkD,mKk->mKD',projector, inv)
#    projector = projector.repeat(250, 0)
#
    DATA +=  centroids.repeat(250, 0)
    #np.einsum('NKD,NK->ND', projector, DATA) + centroids.repeat(250, 0)
    return DATA


# create placeholders and predictions
X = T.Placeholder([BS, args.D], 'float32')
Z = T.Placeholder([BS, Zdim], 'float32')
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
A = T.stack([gradients(G_sample[-1][:,i].sum(), [Z])[0]
             for i in range(args.D)], 1)
print(A.shape)
det = T.sqrt(T.abs(T.det(T.matmul(A.transpose([0, 2, 1]), A))))

# variables
d_variables = sum([l.variables() for l in logits], [])
g_variables = sum([l.variables() for l in G_sample], [])

# updates
updates_d, varsd = optimizers.Adam(disc_loss, d_variables, lr)
updates_g, varsg = optimizers.Adam(gen_loss, g_variables, lr)
updates = {**updates_d, **updates_g}

# functions
f = function(Z, X, outputs = [disc_loss, gen_loss],
             updates = updates)
g = function(Z, outputs=[G_sample[-1]])
h = function(Z, outputs=[det])

# loop over multiple runs
np.random.seed(10)
for run in range(10):

    # reset variables
    for var in varsd+varsg+d_variables+g_variables:
        var.reset()
    DATA = create_dataset(np.random.randn(args.n_modes, args.D))

    # training
    for epoch in tqdm(range(16000), desc='training'):
        for x in batchify(DATA, batch_size=BS, option='random_see_all'):
            z = np.random.rand(BS, Zdim) * 2 -1
            f(z, x)

    # sample points
    z_samples = np.random.rand(50000, Zdim) * 2 -1
    x_samples = list()
    det_samples = list()
    for x in batchify(z_samples, batch_size=BS, option='continuous'):
        x_samples.append(g(x)[0])
        det_samples.append(h(x)[0])
    x_samples = np.concatenate(x_samples)
    det_samples = np.concatenate(det_samples)

    # save into numpy file
    filename = 'determinant_comparison_{}_{}_{}_{}_{}_{}.npz'
    np.savez(filename.format(args.radius, args.std, args.D, args.WG, args.n_modes, run),
             determinant=det_samples, samples=x_samples, data=DATA)
