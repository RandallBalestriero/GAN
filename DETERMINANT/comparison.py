import sys
sys.path.insert(0, "../../TheanoXLA")

import numpy as np
import theanoxla.tensor as T
from theanoxla import layers, losses, optimizers, function, gradients, jacobian_forward, jacobian_backward
from theanoxla.utils import batchify, vq_to_boundary
from sklearn import datasets
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy.special import gamma


parse = argparse.ArgumentParser()
parse.add_argument('--scale', type=float, default=1)
parse.add_argument('--radius', type=float, default=0)
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
    layer = [layers.Dense(X, 16)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 16))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 2))
    return layer


BS = 50
lr = 0.000001

# create dataset
def create_dataset(centroids):
    noise = np.random.randn(args.n_modes * 250, args.D) * args.std
    centroids2 = centroids + 0.
    centroids2[0] += args.radius
    return (noise + centroids2.repeat(250, 0)) * args.scale


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
A = jacobian_backward(G_sample[-1].sum(0), [Z])[0].transpose([1, 0, 2])
det = T.sqrt(T.abs(T.det(T.matmul(A.transpose([0, 2, 1]), A))))

# variables
d_variables = sum([l.variables() for l in logits], [])
g_variables = sum([l.variables() for l in G_sample], [])

# updates
opt_d = optimizers.Adam(disc_loss, d_variables, lr)
opt_g = optimizers.Adam(gen_loss, g_variables, lr)

# functions
train_d = function(Z, X, outputs = [disc_loss],
             updates = opt_d.updates)
train_g = function(Z, X, outputs = [gen_loss],
             updates = opt_g.updates)

g = function(Z, outputs=[G_sample[-1]])
h = function(Z, outputs=[det])

# loop over multiple runs
np.random.seed(10)
for run in range(10):

    # reset variables
    for var in d_variables+g_variables+opt_d.variables+opt_g.variables:
        var.reset()
    DATA = create_dataset(np.random.randn(args.n_modes, args.D))

    loss = list()
    # training
    for epoch in tqdm(range(16000), desc='training'):
        for x in batchify(DATA, batch_size=BS, option='random_see_all'):
            z = np.random.randn(BS, Zdim)
            loss.append(train_d(z, x))
            z = np.random.randn(BS, Zdim)
            loss.append(train_g(z, x))
    loss = np.array(loss)

    # sample points
    z_samples = np.random.randn(50000, Zdim)
    x_samples = list()
    det_samples = list()
    for x in batchify(z_samples, batch_size=BS, option='continuous'):
        x_samples.append(g(x)[0])
        det_samples.append(h(x)[0])
    x_samples = np.concatenate(x_samples)
    det_samples = np.concatenate(det_samples)

    # save into numpy file
    filename = 'determinant_comparison_{}_{}_{}_{}_{}_{}_{}.npz'
    np.savez(filename.format(args.radius, args.scale, args.std, args.D,
                             args.WG, args.n_modes, run),
             determinant=det_samples, samples=x_samples, data=DATA, loss=loss)
