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
import theanoxla
from theanoxla.datasets import mnist

parse = argparse.ArgumentParser()
parse.add_argument('--BS', type=int, default=100)
parse.add_argument('--Z', type=int, default=100)
parse.add_argument('--LR', type=float, default=0.0001)
args = parse.parse_args()


def generator(Z, out_dim=28*28, D=32):
    layer = [layers.Dense(Z, 256)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 512))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 1024))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], out_dim))
    return layer

def discriminator(X, deterministic):
    layer = [layers.Dense(X, 1024)]
    layer.append(layers.Dropout(layer[-1], 0.3, deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    layer.append(layers.Dense(layer[-1], 512))
    layer.append(layers.Dropout(layer[-1], 0.3, deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    layer.append(layers.Dense(layer[-1], 256))
    layer.append(layers.Dropout(layer[-1], 0.3, deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    layer.append(layers.Dense(layer[-1], 2))
    return layer



# create dataset
def create_dataset(centroids=None):
    DATA = np.random.randn(args.n_modes * 250, args.D) * args.std
    centroids2 = centroids + 0.
    centroids2[0] += args.radius
    DATA +=  centroids2.repeat(250, 0)
    return DATA * args.scale


# create placeholders and predictions
X = T.Placeholder([args.BS, 28 * 28], 'float32')
Z = T.Placeholder([args.BS, args.Z], 'float32')
deterministic = T.Placeholder([1,], 'int32')

G_sample = generator(Z)
logits = discriminator(T.concatenate([G_sample[-1], X]), deterministic)
labels = T.concatenate([T.zeros(args.BS, dtype='int32'), T.ones(args.BS, dtype='int32')])

# compute the losses
disc_loss = losses.sparse_crossentropy_logits(labels, logits[-1]).mean()
gen_loss = losses.sparse_crossentropy_logits(1 - labels[:args.BS],
                                             logits[-1][:args.BS]).mean()

# create the vq mask
masks = T.concatenate([G_sample[1] > 0, G_sample[3] > 0], 1)

# compute the slope matrix for the poitns and its determinant
A = jacobian_backward(G_sample[-1].sum(0), [Z])[0].transpose([1, 0, 2])
det = T.matmul(A.transpose([0, 2, 1]), A)
#T.sqrt(T.abs(T.det(T.matmul(A.transpose([0, 2, 1]), A))))

# variables
d_variables = sum([l.variables() for l in logits], [])
g_variables = sum([l.variables() for l in G_sample], [])

# updates
opt_d = optimizers.Adam(disc_loss, d_variables, args.LR)
opt_g = optimizers.Adam(gen_loss, g_variables, args.LR)

# functions
train_d = function(Z, X, deterministic, outputs = [disc_loss],
             updates = opt_d.updates)
train_g = function(Z, X, deterministic, outputs = [gen_loss],
             updates = opt_g.updates)

g = function(Z, deterministic, outputs=[G_sample[-1]])
h = function(Z, deterministic, outputs=[det])


train, valid, test = mnist.load()

DATA = train[0].reshape((-1, 28*28))
DATA -= DATA.mean(1, keepdims=True)
DATA /= DATA.max(1, keepdims=True)

# training
loss = list()
for epoch in tqdm(range(300), desc='training'):
    for x in batchify(DATA, batch_size=args.BS, option='random_see_all'):
        z = np.random.randn(args.BS, args.Z)
        loss.append(train_d(z, x, 0))
        z = np.random.randn(args.BS, args.Z)
        loss.append(train_g(z, x, 0))
    if ((epoch + 1) % 40) == 0:
        print('figure')
        x_samples= g(np.random.randn(args.BS, args.Z), 1)[0]
        plt.figure(figsize=(14, 14))
        for i in range(100):
            plt.subplot(10, 10, 1 + i)
            plt.imshow(x_samples[i].reshape((28, 28)), aspect='auto',
                       cmap='Greys')
        plt.savefig('training_{}_{}_{}_epoch{}.jpg'.format(args.Z, args.BS,
                                                         args.LR, epoch))
        plt.close()

        # sample points
        z_samples = np.random.rand(3000, args.Z) * 2 -1
        x_samples = list()
        det_samples = list()
        for x in batchify(z_samples, batch_size=args.BS, option='continuous'):
            x_samples.append(g(x, 1)[0])
            det_samples.append(np.linalg.svd(h(x, 1)[0], compute_uv=False,
                               hermitian=True))
        x_samples = np.concatenate(x_samples)
        det_samples = np.concatenate(det_samples)

        # save into numpy file
        filename = 'mnist_sampling_{}_{}_{}_epoch{}.npz'
        np.savez(filename.format(args.Z, args.BS, args.LR, epoch),
                 loss=np.array(loss), determinant=det_samples,
                 samples=x_samples, z=z_samples)

