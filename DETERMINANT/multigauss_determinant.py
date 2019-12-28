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
parse.add_argument('--dataset', type=int, default=0)
parse.add_argument('--WG', type=int, default=64)
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
if args.dataset == 0:
    DATA = np.random.randn(1000, 2)
    DATA *= 0.06
    mx, my = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    DATA += np.stack([mx.flatten(), my.flatten()], 1).repeat(1000 // 25, 0)
elif args.dataset == 1:
    DATA = np.random.rand(1000, 2) * 2 - 1
    gauss = np.random.randn(500, 2) * 0.1 + np.array([0.5, 0.5])
    DATA = np.concatenate([DATA, gauss])
elif args.dataset == 2:
    DATA = np.random.rand(1000, 2) * 2 - 1


# create placeholders and predictions
X = T.Placeholder([BS, 2], 'float32')
Z = T.Placeholder([BS, 2], 'float32')
G_sample = generator(Z, 2, args.WG)
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
h = function(Z, outputs=[masks, det])

# do training
for epoch in range(12000):
    for x in batchify(DATA, batch_size=BS, option='random_see_all'):
        z = np.random.rand(BS, 2) * 2 -1
        f(z, x)

# samples from the generator
G = list()
for i in range(10):
        z = np.random.rand(BS, 2) * 2 -1
        G.append(g(z)[0])
G = np.concatenate(G)

#
NN = 400
MIN, MAX = -1, 1
xx, yy = np.meshgrid(np.linspace(MIN, MAX, NN), np.linspace(MIN, MAX, NN))
XX = np.stack([xx.flatten(), yy.flatten()], 1)
O = list()
O2 = list()
for x in batchify(XX, batch_size=BS, option='continuous'):
    a, b = h(x)
    O.append(a)
    O2.append(b)
O = np.concatenate(O)
O2 = np.log(np.concatenate(O2))
print(O2)
partition = vq_to_boundary(O, NN, NN)
partition_location = XX[partition.reshape((-1,)) > 0]

# 
F = list()
for x in batchify(partition_location, batch_size=BS, option='continuous'):
    F.append(g(x)[0])
F = np.concatenate(F)

p2 = np.zeros((NN*NN,))
for i in range(len(F)):
    distances = np.abs(XX - F[i]).max(1)
    istar = distances.argmin()
    if distances[istar] <= 6 / NN:
        p2[istar] = 1

#
H1 = list()
H2 = list()
for i in range(7):
    time = np.linspace(-1, 1, 200)
    H1.append(np.stack([time, time*(np.random.rand()*4-2)], 1))
    H2.append([])
    for x in batchify(H1[-1], batch_size=BS, option='continuous'):
        H2[-1].append(g(x)[0])
    H2[-1] = np.concatenate(H2[-1])


p2 = p2.reshape((NN, NN))

############ GET PROBA
proba = np.exp(O2)
high_samples = np.random.choice(range(len(XX)), size=1000, p=proba/ proba.sum())
print(high_samples)

high_samples = XX[high_samples]
high_samples_out = list()
for x in batchify(high_samples, batch_size=BS, option='continuous'):
    high_samples_out.append(g(x)[0])
high_samples_out = np.concatenate(high_samples_out)

low_samples = np.random.choice(range(len(XX)), size=1000, p=(proba.max()-proba)/(proba.max()-proba).sum())
low_samples = XX[low_samples]
low_samples_out = list()
for x in batchify(low_samples, batch_size=BS, option='continuous'):
    low_samples_out.append(g(x)[0])
low_samples_out = np.concatenate(low_samples_out)

# plots
plt.figure(figsize=(4, 4))
plt.imshow(O2.reshape((NN, NN)), aspect='auto', origin='lower',
           extent=(MIN, MAX, MIN, MAX))
plt.colorbar()
plt.savefig('zspace_logdet_{}_{}.jpg'.format(args.dataset, args.WG))
plt.close()

plt.figure(figsize=(4, 4))
plt.imshow(np.exp(O2).reshape((NN, NN)), aspect='auto', origin='lower',
           extent=(MIN, MAX, MIN, MAX))
plt.colorbar()
plt.savefig('zspace_det_{}_{}.jpg'.format(args.dataset, args.WG))
plt.close()

plt.figure(figsize=(4, 4))
plt.plot(high_samples_out[:, 0], high_samples_out[:, 1], 'rx')
plt.plot(low_samples_out[:, 0], low_samples_out[:, 1], 'gx')
plt.savefig('bisamples_{}_{}.jpg'.format(args.dataset, args.WG))
plt.close()

plt.figure(figsize=(4, 4))
plt.plot(DATA[:, 0], DATA[:, 1], 'x')
plt.plot(G[:, 0], G[:, 1], 'x')
plt.savefig('samples_{}_{}.jpg'.format(args.dataset, args.WG))
plt.close()

