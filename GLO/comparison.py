import sys
sys.path.insert(0, "../../TheanoXLA")

import numpy as np
import theanoxla.tensor as T
from theanoxla import layers, losses, optimizers, function, gradients, jacobians
from theanoxla.utils import batchify, vq_to_boundary
import theanoxla
from sklearn import datasets
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy.special import gamma


parse = argparse.ArgumentParser()
parse.add_argument('--D', type=int, default=2)
parse.add_argument('--N', type=int, default=1000)
parse.add_argument('--W', type=int, default=16)
parse.add_argument('--Z', type=int, default=2)
parse.add_argument('--Zstar', type=int, default=2)
parse.add_argument('--bs', type=int, default=100)
parse.add_argument('--lr', type=float, default=0.005)

args = parse.parse_args()

def encoder(X, out_dim, D=256):
    layer = [layers.Dense(X, D)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], D))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], out_dim))
    return layer



def generator(Z, out_dim, D=16):
    layer = [layers.Dense(Z, D)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], out_dim))
    return layer


# create dataset
def create_dataset(N, D, Z):
    x = np.random.rand(N, D) * 10
    W = np.random.randn(Z, D)
    P = np.dot(W.T, np.dot(np.linalg.inv(np.dot(W, W.T)), W))
    xp = np.dot(P, x.T).T
    return xp


# create placeholders and predictions
X = T.Placeholder([args.bs, args.D], 'float32')
E = encoder(X, args.Z)
G_sample = generator(E[-1], args.D, args.W)

# compute the losses
loss = T.sqrt(((X - G_sample[-1])**2).sum(1)).mean()
# variables
variables = sum([l.variables() for l in G_sample+E], [])

# updates
EPOCHS = int(1000 * 2000 / args.N)
lrs = theanoxla.schedules.PiecewiseConstant(args.lr, {EPOCHS // 3: args.lr/3, int(2 * EPOCHS / 3):args.lr/6, int(4*EPOCHS / 5): args.lr/10})
opt = optimizers.Adam(loss, variables, lrs)

# functions
train_nn = function(X, outputs = loss, updates = opt.updates)
getloss = function(X, outputs = loss)


# loop over multiple runs
np.random.seed(10)
DATA = create_dataset(args.N, args.D, args.Zstar)
L = list()
for run in range(30):

    # reset variables
    for var in variables + opt.variables:
        var.reset()
    lrs.reset()

    # training
    for epoch in tqdm(range(EPOCHS), desc='training'):
        loss = list()
        for x in batchify(DATA, batch_size=args.bs, option='random_see_all'):
            train_nn(x)
        for x in batchify(DATA, batch_size=args.bs, option='continuous'):
            loss.append(getloss(x))
        lrs.update()
        L.append(np.mean(loss))

    # save into numpy file

L = np.array(L).reshape((30, EPOCHS))
filename = 'glo_{}_{}_{}_{}_{}.npz'
np.savez(filename.format(args.N, args.W, args.D, args.Z, args.Zstar), loss=L)
