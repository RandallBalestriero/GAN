import sys
sys.path.insert(0, "../../TheanoXLA")

import numpy as np
import theanoxla.tensor as T
from theanoxla import layers, losses, optimizers, function, gradients, jacobians
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
parse.add_argument('--MODEL', type=str, default='GAN')
parse.add_argument('--RUN', type=int, default=0)
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

def convgenerator(Z, out_dim=28*28, D=32):
    layer = [layers.Dense(Z, 256)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 8 * 6 * 6))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Reshape(layer[-1], (-1, 8, 6, 6)))
    layer.append(layers.Conv2D(layer[-1], W_shape=(8, 8, 3, 3), pad='SAME',
                               input_dilations=(2, 2)))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Conv2D(layer[-1], W_shape=(1, 8, 4, 4), pad='VALID',
                               input_dilations=(3, 3)))
    for l in layer:
        print(l.shape)
    layer.append(layers.Reshape(layer[-1], (-1, 28*28)))
    return layer


def encoder(X, out_dim, deterministic):
    layer = [layers.Dense(X, 512)]
    layer.append(layers.Dropout(layer[-1], 0.3, deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    layer.append(layers.Dense(layer[-1], 256))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    layer.append(layers.Dense(layer[-1], out_dim * 2))
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


def Q(X, deterministic, out_dim):
    layer = [layers.Dense(X, 512)]
    layer.append(layers.Dropout(layer[-1], 0.3, deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    layer.append(layers.Dense(layer[-1], 256))
    layer.append(layers.Dropout(layer[-1], 0.3, deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    layer.append(layers.Dense(layer[-1], out_dim * 2))
    return layer


def reparametrize(mu, logvar):
    std = T.exp(0.5 * logvar)
    eps = T.random.randn(std.shape)
    return eps * std + mu

# create dataset


def create_dataset(centroids=None):
    DATA = np.random.randn(args.n_modes * 250, args.D) * args.std
    centroids2 = centroids + 0.
    centroids2[0] += args.radius
    DATA += centroids2.repeat(250, 0)
    return DATA * args.scale


# create placeholders and predictions
X = T.Placeholder([args.BS, 28 * 28], 'float32')
Z = T.Placeholder([args.BS, args.Z], 'float32')
deterministic = T.Placeholder([1, ], 'int32')


if args.MODEL == 'GAN' or args.MODEL == 'CONVGAN' :
    if 'CONV' in args.MODEL:
        G_sample = convgenerator(Z)
    else:
        G_sample = generator(Z)

    logits = discriminator(T.concatenate([G_sample[-1], X]), deterministic)
    labels = T.concatenate(
        [T.zeros(args.BS, dtype='int32'), T.ones(args.BS, dtype='int32')])

    # compute the losses
    disc_loss = losses.sparse_crossentropy_logits(labels, logits[-1]).mean()
    gen_loss = losses.sparse_crossentropy_logits(1 - labels[:args.BS],
                                                 logits[-1][:args.BS]).mean()

    # compute the slope matrix for the poitns and its determinant
    A = jacobians(G_sample[-1].sum(0), [Z], 'backward')[0].transpose([1, 0, 2])

    # variables
    d_variables = sum([l.variables() for l in logits], [])
    g_variables = sum([l.variables() for l in G_sample], [])

    # updates
    opt_d = optimizers.Adam(disc_loss, d_variables, args.LR)
    opt_g = optimizers.Adam(gen_loss, g_variables, args.LR)

    # functions
    train_d = function(Z, X, deterministic, outputs=[disc_loss],
                       updates=opt_d.updates)
    train_g = function(Z, X, deterministic, outputs=[gen_loss],
                       updates=opt_g.updates)

    def trainit(x):
        z = np.random.rand(args.BS, args.Z)*2-1
        a = train_d(z, x, 0)
        z = np.random.rand(args.BS, args.Z)*2-1
        b = train_g(z, x, 0)
        return a, b

    sample = function(Z, deterministic, outputs=G_sample[-1])
    #h = function(Z, deterministic, outputs=det)
    get_A = function(Z, deterministic, outputs=A)

elif args.MODEL == 'INFOGAN':
    G_sample = generator(Z)
    logits = discriminator(T.concatenate([G_sample[-1], X]), deterministic)
    labels = T.concatenate(
        [T.zeros(args.BS, dtype='int32'), T.ones(args.BS, dtype='int32')])

    q = Q(G_sample[-1], deterministic, Z.shape[1])

    # compute the losses
    disc_loss = losses.sparse_crossentropy_logits(labels, logits[-1]).mean()
    gen_loss = losses.sparse_crossentropy_logits(1 - labels[:args.BS],
                                                 logits[-1][:args.BS]).mean()

    Q_C_mean = q[-1][:, :Z.shape[1]]
    Q_C_logstd = q[-1][:, Z.shape[1]:]
    rec = (Z - Q_C_mean) / (T.exp(Q_C_logstd) + 1e-8)
    q_loss = Q_C_logstd.mean() + 0.5 * T.square(rec).mean()

    # compute the slope matrix for the poitns and its determinant
    A = jacobian_backward(G_sample[-1].sum(0), [Z])[0].transpose([1, 0, 2])
    #det = T.matmul(A.transpose([0, 2, 1]), A)
    #T.sqrt(T.abs(T.det(T.matmul(A.transpose([0, 2, 1]), A))))

    # variables
    d_variables = sum([l.variables() for l in logits], [])
    g_variables = sum([l.variables() for l in G_sample], [])
    q_variables = sum([l.variables() for l in q], [])

    # updates
    opt_d = optimizers.Adam(disc_loss, d_variables, args.LR)
    opt_g = optimizers.Adam(gen_loss-0.05*q_loss, g_variables, args.LR)
    opt_q = optimizers.Adam(q_loss, q_variables, args.LR)

    # functions
    train_d = function(Z, X, deterministic, outputs=[disc_loss],
                       updates=opt_d.updates)
    train_g = function(Z, X, deterministic, outputs=[gen_loss],
                       updates=opt_g.updates)
    train_q = function(Z, deterministic, outputs=[q_loss],
                       updates=opt_q.updates)

    def trainit(x):
        z = np.random.rand(args.BS, args.Z)*2-1
        a = train_d(z, x, 0)
        z = np.random.rand(args.BS, args.Z)*2-1
        b = train_g(z, x, 0)
        z = np.random.rand(args.BS, args.Z)*2-1
        c = train_q(z, 0)
        return a, b, c

    sample = function(Z, deterministic, outputs=G_sample[-1])
    #h = function(Z, deterministic, outputs=det)
    get_A = function(Z, deterministic, outputs=A)


elif 'VAE' == args.MODEL or 'CONVVAE' == args.MODEL:

    E = encoder(X, args.Z, deterministic)
    mu = E[-1][:, :args.Z]
    logvar = E[-1][:, args.Z:]
    if 'CONV' in args.MODEL:
        D = convgenerator(mu + T.exp(logvar/2)*T.random.normal(mu.shape))
        for l in D:
            print(l.shape)
    else:
        D = generator(mu + T.exp(logvar/2)*T.random.normal(mu.shape))
    G_sample = layers.forward(Z, D)

    divergence = (1 + logvar - T.exp(logvar) - mu**2).sum(1).mean()
    rec = ((X - D[-1])**2).sum(1).mean()
    loss = rec - 0.5 * divergence

    A = jacobians(G_sample[-1].sum(0), [Z], 'backward')[0].transpose([1, 0, 2])
    # variables
    variables = sum([l.variables() for l in E+D], [])
    # updates
    opt_d = optimizers.Adam(loss, variables, args.LR)
    # functions
    train_ = function(X, deterministic, outputs=[loss],
                      updates=opt_d.updates)

    def trainit(x):
        return train_(x, 0)

    sample = function(Z, deterministic, outputs=G_sample[-1])
    get_A = function(Z, deterministic, outputs=A)


elif 'BETA' in args.MODEL:

    E = encoder(X, args.Z, deterministic)
    mu = E[-1][:, :args.Z]
    logvar = E[-1][:, args.Z:]
    z = reparametrize(mu, logvar)
    if 'CONV' in args.MODEL:
        D = convgenerator(mu + T.exp(logvar/2)*T.random.normal(mu.shape))
        for l in D:
            print(l.shape)
    else:
        D = generator(mu + T.exp(logvar/2)*T.random.normal(mu.shape))
    G_sample = layers.forward(Z, D)

    kl_loss = (1 + logvar - T.exp(logvar) - mu**2).sum(1).mean()
    rec_loss = ((X - D[-1])**2).sum(1).mean()

    C = discriminator(T.concatenate([z, T.random.shuffle(z, 0)], 0), deterministic)
    beta_loss = (C[-1][:z.shape[0], 0] - C[-1][:z.shape[0], 1]).mean()
    vae_loss = rec_loss - 0.5 * kl_loss + beta_loss
    D_target = T.concatenate([T.zeros(z.shape[0]), T.ones(z.shape[0])]).astype('int32')
    D_loss = losses.sparse_crossentropy_logits(D_target, C[-1]).mean()

    A = jacobians(G_sample[-1].sum(0), [Z], 'backward')[0].transpose([1, 0, 2])
    # variables
    variables_ae = sum([l.variables() for l in E+D], [])
    variables_d = sum([l.variables() for l in C], [])

    # updates
    opt_vae = optimizers.Adam(vae_loss, variables_ae, args.LR)
    opt_d = optimizers.Adam(D_loss, variables_d, args.LR)

    # functions
    train_ae = function(X, deterministic, outputs=vae_loss,
                      updates=opt_vae.updates)
    train_d = function(X, deterministic, outputs=D_loss,
                      updates=opt_d.updates)


    def trainit(x):
        a = train_ae(x, 0)
        b = train_d(x, 0)
        return a, b

    sample = function(Z, deterministic, outputs=G_sample[-1])
    get_A = function(Z, deterministic, outputs=A)


train, valid, test = mnist.load()

DATA = train[0].reshape((-1, 28*28))
DATA -= DATA.mean(1, keepdims=True)
DATA /= DATA.max(1, keepdims=True)

# training
loss = list()
dets = list()
seeds = np.random.rand(10000, args.Z) * 2 - 1
samples = list()
As = list()
bs = list()
for epoch in tqdm(range(300), desc='training'):
    if (epoch % 30) == 0:
        print('figure')
        x_samples = sample(np.random.rand(args.BS, args.Z)*2-1, 1)
        print(x_samples.shape)
        plt.figure(figsize=(14, 14))
        for i in range(100):
            plt.subplot(10, 10, 1 + i)
            plt.imshow(x_samples[i].reshape((28, 28)), aspect='auto',
                       cmap='Greys')
        plt.savefig('training_{}_{}_{}_{}_run{}_epoch{}.jpg'.format(args.Z, args.BS,
                                                                    args.LR, args.MODEL, args.RUN, epoch))
        plt.close()

        # sample points
    if epoch == 0 or epoch == 299:
        x_samples = list()
        A = list()
        cpt = 0
        for x in batchify(seeds, batch_size=args.BS, option='continuous'):
            if cpt < 10:
                x_samples.append(sample(x, 1))
                cpt += 1
            A.append(get_A(x, 1))

        samples.append(np.concatenate(x_samples))
        As.append(np.concatenate(A))

    for x in batchify(DATA, batch_size=args.BS, option='random_see_all'):
        trainit(x)


# save into numpy file
filename = 'mnist_sampling_{}_{}_{}_{}_{}.npz'
np.savez(filename.format(args.Z, args.BS, args.LR, args.MODEL, args.RUN),
         loss=np.array(loss),  # determinant=dets,
         samples=samples, z=seeds, A=As)
