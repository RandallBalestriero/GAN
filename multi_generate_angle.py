import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from multiprocessing import Pool, TimeoutError


mpl.use('Agg')
label_size = 14
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['xtick.labelsize'] = label_size


def one_layer(K, D, x, sigma):

    Z = x.shape[1]
    # create the parameters
    W1 = sigma * np.random.randn(K, Z) / np.sqrt(Z)
    b1 = sigma * np.random.randn(K)
    W2 = sigma * np.random.randn(D, K) / np.sqrt(K)

    # compute the forward pass
    h1 = np.dot(x, W1.T) + b1
    r1 = (h1 > 0).astype('int32') + (h1 < 0).astype('int32') * 0.1
    A = np.einsum('dk,nk,kz->ndz', W2, r1, W1)
    return A, r1

def two_layer(K, D, x, sigma):

    Z = x.shape[1]
    # create the parameters
    W1 = sigma * np.random.randn(K, Z) / np.sqrt(Z)
    b1 = sigma * np.random.randn(K)
    W2 = sigma * np.random.randn(K, K) / np.sqrt(K)
    b2 = sigma * np.random.randn(K)
    W3 = sigma * np.random.randn(D, K) / np.sqrt(K)


    # compute the forward pass

    # layer 1
    h1 = np.dot(x, W1.T) + b1
    r1 = (h1 > 0).astype('int32') + (h1 < 0).astype('int32') * 0.1

    # layer 2
    h2 = np.dot(h1 * r1, W2.T) + b2
    r2 = (h2 > 0).astype('int32') + (h2 < 0).astype('int32') * 0.1

    # compute the matrix A
    A = np.einsum('dk,nk,kp,np,pz->ndz', W3, r2, W2, r1, W1)
    return A, np.concatenate([r1, r2], 1)


def single_run(args):
    K, D, x, sigma, layer = args
    # retreive the slope and q code
    if layer == 1:
        A, q = one_layer(K, D, x, sigma)
    elif layer == 2:
        A, q = two_layer(K, D, x, sigma)

    # check which points represent a change of region
    valid = np.nonzero(np.not_equal(q[::2], q[1::2]).astype('int32').sum(1) == 1)[0]
    valid = np.concatenate([valid * 2, valid * 2 + 1])

    # we only need to keep those indices
    A = A[valid]

    # compute the inverse
    inv = np.linalg.inv(np.einsum('ndz,ndh->nzh', A, A))
    PA = np.einsum('nab,nbc,ndc->nad',A, inv, A)
    diff = PA[::2]-PA[1::2]
    angles = np.linalg.norm(diff, 2, (1, 2))
    return np.arcsin(angles) * 90 * 2/3.14159


Ks = [8 ,16, 32, 64, 128, 256, 512, 1024]
sigma = 1
pool = Pool(processes=32)

for Z in [2, 4, 8, 16]:
    x0 = np.random.randn(Z) - 4
    x1 = np.random.randn(Z) + 4
    x = np.linspace(0, 1, 400).reshape((-1, 1))
    x = x0 * x + (1 - x) * x1
    for L in [1, 2]:
        for D in [Z+1, Z*2, Z*4]:
            Kslocal = [k for k in Ks if k>Z]
            for c, (k, K) in zip(np.linspace(0.3, 0.8, len(Kslocal)), enumerate(Kslocal)):
                args = zip([K] * 300, [D] * 300, [x] * 300, [sigma] * 300,
                           [L] * 300)
                angles = pool.map_async(single_run, args)
                angles = np.concatenate(angles.get())
                print(Z, L, D, K, angles.shape)
                np.savez('anglesv2_{}_{}_{}_{}.npz'.format(L, Z, D, K),
                         angles=angles)


