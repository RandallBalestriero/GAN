import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
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


def single_run(K, D, x, sigma, layer=1):

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
    return np.arcsin(angles) * 180 * 2/3.14159


Ks = [8 ,16, 32, 64, 128, 256, 512]
sigma = 1
INCREASE = 0.005
cmap = cm.get_cmap('Reds')

for L in [2]:
    for Z in [2, 4, 8 ,16]:
        x0 = np.random.randn(Z) - 4
        x1 = np.random.randn(Z) + 4
        x = np.linspace(0, 1, 200).reshape((-1, 1))
        x = x0 * x + (1 - x) * x1
        for D in [Z+1, Z*2, Z*4, Z*8]:
            fig = plt.figure(figsize=(4,5))
            Kslocal = [k for k in Ks if k>Z]
            for c, (k, K) in zip(np.linspace(0.3, 0.8, len(Kslocal)), enumerate(Kslocal)):
                angles = list()
                for i in range(300):
                        angles.append(single_run(K, D, x, sigma, L))
                angles = np.concatenate(angles)
                print(angles.max())
                y_, x_ = np.histogram(angles, 40, density=True)
                plt.fill_between(x_[:-1], y_ + k * INCREASE, k * INCREASE, zorder = 100-K,
                        facecolor=cmap(c))
                plt.plot(x_[:-1], y_ + k * INCREASE, 'w', zorder = 100-K)
                plt.axhline(k * INCREASE, color='k', zorder = 100-K)
            
            plt.yticks([k * INCREASE for k in range(len(Ks))],
                    [r'$' + str(k) + '$' for k in Ks])
            plt.xticks([0, 45, 90, 180])
            plt.xlim([0, 180])
            plt.tight_layout()
            fig.align_ylabels()
            plt.savefig('angles_histo_{}_{}_{}.jpg'.format(L, Z, D))
            plt.close()


