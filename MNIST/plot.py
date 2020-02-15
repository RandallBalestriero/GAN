import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib as mpl
from tabulate import tabulate
mpl.use('Agg')
label_size = 25
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

MODELS = ['GAN', 'CONVGAN', 'VAE', 'CONVVAE']
LRS = [0.0003, 0.0003, 0.0001, 0.0001]
RUNS = [0, 1, 2, 3, 4, 5, 6, 7]

LOSSES = list()
SAMPLES = list()
AS = list()
AS_init = list()

for model, lr in zip(MODELS, LRS):
    for run in RUNS:
        f = np.load('mnist_sampling_10_100_{}_{}_{}.npz'.format(
                                lr, model, run))
        LOSSES.append(f['loss'])
        SAMPLES.append(f['samples'][-1][:20])
        AS.append(f['A'][1])
        AS_init.append(f['A'][0])
        print(f['A'].shape, AS[-1][0,:5,0], AS_init[-1][0,:5,0])


# AS are of shape (MODELS, RUNS, 10K D Z)
AS = np.array(AS).reshape((len(MODELS), len(RUNS)) + AS[-1].shape)
AS_init = np.array(AS_init).reshape(
    (len(MODELS), len(RUNS)) + AS_init[-1].shape)
SAMPLES = np.array(SAMPLES).reshape(
    (len(MODELS), len(RUNS)) + SAMPLES[-1].shape)


for a, name in zip([AS, AS_init], ['learned', 'init']):
    for model, MODEL in enumerate(MODELS):
        # loop over 4 samples
        plt.figure(figsize=(30, 24))
        # loop over the 10 bases
        for run in range(len(RUNS)):
            for i in range(10):
                plt.subplot(8, 10, 1 + i + run * 10)
                plt.imshow(a[model, run, 0, :, i].reshape((28, 28)),
                           aspect='auto', cmap='Greys',
                           vmin=a[model, run, 0].min(),
                           vmax=a[model, run, 0].max())
                plt.xticks([])
                plt.yticks([])
        plt.tight_layout()
        plt.suptitle( r'(' + name + ')Basis vectors from $A_{\omega}$ for model:' +
                 MODEL, fontsize=16)
        plt.savefig('entenglement_basis_{}_{}.jpg'.format(MODEL, name))
        plt.close()

        # loop over the 10 bases
        run = 0
        for i in range(10):
            plt.figure(figsize=(4, 4))
            plt.imshow(a[model, run, 0, :, i].reshape((28, 28)),
                       aspect='auto', cmap='Greys',
                       vmin=a[model, run, 0].min(),
                       vmax=a[model, run, 0].max())
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig('single_entenglement_basis_{}_{}_{}.jpg'.format(MODEL, name, i))
            plt.close()






for model, MODEL in enumerate(MODELS):
    plt.figure(figsize=(30, 24))
    for run in range(len(RUNS)):
        for i in range(10):
            plt.subplot(8, 10, 1 + i + 10 * run)
            plt.imshow(SAMPLES[model, run, i].reshape((28, 28)), aspect='auto',
                       cmap='Greys')
            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()
    plt.suptitle('Random samples for model:{}'.format(MODEL), fontsize=16)
    plt.savefig('entenglement_digits_{}.jpg'.format(MODEL))
    plt.close()


AS /= np.linalg.norm(AS, 2, -2, keepdims=True)
AS_init /= np.linalg.norm(AS_init, 2, -2, keepdims=True)

# those are of dimension
# number of models
# number of runs
# number of points
# Z
# Z
M = np.abs(np.einsum('mrndz,mrndi->mrnzi', AS, AS))
M_init = np.abs(np.einsum('mrndz,mrndi->mrnzi', AS_init, AS_init))
# add extra dim
M = np.stack([M, M_init], 0)

NORM = np.linalg.norm(M-np.eye(10), 2, (-2, -1))

# each below if of shape
# init learned
# number of models
MINMAX = NORM.max(3).mean(2)
MEAN = NORM.mean((2, 3))
STDMINMAX = NORM.max(3).std(2)
STDMEAN = NORM.mean(3).std(2)


V = np.stack([MINMAX, MEAN], 1).reshape((4, -1))
STDV = np.stack([STDMINMAX, STDMEAN], 1).reshape((4, -1))
V = np.round(V, 2).astype('str')
STDV = np.round(STDV, 2).astype('str')

for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        V[i, j] += ' ppm ' + STDV[i, j]


print(tabulate(V, tablefmt="latex"))



asdfsdf

V = list()

for matrix, nname in zip([M_init, M], ['init', 'learn']):
    for model, MODEL in enumerate(MODELS):
#        for func, name in zip([np.mean, np.max], ['mean', 'max']):
#            for ifunc, iname in zip([np.max, np.std], ['max', 'std']):
        # compute the matrix to plot by applying ifunc on the first axis
        # and then applying func on the resulting tensor first axis
        ifunc = np.max
        func = np.max
        iname = 'max'
        name = 'max'
        matrix2plot = func(matrix[model], 1)#ifunc(func(matrix[model], 1), 0)

#        # display the figure via imshow
#        plt.figure(figsize=(6, 6))
#        # we separate two cases, for the mean we set the colormap between
#        # 0 and 1 for the std we do not and show the colorbar
#        if iname == 'mean':
#            plt.imshow(matrix2plot, aspect='auto', interpolation='nearest',
#                       origin='lower', cmap='Greys', vmin=0, vmax=1)
#        else:
#            plt.imshow(matrix2plot, aspect='auto', interpolation='nearest',
#                       origin='lower', cmap='Greys')
#            plt.colorbar()
#        # set up a tick for each column/row
#        plt.yticks(list(range(10)), [str(i) for i in range(1, 11)])
#        plt.xticks(list(range(10)), [str(i) for i in range(1, 11)])
#        # reduce margins
#        plt.tight_layout()
#        # save
#        plt.savefig('imshow_entenglement_{}_{}_{}_{}.jpg'.format(MODEL, nname, name, iname))
#        plt.close()
#
#        # we also have to save the actual smallest/greatest
#        # off diagonal values
        V.append(matrix2plot)

