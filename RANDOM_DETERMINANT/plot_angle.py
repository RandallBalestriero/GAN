import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.special import hyp2f1, gamma
import pickle

mpl.use('Agg')
label_size = 16
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['xtick.labelsize'] = label_size


def plot_hists(angles, L, Z, D, leakiness, Kslocal, INCREASE):

    plt.figure(figsize=(3,4))

    # loop over the case
    for c, (k, K) in zip(np.linspace(0.3, 0.8, len(Kslocal)),
                         enumerate(Kslocal)):
        y_, x_ = np.histogram(angles[k], 40, density=True)
        plt.fill_between((x_[:-1]+x_[1:])/2, y_ + k * INCREASE, k * INCREASE,
                         zorder = 100-K, facecolor=cmap(c))
        plt.plot((x_[:-1]+x_[1:])/2, y_ + k * INCREASE, 'w', zorder = 100-K)
        plt.axhline(k * INCREASE, color='k', zorder = 100-K)
        plt.text(-45, k * INCREASE, r'$' + str(K) + '$', fontsize=18.5)

    K += 10

    for k, K in zip([0, 45, 90], [-5, 35, 80]):
        plt.text(K - 1, -INCREASE/1.05, str(k), fontsize=20)

    plt.xlim([-2, 90])
    plt.ylim([0, INCREASE * 1.95 * len(Ks)])
    plt.tight_layout()
    ax = plt.gca()
    ax.axis('off')
    plt.savefig('det_histo_{}_{}_{}_{}.jpg'.format(L, Z, D, leakiness),
                bbox_inches=mpl.transforms.Bbox([[0.4, 0.09], [3, 4]]))
    plt.close()
    return y_.max()



INCREASE = 0.0065
cmap = cm.get_cmap('Greens')
Ks = [8 ,16, 32, 64, 128, 256, 512, 1024]
sigma = 1
YMAX = 0
for leakiness in [0., 0.1, 0.3]:
    for Z in [2, 4, 8]:#, 16]:
        for L in [1, 2]:
            for D in [Z+1, Z*2, Z*4]:
                Kslocal = [k for k in Ks if k>Z]
                data = list()
                for c, (k, K) in zip(np.linspace(0.3, 0.8, len(Kslocal)), enumerate(Kslocal)):
                    datum = np.load('det_{}_{}_{}_{}_{}.npz'.format(L, Z, D, K, leakiness), allow_pickle=True)
                    data.append(np.log(1 / datum['angles']))
                    print(datum['angles'].shape)
    
                data = np.array(data)
                print(Z, L, D)
                # PLOT HISTOGRAMS
                ymax = plot_hists(data, L, Z, D, leakiness, Kslocal, INCREASE)
                YMAX = max(YMAX, ymax)

