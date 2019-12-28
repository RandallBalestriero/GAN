import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.special import hyp2f1, gamma

mpl.use('Agg')
label_size = 14
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['xtick.labelsize'] = label_size



INCREASE = 0.005
cmap = cm.get_cmap('Reds')
Ks = [8 ,16, 32, 64, 128, 256, 512]

for L in [1, 2]:
    for Z in [2, 4, 8]:
        for D in [Z+1, Z*2, Z*4]:
#            Kslocal = [k for k in Ks if k>Z]
#
#            data = np.load('angles_{}_{}_{}.npz'.format(L, Z, D),
#                           allow_pickle=True)
#            angles = data['angles'] / 2
#
#            plt.figure(figsize=(3,4))
#            for c, (k, K) in zip(np.linspace(0.3, 0.8, len(Kslocal)),
#                                 enumerate(Kslocal)):
#                y_, x_ = np.histogram(angles[k], 40, density=True)
#                plt.fill_between(x_[:-1], y_ + k * INCREASE, k * INCREASE,
#                                 zorder = 100-K, facecolor=cmap(c))
#                plt.plot(x_[:-1], y_ + k * INCREASE, 'w', zorder = 100-K)
#                plt.axhline(k * INCREASE, color='k', zorder = 100-K)
#                plt.text(-45, k * INCREASE, r'$' + str(K) + '$', fontsize=17)
#
#            for k, K in zip([0, 45, 90], [-5, 35, 80]):
#                plt.text(K - 1, -INCREASE/1.4, str(k), fontsize=17)
#
#            plt.xlim([-2, 90])
#            plt.ylim([0, INCREASE * 2.2 * len(Ks)])
#            plt.tight_layout()
#            ax = plt.gca()
#            ax.axis('off')
#            plt.savefig('angles_histo_{}_{}_{}.jpg'.format(L, Z, D))
#            plt.close()

            plt.figure(figsize=(2.5,2.5))
            n, p = D, Z
            theta = np.linspace(0, 3.14159/2, 200)
            first = p*(n-p)*(np.sin(theta)**(p*(n-p)-1))
            g = gamma((p+1)/2)*gamma((n-p+1)/2)/(gamma(0.5)*gamma((n+2)/2))
            hyp = hyp2f1((n-p-1)/2, 1/2, (n+1)/2, np.sin(theta)**2)**(p-1)
            plt.axhline(0., color='k', lw=2)
            plt.plot(theta, first * g * hyp, lw=4)
            for k, K in zip([0, 45, 90], [0, 3.14159/4, 3.14159/2]):
                plt.text(K, -0.25, str(k), fontsize=17)
            plt.xlim([0, 3.14159/2])
            plt.tight_layout()
            ax = plt.gca()
            ax.axis('off')
            plt.savefig('angles_theo_{}_{}.jpg'.format(Z, D))
            plt.close()



