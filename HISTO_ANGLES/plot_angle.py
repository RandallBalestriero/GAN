import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.special import hyp2f1, gamma

mpl.use('Agg')
label_size = 16
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['xtick.labelsize'] = label_size

def analytical_dist(theta, n, p):
    first = p*(n-p)*(np.sin(theta)**(p*(n-p)-1))
    g = gamma((p+1)/2)*gamma((n-p+1)/2)/(gamma(0.5)*gamma((n+2)/2))
    hyp = hyp2f1((n-p-1)/2, 1/2, (n+1)/2, np.sin(theta)**2)**(p-1)
    output = first * g * hyp
    const = (output[1:] + output[:-1])/2
    print(output,const.sum())
    return output/const.sum()


def plot_analytical(n, p, YMAX):
    plt.figure(figsize=(3, 4))

    theta = np.linspace(0, 3.14159/2, 200)
    plt.axhline(0., color='k', lw=2)
    plt.fill_between(theta, analytical_dist(theta, D, Z), facecolor='b')
    for k, K in zip([0, 45, 90], [0, 3.14159/4-0.1, 3.14159/2-0.1]):
        plt.text(K, -0.012, str(k), fontsize=20)
    plt.xlim([0, 3.14159/2])
    plt.ylim([0, YMAX])
    plt.tight_layout()
    ax = plt.gca()
    ax.axis('off')
    plt.savefig('angles_theo_{}_{}.jpg'.format(Z, D),
                bbox_inches=mpl.transforms.Bbox([[0.4, 0.09], [3, 4]]))
    plt.close()

def plot_hists(angles, L, Z, D, Kslocal, INCREASE):

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

    k += 1
    K += 10
    # true case
    x_ = np.linspace(0, 3.14159/2, 200)
    y_ = analytical_dist(x_, D, Z)
    plt.fill_between(x_ * 90 * 2 / 3.14159, y_ + k * INCREASE, k * INCREASE,
                     zorder = 100-K, facecolor=(51/255, 153/255, 1))
    plt.plot(x_, y_ + k * INCREASE, 'w', zorder = 100-K)
    plt.axhline(k * INCREASE, color='k', zorder = 100-K)
    plt.text(-45, k * INCREASE, r'Rand.', fontsize=17.5,
             color=(51/255, 153/255, 1))

    for k, K in zip([0, 45, 90], [-5, 35, 80]):
        plt.text(K - 1, -INCREASE/1.05, str(k), fontsize=20)

    plt.xlim([-2, 90])
    plt.ylim([0, INCREASE * 1.95 * len(Ks)])
    plt.tight_layout()
    ax = plt.gca()
    ax.axis('off')
    plt.savefig('angles_histo_{}_{}_{}.jpg'.format(L, Z, D),
                bbox_inches=mpl.transforms.Bbox([[0.4, 0.09], [3, 4]]))
    plt.close()
    return y_.max()



INCREASE = 0.0065
cmap = cm.get_cmap('Reds')
Ks = [8 ,16, 32, 64, 128, 256, 512, 1024]
sigma = 1
YMAX = 0
for Z in [2, 4, 8, 16]:
    for L in [1, 2]:
        for D in [Z+1, Z*2, Z*4]:
            Kslocal = [k for k in Ks if k>Z]
            data = list()
            for c, (k, K) in zip(np.linspace(0.3, 0.8, len(Kslocal)), enumerate(Kslocal)):
                datum = np.load('anglesv2_{}_{}_{}_{}.npz'.format(L, Z, D, K))
                data.append(datum['angles'])

            data = np.array(data)
            print(Z, L, D)
            # PLOT HISTOGRAMS
            ymax = plot_hists(data, L, Z, D, Kslocal, INCREASE)
            YMAX = max(YMAX, ymax)




