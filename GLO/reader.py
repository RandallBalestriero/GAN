import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('Agg')
label_size = 18
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size


cmap = matplotlib.cm.get_cmap('winter')
filename = 'glo_{}_{}_{}_{}_{}.npz'
X = [100, 120, 140, 160, 180, 200, 300, 400, 500, 1000]
DATA = list()
for D in [16]:
    for N in X:
        for Zstar in [5]:
            for W in [6]:
                data = list()
                for Z in [1, 2, 3, 4, 5, 6, 7]:
                    datum = np.load(filename.format(N, W, D, Z, Zstar))['loss']
                    data.append(datum.min(1))
                data = np.array(data).reshape((7, 30))
                DATA.append(data.min(1))

DATA = np.array(DATA)

plt.figure(figsize=(6, 2.5))
plt.axhline(0, c='k')
plt.axvline(5, c='r')

for i, c in enumerate(np.linspace(0.12, 0.84, DATA.shape[0])):
    plt.plot(np.arange(1, 8), DATA[i], c=cmap(c), lw=2.5)


plt.tight_layout()
plt.savefig('error_with_n.jpg')
plt.close()


