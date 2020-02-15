import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['xtick.labelsize']=18
matplotlib.rcParams['ytick.labelsize']=18



def lrelu(x):
    mask = (x > 0).astype('float32')
    return x * mask + (0.1 * x) * (1 - mask), mask

class DN:

    def __init__(self, Z, D1, D2, D):
        self.W1 = np.random.randn(Z, D1) / np.sqrt(Z)
        self.b1 = np.random.randn(D1)
        self.W2 = np.random.randn(D1, D2) / np.sqrt(D1)
        self.b2 = np.random.randn(D2)
        self.W3 = np.random.randn(D2, D) / np.sqrt(D2)

    def forward(self, x, q):
        if q[0].ndim == 2:
            h1 = np.matmul(x, self.W1) + self.b1
            v1, m1 = lrelu(h1) # (N W)
            h2 = np.matmul(v1 * q[0][:, None, :], self.W2) + self.b2
            v2, m2 = lrelu(h2) # (T N W)
            A = self.W1 * (m1 * q[0][:, None, :])[:, :, None, :] # (T N W W)
            A = np.matmul(A, self.W2 * (m2 * q[1][:, None, :])[:, :, None, :])
        else:
            h1 = np.matmul(x[None, :, :].repeat(q[0].shape[0],0),
                           self.W1 * q[0]) + self.b1 # (T N W)
            v1, m1 = lrelu(h1) # (T N W)
            h2 = np.matmul(v1, self.W2 * q[1]) + self.b2 #(T N W)
            v2, m2 = lrelu(h2) # (T N W)
            A = self.W1 * m1[:, :, None, :] * q[0][:, None, :, :] # (T N W W)
            A = np.matmul(A, self.W2 * m2[:, :, None, :] * q[1][:, None, :, :])
        return A

Z, D = 6, 10
N = 2000
T = 2000
Ws = [6, 8, 10, 12, 24, 48]
ps = [0.1, 0.3]
zs = np.random.randn(N, Z)

plt.figure(figsize=(12, 2.5))
plt.axhline(6, c='k', alpha=0.3)
plt.axhline(0, c='k', alpha=0.3)

for wi, W in enumerate(Ws):
    dn = DN(Z, W, W, D)
    # do the dropout case
    for pi, p in enumerate(ps):
        print(p)
        ranks = list()
        for x in range(10):
            dropouts = (np.random.rand(T, 2 * W) > p).astype('float32')
            As = dn.forward(zs[x*1000:(x+1)*1000], [dropouts[:, :W], dropouts[:, W:]])
            rs = np.linalg.matrix_rank(As)
            pos = pi * len(Ws) + wi + 1 + int(pi>0)
            ranks.append(rs.flatten())
        plt.boxplot(np.concatenate(ranks), showfliers=False, positions=[pos],
                    medianprops={'color': 'r', 'linewidth': 3}, notch=True,
                    patch_artist=True)

    # do the dropconnect case
    for pi, p in enumerate(ps):
        print(p)
        ranks = list()
        for x in range(10):
            dropouts = (np.random.rand(T, Z * W + W * W) > p).astype('float32')
            As = dn.forward(zs[x*1000:(x+1)*1000], [dropouts[:, :Z * W].reshape((T, Z, W)),
                dropouts[:, Z * W:].reshape((T, W, W))])
            rs = np.linalg.matrix_rank(As)
            pos = pi * len(Ws) + wi + 3 + 2*len(Ws) + int(pi>0)
            ranks.append(rs.flatten())
        plt.boxplot(np.concatenate(ranks), showfliers=False, positions=[pos],
                    medianprops={'color': 'r', 'linewidth': 3}, notch=True,
                    patch_artist=True)




#
positions = np.arange(1, len(Ws)+1)[None, :]
positions = positions.repeat(4, 0)
positions += np.arange(4)[:, None] * (len(Ws)+1)
positions = positions.flatten()
labels = np.tile(np.array(Ws), 4).astype('str')


plt.xticks(positions, labels)
plt.xlim([0, positions[-1] + 1])
plt.yticks([0, 2, 4, 6], ['0', '2', '4', '6'])
plt.tight_layout()

plt.savefig('boxplot_dims.jpg')






