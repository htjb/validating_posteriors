import numpy as np
import matplotlib.pyplot as plt

nv = np.array([5, 25, 50, 250])[::-1]

fiducial_model = np.loadtxt('ares_fiducial_model.txt')

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
for i, n in enumerate(nv):
    data = np.loadtxt('ares_fiducial_model_noise_{0}.txt'.format(n))
    axes[0].plot(data[:, 0], data[:, 1], label='\sigma={0}'.format(n))
    delta = data[:, 1] - fiducial_model[:, 1]
    axes[1].hist(delta, label=r'$\sigma=$' + '{:.2f}'.format(np.std(delta)), 
                 bins=20, histtype='step')
axes[1].legend()
axes[1].set_xlabel(r'$\delta T_b - \delta T_b^{fiducial}$')
axes[0].set_xlabel(r'z')
axes[0].set_ylabel(r'$\delta T_b$ [mK]')
plt.tight_layout()
plt.savefig('noise_check.png', dpi=300, bbox_inches='tight')
plt.show()
