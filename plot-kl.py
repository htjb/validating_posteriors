import numpy as np
import matplotlib.pyplot as plt

nd = len(np.arange(6, 55, 0.1))

noise = np.linspace(1, 260, 200)
Dkl = np.linspace(0, 10, 200)

X, Y = np.meshgrid(noise, Dkl)

limit = np.sqrt(2*Y/nd)*X

fig, axes = plt.subplots(1, 1, figsize=(6.3, 3))
cb = axes.contourf(X, Y, limit, cmap='Blues', levels=10)
plt.colorbar(cb, label='Emulator RMSE [mK]')
axes.set_xlabel(r'Noise $\sigma$ [mK]')
axes.set_ylabel(r'KL-Divergence $D_{KL}$ [nats]')

axes.set_xscale('log')
axes.set_yscale('log')
axes.set_ylim(0, 10)
plt.tight_layout()
plt.savefig('kl_divergence-illustration.png', dpi=300, bbox_inches='tight')

plt.show()