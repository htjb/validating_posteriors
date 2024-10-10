import numpy as np
import matplotlib.pyplot as plt

nd = len(np.arange(6, 55, 0.1))

noise = np.linspace(1, 75, 100)
Dkl = np.linspace(0, 5, 100)

X, Y = np.meshgrid(noise, Dkl)

limit = np.sqrt(2*Y/nd)*X

fig, axes = plt.subplots(1, 1, figsize=(6, 4))
cb = axes.contourf(X, Y, limit)
plt.colorbar(cb, label='RMSE')
axes.set_xlabel('Noise Level')
axes.set_ylabel('KL Divergence')

actual_rmse = np.array([0.11, 0.82, 2.56])
label= ['Min Error', 'Mean Error', '95th Percentile Error']

for i,rmse in enumerate(actual_rmse):
    axes.plot(noise, nd/2*(rmse/noise)**2, label=f'{label[i]} = {rmse} mK')
axes.set_ylim(0, 5)
plt.legend()

[axes.axvline(l,color='w', ls='--', ymin=0, ymax=5) for l in [5, 25, 50]]

plt.tight_layout()

plt.show()