import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains

ground_truth = [2e39, 0.2, 1e4, 21, 0.05, 2.8e11, 0.49, -0.61]

FIXED_NOISE = False
ARES = False
nv = 25

if not FIXED_NOISE:
    ground_truth.append(nv)

chains = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_{ARES}_FIXED_NOISE_{FIXED_NOISE}/test')
print(chains)
if FIXED_NOISE:
    names = ['cX', 'fesc', 'Tmin', 'logN', 'fstar', 'Mp', 'gamma_low', 'gamma_high']
else:
    names = ['cX', 'fesc', 'Tmin', 'logN', 'fstar', 'Mp', 'gamma_low', 'gamma_high', 'noise']
chains['cX'] = np.log10(chains['cX'])
chains['Mp'] = np.log10(chains['Mp'])
chains['fstar'] = np.log10(chains['fstar'])
chains['Tmin'] = np.log10(chains['Tmin'])


ground_truth[0] = np.log10(ground_truth[0])
ground_truth[5] = np.log10(ground_truth[5])
ground_truth[4] = np.log10(ground_truth[4])
ground_truth[2] = np.log10(ground_truth[2])

ax = chains.plot_2d(names, figsize=(10, 10))
for i in range(len(ground_truth)):
    ax.axlines({names[i] : ground_truth[i]}, ls='--', color='r')
plt.show()

emulator_true_bias = []
for i in range(len(names)):
    emulator_true_bias.append(np.abs((np.mean(chains[names[i]].values) - ground_truth[i]))/np.std(chains[names[i]].values))
    print(names[i], emulator_true_bias[-1])
