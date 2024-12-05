import numpy as np
from anesthetic import read_chains
import numpy as np
import matplotlib.pyplot as plt

names= ['cX', 'fesc', 'Tmin', 'logN', 'fstar', 'Mp', 'gamma_low', 'gamma_high']

chains = read_chains('ares_fiducial_model_noise_25_ARES_False_FIXED_NOISE_True/test')
chainsrepeate = read_chains('ares_fiducial_model_noise_25_ARES_False_FIXED_NOISE_True_take2/test')

chains['cX'] = np.log10(chains['cX'])
chains['Mp'] = np.log10(chains['Mp'])
chains['fstar'] = np.log10(chains['fstar'])
chains['Tmin'] = np.log10(chains['Tmin'])

chainsrepeate['cX'] = np.log10(chainsrepeate['cX'])
chainsrepeate['Mp'] = np.log10(chainsrepeate['Mp'])
chainsrepeate['fstar'] = np.log10(chainsrepeate['fstar'])
chainsrepeate['Tmin'] = np.log10(chainsrepeate['Tmin'])

axes = chains.plot_2d(names, figsize=(10, 10), label='Run 1')
chainsrepeate.plot_2d(axes, label='Run 2')
plt.tight_layout()
plt.savefig('check-emulator-consistency-ares_fiducial_model_noise_25_ARES_False_FIXED_NOISE_True.png')
plt.show()
