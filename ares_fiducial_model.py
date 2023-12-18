import ares
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior, LogUniformPrior
import numpy as np

pars = ares.util.ParameterBundle('mirocha2017:base')

pars['pop_fesc{0}'] = 0.2 # fesc
pars['pop_rad_yield{1}'] = 2e39 # cx
pars['pop_Tmin{0}'] = 1e4 # Tmin
pars['pop_logN{1}'] = 21 # logN
pars['pq_func_par0[0]{0}'] = 0.05 # fstar
pars['pq_func_par1[0]{0}'] = 2.8e11 # Mp
pars['pq_func_par2[0]{0}'] = 0.49 # gamma low
pars['pq_func_par3[0]{0}'] = -0.61 # gamma high

sim = ares.simulations.Global21cm(**pars)
sim.run()
z = np.arange(6, 55, 0.1)
dT = np.interp(z, sim.history['z'][::-1], sim.history['dTb'][::-1])
plt.plot(1420.4/(1+z), dT)

plt.show()

np.savetxt('ares_fiducial_model.txt', np.c_[z, dT])

#noise = [5, 10, 25, 50]
noise = [1]
for i in range(len(noise)):
    n = np.random.normal(0, noise[i], len(dT))
    np.savetxt('ares_fiducial_model_noise_%d.txt' % noise[i], np.c_[z, dT+n])
    plt.plot(1420.4/(1+z), dT+n, label='Noise = %d mK' % noise[i])
plt.legend()
plt.show()