import ares
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior, LogUniformPrior
import numpy as np

def prior(cube):
    theta = np.zeros_like(cube)
    theta[0] = LogUniformPrior(1e36, 1e41)(cube[0]) # cX
    theta[1] = UniformPrior(0, 1)(cube[1]) # fesc
    theta[2] = LogUniformPrior(3e2, 5e5)(cube[2])# Tmin
    theta[3] = UniformPrior(18, 23)(cube[3]) # logN
    theta[4] = LogUniformPrior(1e-5, 1e0)(cube[4]) #fstar
    theta[5] = LogUniformPrior(1e8, 1e15)(cube[5]) #Mp
    theta[6] = UniformPrior(0, 2)(cube[6]) # gamma low
    theta[7] = UniformPrior(-4, 0)(cube[7]) # gamma high
    return theta

pars = ares.util.ParameterBundle('mirocha2017:base')
import pickle
pickle.dump(pars, open('base_pars.pkl', 'wb'))
pars = pickle.load(open('base_pars.pkl', 'rb'))
for key in pars:
    print(key, pars[key])

sim = ares.simulations.Global21cm(**pars)
sim.run()
z = np.arange(6, 55, 0.1)
dT = np.interp(z, sim.history['z'][::-1], sim.history['dTb'][::-1])
plt.plot(1420.4/(1+z), dT)

pars['pop_fesc{0}'] = 1 # fesc
pars['pop_rad_yield{1}'] = 2e36 # cx
pars['pop_Tmin{0}'] = 1e3 # Tmin
pars['pop_logN{1}'] = 20 # logN
pars['pq_func_par0[0]{0}'] = 0.5 # fstar
pars['pq_func_par1[0]{0}'] = 1e8 # Mp
pars['pq_func_par2[0]{0}'] = 1.5 # gamma low
pars['pq_func_par3[0]{0}'] = -1.5 # gamma high

sim = ares.simulations.Global21cm(**pars)
sim.run()
z = np.arange(6, 55, 0.1)
dT = np.interp(z, sim.history['z'][::-1], sim.history['dTb'][::-1])
plt.plot(1420.4/(1+z), dT)

plt.show()
