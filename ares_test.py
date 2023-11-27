import ares
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior, LogUniformPrior
import numpy as np

def prior(cube):
    theta = np.zeros_like(cube)
    theta[0] = LogUniformPrior(1e36, 1e41)(cube[0]) # cX
    theta[1] = UniformPrior(0, 1)(cube[1]) # fesc
    theta[2] = LogUniformPrior(3e2, 5e5)(cube[2])
    theta[3] = UniformPrior(18, 23)(cube[3]) # logN
    theta[4] = LogUniformPrior(1e-5, 1e0)(cube[4]) #fstar
    theta[5] = LogUniformPrior(1e8, 1e15)(cube[5])
    theta[6] = UniformPrior(0, 2)(cube[6])
    theta[7] = UniformPrior(-4, 0)(cube[7])
    return theta

pars = ares.util.ParameterBundle('mirocha2017:base')
for key in pars:
    print(key, pars[key])

sim = ares.simulations.Global21cm(**pars)

sim.run()

z = np.arange(6, 55, 0.1)

dT = np.interp(z, sim.history['z'][::-1], sim.history['dTb'][::-1])

plt.plot(1420.4/(1+z), dT)
plt.show()
