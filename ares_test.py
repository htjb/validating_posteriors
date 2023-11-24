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

pars_edit = {'pop_sfr_model[0]' :'sfe-func',
                    'pop_fstar[0]': 'pq[0]',
                    'pq_func[0]': 'dpl',
                    'pq_func_par0[0]': 0.05, # fstar, p
                    'pq_func_par1[0]': 2.8e11, # Mturn, q
                    'pq_func_par2[0]': 0.49, # gamma low
                    'pq_func_par3[0]': -0.61, # gamma high
                    'pq_func_par4[0]': 1e10, # normalisation mass
                    'pop_cX' :2.6e39,
                    'pop_Tmin{0}' :1e4,
                    'pop_fesc{0}':0.2,
                    'pop_logN{0}':21,
                    'final_redshift': 6,
                    }

sim = ares.simulations.Global21cm(**pars_edit)
print(sim._pf.Npops)
for key in sim.pf:
    if key in pars_edit.keys():
        print(key, sim.pf[key])

sim.run()
z = np.arange(6, 55, 0.1)

ax, zax = sim.GlobalSignature(color='k', ls='--')

pars = ares.util.ParameterBundle('mirocha2017:base')
sim2 = ares.simulations.Global21cm(**pars)
for key in sim2.pf:
    if key in pars_edit.keys():
        print(key, sim2.pf[key])
    elif 'pop_sfr_model' in key:
        print(key, sim2.pf[key])
    elif 'cX' in key:
        print(key, sim2.pf[key])

sim2.run()
sim2.GlobalSignature(ax=ax, color='b', ls='-')

pars['pop_logN{0}'] = 21
pars['pop_logN{1}'] = 21
sim3 = ares.simulations.Global21cm(**pars)
sim3.run()

for key in pars:
    if key in pars.keys():
        print(key, pars[key])
    elif 'pop_sfr_model' in key:
        print(key, pars[key])
    elif 'cX' in key:
        print(key, pars[key])

sim3.GlobalSignature(ax=ax, color='r', ls='-')

plt.show()
sys.exit(1)



"""pars = ares.util.ParameterBundle('mirocha2017:base')
for key in pars:
    print(key, pars[key])"""

u = np.random.uniform(0, 1, (10, 8))


for i in range(len(u)):
    t = prior(u[i])
    print(t)

    pars = {'pop_sfr_model{0}' :'sfe-func',
                    'pq_func{0}': 'dpl',
                    'pq_func_par0': t[4], # fstar, p
                    'pq_func_par1': t[5], # Mturn, q
                    'pq_func_par2': t[6], # gamma low
                    'pq_func_par3': t[7], # gamma high
                    'pq_func_par4': 1e10, # normalisation mass
                    'pop_rad_yield{0}' :t[0],
                    'pop_fstar{0}': 'pq',
                    'pop_Tmin{0}' :t[2],
                    'pop_fesc{0}':t[1],
                    'pop_logN{0}':t[3],
                    }
    sim2 = ares.simulations.Global21cm(**pars)
    sim2.run()
    dT = np.interp(z, sim2.history['z'][::-1], sim2.history['dTb'][::-1])
    plt.plot(z, dT)
plt.legend()
plt.show()