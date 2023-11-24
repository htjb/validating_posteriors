import ares
import matplotlib.pyplot as plt

fstar = 0.05 # log uniform 10^-5, 1

pars = {'pop_sfr_model{0}' :'sfe-func',
                    'pop_fstar{0}': 'pq',
                    'pq_func{0}': 'dpl',
                    'pq_func_par0{0}': 0.05, # fstar, p
                    'pq_func_par1{0}': 2.8e11, # Mturn, q
                    'pq_func_par2{0}': 0.49, # gamma low
                    'pq_func_par3{0}': -0.61, # gamma high
                    'pq_func_par4{0}': 1e10, # normalisation mass
                    #'pop_cx{0}' :2.6e39,
                    #'pop_Tmin{0}' :1e4,
                    #'pop_fesc{0}':0.2,
                    #'pop_logN{0}':21,
                    }

sim = ares.simulations.Global21cm(**pars)

sim.run()
ax, zax = sim.GlobalSignature()

sim2 = ares.simulations.Global21cm()
sim2.run()
sim2.GlobalSignature(ax=ax)
plt.show()