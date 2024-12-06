import numpy as np
import matplotlib.pyplot as plt
import ares
from pypolychord.priors import UniformPrior, LogUniformPrior
from tqdm import tqdm


def prior(cube):
    theta = np.zeros_like(cube)
    theta[0] = LogUniformPrior(1e36, 1e41)(cube[0])  # cX
    theta[1] = UniformPrior(0, 1)(cube[1])  # fesc
    theta[2] = LogUniformPrior(3e2, 5e5)(cube[2])  # Tmin
    theta[3] = UniformPrior(18, 23)(cube[3])  # logN
    theta[4] = LogUniformPrior(1e-5, 1e0)(cube[4])  # fstar
    theta[5] = LogUniformPrior(1e8, 1e15)(cube[5])  # Mp
    theta[6] = UniformPrior(0, 2)(cube[6])  # gamma low
    theta[7] = UniformPrior(-4, 0)(cube[7])  # gamma high
    return theta


prior_samples = np.array([prior(np.random.uniform(0, 1, 8)) for i in range(2000)])

import pickle

pars = pickle.load(open("base_pars.pkl", "rb"))


def aressignal(z, theta):
    pars["pop_fesc{0}"] = theta[1]  # fesc
    pars["pop_rad_yield{1}"] = theta[0]  # cx
    pars["pop_Tmin{0}"] = theta[2]  # Tmin
    pars["pop_logN{1}"] = theta[3]  # logN
    pars["pq_func_par0[0]{0}"] = theta[4]  # fstar
    pars["pq_func_par1[0]{0}"] = theta[5]  # Mp
    pars["pq_func_par2[0]{0}"] = theta[6]  # gamma low
    pars["pq_func_par3[0]{0}"] = theta[7]  # gamma high

    sim = ares.simulations.Global21cm(**pars, verbose=False)
    sim.run()
    dT = np.interp(z, sim.history["z"][::-1], sim.history["dTb"][::-1])
    del sim
    return dT


validation_set = []
for t in tqdm(range(len(prior_samples))):
    validation_set.append(aressignal(np.arange(6, 55, 0.1), prior_samples[t]))

validation_set = np.array(validation_set)
np.savetxt("signal_data/val_data.txt", prior_samples)
np.savetxt("signal_data/val_labels.txt", validation_set)
