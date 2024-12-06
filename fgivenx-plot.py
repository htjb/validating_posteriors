import numpy as np
import matplotlib.pyplot as plt
from fgivenx import plot_contours
from anesthetic import read_chains
from globalemu.eval import evaluate
import ares
from pypolychord.priors import UniformPrior, LogUniformPrior


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


prior_samples = np.array([prior(np.random.uniform(0, 1, 8)) for i in range(1000)])

import pickle

pars = pickle.load(open("base_pars.pkl", "rb"))
predictor = evaluate(base_dir="emulators/with_AFB_resampling/", logs=[0, 2, 4, 5])
z = np.arange(6, 55, 0.1)


def emusignal(z, theta):
    dT, z = predictor(theta)
    return dT


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


fid = np.loadtxt("ares_fiducial_model.txt")

names = ["cX", "fesc", "Tmin", "logN", "fstar", "Mp", "gamma_low", "gamma_high"]


fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
nv = [5, 25, 50, 250]
for i in range(len(nv)):
    gemu_chains = read_chains(
        f"ares_fiducial_model_noise_{nv[i]}_ARES_False_FIXED_NOISE_True/test"
    )

    gemu_chains = gemu_chains.compress()
    gemu_chains = gemu_chains[names].values

    #if i == 3:
    plot_contours(
        emusignal, z, prior_samples, axes[i, 1], alpha=0.5, colors=plt.cm.Blues_r
    )
    plot_contours(emusignal, z, gemu_chains, axes[i, 1], alpha=0.5)

    ares_chains = read_chains(f'ares_fiducial_model_noise_{nv[i]}_ARES_True_FIXED_NOISE_True/test')
    ares_chains = ares_chains.compress(50)
    ares_chains = ares_chains[names].values


    plot_contours(aressignal, z, prior_samples, axes[i, 0],
                    alpha=0.5, colors=plt.cm.Blues_r)
    plot_contours(aressignal, z, ares_chains, axes[i, 0],
                    alpha=0.5)
    

    z, dT_obs = np.loadtxt("ares_fiducial_model_noise_%d.txt" % nv[i], unpack=True)
    axes[i, 0].plot(z, dT_obs, label="Noise = %d" % nv[i], c="k", alpha=0.8)
    axes[i, 1].plot(z, dT_obs, label="Noise = %d" % nv[i], c='k', alpha=0.8)
    axes[i, 0].plot(fid[:, 0], fid[:, 1], label="Fiducial", c="g")
    axes[i, 1].plot(fid[:, 0], fid[:, 1], label="Fiducial", c='g')


    axes[i, 0].set_ylabel(f"Noise = {nv} mK \n $\delta T_b$ [mK]")
    
axes[3, 0].set_xlabel("z")
axes[3, 1].set_xlabel("z")
axes[0, 0].set_title("ARES")
axes[0, 1].set_title("Emulator")

plt.tight_layout()
plt.savefig("fgivenx.png", dpi=300, bbox_inches="tight")
plt.show()
