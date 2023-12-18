import numpy as np
import matplotlib.pyplot as plt
import ares
import gc
import pypolychord
from pypolychord.priors import UniformPrior, LogUniformPrior
from pypolychord.settings import PolyChordSettings
from globalemu.eval import evaluate
from pypolychord import run_polychord

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
    if not FIXED_NOISE:
        theta[8] = UniformPrior(0, 100)(cube[8]) # noise
    return theta

#pars = ares.util.ParameterBundle('mirocha2017:base')
import pickle
pars = pickle.load(open('base_pars.pkl', 'rb'))

def likelihood(theta):
    if FIXED_NOISE:
        noise = nv
    else:
        noise = theta[8]
        theta = theta[:8]
        
    if ARES:
        pars['pop_fesc{0}'] = theta[1] # fesc
        pars['pop_rad_yield{1}'] = theta[0] # cx
        pars['pop_Tmin{0}'] = theta[2] # Tmin
        pars['pop_logN{1}'] = theta[3] # logN
        pars['pq_func_par0[0]{0}'] = theta[4] # fstar
        pars['pq_func_par1[0]{0}'] = theta[5] # Mp
        pars['pq_func_par2[0]{0}'] = theta[6] # gamma low
        pars['pq_func_par3[0]{0}'] = theta[7] # gamma high

        sim = ares.simulations.Global21cm(**pars, verbose=False)
        sim.run()
        z = np.arange(6, 55, 0.1)
        dT = np.interp(z, sim.history['z'][::-1], sim.history['dTb'][::-1])
        del sim
    else:
        dT, z = predictor(theta)

    logL = np.sum(-0.5*np.log(2*np.pi*noise**2) - 0.5*(dT_obs - dT)**2/noise**2)
    del dT
    gc.collect()
    return logL, []

FIXED_NOISE = True
ARES = False
PROCESSING = True
nv = 5
z, dT_obs = np.loadtxt('ares_fiducial_model_noise_%d.txt' % nv, unpack=True)

if not ARES:
    if PROCESSING:
        predictor = evaluate(base_dir='emulators/with_AFB_resampling/', logs=[0, 2, 4, 5])
    else:
        predictor = evaluate(base_dir='emulators/no_AFB_no_resampling/', logs=[0, 2, 4, 5])

nDims = 8 + (not FIXED_NOISE)

settings = PolyChordSettings(nDims, 0)
if PROCESSING:
    settings.base_dir = f'ares_fiducial_model_noise_{nv}_ARES_{ARES}_FIXED_NOISE_{FIXED_NOISE}'
else:
    settings.base_dir = f'ares_fiducial_model_noise_{nv}_ARES_{ARES}_FIXED_NOISE_{FIXED_NOISE}_no_processing'
settings.read_resume = True

output = run_polychord(likelihood, nDims, 0, settings, prior)
if FIXED_NOISE:
    paramnames = [('cX', 'c_X'), ('fesc', 'f_{esc}'), ('Tmin', 'T_{min}'), 
              ('logN', '\log N'), ('fstar', 'f_*'), ('Mp', 'M_p'), 
              ('gamma_low', '\gamma_{lo}'), ('gamma_high', '\gamma_{hi}')]
else:
    paramnames = [('cX', 'c_X'), ('fesc', 'f_{esc}'), ('Tmin', 'T_{min}'), 
              ('logN', '\log N'), ('fstar', 'f_*'), ('Mp', 'M_p'), 
              ('gamma_low', '\gamma_{lo}'), ('gamma_high', '\gamma_{hi}'), ('noise', '\sigma_{noise}')]
output.make_paramnames_files(paramnames)
