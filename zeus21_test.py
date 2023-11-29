import zeus21
from matplotlib import pyplot as plt
from pypolychord.priors import UniformPrior, LogUniformPrior
import numpy as np
from tqdm import tqdm
import os


def prior(cube):
    theta = np.zeros_like(cube)
    theta[0] = UniformPrior(0, 1)(cube[0]) # fesc
    theta[1] = LogUniformPrior(1e-5, 1e0)(cube[1]) #fstar
    theta[2] = LogUniformPrior(1e8, 1e15)(cube[2]) #Mc
    theta[3] = UniformPrior(0, 2)(cube[3]) # gamma low
    theta[4] = UniformPrior(-4, 0)(cube[4]) # gamma high
    theta[5] = LogUniformPrior(0.003, 3000)(cube[5]) # LX40
    theta[6] = UniformPrior(1, 1.5)(cube[6]) # alpha_Xray
    theta[7] = UniformPrior(100, 3000)(cube[7]) # E0_Xray in ev
    return theta

#set up the CLASS cosmology
from classy import Class
ClassCosmo = Class()
ClassCosmo.compute()

CosmoParams_input = zeus21.Cosmo_Parameters_Input()
ClassyCosmo = zeus21.runclass(CosmoParams_input)
print('CLASS has run, we store the cosmology.')

#define all cosmology (including derived) parameters, and save them to the CosmoParams structure
CosmoParams = zeus21.Cosmo_Parameters(CosmoParams_input, ClassyCosmo) 
CorrFClass = zeus21.Correlations(CosmoParams, ClassyCosmo)
print('Correlation functions saved.')
HMFintclass = zeus21.HMF_interpolator(CosmoParams,ClassyCosmo)
print('HMF interpolator built. This ends the cosmology part -- moving to astrophysics.')

nsamples = 10
parameters, signals = [], []
for i in tqdm(range(nsamples)):
    try:
        theta = prior(np.random.uniform(0, 1, 8))
        #print(theta)
        #set up your astro parameters too, here the peak of f*(Mh) as an example
        epsilon_star = theta[1] # this is fstar at Mc
        Mc = theta[2] # in Msun
        alpha_star = theta[3] # gamma low in ares
        beta_star = theta[4] # gamma high in ares
        fesc = theta[0] # escape fraction
        L40_xray = theta[5] # X-ray luminosity in 10^40 erg/s/SFR where SFR is Msol/yr ... essentially 3*fx in our code
        E0_xray = theta[7] # minimum X-ray energy in eV
        alpha_Xray = theta[6] # X-ray spectral index
        # by default the star formation is a double power law...
        # see here for info on asto parameters https://github.com/JulianBMunoz/Zeus21/blob/main/zeus21/inputs.py
        AstroParams = zeus21.Astro_Parameters(CosmoParams, epsstar=epsilon_star,
                                            Mc=Mc, alphastar=alpha_star, betastar=beta_star,
                                            fesc10=fesc, L40_xray=L40_xray, E0_xray=E0_xray, alpha_xray=alpha_Xray)
        #AP.append(AstroParams)

        ZMIN = 6.0 #down to which z we compute the evolution
        CoeffStructure = zeus21.get_T21_coefficients(CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)
        zlist = CoeffStructure.zintegral
        #print('SFRD and coefficients stored. Move ahead.')
        if np.isnan(np.sum(CoeffStructure.T21avg)):
            print('NaN encountered. Skipping this one.')
        else:
            parameters.append(theta)
            signals.append(CoeffStructure.T21avg)
    except:
        pass

parameters = np.array(parameters)
signals = np.array(signals)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(parameters, signals, test_size=0.2, random_state=42)

if not os.path.exists('zeus21_dat'):
    os.makedirs('zeus21_dat')

np.savetxt('zeus21_dat/train_data.txt', X_train)
np.savetxt('zeus21_dat/train_labels.txt', y_train)
np.savetxt('zeus21_dat/test_data.txt', X_test)
np.savetxt('zeus21_dat/test_labels.txt', y_test)
