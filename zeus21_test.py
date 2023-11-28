import zeus21
from matplotlib import pyplot as plt
import numpy as np


#set up the CLASS cosmology
from classy import Class
ClassCosmo = Class()
ClassCosmo.compute()

#set up your parameters here, as an example the CDM (reduced) density
omega_cdm = 0.12
CosmoParams_input = zeus21.Cosmo_Parameters_Input(omegac = omega_cdm)
ClassyCosmo = zeus21.runclass(CosmoParams_input)
print('CLASS has run, we store the cosmology.')

#define all cosmology (including derived) parameters, and save them to the CosmoParams structure
CosmoParams = zeus21.Cosmo_Parameters(CosmoParams_input, ClassyCosmo) 
CorrFClass = zeus21.Correlations(CosmoParams, ClassyCosmo)
print('Correlation functions saved.')
HMFintclass = zeus21.HMF_interpolator(CosmoParams,ClassyCosmo)
print('HMF interpolator built. This ends the cosmology part -- moving to astrophysics.')

#set up your astro parameters too, here the peak of f*(Mh) as an example
epsilon_star = 0.15 # this is fstar*Mc
Mc = 1e10 # in Msun
alpha_star = 0.5 # gamma low in ares
beta_star = -0.5 # gamma high in ares
fesc = 0.1 # escape fraction
L40_xray = 1.0 # X-ray luminosity in 10^40 erg/s/SFR where SFR is Msol/yr
E0_xray = 30.0 # minimum X-ray energy in keV
alpha_Xray = 1.0 # X-ray spectral index
# by default the star formation is a double power law...
# see here for info on asto parameters https://github.com/JulianBMunoz/Zeus21/blob/main/zeus21/inputs.py
AstroParams = zeus21.Astro_Parameters(CosmoParams, epsstar=epsilon_star,
                                      Mc=Mc, alphastar=alpha_star, betastar=beta_star,
                                      fesc10=fesc, L40_xray=L40_xray, E0_xray=E0_xray, alpha_xray=alpha_Xray)


AP = [AstroParams, zeus21.Astro_Parameters(CosmoParams)]
for i in range(len(AP)):
    ZMIN = 10.0 #down to which z we compute the evolution
    CoeffStructure = zeus21.get_T21_coefficients(CosmoParams, ClassyCosmo, AP[i], HMFintclass, zmin=ZMIN)
    zlist = CoeffStructure.zintegral
    print('SFRD and coefficients stored. Move ahead.')

    plt.plot(zlist,CoeffStructure.T21avg, 'k')
plt.xlabel(r'z');
plt.ylabel(r'$T_{21}$ [mK]');
plt.xlim([10, 25])
plt.show()