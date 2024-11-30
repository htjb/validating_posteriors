# On the accuracy of posterior recovery with neural network emulators

Code associated with ...

The idea is to demonstrate that we can recover accurate posterior distributions
when we use emulators in our inference pipelines. In the paper we define
an upper limit on the amount of incorrect information inferred when using an
emulator as a funciton of the emualtors error and the noise in the data. We then
go on to demonstrate how this limit can be used with an example from
21cm Cosmology. The paper is largely inspired by the questions asked in
and in response to [Dorigo Jones+ 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...959...49D/abstract).

If you have any problems running the code please raise an issue to discuss.

## Repository Contents

There are a few codes needed to run the analysis in the paper and they are:

- ares_fiducial_model.py
    - This generates the noisy versions of the fiducial ares model. This will
    overwrite the existing saved models if you run it again so be careful.
- train_emulators.py 
    - This trains the emulators on the ARES training data available from
    Zenodo, evaluates its performance on the corresponding test data
    and then creates a plot showing the accuracy. The code is set up so that
    different emulators with different combinations of the preprocessing steps
    outlined in the globalemu paper can be easily turned on and off. We 
    recommend having all of the preprocessing steps turned on.
- ares_test.py
    - This shows how to run ARES and generates a pickled file called
    'base_pars.pkl' which is used in other codes. It contains the ARES
    parameters for the fiducial model. This isn't strictly necessary as the
    base parameters can be loaded in again from the ARES library using 
    `pars = ares.util.ParameterBundle('mirocha2017:base')` but it is
    quite helpful to have.
- emulator_accuracy_comparison.py
    - Generates figure 1 in the paper showing how the emulator accuracy
    varies over redshift and with the various preprocessing steps turned on and
    off.
- polychord_call.py
    - This runs the polychord fits using ARES and globalemu to model the fiducial
    signal. There are various toggles that can be turned on and off. They
    determine which noise level is used (`nv`), whether the noise is
    fitted for (`FIXED_NOISE`), whether the emulator with the preprocessing
    on is used (`PROCESSING`) and if ARES or globalemu is used (`ARES`). The
    code saves a set of chains which can then be analysed.
- analysis.py
    - This code takes the chains from `polychord_call.py` and makes nice plots.
    It can plot the posteriors (figures 2, A1 and A2 in the paper). Plot the
    emulator bias metric defined in Dorigo Jones+ 2023 (although we do not use
    this in the paper for the reasons discussed there). It can also calcualte
    the KL divergences between the posteriors evaluated with ARES and globalemu
    and makes figure 3 in the paper.

The file `kappa_HH.txt` is needed by globalemu and is downloaded the first time
the emulator is called if it cannot be found. It is the deexcitation rate for
collisional coupling and is needed for the astrophysics free baseline modelling.

The repository also includes lots of products from the analysis. These include:

- ares_fiducial_model_noise_*/
    - These files contain the chains from polychord. The number corresponds to
    the level of noise in the data, the flag after `ARES` tells you whether
    ARES was used or globalemu and the flag after `FIXED_NOISE` tells you
    whether the noise was fitted or not (True means noise not fitted).
- ares_fiducial_model_noise_*.txt
    - These are the saved realisations of the fiducial signal with different
    levels of Gaussian random noise. The number tells you the standard
    deviation of the noise in mK.
- *_maf_ARES_\*.pkl
    - These are the normalising flows used for the calcualtion of the KL 
    divergence. Flags are as above.
- posterior_ares_fiducial_model_noise_*.png
    - These are the posterior comparisons for the different levels of noise in
    the data.
- accuracy_comparison_ares_emulators.png
    - This is figure 1 in the paper which shows the accuracy of the 
    emulator as a function of redshift with and without the preprocessing steps
    outlined in the globalemu paper.

The training and test data sets used for the emulators are available on
Zenodo at ... The signals have been interpolated onto a redshift range of 6 - 55 in steps
of 0.1.

The file in `ares/` tells ARES where to find certain look up tables that it
needs to run. 

To run the code you will need to install the correct packages. Follow the following
steps (you might have to debug a bit depending on your system)

```bash
python3 -m venv myenv
source myenv/bin/activate
bash environment.sh
```

