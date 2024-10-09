import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains

ground_truth = [2e39, 0.2, 1e4, 21, 0.05, 2.8e11, 0.49, -0.61]

ground_truth[0] = np.log10(ground_truth[0])
ground_truth[5] = np.log10(ground_truth[5])
ground_truth[4] = np.log10(ground_truth[4])
ground_truth[2] = np.log10(ground_truth[2])

prior_lower = [36, 0, np.log10(3e2), 18, -5, 8, 0, -4]
prior_upper = [41, 1, np.log10(5e5), 23, 0, 15, 2, 0]

nvs = [5, 25]
FIXED_NOISE = True
PLOT_POSTERIOR = False
PLOT_BIAS = False
CALCULATE_KL = True

if FIXED_NOISE:
    names = ['cX', 'fesc', 'Tmin', 'logN', 'fstar', 'Mp', 'gamma_low', 'gamma_high']
else:
    names = ['cX', 'fesc', 'Tmin', 'logN', 'fstar', 'Mp', 'gamma_low', 'gamma_high', 'noise']

if PLOT_POSTERIOR:
    for j, nv in enumerate(nvs):
        ARES= [True, False]
        for i, a in enumerate(ARES):

            chains = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_{a}_FIXED_NOISE_{FIXED_NOISE}/test')
            chains['cX'] = np.log10(chains['cX'])
            chains['Mp'] = np.log10(chains['Mp'])
            chains['fstar'] = np.log10(chains['fstar'])
            chains['Tmin'] = np.log10(chains['Tmin'])

            if i == 0:
                ax = chains.plot_2d(names, figsize=(10, 10), label='ARES')
            else:
                chains.plot_2d(ax, label='globalemu')

        for i in range(len(ground_truth)):
            ax.axlines({names[i] : ground_truth[i]}, ls='--', color='r', label='Truth')
        ax.iloc[0, 0].legend(loc='upper left', ncols=3, bbox_to_anchor=(2.25, 1.5))
        plt.savefig(f'posterior_ares_fiducial_model_noise_{nv}_ARES_True_FIXED_NOISE_True.png', dpi=300)
        plt.show()
        plt.close()

if PLOT_BIAS:
    for j, nv in enumerate(nvs):
        chains_ares = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_True_FIXED_NOISE_True/test')
        chains_emu = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_False_FIXED_NOISE_True/test')

        emulator_true_bias = []
        for i in range(len(names)):
            emulator_true_bias.append(np.abs((chains_emu[names[i]].mean()
                                            - chains_ares[names[i]].mean()))/chains_ares[names[i]].std())
        print(f'{nv} :', np.mean(emulator_true_bias), np.max(emulator_true_bias))

        line = np.array([5, 10, 15])
        width = 0.5
        multiplier= 0
        for i in range(len(emulator_true_bias)):
            offset = width*multiplier
            if j ==0:
                rects = plt.bar(line[j]+offset, emulator_true_bias[i], width=width, label=names[i], color='C'+str(i))
                #plt.bar_label(rects, padding=3, rotation=90)
            else:
                plt.bar(line[j]+offset, emulator_true_bias[i], width=width, color='C'+str(i))
                #plt.bar_label(rects, padding=3, rotation=90)
            multiplier += 1

    plt.xticks(line + 3.5*width, ['5', '25', '50'])
    plt.ylim(0.01, 10)
    plt.yscale('log')
    plt.axhline(1, color='k', linestyle='--')
    plt.legend(ncols=2)
    plt.xlabel('Noise')
    plt.ylabel('Bias')
    plt.savefig('bias_comparison.png', dpi=300)
    plt.show()

if CALCULATE_KL:
    from margarine.maf import MAF
    from margarine.marginal_stats import calculate
    import tensorflow as tf

    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    for j, nv in enumerate(nvs):
        z, dT_obs = np.loadtxt('ares_fiducial_model_noise_%d.txt' % nv, unpack=True)

        ares = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_True_FIXED_NOISE_True/test')
        emu = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_False_FIXED_NOISE_True/test')
        ares = ares.compress(15000)
        emu = emu.compress(15000)

        ares['cX'] = np.log10(ares['cX'])
        ares['Mp'] = np.log10(ares['Mp'])
        ares['fstar'] = np.log10(ares['fstar'])
        ares['Tmin'] = np.log10(ares['Tmin'])
        emu['cX'] = np.log10(emu['cX'])
        emu['Mp'] = np.log10(emu['Mp'])
        emu['fstar'] = np.log10(emu['fstar'])
        emu['Tmin'] = np.log10(emu['Tmin'])

        ares_samples = ares[names].values.astype(np.float32)
        emu_samples = emu[names].values.astype(np.float32)
        print(ares_samples.shape, emu_samples.shape)

        nsamples = int(np.min([ares_samples.shape[0], emu_samples.shape[0]]))
        ares_samples = ares_samples[:nsamples]
        emu_samples = emu_samples[:nsamples]

        maf_ares = MAF(ares_samples, weights=ares.get_weights()[:nsamples],
                       theta_min=prior_lower, theta_max=prior_upper)
        maf_emu = MAF(emu_samples, weights=emu.get_weights()[:nsamples],
                      theta_min=prior_lower, theta_max=prior_upper)

        try:
            maf_ares = MAF.load(f'{nv}_maf_ARES_True_FIXED_NOISE_True.pkl')
            maf_emu = MAF.load(f'{nv}_maf_ARES_False_FIXED_NOISE_True.pkl')
        except FileNotFoundError:
            maf_ares.train(10000, early_stop=True)
            maf_emu.train(10000, early_stop=True)
            maf_ares.save(f'{nv}_maf_ARES_True_FIXED_NOISE_True.pkl')
            maf_emu.save(f'{nv}_maf_ARES_False_FIXED_NOISE_True.pkl')

        stats = calculate(maf_ares, prior_de=maf_emu, 
                          samples=maf_ares.sample(nsamples)).statistics()
        print(f'{nv} :', stats['KL Divergence'], ' - ',
              stats['KL Divergence'] - stats['KL Lower Bound'], ' + ',
              stats['KL Upper Bound'] - stats['KL Divergence'])

        nd = len(dT_obs)
        epsilon = np.linspace(0., 0.25, 100)
        min_error = 0.11/nv
        emulator_error = 0.82/nv
        max_error = 2.56/nv
        limit = 0.5*nd*(epsilon)**2
        print(np.array([stats['KL Divergence'] - stats['KL Lower Bound'],
                        stats['KL Upper Bound'] - stats['KL Divergence']]))
        axes[0].errorbar(emulator_error, stats['KL Divergence'],
                      yerr=np.array([stats['KL Divergence'] - stats['KL Lower Bound'],
                        stats['KL Upper Bound'] - stats['KL Divergence']]).reshape(2, 1),
                        #xerr=[[emulator_error - min_error], [max_error - emulator_error]],
                      fmt='x', label=f'Noise = {nv}')
        sigmas = np.linspace(0, 60, 100)
        axes[1].errorbar(nv, emulator_error*nv, 
                         yerr=[[emulator_error*nv - min_error*nv], [max_error*nv - emulator_error*nv]],
                         fmt='x', label=r'$\sigma = $ %.2f mK' % nv)
    axes[1].plot(sigmas, sigmas*np.sqrt(2/nd), c='k')
    axes[1].set_xlabel(r'$\sigma$')
    axes[1].set_ylabel(r'$\mathrm{RMSE}$')
    axes[0].loglog()
    axes[0].plot(epsilon, limit, c='k')
    axes[0].set_xlabel(r'$\mathrm{RMSE}/\sigma$')
    axes[0].set_ylabel(r'$D_{KL}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('kl_divergence.png', dpi=300)
    plt.close()
    #plt.show()

