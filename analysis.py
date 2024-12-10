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

nvs = [5, 25, 50, 250]
FIXED_NOISE = True
PLOT_POSTERIOR = False
PLOT_BIAS = False
CALCULATE_KL = True

if FIXED_NOISE:
    latex_names = [r'$\log c_X$', r'$\log f_\mathrm{esc}$', 
             r'$\log T_\mathrm{min}$', r'$\log N_{HI}$', 
             r'$f_*$', r'$M_c$', r'$\gamma_\mathrm{lo}$', 
             r'$\gamma_\mathrm{hi}$']
    names= ['cX', 'fesc', 'Tmin', 'logN', 'fstar', 'Mp', 'gamma_low', 'gamma_high']
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

            kwargs = dict(ncompress=True, lower_kwargs=dict(nplot_2d=5000), alpha=0.8)

            if i == 0:
                ax = chains.plot_2d(names, figsize=(10, 10), 
                                    label='ARES', **kwargs)
            else:
                chains.plot_2d(ax, label='globalemu', **kwargs)

        for i in range(len(ground_truth)):
            ax.axlines({names[i] : ground_truth[i]}, ls='--', color='r', label='Truth')
        ax.iloc[0, 0].legend(loc='upper left', ncols=3, bbox_to_anchor=(2.25, 1.5))
        plt.savefig(f'posterior_ares_fiducial_model_noise_{nv}_ARES_True_FIXED_NOISE_True.png', dpi=300)
        plt.show()
        plt.close()

if PLOT_BIAS:
    fig, axes = plt.subplots(1, 1, figsize=(6.3, 3))
    line = np.array([5, 10, 15, 20])
    for j, nv in enumerate(nvs):
        chains_ares = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_True_FIXED_NOISE_True/test')
        chains_emu = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_False_FIXED_NOISE_True/test')

        emulator_true_bias = []
        for i in range(len(names)):
            emulator_true_bias.append(np.abs((chains_emu[names[i]].mean()
                                            - chains_ares[names[i]].mean()))/chains_ares[names[i]].std())
        print(f'{nv} :', np.mean(emulator_true_bias), np.max(emulator_true_bias))

        width = 0.5
        multiplier= 0
        for i in range(len(emulator_true_bias)):
            offset = width*multiplier
            if j ==0:
                rects = axes.bar(line[j]+offset, emulator_true_bias[i], 
                                width=width, label=latex_names[i], color='C'+str(i))
            else:
                if nv == 250:
                    print(i, line[j]+offset, emulator_true_bias[i])
                axes.bar(line[j]+offset, emulator_true_bias[i], 
                        width=width, color='C'+str(i))
            multiplier += 1

    plt.xticks(line + 3.5*width, ['5', '25', '50', '250'])
    plt.ylim(0.004, 10)
    plt.yscale('log')
    plt.axhline(1, color='k', linestyle='--')
    plt.legend(ncols=2)
    plt.xlabel(r'$\sigma$ [mK]')
    plt.ylabel('Emulator Bias')
    plt.tight_layout()
    plt.savefig('bias_comparison.png', dpi=300,
                bbox_inches='tight')
    plt.show()

if CALCULATE_KL:
    from margarine.maf import MAF
    from margarine.marginal_stats import calculate
    import tensorflow as tf

    kls = []
    for j, nv in enumerate(nvs):
        z, dT_obs = np.loadtxt('ares_fiducial_model_noise_%d.txt' % nv, unpack=True)

        ares = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_True_FIXED_NOISE_True/test')
        emu = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_False_FIXED_NOISE_True/test')
        ares = ares.compress(5000)
        emu = emu.compress(5000)

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

        maf_ares = MAF(ares_samples,
                       lr=1e-3, number_networks=5, hidden_layers=[250])
        maf_emu = MAF(emu_samples,
                      lr=1e-3, number_networks=5, hidden_layers=[250])

        try:
            maf_ares = MAF.load(f'{nv}_maf_ARES_True_FIXED_NOISE_True.pkl')
            maf_emu = MAF.load(f'{nv}_maf_ARES_False_FIXED_NOISE_True.pkl')
        except FileNotFoundError:
            maf_ares.train(1000, early_stop=True)
            maf_emu.train(1000, early_stop=True)
            maf_ares.save(f'{nv}_maf_ARES_True_FIXED_NOISE_True.pkl')
            maf_emu.save(f'{nv}_maf_ARES_False_FIXED_NOISE_True.pkl')

        stats = calculate(maf_ares, prior_de=maf_emu, 
                          samples=maf_ares.sample(nsamples)).statistics()
        print(stats)
        print(f'{nv} :', stats['KL Divergence'], ' - ',
              stats['KL Divergence'] - stats['KL Lower Bound'], ' + ',
              stats['KL Upper Bound'] - stats['KL Divergence'])
        
        kls.append([stats['KL Divergence'], 
                    stats['KL Divergence'] - stats['KL Lower Bound'],
                    stats['KL Upper Bound'] - stats['KL Divergence']])

    nd = len(np.arange(6, 55, 0.1))

    noise = np.linspace(1, 260, 200)
    Dkl = np.linspace(0, 10, 200)

    X, Y = np.meshgrid(noise, Dkl)

    limit = np.sqrt(2*Y/nd)*X

    fig, axes = plt.subplots(1, 1, figsize=(6.3, 3))
    cb = axes.contourf(X, Y, limit, cmap='Blues', levels=10)
    plt.colorbar(cb, label='Emulator RMSE [mK]')
    axes.set_xlabel(r'Noise $\sigma$ [mK]')
    axes.set_ylabel(r'KL-Divergence $D_{KL}$ [nats]')

    actual_rmse = np.array([0.99, 3.14])
    label= ['_', 'Mean', '_']
    col = ['r', 'g', 'r']
    for i,rmse in enumerate(actual_rmse):
        axes.plot(noise, nd/2*(rmse/noise)**2, 
                    label=f'{label[i]} = {rmse} mK', ls='--',
                    color=col[i])
    axes.set_ylim(0, 10)

    [axes.axvline(l,color='k', ls=':', ymin=0, ymax=5) for l in [5, 25, 50, 250]]

    for i in range(len(nvs)):
        axes.errorbar(nvs[i], kls[i][0]
                        , yerr=np.array([kls[i][1], kls[i][2]]).reshape(2, 1)
                        , fmt='x', label=f'Noise = {nvs[i]} mK',
                        c='C4')

    plt.tight_layout()
    plt.savefig('kl_divergence.png', dpi=300, bbox_inches='tight')

    plt.show()