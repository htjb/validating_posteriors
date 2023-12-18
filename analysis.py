import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains

ground_truth = [2e39, 0.2, 1e4, 21, 0.05, 2.8e11, 0.49, -0.61]

ground_truth[0] = np.log10(ground_truth[0])
ground_truth[5] = np.log10(ground_truth[5])
ground_truth[4] = np.log10(ground_truth[4])
ground_truth[2] = np.log10(ground_truth[2])

nvs = [5, 25, 50]
FIXED_NOISE = True
PLOT_POSTERIOR = False
PLOT_BIAS = True
CALCULATE_KL = False

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

    for j, nv in enumerate(nvs):
        ares = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_True_FIXED_NOISE_True/test')
        emu = read_chains(f'ares_fiducial_model_noise_{nv}_ARES_False_FIXED_NOISE_True/test')
        #ares= chains_ares.compress(5000)
        #emu = chains_emu.compress(5000)

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

        maf_ares = MAF(ares_samples)
        maf_emu = MAF(emu_samples)

        try:
            maf_ares = MAF.load(f'{nv}_maf_ARES_True_FIXED_NOISE_True.pkl')
            maf_emu = MAF.load(f'{nv}_maf_ARES_False_FIXED_NOISE_True.pkl')
        except FileNotFoundError:
            maf_ares.train(10000, early_stop=True)
            maf_emu.train(10000, early_stop=True)
            maf_ares.save(f'{nv}_maf_ARES_True_FIXED_NOISE_True.pkl')
            maf_emu.save(f'{nv}_maf_ARES_False_FIXED_NOISE_True.pkl')

        stats = calculate(maf_emu, prior_de=maf_ares).statistics()
        print(f'{nv} :', stats['KL Divergence'], ' - ',
              stats['KL Divergence'] - stats['KL Lower Bound'], ' + ',
              stats['KL Upper Bound'] - stats['KL Divergence'])
