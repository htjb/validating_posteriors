import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
import matplotlib as mpl

# figure formatting
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', 
     '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

ares_emulator = 'emulators/with_AFB_resampling/'
zeus_emulator = 'zeus_emulators/with_AFB_resampling/'
ares_data = 'signal_data/'
zeus_data = 'zeus21_train_test/'

zeus_std = np.load(zeus_emulator + 'labels_stds.npy')
ares_std = np.load(ares_emulator + 'labels_stds.npy')
print(zeus_std, ares_std)
ares_z = np.arange(6, 55, 0.1)
zeus_z = np.arange(10, 35, 0.1)

ares_AFB = np.loadtxt(ares_emulator + 'AFB.txt')
ares_resample = np.loadtxt(ares_emulator + 'samples.txt')

zeus_AFB = np.loadtxt(zeus_emulator + 'AFB.txt')
zeus_resample = np.loadtxt(zeus_emulator + 'samples.txt')

ares_td = np.loadtxt(ares_data + 'test_data.txt')
ares_tl = np.loadtxt(ares_data + 'test_labels.txt')

zeus_td = np.loadtxt(zeus_data + 'test_data.txt')
zeus_tl = np.loadtxt(zeus_data + 'test_labels.txt')

fig, axes = plt.subplots(2, 3, figsize=(7, 3.5),
                         sharey='col')
ax = axes.flatten()

[ax[0].plot(ares_z, s, alpha=1, lw=0.8) for i, s in enumerate(ares_tl[:100])]
[ax[3].plot(zeus_z, s, alpha=1, lw=0.8) for i, s in enumerate(zeus_tl[:100])]

[ax[1].plot(ares_z, s - ares_AFB, alpha=1, lw=0.8) for i, s in enumerate(ares_tl[:100])]
[ax[4].plot(zeus_z, s - zeus_AFB, alpha=1, lw=0.8) for i, s in enumerate(zeus_tl[:100])]

ares_interp = [np.interp(ares_resample, ares_z, s-ares_AFB) for s in ares_tl[:100]]
# plots the resamples signals with even spaced x ticks
# even if the resampled points are not evenly spaced
[ax[2].plot(np.arange(0, len(ares_resample), 1),
    ares_interp[i]/ares_std, 
    alpha=1, lw=0.8) for i in range(len(ares_interp))]

idx = []
for i in range(len(ares_resample)):
    if np.any(np.isclose(ares_resample[i], 
                         [10, 20, 29.96, 35, 50], 
                         rtol=2e-3, atol=2e-3)):
        idx.append(i)
idx.append(len(ares_resample)-1)

ax[2].set_xticks(idx)
ax[2].set_xticklabels([
    '{:.0f}'.format(a) for a in ares_resample[idx]],
    rotation=45)

zeus_interp = [np.interp(zeus_resample, zeus_z, s-zeus_AFB) for s in zeus_tl[:100]]
[ax[5].plot(np.arange(0, len(zeus_resample), 1),
    zeus_interp[i]/zeus_std, 
    alpha=1, lw=0.8) for i in range(len(zeus_interp))]

idx = []
for i in range(len(zeus_resample)):
    if np.any(np.isclose(zeus_resample[i], 
                         [20], 
                         rtol=2e-3, atol=2e-3)):
        idx.append(i)
idx.append(0)
idx.append(len(zeus_resample)-1)

ax[5].set_xticks(idx)
ax[5].set_xticklabels([
    '{:.0f}'.format(a) for a in zeus_resample[idx]],
    rotation=45)

ax[0].set_ylabel(r'$ T_{21}$ [mK]')
ax[3].set_xlabel(r'$z$')
ax[3].set_ylabel(r'$ T_{21}$ [mK]')
ax[4].set_xlabel(r'$z$')
ax[5].set_xlabel(r'$z$')
ax[1].set_ylabel(r'$ T_{21} - T_{AFB}$ [mK]')
ax[4].set_ylabel(r'$ T_{21} - T_{AFB}$ [mK]')
ax[5].set_ylabel(r'$\frac{T_{21} - T_{AFB}}{\sigma_{T_{21}}}$')
ax[2].set_ylabel(r'$\frac{T_{21} - T_{AFB}}{\sigma_{T_{21}}}$')

ax[5].set_xlim(0, len(zeus_resample)-1)
ax[2].set_xlim(0, len(ares_resample)-1)
for i in range(len(axes)):
    for j in range(axes.shape[-1]):
        if j !=2:
            axes[i, j].set_xlim(6, 55)
            axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(), rotation=45)
        axes[i, j].grid()

plt.tight_layout()
#plt.subplots_adjust(wspace=0.05)
plt.savefig('preprocessing.png', dpi=300)
plt.show()