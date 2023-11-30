import numpy as np
import matplotlib.pyplot as plt

ares_emulator = 'emulators/with_AFB_resampling/'
ares_data = 'signal_data/'
zeus_data = 'zeus21_train_test/'
ares_z = np.arange(6, 55, 0.1)
zeus_z = np.arange(10, 35, 0.1)

ares_AFB = np.loadtxt(ares_emulator + 'AFB.txt')
ares_resample = np.loadtxt(ares_emulator + 'samples.txt')

ares_td = np.loadtxt(ares_data + 'test_data.txt')
ares_tl = np.loadtxt(ares_data + 'test_labels.txt')

zeus_td = np.loadtxt(zeus_data + 'test_data.txt')
zeus_tl = np.loadtxt(zeus_data + 'test_labels.txt')

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
ax = axes.flatten()

[ax[0].plot(ares_z, s, alpha=0.5) for s in ares_tl[:1000]]
[ax[3].plot(zeus_z, s, alpha=0.5) for s in zeus_tl[:1000]]

[ax[1].plot(ares_z, s - ares_AFB, alpha=0.5) for s in ares_tl[:1000]]


[ax[2].plot(ares_z, np.interp(ares_resample, ares_z, s-ares_AFB), alpha=0.5) for s in ares_tl[:1000]]
ax[2].grid()
print([ares_resample[i] for i in range(len(ares_resample)) if i in [0, 20, 40, 60, 80, 100]])
idx = []
for i in range(len(ares_resample)):
    if ares_resample[i] in [0, 20, 40, 60, 80, 100]:
        idx.append(i)

ax[2].set_xticks(ares_resample[idx])

ax[0].set_xlabel(r'$z$')
ax[0].set_ylabel(r'$\delta T_b$ [mK]')
ax[0].set_xlim(6, 55)
ax[3].set_xlabel(r'$z$')
ax[3].set_ylabel(r'$\delta T_b$ [mK]')
ax[3].set_xlim(6, 55)

plt.show()