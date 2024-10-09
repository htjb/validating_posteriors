import numpy as np
import matplotlib.pyplot as plt
from globalemu.eval import evaluate
import matplotlib

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

z = np.arange(6, 55, 0.1)
parameters = np.loadtxt('signal_data/test_data.txt')
labels = np.loadtxt('signal_data/test_labels.txt')

def accuracy(z, parameters, labels):
    sig, _ = predictor(parameters)
    rmse = np.sqrt(np.mean((sig-labels)**2))
    return np.abs((sig-labels)), rmse

emulator = ['emulators/with_AFB_resampling', 
            #'with_AFB_only', 
            #'with_resampling_only', 
            #'emulators/no_AFB_no_resampling',
            'emulators/no_AFB_no_resampling']
label = ['AFB + resampling', 'No Preprocessing']
fig, axes= plt.subplots(1, 1, figsize=(6.3, 3))
for i,e in enumerate(emulator):

    predictor = evaluate(base_dir=e + '/', logs=[0, 2, 4, 5])

    results = [accuracy(z, p, labels[i]) for i, p in enumerate(parameters)]
    rba, rmse = [], []
    for j in range(len(results)):
        rba.append(results[j][0])
        rmse.append(results[j][1])
    axes.plot(z, np.mean(rba, axis=0), label='Mean - '+label[i], color='C'+str(i))
    axes.plot(z, np.percentile(rba, 95, axis=0), linestyle='--', label='95\% - '+ label[i], color='C'+str(i))

    print(np.min(rmse))
    print(np.mean(rmse))
    print(np.max(rmse))

axes.set_ylim(0, 12)

axes.fill_between(np.arange(15, 36, 1), plt.ylim()[0], plt.ylim()[1], color='yellow', alpha=0.2)
axes.fill_between(np.arange(6, 11, 1), plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
axes.fill_between(np.arange(10, 16, 1), plt.ylim()[0], plt.ylim()[1], color='orange', alpha=0.2)
axes.fill_between(np.arange(35, 56, 1), plt.ylim()[0], plt.ylim()[1], color='grey', alpha=0.2)
axes.text(8, 11, 'EoR\n' + r'$f_{esc}$' + '\n' + r'$\log N_{HI}$', fontsize=8,
          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
          horizontalalignment='center',
          verticalalignment='center')
axes.text(13, 8, 'Heating\n' + r'$c_x$', fontsize=8,
          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
          horizontalalignment='center',
          verticalalignment='center')
axes.text(25, 8, 'CD\n' + r'$f_*, M_p, \gamma_{lo}$' + '\n' +
           r'$\gamma_{hi}, T_{min}$', fontsize=8,
           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
           horizontalalignment='center',
           verticalalignment='center')
axes.text(45, 3, 'Dark Ages', fontsize=8,
          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
          horizontalalignment='center',
          verticalalignment='center')
axes.grid()
axes.set_xlim(6, 55)
axes.legend()#loc='upper left', bbox_to_anchor=(0.05, 1.6), fontsize=12)
axes.set_xlabel('z')
axes.set_ylabel(r'$|T_{21} - T_{21}^{\rm{emu}}|$ [mK]')
plt.tight_layout()
plt.savefig('accuracy_comparison_ares_emulators.png', dpi=300, bbox_inches='tight')
plt.show()