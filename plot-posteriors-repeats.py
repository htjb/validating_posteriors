import numpy as np
from anesthetic import read_chains
import anesthetic as ac
import matplotlib.pyplot as plt
from matplotlib import gridspec

names = ["cX", "fesc", "Tmin", "logN",
         "fstar", "Mp", "gamma_low", "gamma_high"]
labels = [r'$\log c_X$', r'$\log f_\mathrm{esc}$',
          r'$\log T_\mathrm{min}$', r'$\log N_{HI}$',
          r'$\log f_*$', r'$\log M_c$', r'$\gamma_\mathrm{lo}$',
          r'$\gamma_\mathrm{hi}$']

chains = read_chains(
    "ares_fiducial_model_noise_25_ARES_False_FIXED_NOISE_True/test")
chainsrepeate = read_chains(
    "ares_fiducial_model_noise_25_ARES_False_FIXED_NOISE_True_take2/test"
)

chains = chains.compress()
chainsrepeate = chainsrepeate.compress()

chainsares = read_chains(
    "ares_fiducial_model_noise_25_ARES_True_FIXED_NOISE_True/test")
chainsaresrepeate = read_chains(
    "ares_fiducial_model_noise_25_ARES_True_FIXED_NOISE_True_take2/test"
)

chainsares = chainsares.compress()
chainsaresrepeate = chainsaresrepeate.compress()

gs = gridspec.GridSpec(1, 2)

fig, ax1 = ac.make_2d_axes(names, figsize=(9, 4.5), subplot_spec=gs[0])
fig, ax2 = ac.make_2d_axes(names, fig=fig, subplot_spec=gs[1])

ax1.iloc[0, 3].set_title(r"     globalemu")
ax2.iloc[0, 3].set_title(r"     ARES")

chains["cX"] = np.log10(chains["cX"])
chains["Mp"] = np.log10(chains["Mp"])
chains["fstar"] = np.log10(chains["fstar"])
chains["Tmin"] = np.log10(chains["Tmin"])

chainsrepeate["cX"] = np.log10(chainsrepeate["cX"])
chainsrepeate["Mp"] = np.log10(chainsrepeate["Mp"])
chainsrepeate["fstar"] = np.log10(chainsrepeate["fstar"])
chainsrepeate["Tmin"] = np.log10(chainsrepeate["Tmin"])

chains.plot_2d(ax1, label="Run 1", upper_kwargs=dict(markersize=0.1))
chainsrepeate.plot_2d(ax1, label="Run 2", upper_kwargs=dict(markersize=0.1))

ax1.iloc[-1, 2].set_xticks([3.2, 4.0])

chainsares["cX"] = np.log10(chainsares["cX"])
chainsares["Mp"] = np.log10(chainsares["Mp"])
chainsares["fstar"] = np.log10(chainsares["fstar"])
chainsares["Tmin"] = np.log10(chainsares["Tmin"])

chainsaresrepeate["cX"] = np.log10(chainsaresrepeate["cX"])
chainsaresrepeate["Mp"] = np.log10(chainsaresrepeate["Mp"])
chainsaresrepeate["fstar"] = np.log10(chainsaresrepeate["fstar"])
chainsaresrepeate["Tmin"] = np.log10(chainsaresrepeate["Tmin"])

chainsares.plot_2d(ax2, label="Run 1", upper_kwargs=dict(markersize=0.1))
chainsaresrepeate.plot_2d(
    ax2, label="Run 2", upper_kwargs=dict(markersize=0.1))

for i in range(len(labels)):
    [ax1.iloc[-1, j].set_xlabel(labels[j], rotation=45)
     for j in range(len(labels))]
    [ax1.iloc[j, 0].set_ylabel(labels[j], rotation=45)
     for j in range(len(labels))]
    [ax2.iloc[-1, j].set_xlabel(labels[j], rotation=45)
     for j in range(len(labels))]
    [ax2.iloc[j, 0].set_ylabel(labels[j], rotation=45)
     for j in range(len(labels))]
    [ax1.iloc[-1, j].set_xticklabels(ax1.iloc[-1, j].get_xticks(),
                                     rotation=90) for j in range(len(labels))]
    [ax2.iloc[-1, j].set_xticklabels(ax2.iloc[-1, j].get_xticks(),
                                     rotation=90) for j in range(len(labels))]


plt.savefig(
    "check-emulator-consistency-ares_fiducial_model" +
    "_noise_25_ARES_False_FIXED_NOISE_True.png",
    dpi=300, bbox_inches="tight")
plt.show()
