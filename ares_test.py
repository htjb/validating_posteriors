import ares
import matplotlib.pyplot as plt

sim = ares.simulations.Global21cm()
sim.run()

sim.GlobalSignature()
plt.show()