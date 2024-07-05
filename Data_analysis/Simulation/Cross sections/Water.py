import numpy as np
import matplotlib.pyplot as plt

from General.simulation.CrossSections import CrossSectionData, plot_CrossSections

loc = r"E:\OneDrive - TU Eindhoven\Master thesis\Cross sections\H2O+e_H2O++2e.txt"
data = CrossSectionData.read_txt(loc)
plot_CrossSections(data, show=True, plot_kwargs={'ylim': 1e-23})
plot_CrossSections(data, show=True, plot_kwargs={'xlim': (10, 20), 'xscale': 'linear'})
sel_data = [dat for dat in data if dat.database_simplified not in ('TRINITI',)]
fig, ax = plot_CrossSections(sel_data, show=False, close=False, plot_kwargs={'ylim': 1e-23})

min_energy, max_energy = min(min(dat.energy) for dat in data), max(max(dat.energy) for dat in data)
energies = np.linspace(1, max_energy, 1000)
def cross_section(energy): return 9.65e-18 * (energy**2.53) * np.exp(-8.99/energy)


ax.plot(energies, cross_section(energies), label='Fit', color='black', linestyle='--')
ax.legend()
ax.set_ylim(1e-23, 1e-10)
plt.show()

