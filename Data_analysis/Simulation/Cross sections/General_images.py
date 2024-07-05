import matplotlib.pyplot as plt
import numpy as np

from General.simulation.CrossSections import CrossSectionData, plot_CrossSections
from General.plotting import linestyles

image_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Simulations\Cross sections'
loc = r"E:\OneDrive - TU Eindhoven\Master thesis\Cross sections\Ar_elastic.txt"
data = CrossSectionData.read_txt(loc)

bigger = data.select_by_values(x_range=(0.2, 0.3), y_range=(2.5e-21, None))
selected = data.select_by_values(x_range=(0.2, 0.3), y_range=(None, 2.5e-21))

names = [dat.database_simplified for dat in data]
line_kwargs = {name: look for (name, look) in zip(names, linestyles.linelook_by(names, markers=True, colors=True))}

line_kwargs_iter = [line_kwargs[dat.database_simplified] for dat in selected]
plot_CrossSections(selected, show=True, plot_kwargs={'ylim': 1e-22, 'xlim': (1e-3, 1e4)}, rotate_markers=True, legend_kwargs=None,
                   line_kwargs_iter=line_kwargs_iter, save_loc=f'{image_loc}/Selection_0.pdf')

selected = selected.select_by_name(starts_with='Puech', inverted=True)
selected = selected.select_by_name(starts_with='SIGLO', inverted=True)
selected = selected.select_by_name(starts_with='Phelps', inverted=True)

line_kwargs_iter = [line_kwargs[dat.database_simplified] for dat in selected]
plot_CrossSections(selected, show=True, plot_kwargs={'ylim': 1e-22, 'xlim': (1e-3, 1e4)}, rotate_markers=True, legend_kwargs=None,
                   line_kwargs_iter=line_kwargs_iter, save_loc=f'{image_loc}/Selection_1.pdf')

diff_data = selected.diff(normalize=False)
line_kwargs_iter = [line_kwargs[dat.database_simplified] for dat in selected]
plot_CrossSections(diff_data, show=True, close=False, plot_kwargs={'yscale': 'linear', 'xlim': (10, 1e2), 'ylim': (-1e-20, 1e-20)}, legend_kwargs=None,
                   line_kwargs_iter=line_kwargs_iter, save_loc=f'{image_loc}/Selection_diff.pdf')

selected = selected.select_by_name(starts_with='Biagi', inverted=True)
selected = selected.select_by_name(starts_with='BSR', inverted=True)

line_kwargs_iter = [line_kwargs[dat.database_simplified] for dat in selected]
plot_CrossSections(selected, show=True, plot_kwargs={'ylim': 1e-22, 'xlim': (1e-3, 1e4)}, rotate_markers=True, legend_kwargs=None,
                   line_kwargs_iter=line_kwargs_iter, save_loc=f'{image_loc}/Selection_2.pdf')
print(line_kwargs)
line_kwargs = {'color': 'C0', 'linestyle': '-', 'marker': None}
fig, ax = plot_CrossSections(selected, show=False, close=False, plot_kwargs={'ylim': 1e-22, 'xlim': (1e-3, 1e4)}, legend_kwargs=None, line_kwargs=line_kwargs)
energies = np.logspace(-3, 4, 1000)
average_erf = selected.average(energies, transition_size=100, transition_type='erf')
ax.plot(energies, average_erf, 'k--')
legend_kwargs = {'handles': [plt.Line2D([0], [0], **line_kwargs), plt.Line2D([0], [0], color='k', linestyle='--')],
                 'labels': ['Values', 'Average']}
plt.legend(**legend_kwargs)
plt.savefig(f'{image_loc}/Selection_interp.pdf')
plt.show()

average = selected.average(energies, transition_type='none')

energies_diff = np.average([energies[1:], energies[:-1]], axis=0)

plt.figure()
plt.plot(energies_diff, np.diff(average_erf), '.-', label='Smoothing')
plt.plot(energies_diff, np.diff(average), '.', label='No smoothing')
plt.legend()
plt.xscale('log')
plt.xlim(1e-3, 1e4)
plt.legend()
plt.savefig(f'{image_loc}/Selection_smoothing.pdf')
plt.show()


selected_biagi = data.select_by_values(x_range=(0.2, 0.3), y_range=(None, 2.5e-21))
selected_biagi = selected_biagi.select_by_name(starts_with='Biagi', inverted=False)
diff_biagi = selected_biagi.diff(normalize=False)
plot_CrossSections(diff_biagi, show=True, close=False, plot_kwargs={'yscale': 'linear', 'xlim': (10, 1e3), 'ylim': (-1e-20, 1e-20)}, legend_kwargs=None,
                   save_loc=f'{image_loc}/Selection_interp.pdf')
