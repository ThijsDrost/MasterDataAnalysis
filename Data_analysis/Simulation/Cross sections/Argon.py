import matplotlib.pyplot as plt

from General.simulation.CrossSections import CrossSectionData, plot_CrossSections, plot_averageCrossSections
from General.plotting import linestyles

image_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Simulations\Cross sections'

loc = r"E:\OneDrive - TU Eindhoven\Master thesis\Cross sections\Ar+e_Ar++2e.txt"
data = CrossSectionData.read_txt(loc)
plot_CrossSections(data, show=True, plot_kwargs={'ylim': 1e-23}, line_kwargs={'marker': '.', 'linestyle': '-'})
selected = data.select_by_name(starts_with='TRINITI', inverted=True)

line_kwargs_iter = linestyles.linelook_by([dat.database_simplified for dat in selected], linestyles=False, colors=True)
plot_CrossSections(selected, show=True, plot_kwargs={'ylim': 1e-23}, line_kwargs_iter=line_kwargs_iter)

plot_averageCrossSections(selected, show=True, plot_kwargs={'ylim': (1e-21, 1e-20), 'xlim': (1e3, 1e4)})
plot_averageCrossSections(selected, show=True)

# %%
loc = r"E:\OneDrive - TU Eindhoven\Master thesis\Cross sections\Ar_elastic.txt"
data = CrossSectionData.read_txt(loc)
plot_kwargs = [{'linestyle': '--', 'color': 'C0', 'marker': None} if data._selector_range(x_range=(0.2, 0.3), y_range=(2.5e-21, None))(dat) else {'linestyle': '-', 'color': 'C1', 'marker': None}
               for dat in data]
legend_kwargs = {'handles': [plt.Line2D([0], [0], **kwargs) for kwargs in [{'linestyle': '--', 'color': 'C0'}, {'linestyle': '-', 'color': 'C1'}]],
                 'labels': ['Upper', 'Lower']}
plot_CrossSections(data, rotate_markers=True, show=True, plot_kwargs={'ylim': 1e-22, 'xlim': (1e-3, 1e7)}, line_kwargs_iter=plot_kwargs,
                   legend_kwargs=None, save_loc=f'{image_loc}/Ar_elastic.pdf')

selected = data.select_by_values(x_range=(0.2, 0.3), y_range=(None, 2.5e-21))
selected = selected.select_by_name(starts_with=('Puech', 'SIGLO', 'Phelps'), inverted=True)
plot_averageCrossSections(selected, show=True, plot_kwargs={'ylim': 1e-22, 'xlim': (1e-3, 1e4)})
