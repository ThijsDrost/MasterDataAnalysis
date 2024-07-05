from General.simulation.CrossSections import CrossSectionData, plot_CrossSections

loc = r"E:\OneDrive - TU Eindhoven\Master thesis\Cross sections\O2+e_O2++2e.txt"
data = CrossSectionData.read_txt(loc)
plot_CrossSections(data, show=True, plot_kwargs={'ylim': 1e-23})
plot_CrossSections(data, show=True, plot_kwargs={'xlim': (10, 20), 'xscale': 'linear'})
sel_data = [dat for dat in data if dat.database_simplified not in ('Phelps', 'TRINITI', 'Itikawa', 'Morgan')]
plot_CrossSections(sel_data, show=True, plot_kwargs={'ylim': 1e-23})
