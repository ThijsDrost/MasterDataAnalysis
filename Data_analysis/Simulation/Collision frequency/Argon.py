from General.simulation.comsol.ComsolAnalyzer import ComsolAnalyzer
from General.simulation.bolsig.Bolsig_plus import BolsigRuns

loc_comsol = r"C:\Users\20222772\Downloads\Output1.txt"
loc_bolsig = r"C:\Users\20222772\Downloads\output.dat"

comsol_data = ComsolAnalyzer.read_txt(loc_comsol, delimiter=',')
cross_sections = BolsigRuns.read_file(loc_bolsig)


plot_kwargs = {'xlabel': 'Energy [eV]', 'ylabel': 'Cross section [m^2]', 'xscale': 'log', 'yscale': 'log'}
fig, ax = comsol_data.plot_vs_vars('be.ebar', 'be.k', show=False, close=False, plot_kwargs=plot_kwargs)
line_kwargs = {'color': 'k', 'linestyle': '--'}
cross_sections.plot_cross_section_with_energy(show=True, plot_kwargs=plot_kwargs, fig_ax=(fig, ax), line_kwargs=line_kwargs)
