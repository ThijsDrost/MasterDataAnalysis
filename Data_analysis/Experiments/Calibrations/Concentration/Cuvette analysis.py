import matplotlib.pyplot as plt

from General.Analysis import CalibrationAnalyzer
from General.Data_handling import drive_letter
from General.Plotting import Names

save_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Results\Calibrations'

base_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration'
NO2_loc = rf'{base_loc}\NO2 cuvette\data.hdf5'
NO3_loc = rf'{base_loc}\NO3 cuvette\data.hdf5'
H2O2_loc = rf'{base_loc}\H2O2 cuvette\data.hdf5'

NO2_analyzer = CalibrationAnalyzer.standard(NO2_loc, 'NO2-', f'{Names.NO2} [mM]')
NO3_analyzer = CalibrationAnalyzer.standard(NO3_loc, 'NO3-', f'{Names.NO3} [mM]')
H2O2_analyzer = CalibrationAnalyzer.standard(H2O2_loc, 'H2O2', f'{Names.H2O2} [mM]')

NO2_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (200, 300)}, relative=True, save_loc=save_loc, save_suffix='_NO2_concentration_cuvette_200-300.pdf')
NO3_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (200, 300)}, relative=True, save_loc=save_loc, save_suffix='_NO3_concentration_cuvette_200-300.pdf')
H2O2_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (200, 300)}, relative=True, save_loc=save_loc, save_suffix='_H2O2_concentration_cuvette_200-300.pdf')

NO2_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (300, 400)}, save_loc=save_loc, save_suffix='_NO2_concentration_cuvette_300-400.pdf')
NO3_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (300, 400)}, save_loc=save_loc, save_suffix='_NO3_concentration_cuvette_300-400.pdf')
H2O2_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (300, 400)}, save_loc=save_loc, save_suffix='_H2O2_concentration_cuvette_300-400.pdf')

fig_ax = NO2_analyzer.pearson_r_vs_wavelength(close=False, show=False, labels=(Names.NO2,), colors=('C0',))
NO3_analyzer.pearson_r_vs_wavelength(close=False, show=False, fig_ax=fig_ax, labels=(Names.NO3,), colors=('C1',))
H2O2_analyzer.pearson_r_vs_wavelength(show=True, plot_kwargs={'xlim': (200, 300), 'ylim': (0.99, 1)}, close=False, fig_ax=fig_ax, labels=(Names.H2O2,),
                                      legend_kwargs={'title': 'Specie'}, colors=('C2',), save_loc=save_loc, save_suffix='_pearson_r_cuvette.png')

fig_ax = NO2_analyzer.one_minus_pearson_r_vs_wavelength(close=False, show=False, labels=(Names.NO2,), colors=('C0',))
NO3_analyzer.one_minus_pearson_r_vs_wavelength(close=False, show=False, fig_ax=fig_ax, labels=(Names.NO3,), colors=('C1',))
H2O2_analyzer.one_minus_pearson_r_vs_wavelength(show=True, plot_kwargs={'xlim': (200, 300), 'ylim': (1e-10, 1), 'yscale': 'log'}, close=False, fig_ax=fig_ax,
                                                labels=(Names.H2O2,), legend_kwargs={'title': 'Specie'}, colors=('C2',), save_loc=save_loc,
                                                save_suffix='_one_minus_pearson_r_cuvette.png')


