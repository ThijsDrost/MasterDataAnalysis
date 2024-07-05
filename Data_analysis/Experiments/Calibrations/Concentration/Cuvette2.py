import numpy as np
import h5py
import os
import shutil

from General.experiments.absorption import CalibrationAnalyzer
from General.import_funcs import drive_letter
from General.plotting import Names
from General.experiments.hdf5.Conversion import make_hdf5

# %%
data_loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_05_31_cali'
NO2_files = ['NO2 0.2g_L.txt', 'NO2 0.4g_L.txt', 'NO2 0.6g_L.txt', 'NO2 0.8g_L.txt', 'NO2 1g_L.txt', ]

dark_loc = rf'{data_loc}\dark.txt'
reference_loc = rf'{data_loc}\reference3.txt'

shutil.rmtree(rf'{data_loc}\NO2')
os.mkdir(rf'{data_loc}\NO2')
concs = []
for index, file in enumerate(NO2_files):
    concs.append(float(file.split(' ')[1].split('g')[0])/46.0055)
    shutil.copy(rf'{data_loc}\{file}', rf'{data_loc}\NO2\{index}_1.txt')
shutil.copy(dark_loc, rf'{data_loc}\NO2\dark.txt')
shutil.copy(reference_loc, rf'{data_loc}\NO2\reference.txt')
with open(rf'{data_loc}\NO2\NO2-.txt', 'w') as file:
    for index, conc in enumerate(concs):
        file.write(f'{index}\t{conc}\n')

NO2_files = ['NO3 0.1g_L.txt', 'NO3 0.4 g_L.txt', 'NO3 0.7 g_L.txt', 'NO3 1g_L.txt']

dark_loc = rf'{data_loc}\dark.txt'
reference_loc = rf'{data_loc}\reference2.txt'

shutil.rmtree(rf'{data_loc}\NO3')
os.mkdir(rf'{data_loc}\NO3')
concs = []
for index, file in enumerate(NO2_files):
    concs.append(float(file.split(' ')[1].split('g')[0])/62.0049)
    shutil.copy(rf'{data_loc}\{file}', rf'{data_loc}\NO3\{index}_1.txt')
shutil.copy(dark_loc, rf'{data_loc}\NO3\dark.txt')
shutil.copy(reference_loc, rf'{data_loc}\NO3\reference.txt')
with open(rf'{data_loc}\NO3\NO3-.txt', 'w') as file:
    for index, conc in enumerate(concs):
        file.write(f'{index}\t{conc}\n')

data_loc2 = rf'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_06_03_cali'
H2O2_files = ['0.1mL.txt', '0.2mL.txt', '0.3mL.txt', '0.4mL.txt', '0.5mL.txt']

dark_loc = rf'{data_loc2}\dark.txt'
reference_loc = rf'{data_loc2}\reference.txt'

shutil.rmtree(rf'{data_loc2}\H2O2')
os.mkdir(rf'{data_loc2}\H2O2')
concs = []
for index, file in enumerate(H2O2_files):
    concs.append(0.098*float(file.split('mL')[0])/2.5)
    shutil.copy(rf'{data_loc2}\{file}', rf'{data_loc2}\H2O2\{index}_1.txt')
shutil.copy(dark_loc, rf'{data_loc2}\H2O2\dark.txt')
shutil.copy(reference_loc, rf'{data_loc2}\H2O2\reference.txt')
with open(rf'{data_loc2}\H2O2\H2O2.txt', 'w') as file:
    for index, conc in enumerate(concs):
        file.write(f'{index}\t{conc}\n')


# %%
make_hdf5(rf'{data_loc}\NO2', 'closest', 'absorbance',
          {'measurement': 'mean', 'dark': 'mean', 'reference': 'mean'}, rf'{data_loc}\NO2\data.hdf5')
make_hdf5(rf'{data_loc}\NO3', 'closest', 'absorbance',
          {'measurement': 'mean', 'dark': 'mean', 'reference': 'mean'}, rf'{data_loc}\NO3\data.hdf5')
make_hdf5(rf'{data_loc2}\H2O2', 'closest', 'absorbance',
          {'measurement': 'mean', 'dark': 'mean', 'reference': 'mean'}, rf'{data_loc2}\H2O2\data.hdf5')
# %%
NO2_analyzer = CalibrationAnalyzer.standard(rf'{data_loc}\NO2\data.hdf5', 'NO2-', f'{Names.NO2} [mM]')
NO3_analyzer = CalibrationAnalyzer.standard(rf'{data_loc}\NO3\data.hdf5', 'NO3-', f'{Names.NO3} [mM]')
H2O2_analyzer = CalibrationAnalyzer.standard(rf'{data_loc2}\H2O2\data.hdf5', 'H2O2', f'{Names.NO3} [mM]')


save_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Results\Calibrations'
base_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration'


# NO2_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (200, 300)}, relative=True, num=None) #, save_loc=save_loc, save_suffix='_NO2_concentration_cuvette_200-300.pdf')
# NO3_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (200, 300)}, relative=True) #, save_loc=save_loc, save_suffix='_NO3_concentration_cuvette_200-300.pdf')
# H2O2_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (200, 300)}, relative=True) #, save_loc=save_loc, save_suffix='_H2O2_concentration_cuvette_200-300.pdf')

NO2_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (250, 400)}, save_loc=save_loc, save_suffix='_NO2_concentration_cuvette2_250-400.pdf')
NO3_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (250, 400)}, save_loc=save_loc, save_suffix='_NO3_concentration_cuvette2_250-400.pdf')
H2O2_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (250, 400)}, save_loc=save_loc, save_suffix='_H2O2_concentration_cuvette2_250-400.pdf')

fig_ax = NO2_analyzer.pearson_r_vs_wavelength(close=False, show=False, labels=(Names.NO2,), colors=('C0',))
NO3_analyzer.pearson_r_vs_wavelength(close=False, show=False, fig_ax=fig_ax, labels=(Names.NO3,), colors=('C1',))
H2O2_analyzer.pearson_r_vs_wavelength(close=False, show=True, fig_ax=fig_ax, labels=(Names.H2O2,), colors=('C2',))

plot_kwargs = {'yscale': 'log'}
fig_ax = NO2_analyzer.one_minus_pearson_r_vs_wavelength(True, close=False, show=False, labels=(Names.NO2,), colors=('C0',),
                                                        plot_kwargs=plot_kwargs)
NO3_analyzer.one_minus_pearson_r_vs_wavelength(True, close=False, show=False, fig_ax=fig_ax, labels=(Names.NO3,), colors=('C1',))
H2O2_analyzer.one_minus_pearson_r_vs_wavelength(True, close=False, show=True, fig_ax=fig_ax, labels=(Names.H2O2,), colors=('C2',),
                                                save_loc=save_loc, save_suffix='_one_minus_pearson_r_cuvette2.pdf')

plot_kwargs = {'yscale': 'log'}
fig_ax = NO2_analyzer.one_minus_pearson_r_vs_wavelength(False, close=False, show=False, labels=(Names.NO2,), colors=('C0',),
                                                        plot_kwargs=plot_kwargs)
NO3_analyzer.one_minus_pearson_r_vs_wavelength(False, close=False, show=False, fig_ax=fig_ax, labels=(Names.NO3,), colors=('C1',))
H2O2_analyzer.one_minus_pearson_r_vs_wavelength(False, close=False, show=True, fig_ax=fig_ax, labels=(Names.H2O2,), colors=('C2',))


