import numpy as np
import matplotlib.pyplot as plt

from General.experiments.absorption import CalibrationAnalyzer, pH
from General.experiments.absorption.pH_analysis import theoretical_ratio
from General.import_funcs import drive_letter

save_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Results\Calibrations'
base_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements'

NO2_loc = rf'{base_loc}\24_01_25 NO2- pH_2203047U1\data.hdf5'
NO3_loc = rf'{base_loc}\Calibration\NO3 pH\data.hdf5'
H2O2_loc = rf'{base_loc}\Calibration\H2O2 pH 2\data.hdf5'
H2O2_loc2 = rf'{base_loc}\24_02_16\H2O2\data.hdf5'

NO2_analyzer = CalibrationAnalyzer.standard(NO2_loc, 'pH', 'pH')
NO3_analyzer = CalibrationAnalyzer.standard(NO3_loc, 'pH', 'pH')
H2O2_analyzer = CalibrationAnalyzer.standard(H2O2_loc, 'pH', 'pH')
H2O2_analyzer2 = CalibrationAnalyzer.standard(H2O2_loc2, 'pH', 'pH')

NO2_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (280, 400)}, variable_range=(2.5, 7))
NO3_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (200, 300)}, variable_range=(2.5, 7))
H2O2_analyzer.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (200, 300)}, variable_range=(2.5, 7))
H2O2_analyzer2.absorbance_vs_wavelength_with_variable(show=True, plot_kwargs={'xlim': (200, 300)}, variable_range=(2.5, 7))

intensity = H2O2_analyzer2.data_set.absorbances[-1]
fig, ax = plt.subplots()
ax.plot(H2O2_analyzer2.data_set.wavelength, intensity, 'k--', label='Bottle', zorder=10)
H2O2_analyzer2.absorbance_vs_wavelength_with_variable(show=False, close=False, fig_ax=(fig, ax), plot_kwargs={'xlim': (200, 300)}, variable_range=(2.5, 7))
plt.show()

# %%
NO2_analyzer.wavelength_range_ratio_vs_variable(*pH.NO2_ranges, show=False, close=False, plot_kwargs={'xticks': np.arange(2, 6, 0.5)},
                                                labels=('Measurements',))
ratios = []
ph_vals = np.linspace(2, 6,  100)
for ph in ph_vals:
    ratios.append(theoretical_ratio(ph, 15))
plt.plot(ph_vals, ratios, label='Theoretical')
plt.xlim(2, 6)
plt.legend()
plt.savefig('NO2_ratio_vs_pH.png')
plt.show()
