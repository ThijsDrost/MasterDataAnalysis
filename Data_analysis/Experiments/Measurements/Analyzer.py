import numpy as np
import matplotlib.pyplot as plt

from General.import_funcs import drive_letter
from General.experiments.absorption import MeasurementsAnalyzer, CalibrationAnalyzer
from General.experiments.absorption.Models import multi_species_model
from General.plotting import Names


species = ('NO2-', 'NO3-', 'H2O2')
loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_23\Air 12kV\data.hdf5'

analyzer = MeasurementsAnalyzer.read_hdf5(loc)

# %%
analyzer.absorbance_with_wavelength_over_time(100, show=True, plot_kwargs={'xlim': (200, 300)}, line_kwargs={'linewidth': 1})
# analyzer.absorbance_with_wavelength_over_time(50, show=True, plot_kwargs={'xlim': (300, 400), 'ylim': (-0.00025, 0.0025)}, line_kwargs={'linewidth': 1})
analyzer.absorbance_over_time_with_wavelength(100, show=True, min_absorbance=0.025, time_num=300, line_kwargs={'linewidth': 1})

# %%
model_loc = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Cuvette.hdf5'
model = multi_species_model(model_loc, add_zero=True)
analyzer.fit(model, show=True, average_num=100, wavelength_range=(230, 350), export_fit_loc=r"C:\Users\20222772\Downloads\Output\plots")

# %%


# %%
O3_wav, O3_abs = np.loadtxt(r"C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Ozone.txt").T
H2O2_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\H2O2 cuvette\data.hdf5'
H2O2_analyzer = CalibrationAnalyzer.standard(H2O2_loc, 'H2O2', f'{Names.H2O2} [mM]')
H2O2_intensity = H2O2_analyzer.data_set.get_absorbances(var_value=np.max(H2O2_analyzer.data_set.variable)).mean(axis=0)
H2O2_wavelength = H2O2_analyzer.data_set.get_wavelength()
H2O2_wavelength, H2O2_intensity = H2O2_wavelength[200 < H2O2_wavelength], H2O2_intensity[200< H2O2_wavelength]

intensity = np.interp(analyzer.data_set.get_wavelength(), O3_wav, O3_abs)


def gaussian(x, a, b, c):
    return a*np.exp(-((x-b)/c)**2)


factor = 0
factor2 = 1/8
factor3 = 1/120
plt.figure()
plt.plot(analyzer.data_set.get_wavelength(), analyzer.data_set.get_absorbances()[0])
plt.plot(analyzer.data_set.get_wavelength(), intensity*factor)
plt.plot(analyzer.data_set.get_wavelength(), H2O2_intensity*factor2)
plt.plot(analyzer.data_set.get_wavelength(), gaussian(analyzer.data_set.get_wavelength(), 0.009, 267.5, 25))
plt.plot(analyzer.data_set.get_wavelength(), analyzer.data_set.get_absorbances()[0] - intensity*factor - H2O2_intensity*factor2)
plt.xlim(230, 500)
plt.ylim(0, 0.02)
plt.show()
