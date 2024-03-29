import numpy as np

from General.Data_handling import DataSet, drive_letter
from General.Analysis.Models import export_models

models_loc = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models'

# %% Cuvette
base_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration'
NO2_loc = rf'{base_loc}\NO2 cuvette\data.hdf5'
NO3_loc = rf'{base_loc}\NO3 cuvette\data.hdf5'
H2O2_loc = rf'{base_loc}\H2O2 cuvette\data.hdf5'

NO2_data = DataSet.read_hdf5(NO2_loc, 'NO2-')
NO3_data = DataSet.read_hdf5(NO3_loc, 'NO3-')
H2O2_data = DataSet.read_hdf5(H2O2_loc, 'H2O2')

NO2_conc = np.max(NO2_data.variable)
NO3_conc = np.max(NO3_data.variable)
H2O2_conc = np.max(H2O2_data.variable)

NO2_max = NO2_data.get_absorbances(var_value=NO2_conc).mean(axis=0)
NO3_max = NO3_data.get_absorbances(var_value=NO3_conc).mean(axis=0)
H2O2_max = H2O2_data.get_absorbances(var_value=H2O2_conc).mean(axis=0)

O3_wav, O3_absorbance = np.loadtxt(r"C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Ozone.txt").T

models = {'NO2': {'wavelength': NO2_data.get_wavelength(), 'absorbance': NO2_max, 'concentration': [NO2_conc/68.9953]},
          'NO3': {'wavelength': NO3_data.get_wavelength(), 'absorbance': NO3_max, 'concentration': [NO3_conc/84.9947]},
          'H2O2': {'wavelength': H2O2_data.get_wavelength(), 'absorbance': H2O2_max, 'concentration': [H2O2_conc]},
          'O3': {'wavelength': O3_wav, 'absorbance': O3_absorbance, 'concentration': [np.max(O3_absorbance)/3300]},
          'O3_2': {'wavelength': O3_wav+8, 'absorbance': O3_absorbance, 'concentration': [np.max(O3_absorbance)/3300]}}

export_models(models, rf'{models_loc}\Cuvette.hdf5', 'mol/L')
