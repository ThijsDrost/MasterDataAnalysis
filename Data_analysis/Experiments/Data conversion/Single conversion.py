import os

from General.Data_handling import make_hdf5, drive_letter

# SPECTRUM_TYPE = ['spectrum', 'absorbance', 'intensity']  # spectrum = measurement-dark, absorbance = -log10((measurement-dark)/(reference-dark)), intensity = measurement
# FIND_METHODS = ['ref_interpolate', 'interpolate', 'closest_before', 'closest']
# MULTI_METHODS = ['mean', 'median']

spectrometers = ('2112120U1', '2203047U1')
loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_01_25 NO2- Conc_2203047U1'
find_method = 'closest_before'
multi_method = {'measurement': 'mean', 'dark': 'mean', 'reference': 'mean'}
spectrum_type = 'absorbance'


make_hdf5(rf'{loc}', find_method, spectrum_type, multi_method, f'{loc}/data.hdf5')
