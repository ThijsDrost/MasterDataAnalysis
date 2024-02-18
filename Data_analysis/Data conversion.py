import os

from General.Data_handling import make_hdf5, drive_letter

# SPECTRUM_TYPE = ['spectrum', 'absorbance', 'intensity']  # spectrum = measurement-dark, absorbance = -log10((measurement-dark)/(reference-dark)), intensity = measurement
# FIND_METHODS = ['ref_interpolate', 'interpolate', 'closest_before', 'closest']
# MULTI_METHODS = ['mean', 'median']

spectrometers = ('2112120U1', '2203047U1')
loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_16\Stability'
dirs = [d for d in os.scandir(loc) if d.is_dir()]
dirs = [d for d in dirs if any(s in d.name for s in spectrometers)]

for d in dirs:
    make_hdf5(d.path, 'closest', 'intensity', {'measurement': 'mean'}, f'{d.path}.hdf5')

