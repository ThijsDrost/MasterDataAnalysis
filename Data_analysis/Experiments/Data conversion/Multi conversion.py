import os

from General.experiments.hdf5.makeHDF5 import make_hdf5

# SPECTRUM_TYPE = ['spectrum', 'absorbance', 'intensity']  # spectrum = measurement-dark, absorbance = -log10((measurement-dark)/(reference-dark)), intensity = measurement
# FIND_METHODS = ['ref_interpolate', 'interpolate', 'closest_before', 'closest']
# MULTI_METHODS = ['mean', 'median']

spectrometers = ('2112120U1', '2203047U1')
loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_23\Air 10 kV'
dirs = [d for d in os.scandir(loc) if d.is_dir()]
dirs = [d for d in dirs if any(s in d.name for s in spectrometers)]

for d in dirs:
    make_hdf5(d.path, 'closest', 'intensity', {'measurement': 'mean'}, f'{d.path}.hdf5')

