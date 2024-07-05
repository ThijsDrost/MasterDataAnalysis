import numpy as np
import matplotlib.pyplot as plt
import scipy

from General.experiments.oes.ratio_fit import RatioFit, N2_RANGES_SEL
from General.experiments.hdf5.readHDF5 import read_hdf5
from General.experiments.oes import OESData
from General.simulation.specair.specair import N2SpecAirSimulations, Spectrum, SpecAirSimulations
from General.plotting.plot import lines

# %%
data_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results\Air_2slm_Ar_2slm_9kV_0.3us.hdf5'
data = read_hdf5(data_loc)
emission: OESData = data['emission'].remove_dead_pixels()

wav_range = 320, 410
mask = (emission.wavelengths > wav_range[0]) & (emission.wavelengths < wav_range[1])

simulation_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\N2_fps_rot_500-5000_vib_1000-11000_elec_12000.hdf5'
simulation = N2SpecAirSimulations(simulation_loc)

# %%
corrected_data = emission.remove_background_interp_off()
corrected_data.total_intensity_vs_time()

# %%
