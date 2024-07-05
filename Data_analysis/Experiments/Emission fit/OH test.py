import numpy as np
import scipy
import matplotlib.pyplot as plt

from General.simulation.specair.specair import Spectrum, SpecAirSimulations
from General.experiments.hdf5.readHDF5 import read_hdf5
from General.experiments.oes import OESData
from General.plotting import plot
from General.experiments.spectrum import TemporalSpectrum

loc = r"E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\OH_AX_rot_600-2400_vib_1000-11000_elec_12000.hdf5"
wavs = np.linspace(-6, 6, 250)
intensity = scipy.stats.norm.pdf(wavs, 0, 0.5)
wav_range = (275, 340)
wavelengths = np.linspace(*wav_range, 20)
interp = SpecAirSimulations.from_hdf5(loc, Spectrum(wavs, intensity), wavelengths)

# %%
rotational_values = np.linspace(600, 2400, 10)
rotational_spectra = [interp(rotational_value, 3000) for rotational_value in rotational_values]
rotational_spectra = [spec/np.sum(spec) for spec in rotational_spectra]
plot.lines(wavelengths, rotational_spectra, labels=[f"{value:.0f}" for value in rotational_values], legend_kwargs={})

# %%
vibrational_values = np.linspace(1000, 11000, 10)
vibrational_spectra = [interp(2000, vibrational_value) for vibrational_value in vibrational_values]
vibrational_spectra = [spec/np.sum(spec) for spec in vibrational_spectra]
plot.lines(wavelengths, vibrational_spectra, labels=[f"{value:.0f}" for value in vibrational_values], legend_kwargs={})

# %%
data_loc = r"E:\OneDrive - TU Eindhoven\Master thesis\Results\Ar_3slm_4.5kV_0.3us.hdf5"
data: OESData = read_hdf5(data_loc)['emission']
plot_kwargs = {'xlim': wav_range}
data.intensity_vs_wavelength_with_time(plot_kwargs=plot_kwargs, background_index=-1, block_average=100)

# %%
avg_data = data.remove_baseline((400, 600))
b_data = avg_data.remove_background_interp((None, 10), (200, None))
avg_data.total_intensity_vs_index()
avg_data.intensity_vs_wavelength_with_time(plot_kwargs=plot_kwargs, background_index=-1)
# b_data.intensity_vs_wavelength_with_time(plot_kwargs=plot_kwargs)
# b_data.total_intensity_vs_index()

# %%
spectrum = b_data.spectrum.wavelength_range(*wav_range)
meas_interp = SpecAirSimulations.from_hdf5(loc, Spectrum(wavs, intensity), spectrum.wavelengths)

# %%
model = meas_interp.model()
params = model.make_params()
fit_index = 100
result = model.fit(spectrum.intensities[fit_index], params)
plt.figure()
plt.plot(spectrum.wavelengths, result.best_fit)
plt.plot(spectrum.wavelengths, spectrum.intensities[fit_index])
plt.show()
print(result.fit_report())

# %%
model = meas_interp.model()
params = model.make_params()
indexes = (20, 180)

t_rot = []
t_rot_std = []
t_vib = []
t_vib_std = []

for index in range(*indexes):
    result = model.fit(spectrum.intensities[index], params)
    t_rot.append(result.params['rot_energy'].value)
    t_rot_std.append(result.params['rot_energy'].stderr)
    t_vib.append(result.params['vib_energy'].value)
    t_vib_std.append(result.params['vib_energy'].stderr)

t_vib = np.array(t_vib)
t_vib_std = np.array(t_vib_std)
t_rot = np.array(t_rot)
t_rot_std = np.array(t_rot_std)

plt.figure()
plt.errorbar(spectrum.times[indexes[0]:indexes[1]], t_vib)
plt.show()

plt.figure()
plt.errorbar(spectrum.times[indexes[0]:indexes[1]], t_rot)
plt.show()
