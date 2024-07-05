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

# %%
emission.intensity_vs_wavelength_with_time()

new_emission: OESData = emission.remove_background_interp((None, 10), (190, None))

plot_kwargs = {'xlim': (290, 430)}
new_emission.intensity_vs_wavelength_with_time(plot_kwargs=plot_kwargs)

plot_kwargs = {'xlim': (700, 830)}
new_emission.intensity_vs_wavelength_with_time(plot_kwargs=plot_kwargs)

# %%
wav_range = (270, 450)
loc = r'E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\N2_fps_rot_500-5000_vib_1000-11000_elec_12000.hdf5'

fwhm = 0.5
mask = (new_emission.spectrum.wavelengths > wav_range[0]) & (new_emission.spectrum.wavelengths < wav_range[1])
wavs = new_emission.spectrum.wavelengths[mask]
intensity = new_emission.spectrum.intensities[:, mask]

ratio_fitter = RatioFit(loc, fwhm, wavs)
ratio_results = []
for inten in intensity[10:190]:
    ratio_results.append(ratio_fitter.fit(wavs, inten))

ratio_results = np.array(ratio_results)

# %%
lower, upper = np.transpose(ratio_results, axes=(2, 0, 1))[:, :, 1:-3]

plt.figure()
plt.plot(lower)
plt.plot(upper)
plt.show()

plt.figure()
plt.fill_between(np.arange(len(lower)), lower[:, 0], upper[:, 0], alpha=1)
plt.show()

# %%
wav_range = (325, 450)
mask = (new_emission.spectrum.wavelengths > wav_range[0]) & (new_emission.spectrum.wavelengths < wav_range[1])
wavs = new_emission.spectrum.wavelengths[mask]
intensity = new_emission.spectrum.intensities[:, mask]

peak_wavs = np.linspace(-3, 3, 100)
peak = scipy.stats.norm.pdf(peak_wavs, 0, fwhm)
spectrum_fitter: N2SpecAirSimulations = N2SpecAirSimulations.from_hdf5(loc, Spectrum(peak_wavs, peak), wavs)


# %%
spectrum_fitter.first_global_fit_result_plot(intensity[100])
values, _ = spectrum_fitter.best_global_fit_result_plot(intensity[100])
print(values)

# %%
model = spectrum_fitter.model()
params = model.make_params()
params['rot_energy'].set(value=800)
fit_results = []
mask = spectrum_fitter.total_mask
for inten in [intensity[10], intensity[180]]:
    fit = model.fit(inten, params)
    values = (fit.best_values['rot_energy'], fit.best_values['vib_energy'])
    fit_results.append(values)
    plt.figure()
    plt.plot(wavs, inten)
    plt.plot(wavs, fit.best_fit)
    plt.show()
    # break
# %%
wav_range = (270, 450)
mask = (new_emission.spectrum.wavelengths > wav_range[0]) & (new_emission.spectrum.wavelengths < wav_range[1])
wavs = new_emission.spectrum.wavelengths[mask]
intensity = new_emission.spectrum.intensities[:, mask]

peak_wavs = np.linspace(-3, 3, 100)
peak = scipy.stats.norm.pdf(peak_wavs, 0, fwhm)

loc_OH = r'E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\OH_AX_rot_600-2400_vib_1000-11000_elec_12000.hdf5'
loc_N2 = loc
spectrum_fitter_OH = SpecAirSimulations.from_hdf5(loc_OH, Spectrum(peak_wavs, peak), wavs)
spectrum_fitter_N2 = N2SpecAirSimulations.from_hdf5(loc, Spectrum(peak_wavs, peak), wavs)
# %%
model = spectrum_fitter_N2.model(prefix='N2_') + spectrum_fitter_OH.model(prefix='OH_')
params = model.make_params()
# params['rot_energy'].set(value=800)
fit_results = []
mask = spectrum_fitter_N2.total_mask
for inten in [intensity[10], intensity[180]]:
    fit = model.fit(inten, params)
    values = (fit.best_values['N2_rot_energy'], fit.best_values['N2_vib_energy'])
    fit_results.append(values)
    plt.figure()
    plt.plot(wavs, inten)
    plt.plot(wavs, fit.best_fit)
    plt.show()
    # break

# %%
model = spectrum_fitter_N2.model(prefix='N2_') + spectrum_fitter_OH.model(prefix='OH_')
params = model.make_params()
# params['rot_energy'].set(value=800)
fit_results = []
for inten in intensity[10:190]:
    fit = model.fit(inten, params)
    values = (fit.best_values['N2_rot_energy'], fit.best_values['N2_vib_energy'],
              fit.best_values['OH_rot_energy'], fit.best_values['OH_vib_energy'])
    fit_results.append(values)

# %%
t_vibs_N2 = [t[1] for t in fit_results]
t_vibs_OH = [t[3] for t in fit_results]
t_vibs_N2_r = [t[0] for t in fit_results]
t_vibs_OH_r = [t[2] for t in fit_results]

lines([t_vibs_N2])
lines([t_vibs_OH])
lines([t_vibs_N2_r])
lines([t_vibs_OH_r])


# %%
values = spectrum_fitter(3_000, 10_000)
plt.figure()
plt.plot(intensity[100])
plt.plot(4000*values)
plt.show()
