from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from General.experiments.hdf5.readHDF5 import read_hdf5
from General.experiments.oes import OESData
from General.experiments import WavelengthCalibration
from General.itertools import argmax
from General.plotting import plot, cbar

# %%
save_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Theory'
data_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results\Ar_3slm_5kV_0.3us.hdf5'
data: OESData = read_hdf5(data_loc)['emission']

# %%
data = data.remove_dead_pixels()
plot_kwargs = {'xlim': (840, 850)}
data.spectrum.plot_intensity_with_time(plot_kwargs=plot_kwargs)

# %%

plot_kwargs = {'xlim': (690, 940)}
data.spectrum.plot_intensity_with_time(plot_kwargs=plot_kwargs)

# %%
this_save_loc = rf'{save_loc}\Argon_spectrum.pdf'
data.spectrum.plot_spectrum(100, plot_kwargs=plot_kwargs, normalize=True, save_loc=this_save_loc)

data.spectrum.plot_spectrum(100, plot_kwargs=plot_kwargs, normalize=True)

# %%
wavelengths = data.spectrum.wavelengths
mask = (wavelengths > 690) & (wavelengths < 940)
peaks = find_peaks(data.spectrum.intensities[100][mask], prominence=1000)
peak_wavelengths = wavelengths[mask][peaks[0]]

fig, ax = data.spectrum.plot_spectrum(100, plot_kwargs=plot_kwargs, show=False)
for wav in peak_wavelengths:
    ax.axvline(wav, color='r', linestyle='--')
plt.show()
print(peak_wavelengths)
argon_lines = []

# %%
intensities = data.spectrum.intensities[100]

cmap = plt.get_cmap('turbo')
norm = plt.Normalize(vmin=min(peak_wavelengths), vmax=max(peak_wavelengths))

peak_loc = np.empty_like(peak_wavelengths)
fig, ax = plt.subplots()
for i, wav in enumerate(peak_wavelengths):
    index = np.searchsorted(wavelengths, wav)
    index = index - 1 + argmax(intensities[index-1:index+2])
    x, y = WavelengthCalibration.quadratic_peak_xy(wavelengths[index-1:index+2], intensities[index-1:index+2])
    peak_loc[i] = x

    mask = (wavelengths > x-2.5) & (wavelengths < x+2.5)
    peak_inten = intensities[mask]
    peak_wav = wavelengths[mask]
    color = cmap(norm(wav))

    ax.plot(peak_wav-x, peak_inten/y, c=color)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Peak wavelength [nm]')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity [A.U.]')
plt.tight_layout()
plt.grid()
plt.xlim(-2.5, 2.5)
plt.ylim(0)
plt.show()

# %%
time, values, peak_locs = data.peak_intensity('ar')
values[values < 0] = 0

p_wavs = data.peaks('ar')
color, mapp = cbar.cbar_norm_colors(p_wavs, 'turbo')

cbar_kwargs = {'label': 'Peak wavelength [nm]', 'mappable': mapp}
plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Intensity [A.U.]'}
plot.lines((time-time[0])/60, values, colors=color, cbar_kwargs=cbar_kwargs, plot_kwargs=plot_kwargs)
plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Normalized intensity [A.U.]'}
plot.lines((time-time[0])/60, values/values[:, 100][:, None], colors=color, cbar_kwargs=cbar_kwargs, plot_kwargs=plot_kwargs)

cmap = plt.get_cmap('turbo')
boundaries = [p_wavs[0]-5] + [(p_wavs[i] + p_wavs[i+1])/2 for i in range(len(p_wavs)-1)] + [p_wavs[-1]+5]
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
colors = cmap(norm(p_wavs))

fig, ax = plt.subplots()
for i, wav in enumerate(p_wavs):
    ax.plot((time-time[0])/60, values[i], c=colors[i])
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Peak wavelength [nm]',
             ticks=[int(x) for x in p_wavs])
plt.xlabel('Time [min]')
plt.ylabel('Intensity [A.U.]')
plt.grid()
plt.tight_layout()
plt.show()

data.peak_intensity_vs_wavelength_with_time('ar')
data.peak_intensity_vs_wavelength_with_time('ar', norm=True)

# %%
# data.intensity_vs_wavelength_with_time(plot_kwargs={'xlim': (775, 780)})
data.intensity_vs_wavelength_with_time(plot_kwargs={'xlim': (762, 767)})
