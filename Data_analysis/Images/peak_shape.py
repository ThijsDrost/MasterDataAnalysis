import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress, ttest_ind
import lmfit
from scipy.stats import norm

from General.import_funcs import drive_letter
from General.experiments import SpectroData, WavelengthCalibration
from General.plotting import plot, cbar, linestyles

plt.rcParams.update({'font.size': 14})

# loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_23'
# spectrum = r'Hg-Ar lamp.txt'
# dark = r'Dark.txt'
# save_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Setup\peak_fit'
# spectrum = SpectroData.read_data(f'{loc}\\{spectrum}')
# dark = SpectroData.read_data(f'{loc}\\{dark}')
#
# wavelengths = spectrum.spectrum.wavelengths
# intensity = spectrum.spectrum.intensities - dark.spectrum.intensities
# #%%
# wavelength = 435.8335
# mask = (wavelengths > wavelength - 1.5) & (wavelengths < wavelength + 1.5)
# line_kwargs = {'linestyle': '-', 'marker': 'o'}
# plot_kwargs = {'xlabel': 'Wavelength [nm]', 'ylabel': 'Intensity [A.U]', 'ylim': 0}
# plot.lines(wavelengths[mask], intensity[mask], line_kwargs=line_kwargs, plot_kwargs=plot_kwargs, save_loc=f'{save_loc}_1.pdf')
# #%%
# top_index = np.argmax(intensity[mask])
# fig_ax = plot.lines(wavelengths[mask], intensity[mask], line_kwargs=line_kwargs, plot_kwargs=plot_kwargs, show=False)
# plot.lines([wavelengths[mask][top_index]], [intensity[mask][top_index]], fig_ax=fig_ax, line_kwargs={'marker': 'x', 'color': 'red'}, plot_kwargs=plot_kwargs, save_loc=f'{save_loc}_2.pdf')
# #%%
# top_index = np.argmax(intensity[mask])
# fig_ax = plot.lines(wavelengths[mask], intensity[mask], line_kwargs=line_kwargs, plot_kwargs=plot_kwargs, show=False)
# plot.lines([wavelengths[mask][top_index]], [intensity[mask][top_index]], fig_ax=fig_ax, line_kwargs={'marker': 'x', 'color': 'red'}, show=False)
# plot.lines([wavelengths[mask][top_index-1], wavelengths[mask][top_index+1]], [intensity[mask][top_index-1], intensity[mask][top_index+1]], fig_ax=fig_ax, line_kwargs={'marker': 'x', 'color': 'green', 'linestyle': 'none'}, plot_kwargs=plot_kwargs, save_loc=f'{save_loc}_3.pdf')
# #%%
# top_index = np.argmax(intensity[mask])
# plot_kwargs['ylim'] = (0, 8900)
# fig_ax = plot.lines(wavelengths[mask], intensity[mask], line_kwargs=line_kwargs, plot_kwargs=plot_kwargs, show=False)
# quadratic = WavelengthCalibration.quadratic(wavelengths[mask][top_index-1:top_index+2], intensity[mask][top_index-1:top_index+2])
# wavs = np.linspace(wavelengths[mask][0], wavelengths[mask][-1], 1000)
# line = quadratic[0]*(wavs**2) + quadratic[1]*wavs + quadratic[2]
# maskerino = (line > 0) & (line < 1.1*intensity[mask][top_index])
# plot.lines(wavs[maskerino], line[maskerino], fig_ax=fig_ax, line_kwargs={'color': 'orange'}, show=False)
# peak = WavelengthCalibration.quadratic_peak(wavelengths[mask][top_index-1:top_index+2], intensity[mask][top_index-1:top_index+2])
# plot.lines([peak], [quadratic[0]*(peak**2) + quadratic[1]*peak + quadratic[2]], fig_ax=fig_ax, line_kwargs={'marker': 'x', 'color': 'red'}, plot_kwargs=plot_kwargs, save_loc=f'{save_loc}_4.pdf')
#%%
def gauss(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))


pixels = np.arange(-100, 101)
sigmas = np.linspace(0.25, 4, 251)
pixel_offsets = np.linspace(0, 0.5, 251)
locs = np.zeros((len(sigmas), len(pixel_offsets)))
heights = np.zeros_like(locs)


def find_values(pixels, pixel_offset, sigma):
    y_vals = gauss(pixels, 1, pixel_offset, sigma)
    index = np.argmax(y_vals)
    if (y_vals[index] < 0.999) and (pixel_offset == 0):
        print('Hmmmmm', i, j)
    loc, height = WavelengthCalibration.quadratic_peak_xy(pixels[index - 1:index + 2], y_vals[index - 1:index + 2])
    return loc - pixel_offset, height - 1



for i, sigma in enumerate(sigmas):
    for j, pixel_offset in enumerate(pixel_offsets):
        loc, height = find_values(pixels, pixel_offset, sigma)
        locs[i, j] = loc
        heights[i, j] = height


# %%
values = [1.34, 0.84]
for val in values:
    temp_locs = np.zeros(len(pixel_offsets))
    temp_heights = np.zeros(len(pixel_offsets))
    for j, pixel_offset in enumerate(pixel_offsets):
        temp_locs[j], temp_heights[j] = find_values(pixels, pixel_offset, val)

    print(f'For sigma = {val}: max, avg')
    print(f'Loc: {np.max(abs(temp_locs)):.2e}, {np.mean(abs(temp_locs)):.2e}')
    print(f'Height: {np.max(abs(temp_heights)):.2e}, {np.mean(abs(temp_heights)):.2e}')
# %%
save_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Appendices\peak_shape'

plt.figure()
plt.imshow(abs(locs), aspect='auto', extent=(pixel_offsets[0], pixel_offsets[-1], sigmas[0], sigmas[-1]), cmap='tab20c',
           origin='lower', norm='log', vmin=1e-5, vmax=1e0)
plt.colorbar(label='Relative peak location error')
plt.xlabel('Pixel offset')
plt.ylabel('Relative peak width')
plt.tight_layout()
plt.savefig(rf'{save_loc}\locs_err_img.pdf')
plt.show()

plt.figure()
plt.imshow(abs(heights), aspect='auto', extent=(pixel_offsets[0], pixel_offsets[-1], sigmas[0], sigmas[-1]), cmap='tab20c',
           origin='lower', norm='log', vmin=1e-5, vmax=1e0)
plt.colorbar(extend='min', label='Relative peak height error')
plt.xlabel('Pixel offset')
plt.ylabel('Relative peak width')
plt.tight_layout()
plt.savefig(rf'{save_loc}\heights_err_img.pdf')
plt.show()


#%%
colors, sm = cbar.cbar_norm_colors(sigmas)
cbar_kwargs = {'label': r'$\sigma$', 'mappable': sm}
plot_kwargs = {'xlabel': 'Pixel offset', 'ylabel': 'Relative height error', 'ylim': (1e-5, None)}
plot.semilogy(pixel_offsets, abs(heights), colors=colors, cbar_kwargs=cbar_kwargs, plot_kwargs=plot_kwargs)
plot_kwargs['ylim'] = (3e-4, 3e-1)
plot_kwargs['ylabel'] = 'Relative location error'
plot.semilogy(pixel_offsets, abs(locs), colors=colors, cbar_kwargs=cbar_kwargs, plot_kwargs=plot_kwargs)

plot_kwargs = {'xlabel': "Relative peak width", 'ylabel': "Relative location error", 'grid_which': 'both'}
plot.semilogy(sigmas, [np.max(abs(locs), axis=1), np.mean(abs(locs), axis=1)], plot_kwargs=plot_kwargs,
              labels=['Maximum', 'Mean'], save_loc=rf'{save_loc}\locs_err.pdf')

plot_kwargs = {'xlabel': "Relative peak width", 'ylabel': "Relative height error"}
plot.semilogy(sigmas, [np.max(abs(heights), axis=1), np.mean(abs(heights), axis=1)], plot_kwargs=plot_kwargs,
              labels=['Maximum', 'Mean'], save_loc=rf'{save_loc}\heights_err.pdf')
