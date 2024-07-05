import numpy as np
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from General.simulation.specair.specair import Spectrum, N2SpecAirSimulations
from General.plotting import plot, cbar


def ranges(wav, intensity, ranges: tuple[tuple[float, float], ...]):
    results = []
    for r in ranges:
        mask = (wav > r[0]) & (wav < r[1])
        results.append(intensity[mask].sum())
    return results


# %%
save_folder = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Theory'
loc = r"E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\N2_fps_rot_500-5000_vib_1000-11000_elec_12000.hdf5"
fwhm = 0.5
wav_range = (270, 450)
wavelengths = np.linspace(*wav_range, 200)

wavs = np.linspace(-6, 6, 250)
peak = scipy.stats.norm.pdf(wavs, 0, fwhm)

meas_interp = N2SpecAirSimulations.from_hdf5(loc, Spectrum(wavs, peak), wavelengths)

# %%
lines = [274., 286., 302., 321, 342., 362., 384., 403]
ranges_ = [(274.0, 282.0), (286.0, 298.0), (302.0, 317), (321, 338.0), (343.0, 357.0), (363.0, 379.0), (385.0, 398)]
# %%
data = []
t_rots = np.linspace(500, 5000, 7)
for t_rot in t_rots:
    data.append(meas_interp(t_rot, 6000))

plot_kwargs = {'xlabel': 'Wavelength [nm]', 'ylabel': 'Intensity [A.U.]'}
colors, mappable = cbar.cbar_norm_colors(t_rots/1000)
cbar_kwargs = {'label': 'T$_{rot}$ [$10^3$K]', 'mappable': mappable}

save_loc = rf'{save_folder}\N2_fps_rot.pdf'
plot.lines(wavelengths, data, colors=colors, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, save_loc=save_loc, show=False)

colors, _ = cbar.cbar_norm_colors(len(ranges_), 'plasma')
for c, r in zip(colors, ranges_):
    plt.fill_betweenx([0, 1], *r, color=c, alpha=0.5)
plt.savefig(rf'{save_folder}\N2_fps_rot_ranges.pdf')
plt.show()


data = []
t_vibs = np.linspace(1000, 11000, 10)
for t_vib in t_vibs:
    data.append(meas_interp(2000, t_vib))

plot_kwargs = {'xlabel': 'Wavelength [nm]', 'ylabel': 'Intensity [A.U.]'}
colors, mappable = cbar.cbar_norm_colors(t_vibs/1000)
cbar_kwargs = {'label': 'T$_{vib}$ [$10^3$K]', 'mappable': mappable}
save_loc = rf'{save_folder}\N2_fps_vib.pdf'
plot.lines(wavelengths, data, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, colors=colors, save_loc=save_loc, show=False)

colors, _ = cbar.cbar_norm_colors(len(ranges_), 'plasma')
for c, r in zip(colors, ranges_):
    plt.fill_betweenx([0, 1], *r, color=c, alpha=0.5)
plt.savefig(rf'{save_folder}\N2_fps_vib_ranges.pdf')
plt.show()
# %%
result = []
i_vals = np.linspace(-5, 10, 16)

t_rots = np.linspace(1200, 2000, 10)
t_vibs = np.linspace(2000, 11000, 12)


ratios = []
for t_vib in t_vibs:
    ratio = []
    for t_rot in t_rots:
        res = meas_interp(t_rot, t_vib)
        range_vals = ranges(wavelengths, res, ranges_)
        ratio.append(range_vals)

    values = []
    for index in range(len(ranges_)):
        max_val = max([r[index] for r in ratio])
        min_val = min([r[index] for r in ratio])
        values.append((min_val, max_val))

    ratios.append(ratio)
    result.append(values)

result = np.array(result)
ratios = np.array(ratios)

cmap = plt.get_cmap('jet')

ref_ratios = ratios[:, :, 3]
test_ratios = ratios.copy()
for index in range(len(ranges_)):
    test_ratios[:, :, index] /= ref_ratios


handles = []
labels = []

plt.figure()
for index in range(len(ranges_)):
    plt.fill_between(t_vibs, result[:, index, 0], result[:, index, 1], color=cmap(index/(len(ranges_)-1)), alpha=0.4)
    handles.append(plt.Line2D([0], [0], color=cmap(index/(len(ranges_)-1)), lw=10, alpha=0.4))
    labels.append(f'{ranges_[index][0]}-{ranges_[index][1]}')
plt.legend(handles, labels)
plt.xlabel('T$_{vib}$ [K]')
plt.ylabel('Intensity [A.U.]')
plt.tight_layout()
plt.savefig(rf'{save_folder}\N2_fps_vib_ratio_ranges.pdf')
plt.show()


plt.figure()
for index in range(len(ranges_)):
    values = test_ratios[:, :, index]
    max_values = values.max(axis=1)
    min_values = values.min(axis=1)
    plt.fill_between(t_vibs, min_values, max_values, color=cmap(index/(len(ranges_)-1)), alpha=0.4)
plt.legend(handles, labels)
plt.xlabel('T$_{vib}$ [K]')
plt.ylabel('Intensity [A.U.]')
plt.tight_layout()
plt.savefig(rf'{save_folder}\N2_fps_vib_ratio_ranges_norm.pdf')
plt.show()

# %%

colors, _ = cbar.cbar_norm_colors(len(ranges_), 'plasma')
labels = [f'{index}' for index in range(len(ranges_))]

save_loc = rf'{save_folder}\N2_fps_peak_iten.pdf'
errors = (ratios.max(axis=1) - ratios.min(axis=1))
plot_kwargs = {'xlabel': 'T$_{vib}$ [$10^3$ K]', 'ylabel': 'Peak intensity (A.U.)'}
legend_kwargs = {'labels': labels, 'title': 'Range'}
plot.errorrange(t_vibs/1000, ratios.mean(axis=1).T, yerr=errors.T, colors=colors, plot_kwargs=plot_kwargs, show=True,
                legend_kwargs=legend_kwargs, save_loc=save_loc)

save_loc = rf'{save_folder}\N2_fps_peak_iten_ratio.pdf'
plot_kwargs = {'xlabel': 'T$_{vib}$ [$10^3$ K]', 'ylabel': 'Relative peak intensity (A.U.)'}
errors = (test_ratios.max(axis=1) - test_ratios.min(axis=1))
plot.errorrange(t_vibs/1000, test_ratios.mean(axis=1).T, yerr=errors.T, colors=colors, plot_kwargs=plot_kwargs, show=True,
                legend_kwargs=legend_kwargs, save_loc=save_loc)

