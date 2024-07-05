import numpy as np
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from General.simulation.specair.specair import Spectrum, SpecAirSimulations
from General.plotting import plot, cbar


def ranges(wav, intensity, ranges: tuple[tuple[float, float], ...]):
    results = []
    for r in ranges:
        mask = (wav > r[0]) & (wav < r[1])
        results.append(intensity[mask].sum())
    return results


# %%
loc = r"E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\OH_AX_rot_600-2400_vib_1000-11000_elec_12000.hdf5"
save_folder = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Theory'
fwhm = 1

wavs = np.linspace(-6, 6, 250)
peak = scipy.stats.norm.pdf(wavs, 0, fwhm)

wav_range = (270, 340)
wavelengths = np.linspace(*wav_range, 100)
ranges_ = ((270., 297.), (303., 340.))

meas_interp = SpecAirSimulations.from_hdf5(loc, Spectrum(wavs, peak), wavelengths)

# %%
data = []
ranges_result = []
t_rots = np.linspace(600, 2400, 7)
for t_rot in t_rots:
    res = meas_interp(t_rot, 6000)
    data.append(res)

plot_kwargs = {'xlabel': 'Wavelength [nm]', 'ylabel': 'Intensity [A.U.]'}
colors, mappable = cbar.cbar_norm_colors(t_rots/1000)
cbar_kwargs = {'label': 'T$_{rot}$ [$10^3$K]', 'mappable': mappable}

save_loc = rf'{save_folder}\OH_A-X_rot.pdf'
plot.lines(wavelengths, data, colors=colors, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, show=False,
           save_loc=save_loc)
plt.fill_betweenx([0, 1], *ranges_[0], color='C0', alpha=0.5)
plt.fill_betweenx([0, 1], *ranges_[1], color='C1', alpha=0.5)
save_loc = rf'{save_folder}\OH_A-X_rot_ranges.pdf'
plt.savefig(save_loc)
plt.show()

data = []
t_vibs = np.linspace(1000, 11000, 10)
for t_vib in t_vibs:
    data.append(meas_interp(1800, t_vib))

plot_kwargs = {'xlabel': 'Wavelength [nm]', 'ylabel': 'Intensity [A.U.]'}
colors, mappable = cbar.cbar_norm_colors(t_vibs/1000)
cbar_kwargs = {'label': 'T$_{vib}$ [$10^3$K]', 'mappable': mappable}
save_loc = rf'{save_folder}\OH_A-X_vib.pdf'
plot.lines(wavelengths, data, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, colors=colors, show=False,
           save_loc=save_loc)
plt.fill_betweenx([0, 1], *ranges_[0], color='C0', alpha=0.5)
plt.fill_betweenx([0, 1], *ranges_[1], color='C1', alpha=0.5)
save_loc = rf'{save_folder}\OH_A-X_vib_ranges.pdf'
plt.savefig(save_loc)
plt.show()

# %%
ratios = []
ratios2 = []

t_rots = np.linspace(600, 2400, 10)
t_vibs = np.linspace(1500, 11000, 100)

for t_rot in t_rots:
    ratio = []
    for t_vib in t_vibs:
        res = meas_interp(t_rot, t_vib)
        range_vals = ranges(wavelengths, res, ranges_)
        ratio.append(range_vals[1] / range_vals[0])
    ratios.append(ratio)

ratios = np.array(ratios)
ratios2 = ratios / ratios[0]

plot_kwargs = {'xlabel': 'T$_{vib}$ [K]', 'ylabel': 'ratio'}
colors, mappable = cbar.cbar_norm_colors(t_rots/1000)
cbar_kwargs = {'label': 'T$_{rot}$ [$10^3$K]', 'mappable': mappable}
save_loc = rf'{save_folder}\OH_A-X_ratio.pdf'
plot.semilogy(t_vibs, ratios, plot_kwargs=plot_kwargs, save_loc=save_loc, cbar_kwargs=cbar_kwargs, colors=colors)
save_loc = rf'{save_folder}\OH_A-X_ratio2.pdf'
plot.lines(t_vibs, ratios, plot_kwargs=plot_kwargs, save_loc=save_loc, cbar_kwargs=cbar_kwargs, colors=colors)

plot_kwargs = {'xlabel': 'T$_{vib}$ [K]', 'ylabel': 'relative ratio'}
save_loc = rf'{save_folder}\OH_A-X_relative_ratio.pdf'
plot.lines(t_vibs, ratios2, plot_kwargs=plot_kwargs, save_loc=save_loc, cbar_kwargs=cbar_kwargs, colors=colors)

# %%
ratio_values = np.linspace(1.1*ratios.min(), 0.8*ratios.max(), 50)
interps = [interp1d(ratios[i], t_vibs) for i in range(ratios.shape[0])]
min_max_values = []
for ratio in ratio_values:
    t_vibs_vals = [interp(ratio) for interp in interps]
    min_max_values.append((min(t_vibs_vals), max(t_vibs_vals)))

avg_values = [(v[0]+v[1])/2 for v in min_max_values]

plt.figure()
plt.plot(ratio_values, avg_values, '--')
plt.fill_between(ratio_values, [v[0] for v in min_max_values], [v[1] for v in min_max_values], alpha=0.5)
plt.show()

# %%
min_val = []
max_val = []
min_max_val = []
i_vals = np.linspace(-5, 10, 16)
for i in i_vals:
    ranges_ = ((270., 295.+i), (300.+i, 340.))
    ratios = []
    ratios2 = []

    t_rots = np.linspace(600, 2400, 10)
    t_vibs = np.linspace(2500, 11000, 10)

    for t_rot in t_rots:
        ratio = []
        for t_vib in t_vibs:
            res = meas_interp(t_rot, t_vib)
            range_vals = ranges(wavelengths, res, ranges_)
            ratio.append(range_vals[1] / range_vals[0])
        ratios.append(ratio)

    ratios = np.array(ratios)
    ratios2 = ratios / ratios[0]

    min_val.append(ratios2.min())
    max_val.append(ratios2.max())
    min_max_val.append((ratios2.max()-ratios2.min()))

labels = ['min', 'max', 'max-min']
plot.lines(i_vals, [min_val, max_val, min_max_val], labels=labels, plot_kwargs={'xlabel': 'i', 'ylabel': 'ratio'})