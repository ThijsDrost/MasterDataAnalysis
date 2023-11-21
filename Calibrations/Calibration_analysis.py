import os

import numpy as np
import matplotlib.pyplot as plt
import scipy
import lmfit

from Data_handling.Import import import_hdf5, DataSet


loc = r'D:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO3 pH\data.hdf5'
image_loc = r'D:\OneDrive - TU Eindhoven\Master thesis\Plots\Calibration\NO3 pH'
dependent = 'pH'
variable_name = 'pH'
variable_factor = 1
wavelength_range = [180, 400]
r2_values = [0.99, 1]
wavelength_plot_every = 5
plot_measurement_num = 2
baseline_correction = [300, 400]


if not os.path.exists(image_loc):
    os.makedirs(image_loc)


def save_loc(loc):
    return os.path.join(image_loc, loc)


temp = import_hdf5(loc, dependent)
data = DataSet.from_simple(import_hdf5(loc, dependent), wavelength_range, baseline_correction, plot_measurement_num)

for value in np.unique(data.variable):
    plt.figure()
    for index, num in enumerate(data.measurement_num[data.variable == value]):
        plt.plot(data.wavelength_masked,
                 data.absorbances_masked[(data.measurement_num == num) & (data.variable == value)].T, f'C{index}', label=num)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.legend(title='num')
    plt.tight_layout()
    plt.savefig(save_loc(f'absorbance vs wavelength {value} {data.variable_name}.png'))
    plt.close()

cmap = plt.get_cmap('jet')
for value in np.unique(data.variable):
    plt.figure()
    wav_abs_mask = data.absorbances_masked[(data.variable == value)][-1, :] > 0.02
    for index, wav in enumerate(data.wavelength_masked[wav_abs_mask][::wavelength_plot_every]):
        vals = data.absorbances_masked[(data.variable == value)].T[::wavelength_plot_every][index]
        plt.plot(data.measurement_num[data.variable == value], vals / vals[-1], 'o-',
                 color=cmap(index / len(data.wavelength_masked[wav_abs_mask][::wavelength_plot_every])))
    plt.xlabel('Measurement measurement_num')
    plt.ylabel('Relative absorbance')
    plt.xticks(data.measurement_num[data.variable == value])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data.wavelength_masked[wav_abs_mask][0],
                                                             vmax=data.wavelength_masked[wav_abs_mask][-1]))
    plt.colorbar(sm, label='Wavelength (nm)', ax=plt.gca())
    plt.tight_layout()
    plt.savefig(save_loc(f'absorbance vs measurement measurement_num {value} {data.variable_name}.png'))
    plt.close()
# %%
# plot absorbance vs variable
plt.figure()
for index, var in enumerate(np.unique(data.variable)):
    plt.plot(data.wavelength_masked, data.absorbances_masked[(data.measurement_num == plot_measurement_num) & (data.variable == var)].T, f'C{index}',
             label=variable_factor * var)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.legend(title=data.variable_name)
plt.tight_layout()
plt.savefig(save_loc(f'absorbance vs wavelength.png'))
plt.close()

cmap = plt.get_cmap('jet')
plt.figure()
for index, wav in enumerate(data.wavelength_masked[::wavelength_plot_every]):
    plt.plot(variable_factor * data.variable_num, data.absorbances_masked_num.T[::wavelength_plot_every][index],
             color=cmap(index / len(data.wavelength_masked[::wavelength_plot_every])))
plt.xlabel(data.variable_name)
plt.ylabel('Absorbance')
plt.xticks(variable_factor * data.variable_num)
# make a cmap for the plot
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data.wavelength_masked[0], vmax=data.wavelength_masked[-1]))
plt.colorbar(sm, label='Wavelength (nm)', ax=plt.gca())
plt.tight_layout()
plt.savefig(save_loc(f'absorbance vs {data.variable_name}.png'))
plt.close()

cmap = plt.get_cmap('jet')
plt.figure()
for index, wav in enumerate(data.wavelength_masked[::wavelength_plot_every]):
    vals = data.absorbances_masked_num.T[::wavelength_plot_every][index]
    plt.plot(variable_factor * data.variable_num, vals / vals[-1],
             color=cmap(index / len(data.wavelength_masked[::wavelength_plot_every])))
plt.xlabel(variable_name)
plt.ylabel('Absorbance')
plt.xticks(variable_factor * data.variable_num)
# make a cmap for the plot
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data.wavelength_masked[0], vmax=data.wavelength_masked[-1]))
plt.colorbar(sm, label='Wavelength (nm)', ax=plt.gca())
plt.tight_layout()
plt.savefig(save_loc(f'absorbance vs {data.variable_name} relative.png'))
plt.close()
# %%
# pearson r for each wavelength
if baseline_correction is not None:
    linearity = np.zeros(len(data.wavelength_masked))
    for i in range(len(data.wavelength_masked)):
        linearity[i] = scipy.stats.pearsonr(data.variable, data.absorbances_masked_corrected[:, i])[0]

    linearity_corrected_num = np.zeros(len(data.wavelength_masked))
    for i in range(len(data.wavelength_masked)):
        linearity_corrected_num[i] = scipy.stats.pearsonr(data.variable_num, data.absorbances_masked_corrected_num[:, i])[0]

linearity_uncorrected = np.zeros(len(data.wavelength_masked))
for i in range(len(data.wavelength_masked)):
    linearity_uncorrected[i] = scipy.stats.pearsonr(data.variable, data.absorbances_masked[:, i])[0]

linearity_uncorrected_num = np.zeros(len(data.wavelength_masked))
for i in range(len(data.wavelength_masked)):
    linearity_uncorrected_num[i] = scipy.stats.pearsonr(data.variable_num, data.absorbances_masked_num[:, i])[0]

linearity_best_num = np.zeros(len(data.wavelength_masked))
for i in range(len(data.wavelength_masked)):
    linearity_best_num[i] = scipy.stats.pearsonr(data.variable_best_num, data.absorbances_masked_best_num[:, i])[0]

plt.figure()
if baseline_correction is not None:
    plt.plot(data.wavelength_masked, linearity ** 2, label='corrected')
    plt.plot(data.wavelength_masked, linearity_corrected_num ** 2, label='corrected num')
plt.plot(data.wavelength_masked, linearity_uncorrected ** 2, label='uncorrected')
plt.plot(data.wavelength_masked, linearity_uncorrected_num ** 2, label='uncorrected num')
plt.plot(data.wavelength_masked, linearity_best_num ** 2, label='best num')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Linearity coefficient')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(save_loc(f'linearity vs wavelength method comparison.png'))
plt.close()

r2_mask = ((r2_values[0] < linearity_uncorrected ** 2) & (linearity_uncorrected ** 2 < r2_values[1])) | (
            (r2_values[0] < linearity_uncorrected_num ** 2) & (linearity_uncorrected_num ** 2 < r2_values[1])) | (
                      (r2_values[0] < linearity_best_num ** 2) & (linearity_best_num ** 2 < r2_values[1]))
if baseline_correction is not None:
    r2_mask = r2_mask | ((r2_values[0] < linearity ** 2) & (linearity ** 2 < r2_values[1])) | (
                (r2_values[0] < linearity_corrected_num ** 2) & (linearity_corrected_num ** 2 < r2_values[1]))

wavs = data.wavelength_masked[r2_mask]
dw = np.diff(wavs)
index = np.nonzero(dw > 10)[0]
if len(index) > 0:
    r2_mask[index[0] + 1:] = False
# r2_mask[index+1:] = False

plt.figure()
if baseline_correction is not None:
    plt.plot(data.wavelength_masked[r2_mask], linearity[r2_mask] ** 2, label='corrected')
    plt.plot(data.wavelength_masked[r2_mask], linearity_corrected_num[r2_mask] ** 2, label='corrected num')
plt.plot(data.wavelength_masked[r2_mask], linearity_uncorrected[r2_mask] ** 2, label='uncorrected')
plt.plot(data.wavelength_masked[r2_mask], linearity_uncorrected_num[r2_mask] ** 2, label='uncorrected num')
plt.plot(data.wavelength_masked[r2_mask], linearity_best_num[r2_mask] ** 2, label='best num')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Linearity coefficient')
plt.grid()
plt.legend()
plt.tight_layout()
plt.ylim(*r2_values)
plt.savefig(save_loc(f'linearity vs wavelength zoomed method comparison.png'))
plt.close()
# %%
# linear fit for each wavelength
if baseline_correction is not None:
    slope = np.zeros(len(data.wavelength_masked))
    intercept = np.zeros(len(data.wavelength_masked))
    for i in range(len(data.wavelength_masked)):
        slope[i], intercept[i] = np.polyfit(data.variable, data.absorbances_masked_corrected[:, i], 1)

    slope_corrected_num = np.zeros(len(data.wavelength_masked))
    intercept_corrected_num = np.zeros(len(data.wavelength_masked))
    for i in range(len(data.wavelength_masked)):
        slope_corrected_num[i], intercept_corrected_num[i] = np.polyfit(data.variable_num, data.absorbances_masked_corrected_num[:, i], 1)

slope_uncorrected = np.zeros(len(data.wavelength_masked))
intercept_uncorrected = np.zeros(len(data.wavelength_masked))
for i in range(len(data.wavelength_masked)):
    slope_uncorrected[i], intercept_uncorrected[i] = np.polyfit(data.variable, data.absorbances_masked[:, i], 1)

slope_uncorrected_num = np.zeros(len(data.wavelength_masked))
intercept_uncorrected_num = np.zeros(len(data.wavelength_masked))
for i in range(len(data.wavelength_masked)):
    slope_uncorrected_num[i], intercept_uncorrected_num[i] = np.polyfit(data.variable_num, data.absorbances_masked_num[:, i], 1)

slope_best_num = np.zeros(len(data.wavelength_masked))
intercept_best_num = np.zeros(len(data.wavelength_masked))
for i in range(len(data.wavelength_masked)):
    slope_best_num[i], intercept_best_num[i] = np.polyfit(data.variable_best_num, data.absorbances_masked_best_num[:, i], 1)

plt.figure()
if baseline_correction is not None:
    plt.plot(data.wavelength_masked, slope, label='corrected')
    plt.plot(data.wavelength_masked, slope_corrected_num, label='corrected num')
plt.plot(data.wavelength_masked, slope_uncorrected, label='uncorrected')
plt.plot(data.wavelength_masked, slope_uncorrected_num, label='uncorrected num')
plt.plot(data.wavelength_masked, slope_best_num, label='best num')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Slope')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(save_loc(f'slope vs wavelength method comparison.png'))
plt.close()

# plot relative slope
plt.figure()
if baseline_correction is not None:
    plt.plot(data.wavelength_masked, slope / slope_uncorrected_num, label='corrected')
    plt.plot(data.wavelength_masked, slope_corrected_num / slope_uncorrected_num, label='corrected num')
plt.plot(data.wavelength_masked, slope_uncorrected / slope_uncorrected_num, label='uncorrected')
plt.plot(data.wavelength_masked, slope_uncorrected_num / slope_uncorrected_num, label='uncorrected num')
plt.plot(data.wavelength_masked, slope_best_num / slope_uncorrected_num, label='best num')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative slope')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(save_loc(f'slope vs wavelength relative method comparison.png'))
plt.close()

# plot relative slope
plt.figure()
if baseline_correction is not None:
    plt.plot(data.wavelength_masked, slope / slope_uncorrected_num, label='corrected')
    plt.plot(data.wavelength_masked, slope_corrected_num / slope_uncorrected_num, label='corrected num')
plt.plot(data.wavelength_masked, slope_uncorrected / slope_uncorrected_num, label='uncorrected')
plt.plot(data.wavelength_masked, slope_uncorrected_num / slope_uncorrected_num, label='uncorrected num')
plt.plot(data.wavelength_masked, slope_best_num / slope_uncorrected_num, label='best num')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative slope')
plt.legend()
plt.grid()
plt.tight_layout()
plt.ylim(0.9, 1.1)
plt.savefig(save_loc(f'slope vs wavelength relative method comparison zoom.png'))
plt.close()

plt.figure()
if baseline_correction is not None:
    plt.plot(data.wavelength_masked, intercept, label='corrected')
    plt.plot(data.wavelength_masked, intercept_corrected_num, label='corrected num')
plt.plot(data.wavelength_masked, intercept_uncorrected, label='uncorrected')
plt.plot(data.wavelength_masked, intercept_uncorrected_num, label='uncorrected num')
plt.plot(data.wavelength_masked, intercept_best_num, label='best num')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intercept')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(save_loc(f'intercept vs wavelength method comparison.png'))
plt.close()
# %%
# linear fit for each wavelength
lin_model = lmfit.models.LinearModel()
params = lin_model.make_params()
params['intercept'].value = 0
params['intercept'].vary = False

if baseline_correction is not None:
    slope = np.zeros(len(data.wavelength_masked))
    slope_std = np.zeros(len(data.wavelength_masked))
    for i in range(len(data.wavelength_masked)):
        result = lin_model.fit(data.absorbances_masked_corrected[:, i], params, x=data.variable)
        slope[i] = result.params['slope'].value
        slope_std[i] = result.params['slope'].stderr

    slope_corrected_num = np.zeros(len(data.wavelength_masked))
    slope_corrected_num_std = np.zeros(len(data.wavelength_masked))
    for i in range(len(data.wavelength_masked)):
        result = lin_model.fit(data.absorbances_masked_corrected_num[:, i], params, x=data.variable_num)
        slope_corrected_num[i] = result.params['slope'].value
        slope_corrected_num_std[i] = result.params['slope'].stderr

slope_uncorrected = np.zeros(len(data.wavelength_masked))
slope_uncorrected_std = np.zeros(len(data.wavelength_masked))
for i in range(len(data.wavelength_masked)):
    result = lin_model.fit(data.absorbances_masked[:, i], params, x=data.variable)
    slope_uncorrected[i] = result.params['slope'].value
    slope_uncorrected_std[i] = result.params['slope'].stderr

slope_uncorrected_num = np.zeros(len(data.wavelength_masked))
slope_uncorrected_num_std = np.zeros(len(data.wavelength_masked))
for i in range(len(data.wavelength_masked)):
    result = lin_model.fit(data.absorbances_masked_num[:, i], params, x=data.variable_num)
    slope_uncorrected_num[i] = result.params['slope'].value
    slope_uncorrected_num_std[i] = result.params['slope'].stderr

slope_best_num = np.zeros(len(data.wavelength_masked))
slope_best_num_std = np.zeros(len(data.wavelength_masked))
for i in range(len(data.wavelength_masked)):
    result = lin_model.fit(data.absorbances_masked_best_num[:, i], params, x=data.variable_best_num)
    slope_best_num[i] = result.params['slope'].value
    slope_best_num_std[i] = result.params['slope'].stderr

plt.figure()
if baseline_correction is not None:
    plt.errorbar(data.wavelength_masked, slope, yerr=slope_std, label='corrected')
    plt.errorbar(data.wavelength_masked, slope_corrected_num, yerr=slope_corrected_num_std, label='corrected num')
plt.errorbar(data.wavelength_masked, slope_uncorrected, yerr=slope_uncorrected_std, label='uncorrected')
plt.errorbar(data.wavelength_masked, slope_uncorrected_num, yerr=slope_uncorrected_num_std, label='uncorrected num')
plt.errorbar(data.wavelength_masked, slope_best_num, yerr=slope_best_num_std, label='best num')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Slope')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(save_loc(f'slope vs wavelength method comparison err.png'))
plt.close()

# plot wavelength vs error
plt.figure()
if baseline_correction is not None:
    plt.plot(data.wavelength_masked, slope_std, label='corrected')
    plt.plot(data.wavelength_masked, slope_corrected_num_std, label='corrected num')
plt.plot(data.wavelength_masked, slope_uncorrected_std, label='uncorrected')
plt.plot(data.wavelength_masked, slope_uncorrected_num_std, label='uncorrected num')
plt.plot(data.wavelength_masked, slope_best_num_std, label='best num')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig(save_loc(f'slope vs wavelength relative error method comparison.png'))
plt.close()

# plot relative slope
mask = data.absorbances_masked[-1] > 0.05 * np.max(data.absorbances_masked[-1])
y_min, y_max = np.min(slope_uncorrected[mask] / data.absorbances_masked[-1][mask]), np.max(
    slope_uncorrected[mask] / data.absorbances_masked[-1][mask])
dy = y_max - y_min
y_min -= 0.1 * dy
y_max += 0.1 * dy
plt.figure()
if baseline_correction is not None:
    plt.plot(data.wavelength_masked, slope / data.absorbances_masked_corrected[-1], label='corrected')
    plt.plot(data.wavelength_masked, slope_corrected_num / data.absorbances_masked_corrected_num[-1], label='corrected num')
plt.plot(data.wavelength_masked, slope_uncorrected / data.absorbances_masked[-1], label='uncorrected')
plt.plot(data.wavelength_masked, slope_uncorrected_num / data.absorbances_masked_num[-1], label='uncorrected num')
plt.plot(data.wavelength_masked, slope_best_num / data.absorbances_masked_best_num[-1], label='best num')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative slope')
plt.legend()
plt.grid()
plt.ylim(y_min, y_max)
plt.tight_layout()
# plt.savefig(save_loc(f'slope vs wavelength relative method comparison.png'))
plt.close()


# %%
def residual(pars, x, reference):
    a = pars['a'].value
    return x - a * reference


params = lmfit.Parameters()
params.add('a', value=1, vary=True)
ratio = []
ratio_std = []
for i in data.absorbances_masked:
    result = lmfit.minimize(residual, params, args=(i,), kws={'reference': data.absorbances_masked[-1]})
    ratio.append(result.params['a'].value)
    ratio_std.append(result.params['a'].stderr)
ratio = np.array(ratio)
ratio_std = np.array(ratio_std)

plt.figure()
plt.errorbar(data.variable, ratio, yerr=ratio_std, capsize=2, fmt='.')
plt.xlabel(data.variable_name)
plt.ylabel('Ratio')
plt.grid()
plt.tight_layout()
plt.savefig(save_loc(f'Relative intensity vs {data.variable_name}.png'))
plt.close()

lines, labels = [], []
plt.figure()
for index, var in enumerate(np.unique(data.variable)):
    plt.plot(data.wavelength_masked, data.absorbances_masked[data.variable == var].T / ratio[data.variable == var], f'C{index}', label=var)
    lines.append(plt.Line2D([0], [0], color=f'C{index}'))
    labels.append(f'{var}')
# plt.plot(data.wavelength_masked, data.absorbances_masked.T/ratio, label=data.variable)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized absorbance (A.U.)')
plt.legend(lines, labels, title=data.variable_name)
plt.grid()
plt.tight_layout()
plt.savefig(save_loc(f'Normalized absorbance vs wavelength.png'))
plt.close()


# %%
def residual(pars, x, concentration):
    a = pars['a'].value
    return x - a * concentration * x[-1]


params = lmfit.Parameters()
params.add('a', value=1, vary=True)

if baseline_correction is not None:
    result = lmfit.minimize(residual, params, args=(data.absorbances_masked_corrected,),
                            kws={'concentration': data.variable[:, np.newaxis]})
    result_num = lmfit.minimize(residual, params, args=(data.absorbances_masked_corrected_num,),
                                kws={'concentration': data.variable_num[:, np.newaxis]})
result_uncorr = lmfit.minimize(residual, params, args=(data.absorbances_masked,),
                               kws={'concentration': data.variable[:, np.newaxis]})
result_uncorr_num = lmfit.minimize(residual, params, args=(data.absorbances_masked_num,),
                                   kws={'concentration': data.variable_num[:, np.newaxis]})
result_best_num = lmfit.minimize(residual, params, args=(data.absorbances_masked_best_num,),
                                 kws={'concentration': data.variable_best_num[:, np.newaxis]})

print(
f"""
# Fit report
corrected: {result.params['a'].value:.3f} ± {result.params['a'].stderr:.3f}
corrected num: {result_num.params['a'].value:.3f} ± {result_num.params['a'].stderr:.3f}
uncorrected: {result_uncorr.params['a'].value:.3f} ± {result_uncorr.params['a'].stderr:.3f}
uncorrected num: {result_uncorr_num.params['a'].value:.3f} ± {result_uncorr_num.params['a'].stderr:.3f}
best num: {result_best_num.params['a'].value:.3f} ± {result_best_num.params['a'].stderr:.3f}
""")

plt.figure()
plt.plot(data.wavelength_masked, (data.absorbances_masked_num / (result.params['a'].value * data.variable_num[:, np.newaxis])).T)
plt.close()