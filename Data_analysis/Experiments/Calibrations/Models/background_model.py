import os

import numpy as np
import matplotlib.pyplot as plt

from General.experiments.hdf5.readHDF5 import read_hdf5
from General.plotting import plot
import General.numpy_funcs as npf

# %%
loc2 = r'E:\OneDrive - TU Eindhoven\Master thesis\Results'
index = 0
wavs = []
datas = []
for file_name in os.listdir(loc2):
    if file_name in ('Air_2slm_Ar_2slm_10kV_1.5us.hdf5',):
        continue
    if not file_name.endswith('.hdf5'):
        continue
    vals = file_name.split('_')
    if vals[-2][:2] == '10':
        continue
    index += 1
    print(file_name)
    data2 = read_hdf5(rf'{loc2}\{file_name}')
    absorption = data2['absorbance']
    oes = data2['emission']
    wavelength_ranges = ((270, 330), (654, 659), (690, 860), (776, 779))
    oes = oes.remove_background_interp_off(is_on_kwargs={'wavelength_range': wavelength_ranges, 'relative_threshold': 0.3})
    mask = oes.is_on(wavelength_range=wavelength_ranges, relative_threshold=0.25)
    on_time = oes.times[np.argwhere(np.diff(mask.astype(int)) == 1)[0, 0] - 1]

    absorbances = absorption.get_absorbances(masked=False)
    mask = (400 < absorption.get_wavelength(masked=False)) & (absorption.get_wavelength(masked=False) < 500)
    average = np.mean(absorbances[:, mask], axis=1)

    mask_100 = ((absorption.variable - absorption.variable[0]) <= 100) & (50 <= (absorption.variable - absorption.variable[0]))
    normed_absorbance = absorption.get_absorbances(corrected=True)[mask_100] - average[:, None][mask_100]
    times = absorption.variable[mask_100] - absorption.variable[0]

    wav_mask = absorption.get_wavelength() > 225
    vals = np.average(normed_absorbance/np.max(normed_absorbance, axis=1)[:, None], axis=0)

    wavs.append(absorption.get_wavelength())
    datas.append(vals/max(vals[wav_mask]))

# %%
avg_datas = [npf.block_average(data, 10) for data in datas]
avg_wavs = [npf.block_average(wav, 10) for wav in wavs]

fig, ax = plot.lines(avg_wavs, avg_datas, line_kwargs={'color': 'k', 'alpha': 0.25, 'linewidth':1}, show=False, close=False)
average = np.nanmean(avg_datas, axis=0)
line_kwargs = {'color': 'r', 'linewidth': 1, 'linestyle': '-'}
plot_kwargs = {'xlabel': 'Wavelength [nm]', 'ylabel': 'Normalized absorbance', 'xlim': (225, 400), 'ylim': (-0.1, 1.1)}
save_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Measurement techniques\drift_fit_1.pdf'
plot.lines(avg_wavs[0], average, line_kwargs=line_kwargs, labels=['Average'], plot_kwargs=plot_kwargs, save_loc=save_loc,
           fig_ax=(fig, ax), show=True)

# %%
avg_datas = [npf.block_average(data, 10) for data in datas]
avg_wavs = [npf.block_average(wav, 10) for wav in wavs]
avg_datas = [value - np.nanmean(value[w > 350]) for value, w in zip(avg_datas, avg_wavs)]

fig, ax = plot.lines(avg_wavs, avg_datas, line_kwargs={'color': 'k', 'alpha': 0.25, 'linewidth':1}, show=False, close=False)
average = np.nanmean(avg_datas, axis=0)
save_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Measurement techniques\drift_fit_2.pdf'
plot.lines(avg_wavs[0], average, line_kwargs=line_kwargs, labels=['Average'], plot_kwargs=plot_kwargs, save_loc=save_loc,
           fig_ax=(fig, ax), show=True)

# %%
differences = np.array([np.average(np.abs(data - average)) for data in avg_datas])
selected = np.argsort(differences)[::-1][(len(avg_datas) // 2):]
avg_datas_sel = np.array(avg_datas)[selected]
avg_wavs_sel = np.array(avg_wavs)[selected]

fig, ax = plot.lines(avg_wavs_sel, avg_datas_sel, line_kwargs={'color': 'k', 'alpha': 0.25, 'linewidth':1}, show=False, close=False)
average = np.nanmean(avg_datas_sel, axis=0)
save_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Measurement techniques\drift_fit_3.pdf'
plot.lines(avg_wavs[0], average, line_kwargs=line_kwargs, labels=['Average'], plot_kwargs=plot_kwargs, save_loc=save_loc,
           fig_ax=(fig, ax), show=True)

average[avg_wavs[0] > 340] = 0
loc = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\background.txt'
np.savetxt(loc, np.array([avg_wavs[0], average]).T)
