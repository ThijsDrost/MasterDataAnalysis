import numpy as np
import matplotlib.pyplot as plt

from General.Analysis import Analyzer, Models
from General.Data_handling import drive_letter, InterpolationDataSet, import_hdf5, DataSet
from General.Data_handling.Import import SimpleDataSet
from General.Numpy_funcs import moving_average, block_averages

# %%
line_width = 2  # nm
lines = [215, 225, 235, 260]  # nm
ranges = [(lines[i] - line_width, lines[i] + line_width) for i in range(len(lines))]


analyzer_H2O2 = Analyzer.standard(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\H2O2 cuvette\data.hdf5',
                                  'H2O2', 'H2O2 [mM]')
analyzer_NO3 = Analyzer.standard(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO3 cuvette\data.hdf5',
                                  'NO3-', 'NO3- [mM]')
analyzer_NO2 = Analyzer.standard(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2 cuvette\data.hdf5',
                                  'NO2-', 'NO2- [mM]')

interp_H2O2 = InterpolationDataSet.from_dataset(analyzer_H2O2.data_set, num=1)
interp_NO3 = InterpolationDataSet.from_dataset(analyzer_NO3.data_set, num=1)
interp_NO2 = InterpolationDataSet.from_dataset(analyzer_NO2.data_set, num=1)

loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2_H2O2 cuvette\data.hdf5'
data = import_hdf5(loc, 'timestamp_s')


# %%
def make_lines(wav, absorbance, ranges):
    return np.array([np.average(absorbance[(r[0] <= wav) & (wav <= r[1])]) for r in ranges])


model_lines = Models.make_lines_model(ranges, corrected=True, num=1, H2O2=analyzer_H2O2.data_set,
                                      NO3=analyzer_NO3.data_set, NO2=analyzer_NO2.data_set)
wav_range = (205, 350)
model_spectra = Models.make_spectra_model(wav_range, corrected=True, num=1, H2O2=analyzer_H2O2.data_set,
                                          NO3=analyzer_NO3.data_set, NO2=analyzer_NO2.data_set)

bunch_size = 20
for j in range(len(data)):
    dat = data[j]
    fit_data = DataSet.from_simple(dat)
    wavelength = fit_data.get_wavelength(False)
    bunched_data = block_averages(fit_data.get_absorbances(masked=False), bunch_size)

    conc_lines = np.zeros((len(bunched_data), 3))
    conc_std_lines = np.zeros((len(bunched_data), 3))
    conc_spectra = np.zeros((len(bunched_data), 3))
    conc_std_spectra = np.zeros((len(bunched_data), 3))

    for i in range(len(bunched_data)):
        fit_datas = make_lines(wavelength, bunched_data[i], ranges)

        params = model_lines.make_params()
        result = model_lines.fit(data=fit_datas, params=params)
        conc_lines[i] = [result.params['H2O2_conc'].value, result.params['NO3_conc'].value, result.params['NO2_conc'].value]
        conc_std_lines[i] = [result.params['H2O2_conc'].stderr, result.params['NO3_conc'].stderr, result.params['NO2_conc'].stderr]

        params = model_spectra.make_params()
        mask = (wav_range[0] <= wavelength) & (wavelength <= wav_range[1])
        wav_masked = wavelength[mask]
        intensity = bunched_data[i][mask]
        result = model_spectra.fit(data=intensity, params=params, wav=wav_masked)
        conc_spectra[i] = [result.params['H2O2_conc'].value, result.params['NO3_conc'].value, result.params['NO2_conc'].value]
        conc_std_spectra[i] = [result.params['H2O2_conc'].stderr, result.params['NO3_conc'].stderr, result.params['NO2_conc'].stderr]

    bunched_var = block_averages(dat.variable, bunch_size)
    plt.figure()
    plt.title(j)
    p1 = plt.plot(bunched_var, conc_lines[:, 0], 'C0', label='H2O2')[0]
    f1 = plt.fill_between(bunched_var, conc_lines[:, 0] - conc_std_lines[:, 0], conc_lines[:, 0] + conc_std_lines[:, 0],
                          color='C0', alpha=0.4, linewidth=0)
    p1_2 = plt.plot(bunched_var, conc_spectra[:, 0], 'C3', label='H2O2')[0]
    f1_2 = plt.fill_between(bunched_var, conc_spectra[:, 0] - conc_std_spectra[:, 0], conc_spectra[:, 0] + conc_std_spectra[:, 0],
                            color='C3', alpha=0.4, linewidth=0)

    p2 = plt.plot(bunched_var, conc_lines[:, 1], 'C1', label='NO3')[0]
    f2 = plt.fill_between(bunched_var, conc_lines[:, 1] - conc_std_lines[:, 1], conc_lines[:, 1] + conc_std_lines[:, 1],
                          color='C1', alpha=0.4, linewidth=0)
    p2_2 = plt.plot(bunched_var, conc_spectra[:, 1], 'c', label='NO3')[0]
    f2_2 = plt.fill_between(bunched_var, conc_spectra[:, 1] - conc_std_spectra[:, 1], conc_spectra[:, 1] + conc_std_spectra[:, 1],
                            color='c', alpha=0.4, linewidth=0)

    p3 = plt.plot(bunched_var, conc_lines[:, 2], 'C2', label='NO2')[0]
    f3 = plt.fill_between(bunched_var, conc_lines[:, 2] - conc_std_lines[:, 2], conc_lines[:, 2] + conc_std_lines[:, 2],
                          color='C2', alpha=0.4, linewidth=0)
    p3_2 = plt.plot(bunched_var, conc_spectra[:, 2], 'm', label='NO2')[0]
    f3_2 = plt.fill_between(bunched_var, conc_spectra[:, 2] - conc_std_spectra[:, 2], conc_spectra[:, 2] + conc_std_spectra[:, 2],
                            color='m', alpha=0.4, linewidth=0)

    labels = ['H2O2 r', 'NO3 r', 'NO2 r', 'H2O2 s', 'NO3 s', 'NO2 s']
    plt.legend([(p1, f1), (p2, f2), (p3, f3), (p1_2, f1_2), (p2_2, f2_2), (p3_2, f3_2)], labels)
    # plt.savefig(f'D:/Downloads/test_image.pdf')
    plt.show()

    print(f'{j}/{len(data)}')

# %%
ranges1 = [(344, 348), (356, 360), (370, 374), (384, 388)]
ranges2 = [(350, 354), (363, 367), (377, 381)]
datas = []
for i in range(len(data)):
    avg_inten = block_averages(data[i].absorbances, 25)
    avg_time = block_averages(data[i].variable, 25)
    num = np.full(len(avg_time), 1)
    avg_data = SimpleDataSet(data[i].wavelength, avg_inten, avg_time, num, 'time')
    analyzer = Analyzer.from_DataSet(DataSet.from_simple(avg_data), 1, 'time [s]')
    analyzer.wavelength_range_ratio_vs_variable(ranges1, ranges2, variable_val_ticks=False)
    conc_lines = np.zeros(len(avg_inten))
    intensities = analyzer.data_set.get_absorbances(masked=False)
    for j in range(len(conc_lines)):
        dat = make_lines(avg_data.wavelength, intensities[j], ranges1)
        dat2 = make_lines(avg_data.wavelength, intensities[j], ranges2)
        conc_lines[j] = np.average(dat) / np.average(dat2)
    datas.append(np.average(conc_lines))

plt.figure()
plt.plot(datas)
plt.show()
