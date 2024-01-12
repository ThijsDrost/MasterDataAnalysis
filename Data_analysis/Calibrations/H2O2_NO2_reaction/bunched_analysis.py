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

interp_H2O2 = InterpolationDataSet.from_dataset(analyzer_H2O2, num=1)
interp_NO3 = InterpolationDataSet.from_dataset(analyzer_NO3, num=1)
interp_NO2 = InterpolationDataSet.from_dataset(analyzer_NO2, num=1)

model = Models.make_lines_model(ranges, corrected=True, num=1, H2O2=analyzer_H2O2, NO3=analyzer_NO3, NO2=analyzer_NO2)

loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2_H2O2 cuvette\data.hdf5'
data = import_hdf5(loc, 'timestamp_s')


# %%
def make_lines(wav, absorbance, ranges):
    return np.array([np.average(absorbance[(r[0] <= wav) & (wav <= r[1])]) for r in ranges])


for j in range(len(data)):
    dat = data[j]
    values = np.zeros((len(dat), 3))
    values_std = np.zeros((len(dat), 3))
    fit_data = DataSet.from_simple(dat)
    fit_data = fit_data.get_wavelength(False), fit_data.get_absorbances(masked=False)
    for i in range(len(dat)):
        fit_datas = make_lines(fit_data[0], fit_data[1][i], ranges)
        params = model.make_params()
        result = model.fit(data=fit_datas, params=params)
        values[i] = [result.params['H2O2'].value, result.params['NO3'].value, result.params['NO2'].value]
        values_std[i] = [result.params['H2O2'].stderr, result.params['NO3'].stderr, result.params['NO2'].stderr]

    plt.figure()
    plt.title(j)
    p1 = plt.plot(dat.variable, values[:, 0], 'C0', label='H2O2')[0]
    f1 = plt.fill_between(dat.variable, values[:, 0] - values_std[:, 0], values[:, 0] + values_std[:, 0], color='C0', alpha=0.4)
    p2 = plt.plot(dat.variable, values[:, 1], 'C1', label='NO3')[0]
    f2 = plt.fill_between(dat.variable, values[:, 1] - values_std[:, 1], values[:, 1] + values_std[:, 1], color='C1', alpha=0.4)
    p3 = plt.plot(dat.variable, values[:, 2], 'C2', label='NO2')[0]
    f3 = plt.fill_between(dat.variable, values[:, 2] - values_std[:, 2], values[:, 2] + values_std[:, 2], color='C2', alpha=0.4)
    labels = ['H2O2', 'NO3', 'NO2']
    plt.legend([(p1, f1), (p2, f2), (p3, f3)], labels)
    plt.show()

    print(f'{j}/{len(data)}')
    break


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
    values = np.zeros(len(avg_inten))
    intensities = analyzer.get_absorbances(masked=False)
    for j in range(len(values)):
        dat = make_lines(avg_data.wavelength, intensities[j], ranges1)
        dat2 = make_lines(avg_data.wavelength, intensities[j], ranges2)
        values[j] = np.average(dat) / np.average(dat2)
    datas.append(np.average(values))

plt.figure()
plt.plot(datas)
plt.show()
