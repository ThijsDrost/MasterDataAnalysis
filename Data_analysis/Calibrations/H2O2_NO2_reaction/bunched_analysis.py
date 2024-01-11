import numpy as np
import matplotlib.pyplot as plt

from General.Analysis import Analyzer, Models
from General.Data_handling import drive_letter, InterpolationDataSet, import_hdf5, DataSet


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
    for i in range(len(dat)):
        fit_data = DataSet.from_simple(dat)
        fit_data = make_lines(fit_data.get_wavelength(False), fit_data.get_absorbances(masked=False), ranges)
        params = model.make_params()
        result = model.fit(data=fit_data, params=params)
        values[i] = [result.params['H2O2'].value, result.params['NO3'].value, result.params['NO2'].value]

    plt.figure()
    plt.title(j)
    plt.plot(dat.variable, values[:, 0], label='H2O2')
    plt.plot(dat.variable, values[:, 1], label='NO3')
    plt.plot(dat.variable, values[:, 2], label='NO2')
    plt.legend()
    plt.show()

    print(f'{j}/{len(data)}')
