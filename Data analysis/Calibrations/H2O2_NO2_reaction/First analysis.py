import numpy as np
import matplotlib.pyplot as plt
import lmfit

from General.Data_handling.Import import import_hdf5, drive_letter

dependent = 'H2O2'
loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\H2O2 cuvette\data.hdf5'
data = import_hdf5(loc, dependent)
mask = (data.variable == np.max(data.variable)) & (data.measurement_num == 3)
ref_H2O2 = data.absorbances[mask][0]
H2O2_ref_val = np.max(data.variable)*1000

dependent = 'NO2-'
loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2 cuvette\data.hdf5'
data = import_hdf5(loc, dependent)
mask = (data.variable == np.max(data.variable)) & (data.measurement_num == 3)
ref_NO2= data.absorbances[mask][0]
NO2_ref_val = np.max(data.variable)*1000

dependent = 'NO3-'
image_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Plots\Calibration\NO3'
loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO3 cuvette\data.hdf5'
data = import_hdf5(loc, dependent)
mask = (data.variable == np.max(data.variable)) & (data.measurement_num == 3)
ref_NO3 = data.absorbances[mask][0]
NO3_ref_val = np.max(data.variable)*1000

loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2_H2O2 cuvette\data.hdf5'
data = import_hdf5(loc, 'timestamp_s')


# %%
wav_range = [210, 300]
# exclude_range = [215, 225]
mask = (data[0].wavelength > wav_range[0]) & (data[0].wavelength < wav_range[1]) #& ((data[0].wavelength < exclude_range[0]) | (data[0].wavelength > exclude_range[1]))
ref_H2O2_m = ref_H2O2[mask]
ref_NO2_m = ref_NO2[mask]
ref_NO3_m = ref_NO3[mask]

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float, axis=1)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def model(wav, cH2O2, cNO2, cNO3):
    return cH2O2*ref_H2O2_m + cNO2*ref_NO2_m + cNO3*ref_NO3_m


fit_model = lmfit.Model(model)
params = lmfit.Parameters()
params.add('cH2O2', value=1, min=0)
params.add('cNO2', value=1, min=0)
params.add('cNO3', value=1, min=0)

number = 1
avg_num = 25
sample_num = len(data[number].absorbances) // avg_num

test = np.array([np.average(data[number].absorbances[i*avg_num:(i+1)*avg_num], axis=0) for i in range(sample_num)])
time_values = data[number].variable - np.min(data[number].variable)
times = np.array([np.average(time_values[i*avg_num:(i+1)*avg_num]) for i in range(sample_num)])

cmap = plt.get_cmap('jet')
plt.figure()
for val, tims in zip(test, times):
    plt.plot(data[0].wavelength, val, c=cmap((tims-times[0])/(times[-1]-times[0])))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=times[0], vmax=times[-1]))
plt.colorbar(sm, ax=plt.gca())
plt.xlim(195, 300)
plt.show()

H2O2_res = []
H2O2_std = []
NO2_res = []
NO2_std = []
NO3_res = []
NO3_std = []
for index, measurement in enumerate(test):
    fit = fit_model.fit(measurement[mask], params, wav=data[number].wavelength[mask])

    fit.plot()
    plt.savefig(rf'E:\OneDrive - TU Eindhoven\Master thesis\Plots\Calibration\NO2_H2O2\{index}.png')
    plt.close()

    H2O2_res.append(fit.params['cH2O2'].value)
    H2O2_std.append(fit.params['cH2O2'].stderr)
    NO2_res.append(fit.params['cNO2'].value)
    NO2_std.append(fit.params['cNO2'].stderr)
    NO3_res.append(fit.params['cNO3'].value)
    NO3_std.append(fit.params['cNO3'].stderr)

H2O2_res = np.array(H2O2_res, dtype=float)
H2O2_std = np.array(H2O2_std, dtype=float)
NO2_res = np.array(NO2_res, dtype=float)
NO2_std = np.array(NO2_std, dtype=float)
NO3_res = np.array(NO3_res, dtype=float)
NO3_std = np.array(NO3_std, dtype=float)
H2O2_std[np.isnan(H2O2_std)] = 0
NO2_std[np.isnan(NO2_std)] = 0
NO3_std[np.isnan(NO3_std)] = 0

# %%
plt.figure()
plt.errorbar(times, H2O2_res*H2O2_ref_val, yerr=H2O2_std*H2O2_ref_val, fmt='.', capsize=2)
plt.errorbar(times, 2*NO2_res*NO2_ref_val, yerr=2*NO2_std*NO2_ref_val, fmt='.', capsize=2)
plt.errorbar(times, NO3_res*NO3_ref_val, yerr=NO3_std*NO3_ref_val, fmt='.', capsize=2)
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.legend(['H2O2', 'NO2', 'NO3'])
plt.grid()
plt.show()

plt.figure()
# plt.errorbar(times, 2*NO2_res*NO2_ref_val+NO3_res*NO3_ref_val, yerr=2*NO2_std*NO2_ref_val+NO3_std*NO3_ref_val, fmt='.',
#              capsize=2)
plt.plot(times, 2*NO2_res*NO2_ref_val+NO3_res*NO3_ref_val, '-')
plt.fill_between(times, 2*NO2_res*NO2_ref_val+NO3_res*NO3_ref_val-2*NO2_std*NO2_ref_val-NO3_std*NO3_ref_val,
                 2*NO2_res*NO2_ref_val+NO3_res*NO3_ref_val+2*NO2_std*NO2_ref_val+NO3_std*NO3_ref_val, alpha=0.5)
plt.show()

