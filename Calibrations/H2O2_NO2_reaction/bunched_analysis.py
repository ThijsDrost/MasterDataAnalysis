import numpy as np
import matplotlib.pyplot as plt
import lmfit

from Data_handling.Import import import_hdf5, drive_letter

index = 5

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
# %%
loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2_H2O2 cuvette\data.hdf5'
data = import_hdf5(loc, 'timestamp_s')

# %%
wavelength = data.wavelength
wav_mask = (wavelength > 200) & (wavelength < 400)
wav_m = wavelength[wav_mask]
ref_H2O2_m = ref_H2O2[wav_mask]
ref_NO2_m = ref_NO2[wav_mask]
ref_NO3_m = ref_NO3[wav_mask]

plt.figure()
plt.plot(wav_m, ref_H2O2_m, label='H2O2')
plt.plot(wav_m, ref_NO2_m, label='NO2-')
plt.plot(wav_m, ref_NO3_m, label='NO3-')
plt.legend()
plt.show()

plt.figure()
plt.plot(wav_m, ref_H2O2_m, label='H2O2')
plt.plot(wav_m, ref_NO2_m, label='NO2-')
plt.plot(wav_m, ref_NO3_m, label='NO3-')
plt.legend()
plt.ylim(-0.0025, 0.02)
plt.show()

