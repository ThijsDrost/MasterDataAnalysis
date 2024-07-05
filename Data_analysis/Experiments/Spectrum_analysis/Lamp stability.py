from General.experiments.absorption.DataSets import import_hdf5
from General.import_funcs import drive_letter

import numpy as np
import matplotlib.pyplot as plt

loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\Stability\data.hdf5'
data = import_hdf5(loc, 'timestamp_s')
wav_mask = (data.wavelength > 200) & (data.wavelength < 400)
times = (data.variable - data.variable[0])/60

cmap = plt.get_cmap('jet')
plt.figure()
for t, absorbance in zip(times[1:], data.absorbances[1:]):
    plt.plot(data.wavelength, absorbance/data.absorbances[0], c=cmap((t-times[0])/(times[-1]-times[0])))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative intensity')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=times[-1]))
plt.colorbar(sm, ax=plt.gca(), label='Time (min)')
plt.tight_layout()
plt.show()

wav = data.wavelength
plt.figure()
for w, a in zip(wav, data.absorbances.T):
    plt.plot(times, a/a[0], c=cmap((w-wav[0])/(wav[-1]-wav[0])), linewidth=0.1)
plt.plot(times, np.mean(data.absorbances/data.absorbances[0], axis=1), c='k', label='Mean')
mean2 = np.mean(data.absorbances, axis=1)
plt.plot(times, mean2/mean2[0], 'k--', label='Mean2')
plt.xlabel('Time (min)')
plt.ylabel('Relative intensity')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=wav[0], vmax=wav[-1]))
plt.colorbar(sm, ax=plt.gca(), label='Wavelength (nm)')
plt.tight_layout()
plt.show()
