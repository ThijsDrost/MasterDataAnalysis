import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from General.Analysis import CalibrationAnalyzer
from General.Data_handling import drive_letter
from General.Plotting import Names

plt.rcParams.update({'font.size': 14})


analyzer_H2O2 = CalibrationAnalyzer.standard(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\H2O2 cuvette\data.hdf5', 'H2O2', 'H2O2 [mM]')
analyzer_NO3 = CalibrationAnalyzer.standard(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO3 cuvette\data.hdf5', 'NO3-', 'NO3- [mM]')
analyzer_NO2 = CalibrationAnalyzer.standard(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2 cuvette\data.hdf5', 'NO2-', 'NO2- [mM]')

line_width = 2  # nm
lines = [215, 225, 235, 260]  # nm
ranges = [(lines[i] - line_width, lines[i] + line_width) for i in range(len(lines))]

H2O2 = analyzer_H2O2.data_set.get_wavelength(True), analyzer_H2O2.data_set.get_absorbances()[-1]
NO3 = analyzer_NO3.data_set.get_wavelength(True), analyzer_NO3.data_set.get_absorbances()[-1]
NO2 = analyzer_NO2.data_set.get_wavelength(True), analyzer_NO2.data_set.get_absorbances()[-1]

x_ticks = np.linspace(210, 300, 10)

plt.figure()
plt.plot(H2O2[0], H2O2[1]/np.max(H2O2[1]), label=Names.H2O2)
plt.plot(NO3[0], NO3[1]/np.max(NO3[1]), label=Names.NO3)
plt.plot(NO2[0], NO2[1]/np.max(NO2[1]), label=Names.NO2)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Normalized absorbance')
for r in ranges:
    plt.axvspan(r[0], r[1], color='grey', alpha=0.5)
plt.legend()
plt.xlim(207, 300)
plt.xticks(x_ticks)
plt.grid()
plt.tight_layout()
plt.savefig(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Method\norm_absorbances_200-300.pdf')
plt.show()

plt.figure()
plt.plot(H2O2[0], H2O2[1]/np.max(H2O2[1]), label='H$_2$O$_2$')
plt.plot(NO3[0], NO3[1]/np.max(NO3[1]), label='NO$_3^-$')
plt.plot(NO2[0], NO2[1]/np.max(NO2[1]), label='NO$_2^-$')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Normalized absorbance')
for r in ranges:
    plt.axvspan(r[0], r[1], color='grey', alpha=0.5)
plt.legend()
plt.xlim(207, 300)
plt.xticks(x_ticks)
plt.grid()
plt.tight_layout()
plt.savefig(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Method\norm_absorbances_ranges_200-300.pdf')
plt.show()

hno2 = np.loadtxt(rf"{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Spectral_data\HNO2.txt", unpack=True)
no2 = np.loadtxt(rf"{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Spectral_data\NO2-.txt", unpack=True)

ranges1 = [(356, 360), (369.5, 373.5), (384, 388)]
ranges2 = [(349, 353), (362.5, 366.5), (377.5, 381.5)]

plt.figure()
for r in ranges1:
    plt.axvspan(r[0], r[1], alpha=0.5, color='C2')
for r in ranges2:
    plt.axvspan(r[0], r[1], alpha=0.5, color='C4')
plt.plot(no2[0], no2[1]/np.max(hno2[1]), label=Names.NO2)
plt.plot(hno2[0], hno2[1]/np.max(hno2[1]), label=Names.HNO2)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Absorbance')
plt.legend()
plt.xlim(270, 400)
plt.grid()
plt.tight_layout()
plt.savefig(r'C:\Users\20222772\Downloads\test.png')
plt.show()

data1 = np.average([np.average(no2[1][(no2[0] > r[0]) & (no2[0] < r[1])]) for r in ranges1])
data2 = np.average([np.average(no2[1][(no2[0] > r[0]) & (no2[0] < r[1])]) for r in ranges2])
data3 = np.average([np.average(hno2[1][(hno2[0] > r[0]) & (hno2[0] < r[1])]) for r in ranges1])
data4 = np.average([np.average(hno2[1][(hno2[0] > r[0]) & (hno2[0] < r[1])]) for r in ranges2])
print(data1/data2, data3/data4)
