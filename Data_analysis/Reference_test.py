import matplotlib.pyplot as plt
import numpy as np

from General.Data_handling import drive_letter, SpectroData

base_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_01_25 NO2- pH'

spectrometer = '2203047U1'  # '2112120U1' or '2203047U1'

file_locs = [rf'{base_loc}\{i}_{spectrometer}.TXT' for i in range(1, 10)]
refs_loc = [rf'{base_loc}\Reference{i}_{spectrometer}.TXT' for i in range(1, 3)]
dark_loc = rf'{base_loc}\Dark_{spectrometer}.TXT'

datas = [np.mean(SpectroData.read_data(loc).intensity, axis=1) for loc in file_locs]
refs = [np.mean(SpectroData.read_data(loc).intensity, axis=1) for loc in refs_loc]
dark = SpectroData.read_txt(dark_loc)
dark, wavelength = np.mean(dark.intensity, axis=1), dark.wavelength

x_lim = (200, 400)
mask = (wavelength > x_lim[0]) & (wavelength < x_lim[1])

for j, ref in enumerate(refs):
    plt.figure()
    plt.title(refs_loc[j].split('\\')[-1])
    for i, data in enumerate(datas):
        absorbance = -np.log10((data - dark) / (ref - dark))
        plt.plot(wavelength[mask], absorbance[mask], label=i)
    plt.xlim(*x_lim)
    # plt.ylim(0, 0.1)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.legend()
    plt.show()
