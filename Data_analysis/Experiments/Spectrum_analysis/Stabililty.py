import matplotlib.pyplot as plt
import numpy as np

from General.Data_handling import drive_letter, import_hdf5

base_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_16\Stability'
spectrometers = ('2112120U1', '2203047U1')
names = ('Long fast', 'Long medium', 'Long slow', 'Short fast', 'Short medium', 'Short slow', 'Short stationary', 'Syringe short')

data = [[import_hdf5(f'{base_loc}/{name}_{spectrometer}.hdf5', 'timestamp_s') for name in names] for spectrometer in spectrometers]

wav_range = (200, 400)

for dat, spectrometer in zip(data, spectrometers):
    for d, name in zip(dat, names):
        times = d.variable - d.variable[0]
        mask = (d.wavelength > wav_range[0]) & (d.wavelength < wav_range[1])
        absorbances = np.sum(d.absorbances[:, mask], axis=1)
        absorbances = absorbances/np.max(absorbances)
        std = np.std(absorbances)
        change = np.mean(absorbances[-10:]) - np.mean(absorbances[:10])
        print(f'{spectrometer} {name}: std = {100*std:.2f}, change (2 min) = {100*np.abs(change):.2f}')

# %%
for dat, spectrometer in zip(data, spectrometers):
    plt.figure()
    for d, name in zip(dat, names):
        times = d.variable - d.variable[0]
        mask = (d.wavelength > wav_range[0]) & (d.wavelength < wav_range[1])
        absorbances = np.sum(d.absorbances[:, mask], axis=1)
        absorbances = absorbances/np.max(absorbances)
        plt.plot(times, absorbances, label=f'{name}')
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized total intensity [a.u.]')
    plt.legend()
    plt.show()

# %%
for dat, spectrometer in zip(data, spectrometers):
    plt.figure()
    for d, name in zip(dat, names):
        times = d.variable - d.variable[0]
        mask = (d.wavelength > wav_range[0]) & (d.wavelength < wav_range[1])
        absorbances = np.sum(d.absorbances[:, mask], axis=1)
        absorbances = absorbances - np.mean(absorbances)
        absorbances = absorbances/np.max(absorbances)
        corr = np.correlate(absorbances, absorbances, mode='full')
        corr = corr[corr.size//2:]
        plt.plot(corr, label=f'{name}')
    plt.xlabel('Time [s]')
    plt.ylabel('Correlation')
    plt.legend()
    plt.xlim(0, 100)
    plt.show()
