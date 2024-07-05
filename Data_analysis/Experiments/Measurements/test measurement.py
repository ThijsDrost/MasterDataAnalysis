import os

import numpy as np
import matplotlib.pyplot as plt

from General.experiments import SpectroData

loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_16\Argon'

dark = rf'{loc}\Dark_2203047U1'
reference = rf'{loc}\Reference moving'
measurements = rf'{loc}\Measurments'

dark_files = os.scandir(dark)
reference_files = os.scandir(reference)
measurement_files = os.scandir(measurements)

out = [[], []]

for index, files in enumerate((dark_files, reference_files)):
    for file in files:
        if '2203047U1' not in file.name:
            continue
        data = SpectroData.read_data(file.path)
        if len(data.intensity) > 1000:
            out[index].append(data.intensity)
    out[index] = np.average(out[index], axis=0)
    wavelength = data.wavelength
    print(index)

# %%
values = []
times = []
for file in measurement_files:
    data = SpectroData.read_data(file.path)
    if len(data.intensity) == 2048:
        absorbance = -np.log10((data.intensity - out[0]) / (out[1] - out[0]))
        values.append(absorbance)
        times.append(data.time_ms)

# %%
baseline_corrected = []
baseline_mask = (wavelength > 500) & (wavelength < 600)
for vals in values:
    baseline_corrected.append(vals - np.average(vals[baseline_mask]))

# %%
plt.rcParams.update({'font.size': 14})
every = 20

cmap = plt.get_cmap('jet')
times = np.array(times)
times -= times[0]

fig, ax = plt.subplots()
for t, v in zip(times[::every], baseline_corrected[::every]):
    plt.plot(wavelength, v, color=cmap(t/times[-1]))
plt.xlabel('Wavelength [nm]')
plt.ylabel('Absorbance')
normalizer = plt.Normalize(times[0], times[-1]/(1000*60))
plt.colorbar(plt.cm.ScalarMappable(normalizer, cmap=cmap), label='Time [min]', ax=plt.gca())
plt.tight_layout()
plt.xlim(200, 400)
plt.grid()
plt.show()

