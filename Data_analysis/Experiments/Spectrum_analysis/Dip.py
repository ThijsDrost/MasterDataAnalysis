import numpy as np
import matplotlib.pyplot as plt
import lmfit

from General.import_funcs import drive_letter
from General.experiments import WavelengthCalibration
from General.experiments import SpectroData

plt.rcParams.update({'font.size': 14})

loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_16\H2O2'
ref = SpectroData.read_data(rf'{loc}\Reference.txt')
dark = SpectroData.read_data(rf'{loc}\Dark.txt')
data = SpectroData.read_data(rf'{loc}\6.txt')

ref_intensity = ref.intensity
dark_intensity = dark.intensity
data_intensity = data.intensity

ratio = (data_intensity - dark_intensity) / (ref_intensity - dark_intensity)

wav_range = (210, 225)
mask = (data.wavelength > wav_range[0]) & (data.wavelength < wav_range[1])

fig, ax = plt.subplots()
ax.plot(data.wavelength[mask], ratio[mask])
plt.show()


# %%
def extract_peak(wav, ratio):
    fit_range = ((210, 214), (221, 225))
    fit_mask = (wav > fit_range[0][0]) & (wav < fit_range[0][1]) | (wav > fit_range[1][0]) & (wav < fit_range[1][1])
    x = wav[fit_mask]
    y = ratio[fit_mask]
    model = lmfit.models.QuadraticModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)
    return ratio - model.eval(result.params, x=wav), result


datas = [0, 1, 2, 3, 4, 5]
model = lmfit.models.QuadraticModel()
peaks = []
for data in datas:
    data = SpectroData.read_data(rf'{loc}\{data}')
    data_intensity = np.average(data.intensity, axis=1)
    ratio = (data_intensity - dark_intensity) / (ref_intensity - dark_intensity)
    values, result = extract_peak(data.wavelength, ratio)
    # values = model.eval(result[1].params, x=data.wavelength[mask])
    plt.figure()
    plt.plot(data.wavelength[mask], ratio[mask])
    plt.plot(data.wavelength[mask], ratio[mask]-values[mask])
    plt.show()

    max_index = np.argmax(values[mask])
    peak_loc = WavelengthCalibration.quadratic_peak(data.wavelength[mask][max_index-1:max_index+2], values[mask][max_index-1:max_index+2])
    quadratic = WavelengthCalibration.quadratic(data.wavelength[mask][max_index-1:max_index+2], values[mask][max_index-1:max_index+2])
    peak_height = quadratic[2] - (quadratic[1]**2)/(4*quadratic[0])
    peak = values[mask]/peak_height
    peaks.append((data.wavelength[mask]-peak_loc, peak))

plt.figure()
for peak in peaks:
    plt.plot(peak[0], peak[1], 'o-')
plt.grid()
plt.xlim(-2.5, 2.5)
plt.show()


# %%
num = 10000
wavert_range = 200, 250
value = 0.5
wavs = np.linspace(*wavert_range, num)
intensities = np.ones(num)
intensities[(224 < wavs) & (wavs < 225)] = value

peak_wavs = np.linspace(-2.5, 2.5, int(num/10))
peak = (peak_wavs[1]-peak_wavs[0])*np.exp(-0.5*peak_wavs**2 / 0.5**2)/(0.5*np.sqrt(2*np.pi))
measured = np.convolve(intensities, peak, mode='same')
measured = 1-measured

plt.figure()
plt.plot(wavs, measured)
plt.plot(peak_wavs+224.5, 0.85*(1-value)*peak/(peak_wavs[1]-peak_wavs[0]))
plt.xlim(222, 227)
plt.ylim(0, 0.7)
plt.grid()
plt.show()



