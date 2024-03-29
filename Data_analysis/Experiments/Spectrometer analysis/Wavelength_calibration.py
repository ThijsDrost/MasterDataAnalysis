import matplotlib.pyplot as plt
import numpy as np
import lmfit

from General.Data_handling import drive_letter, SpectroData
from General.Analysis import WavelengthCalibration

plt.rcParams.update({'font.size': 14})

loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_23'
data = SpectroData.read_data(rf'{loc}\Hg-Ar lamp.txt')
dark = SpectroData.read_data(rf'{loc}\Dark.txt')
data2 = SpectroData.read_data(rf'{loc}\Hg-Ar lamp 2.txt')
dark2 = SpectroData.read_data(rf'{loc}\Dark 2.txt')

peaks, wavs = WavelengthCalibration.wavelength_calibration3(data.wavelength, data.intensity - dark.intensity, min_intensity=1)
peaks2, wavs2 = WavelengthCalibration.wavelength_calibration3(data2.wavelength, data2.intensity - dark2.intensity)

wav_range = data.wavelength[0], data.wavelength[-1]
wav_delta = wav_range[1] - wav_range[0]
wav_range2 = data2.wavelength[0], data2.wavelength[-1]
wav_delta2 = wav_range2[1] - wav_range2[0]

print(f'Pixel size 1: {np.diff(data.wavelength).min():.3f} to {np.diff(data.wavelength).max():.3f} nm, '
      f'2: {np.diff(data2.wavelength).min():.3f} to {np.diff(data2.wavelength).max():.3f} nm')

# %%
plt.figure()
plt.errorbar(peaks, wavs-peaks, 0.016, fmt='o', capsize=2, label='Spectrometer 1')
plt.grid()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Difference [nm]')
plt.tight_layout()
plt.savefig(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Method\Spec1_Wavelength_calibration.pdf')
plt.show()

plt.figure()
plt.errorbar(peaks2, wavs2-peaks2, 0.08, fmt='C1o', capsize=2, label='Spectrometer 1')
plt.grid()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Difference [nm]')
plt.tight_layout()
plt.savefig(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Method\Spec2_Wavelength_calibration.pdf')
plt.show()

plt.figure()
plt.plot(peaks, wavs-peaks, 'o', label='Spectrometer 1')
plt.plot(peaks2, wavs2-peaks2, 'o', label='Spectrometer 2')
plt.grid()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Difference [nm]')
plt.tight_layout()
plt.legend()
plt.show()


# %%
def error(x, sigma_a, sigma_b, sigma_c, sigma_t):
    return np.sqrt((sigma_a * x**2) ** 2 + (sigma_b * x) ** 2 + sigma_c ** 2 + sigma_t ** 2)


def model(wavs, peaks, save_name=None):
    wav_range = (peaks[0], peaks[-1])
    wav_delta = wav_range[1] - wav_range[0]
    plot_range = (wav_range[0]-0.05*wav_delta, wav_range[1]+0.05*wav_delta)
    middle_wav = (wav_range[0] + wav_range[1]) / 2

    model = lmfit.models.QuadraticModel()
    params = model.guess(wavs - peaks, x=peaks - middle_wav)
    result = model.fit(wavs - peaks, params, x=peaks - middle_wav)
    result.plot_fit()

    error_wavs = np.linspace(*plot_range, 100)
    error_values = error(error_wavs-middle_wav, result.params['a'].stderr, result.params['b'].stderr, result.params['c'].stderr, 0.0)
    best_values = model.eval(result.params, x=error_wavs-middle_wav)
    plt.figure()
    data = plt.errorbar(peaks, wavs - peaks, yerr=0.016, fmt='oC0', capsize=2)
    line = plt.plot(error_wavs, best_values, '-C1')
    fill = plt.fill_between(error_wavs, best_values - error_values, best_values + error_values, color='C1', alpha=0.3)
    plt.grid()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Difference [nm]')
    plt.legend((data, (line[0], fill)), ('Data', 'Fit'))
    plt.xlim(*plot_range)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
    return result


image_loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Method\Wavelength_calibration'
result1 = model(wavs, peaks, save_name=image_loc + '_1.pdf')
result2 = model(wavs2, peaks2, save_name=image_loc + '_2.pdf')
errors1 = error(peaks - (peaks[0] + peaks[-1]) / 2, result1.params['a'].stderr, result1.params['b'].stderr, result1.params['c'].stderr, 0.016)
errors2 = error(peaks2 - (peaks2[0] + peaks2[-1]) / 2, result2.params['a'].stderr, result2.params['b'].stderr, result2.params['c'].stderr, 0.08)

plt.figure()
plt.errorbar(peaks2, wavs2-peaks2-result2.best_fit,  errors2, capsize=2, fmt='o', label='Spectrometer 2')
plt.errorbar(peaks, wavs-peaks-result1.best_fit, errors1, capsize=2, fmt='o', label='Spectrometer 1')
plt.grid()
plt.legend()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Difference [nm]')
plt.tight_layout()
plt.savefig(image_loc + '_calibration_result.pdf')
plt.show()

# %%
plt.figure()
plt.plot(data2.wavelength, data2.intensity - dark2.intensity, 'C1')
plt.grid()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity')
plt.yscale('log')
plt.ylim(0.1, 1e5)
plt.xlim(200, 1050)
plt.tight_layout()
plt.savefig(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Method\Hg-Ar lamp.pdf')
plt.show()

# %%
wavelength = data.wavelength
intensity = data.intensity - dark.intensity
for start_wavelength in (200, 300, 400, 500, 600, 700, 800, 900):
    plt.figure()
    mask = (wavelength > start_wavelength) & (wavelength < start_wavelength + 100)
    plt.plot(wavelength[mask], intensity[mask])
    for p in peaks:
        if start_wavelength < p < start_wavelength + 100:
            plt.axvline(p, color='r')
    plt.grid()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity')
    plt.yscale('log')
    plt.show()

# %%
wavelength, intensity = data2.wavelength, data2.intensity - dark2.intensity

wav_range = (395, 400)
mask = (wavelength > wav_range[0]) & (wavelength < wav_range[1])
plt.figure()
plt.plot(wavelength[mask], intensity[mask])
plt.grid()
plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity')
plt.yscale('log')
plt.show()
