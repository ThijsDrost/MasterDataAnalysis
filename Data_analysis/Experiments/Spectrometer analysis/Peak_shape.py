import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress, ttest_ind
import lmfit

from General.Data_handling import drive_letter, SpectroData
from General.Analysis import WavelengthCalibration


def main():
    plt.rcParams.update({'font.size': 14})

    loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_23'
    data_locs = ("Hg-Ar lamp.txt", "Hg-Ar lamp 2.txt")
    dark_locs = ("Dark.txt", "Dark 2.txt")
    for index, (data_loc, dark_loc) in enumerate(zip(data_locs, dark_locs), start=1):
        data = SpectroData.read_data(rf'{loc}\{data_loc}')
        dark = SpectroData.read_data(rf'{loc}\{dark_loc}')
        peaks, wavs = WavelengthCalibration.wavelength_calibration3(data.wavelength, data.intensity - dark.intensity)
        print(f'Pixel size: {np.diff(data.wavelength).min():.3f} to {np.diff(data.wavelength).max():.3f} nm')
        peak_shape(data.wavelength, data.intensity - dark.intensity, wavs, 2.0, min_rel_height=0.9,
                   save_name=f'{drive_letter()}:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Appendices\\spec{index}_')


def peak_shape(wavelengths, intensities, peak_locs, width, save_name, min_rel_height=0.75, cmap='jet'):
    peak_locs = peak_locs[~np.isnan(peak_locs)]
    if np.sum(np.isnan(peak_locs)) > 0:
        print(f'{np.sum(np.isnan(peak_locs))} NaN values were removed')

    w_min = wavelengths[0]
    w_max = wavelengths[-1]
    w_d = w_max - w_min

    wav_vals = []
    left = []
    right = []
    interps = []
    fwhm = []

    cmap = plt.get_cmap(cmap)
    plt.figure()
    rejected = 0
    for wav in peak_locs:
        mask = (wavelengths > (wav - width - 1)) & (wavelengths < (wav + width + 1))
        min_int = np.min(intensities[mask])
        max_int = np.max(intensities[mask])

        if (max_int-min_int)/max_int < min_rel_height:
            rejected += 1
            continue

        wavs = wavelengths[mask]
        inten = (intensities[mask] - min_int)/(max_int - min_int)

        mask1 = (wavs <= wav)
        mask2 = (wavs >= wav)
        left.append((-wavs[mask1][::-1]+wav, inten[mask1][::-1]))
        right.append((wavs[mask2]-wav, inten[mask2]))
        wav_vals.append(wav)

        interp_mask = (wavelengths > (wav - width - 1)) & (wavelengths < (wav + width + 1))
        interp_inten = (intensities[interp_mask] - min_int)/(max_int - min_int)
        interp = np.interp(np.linspace(wav - width, wav + width, 21), wavelengths[interp_mask], interp_inten)
        interps.append(interp)

        interp_fwhm_l = np.interp(0.5, left[-1][1][::-1], left[-1][0][::-1])
        interp_fwhm_r = np.interp(0.5, right[-1][1][::-1], right[-1][0][::-1])
        fwhm.append((interp_fwhm_l, interp_fwhm_r))

        plt.plot(wavelengths[interp_mask]-wav, interp_inten, color=cmap((wav - w_min) / w_d))
    plt.grid()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(w_min, w_max))
    plt.colorbar(sm, label='Wavelength [nm]', ax=plt.gca())
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity')
    plt.xlim(-width, width)
    plt.tight_layout()
    plt.savefig(rf'{save_name}Hg-Ar lamp_lines.pdf')
    plt.show()

    wav_vals_l, wav_vals_r = np.array(wav_vals), np.array(wav_vals)
    fwhm_l = np.array([f[0] for f in fwhm])
    fwhm_r = np.array([f[1] for f in fwhm])

    mask_l, mask_r = np.ones_like(fwhm_l, dtype=bool), np.ones_like(fwhm_r, dtype=bool)
    for i in range(2):
        mask_l = np.abs(fwhm_l-np.mean(fwhm_l[mask_l])) < 3*np.std(fwhm_l[mask_l])
        mask_r = np.abs(fwhm_r-np.mean(fwhm_r[mask_r])) < 3*np.std(fwhm_r[mask_r])
    maskerino = mask_l & mask_r

    fwhm_l, wav_vals_l = fwhm_l[maskerino], wav_vals_l[maskerino]
    fwhm_r, wav_vals_r = fwhm_r[maskerino], wav_vals_r[maskerino]
    fwhm_l_std = np.std(fwhm_l)
    fwhm_r_std = np.std(fwhm_r)
    fwhm_l_avg = np.average(fwhm_l)
    fwhm_r_avg = np.average(fwhm_r)

    print(f'Rejected: \nIntensity: {rejected} \nLeft: {np.sum((~mask_l).astype(int))} Right: {np.sum((~mask_r).astype(int))} Total: {np.sum((~maskerino).astype(int))}')
    print(f'Peaks not rejected: {len(wav_vals_l)}')

    plt.figure()
    plt.plot(wav_vals_l, fwhm_l, 'o', label='Left')
    plt.axhline(fwhm_l_avg, color='C0', linestyle='--')
    plt.axhspan(fwhm_l_avg-fwhm_l_std, fwhm_l_avg+fwhm_l_std, color='C0', alpha=0.5)

    plt.plot(wav_vals_r, fwhm_r, 'o', label='Right')
    plt.axhline(fwhm_r_avg, color='C1', linestyle='--')
    plt.axhspan(fwhm_r_avg-fwhm_r_std, fwhm_r_avg+fwhm_r_std, color='C1', alpha=0.5)
    print(f"HWHM:\nLeft:{fwhm_l_avg:.3f} +/- {fwhm_l_std:.3f} nm, Right:{fwhm_r_avg:.3f} +/- {fwhm_r_std:.3f} nm")

    plt.grid()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('HWHM [nm]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(rf'{save_name}Hg-Ar lamp_lines_fwhm.pdf')
    plt.show()

    wav_offset = wavelengths[0] + (wavelengths[-1] - wavelengths[0])/2
    left_fit = linregress(wav_vals_l-wav_offset, fwhm_l)
    right_fit = linregress(wav_vals_r-wav_offset, fwhm_r)

    peak_locs = np.array(peak_locs)
    dwav = peak_locs.max() - peak_locs.min()
    wav_values = np.linspace(peak_locs.min()-dwav/20, peak_locs.max()+dwav/20, 1000)

    def error(x, sig_a, sig_b):
        return np.sqrt((sig_a * x) ** 2 + sig_b ** 2)

    fit_l = left_fit.slope*(wav_values-wav_offset) + left_fit.intercept
    fit_l_errors = error(wav_values-wav_offset, left_fit.stderr, left_fit.intercept_stderr)

    fit_r = right_fit.slope*(wav_values-wav_offset) + right_fit.intercept
    fit_r_errors = error(wav_values-wav_offset, right_fit.stderr, right_fit.intercept_stderr)

    print('Fits:')
    print(f'Left: {left_fit.slope:.2e} +/- {left_fit.stderr:.2e} * (x-{wav_offset:.1f}) + {left_fit.intercept:.2e} +/- {left_fit.intercept_stderr:.2e} nm')
    print(f'Right: {right_fit.slope:.2e} +/- {right_fit.stderr:.2e} * (x-{wav_offset:.1f}) + {right_fit.intercept:.2e} +/- {right_fit.intercept_stderr:.2e} nm')

    plt.figure()
    plt.plot(wav_vals_l, fwhm_l, 'o', label='Left')
    plt.plot(wav_values, fit_l, 'C0--')
    plt.fill_between(wav_values, fit_l-fit_l_errors, fit_l+fit_l_errors, alpha=0.5)

    plt.plot(wav_vals_r, fwhm_r, 'o', label='Right')
    plt.plot(wav_values, fit_r, 'C1--')
    plt.fill_between(wav_values, fit_r-fit_r_errors, fit_r+fit_r_errors, alpha=0.5)

    plt.grid()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('HWHM [nm]')
    plt.legend()
    plt.xlim(wav_values[0], wav_values[-1])
    plt.tight_layout()
    plt.savefig(rf'{save_name}Hg-Ar lamp_lines_fwhm_fit.pdf')
    plt.show()

    plt.figure()
    for interp, w, m in zip(interps, wav_vals, maskerino):
        if m:
            plt.plot(np.linspace(0, width, 11), interp[:11][::-1], '--', color=cmap((w - w_min) / w_d))
            plt.plot(np.linspace(0, width, 11), interp[10:], '-', color=cmap((w - w_min) / w_d))
    plt.grid()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(w_min, w_max))
    plt.colorbar(sm, label='Wavelength [nm]', ax=plt.gca())
    plt.xlabel('Distance from center [nm]')
    plt.ylabel('Relative intensity [A.U.]')
    plt.xlim(0, width)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(rf'{save_name}Hg-Ar lamp_lines_symmetry.pdf')
    plt.show()

    interps = np.array(interps)
    interp_left = np.average(interps[mask_l], axis=0)[:11][::-1]
    interp_left_std = np.std(interps[mask_l], axis=0)[:11][::-1]
    interp_right = np.average(interps[mask_r], axis=0)[10:]
    interp_right_std = np.std(interps[mask_r], axis=0)[10:]
    plt.figure()
    plt.plot(np.linspace(0, width, 11), interp_left, label='Left')
    plt.fill_between(np.linspace(0, width, 11), interp_left - interp_left_std, interp_left + interp_left_std, alpha=0.5)
    plt.plot(np.linspace(0, width, 11), interp_right, label='Right')
    plt.fill_between(np.linspace(0, width, 11), interp_right - interp_right_std, interp_right + interp_right_std, alpha=0.5)
    plt.grid()
    plt.legend()
    plt.xlabel('Distance from center [nm]')
    plt.ylabel('Relative intensity [A.U.]')
    plt.xlim(0, width)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(rf'{save_name}Hg-Ar lamp_lines_symmetry_average.pdf')
    plt.show()

    print(f'Interp HWHM vals:\nLeft: {np.average(fwhm_l):.3f} +/- {np.std(fwhm_l):.3f} nm Right: {np.average(fwhm_r):.3f} +/- {np.std(fwhm_r):.3f} nm')
    p_val = ttest_ind(fwhm_l, fwhm_r, equal_var=False)[1]
    print(f'P-value: {p_val:.2e}')

    print('Pearson R:')
    print(f'left: {pearsonr(wav_vals_l, fwhm_l)}')
    print(f'right: {pearsonr(wav_vals_r, fwhm_r)}')

    interps_avg = np.average(interps, axis=0)
    model = lmfit.models.VoigtModel()
    params = model.guess(interps_avg, x=np.linspace(-width, width, 21))
    params['gamma'].set(value=params['sigma'].value, min=0, vary=True)
    params['sigma'].set(min=0, vary=True)
    params['center'].set(value=0, vary=False)
    model += lmfit.models.ConstantModel()
    params.add('c', value=0, vary=True)
    print('Voigt fit:')

    l_result = model.fit(interps_avg[:11][::-1], params, x=np.linspace(0, width, 11))
    if l_result.params["gamma"].value <= 1e-3:
        gaussian = lmfit.models.GaussianModel()
        g_params = gaussian.guess(interps_avg[:11][::-1], x=np.linspace(0, width, 11))
        gaussian += lmfit.models.ConstantModel()
        g_params.add('c', value=0, vary=True)
        l_result = gaussian.fit(interps_avg[:11][::-1], g_params, x=np.linspace(0, width, 11))
        try:
            print(f'left: sigma: {l_result.params["sigma"].value:.3f} +/- {l_result.params["sigma"].stderr:.3f} nm, hwhm: {l_result.params["fwhm"].value/2:.3f} +/- {l_result.params["fwhm"].stderr/2:.3f} nm')
        except TypeError:
            print(f'left: sigma: {l_result.params["sigma"].value:.3f} nm, hwhm: {l_result.params["fwhm"].value/2:.3f} nm')
            print(l_result.fit_report())
    else:
        try:
            print(f'left: sigma: {l_result.params["sigma"].value:.3f} +/- {l_result.params["sigma"].stderr:.3f} nm, '
                  f'gamma: {l_result.params["gamma"].value:.3f} +/- {l_result.params["gamma"].stderr:.3f} nm, '
                  f'hwhm: {l_result.params["fwhm"].value/2:.3f} +/- {l_result.params["fwhm"].stderr/2:.3f} nm')
        except TypeError:
            print(f'left: sigma: {l_result.params["sigma"].value:.3f} nm, gamma: {l_result.params["gamma"].value:.3f} nm, '
                  f'hwhm: {l_result.params["fwhm"].value/2:.3f} nm')
            print(l_result.fit_report())

    r_result = model.fit(interps_avg[10:], params, x=np.linspace(0, width, 11))
    if r_result.params["gamma"].value <= 1e-3:
        gaussian = lmfit.models.GaussianModel()
        g_params = gaussian.guess(interps_avg[10:], x=np.linspace(0, width, 11))
        gaussian += lmfit.models.ConstantModel()
        g_params.add('c', value=0, vary=True)
        r_result = gaussian.fit(interps_avg[10:], g_params, x=np.linspace(0, width, 11))
        try:
            print(f'right: sigma: {r_result.params["sigma"].value:.3f} +/- {r_result.params["sigma"].stderr:.3f} nm, '
                  f'hwhm: {r_result.params["fwhm"].value/2:.3f} +/- {r_result.params["fwhm"].stderr/2:.3f} nm')
        except TypeError:
            print(f'right: sigma: {r_result.params["sigma"].value:.3f} nm, hwhm: {r_result.params["fwhm"].value/2:.3f} nm')
            print(r_result.fit_report())
    else:
        try:
            print(f'right: sigma: {r_result.params["sigma"].value:.3f} +/- {r_result.params["sigma"].stderr:.3f} nm, '
                  f'gamma: {r_result.params["gamma"].value:.3f} +/- {r_result.params["gamma"].stderr:.3f} nm, '
                  f'hwhm: {r_result.params["fwhm"].value/2:.3f} +/- {r_result.params["fwhm"].stderr/2:.3f} nm')
        except TypeError:
            print(f'right: sigma: {r_result.params["sigma"].value:.3f} nm, gamma: {r_result.params["gamma"].value:.3f} nm, '
                  f'hwhm: {r_result.params["fwhm"].value/2:.3f} nm')
            print(r_result.fit_report())

    plt.figure()
    plt.plot(np.linspace(0, width, 11), interps_avg[:11][::-1], 'C0o', label='Left')
    plt.plot(np.linspace(0, width, 11), l_result.best_fit, 'C0-')
    plt.plot(np.linspace(0, width, 11), interps_avg[10:], 'C1o', label='Right')
    plt.plot(np.linspace(0, width, 11), r_result.best_fit, 'C1-')
    plt.grid()
    plt.xlabel('Distance from center [nm]')
    plt.ylabel('Relative intensity [A.U.]')
    plt.legend()
    plt.xlim(0, width)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(rf'{save_name}Hg-Ar lamp_lines_symmetry_average_voigt.pdf')
    plt.show()


if __name__ == "__main__":
    main()
