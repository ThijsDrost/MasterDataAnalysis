import warnings

import numpy as np
import lmfit
import pandas as pd


class WavelengthCalibration:
    # peak 801.0472 is midway between peaks 800.6157 801.4786

    peaks = [
        253.6517, 265.2039, 265.3679, 275.28, 289.36, 296.7280, 302.1498, 313.1548, 334.1481, 365.0156, 404.6565, 435.8335, 546.0735, 576.9598,
        579.0663, 696.5431, 706.7218, 714.7042, 727.2936, 738.3980, 750.3869, 763.5106, 772.3761, 794.8176, 801.0472, 811.5311, 826.4522, 842.4648, 852.1142,
        866.7944, 912.2967, 922.4499, 965.7786, 9784.503, 1013.976
    ]

    avantes_peaks = [
        253.6517, 265.2039, 265.3679, 275.28, 289.36, 296.7280, 302.1498, 312.57, 334.1481, 365.0156, 404.6565, 407.78, 434.75, 435.8335, 546.0735, 576.9598,
        579.0663, 696.5431, 706.7218, 727.2936, 738.3980, 763.5106, 772.40, 794.8176, 801.0472, 811.5311, 842.4648, 852.1142, 912.2967, 922.4499
    ]

    broad_peaks = [
        255, 395
    ]

    @staticmethod
    def wavelength_calibration(wavelengths, intensities, peak_width: tuple[float, ...] = None, wavelength_range: tuple[float, float] = None, peak_type='Gauss'):
        if peak_type.lower() == 'gauss':
            model = lmfit.models.GaussianModel
        elif peak_type.lower() == 'lorentz':
            model = lmfit.models.LorentzianModel
        elif peak_type.lower() == 'voigt':
            model = lmfit.models.VoigtModel
        else:
            raise ValueError(f'Unknown peak type `{peak_type}`, allowed values are: `Gauss`, `Lorentz`, or `Voigt`')

        if peak_type.lower() == 'voigt' and len(peak_width) > 2:
            raise ValueError(f'Voigt model only accepts one or two peak widths, not {len(peak_width)}')
        if peak_type.lower() in ('gauss', 'lorentz') and len(peak_width) > 1:
            raise ValueError(f'Gauss and Lorentz models only accept one peak width, not {len(peak_width)}')

        if peak_width is not None:
            if peak_type.lower() == 'voigt':
                if len(peak_width) == 1:
                    peak_width = {'sigma': peak_width[0], 'gamma': peak_width[0]}
                else:
                    peak_width = {'sigma': peak_width[0], 'gamma': peak_width[1]}
            if peak_type.lower() in ('gauss', 'lorentz'):
                peak_width = {'sigma': peak_width[0]}

        if wavelength_range is None:
            wavelength_range = (wavelengths[0], wavelengths[-1])

        total_width = peak_width['sigma'] + peak_width.get('gamma', 0)
        peaks = [p for p in WavelengthCalibration.peaks if (wavelength_range[0]+3*total_width) < p < (wavelength_range[1]-3*total_width)]

        composite_model = lmfit.models.ConstantModel(prefix='background_')
        params = composite_model.make_params()
        params.add('sigma', value=peak_width['sigma'], min=0, vary=True)
        if peak_type.lower() == 'voigt':
            params.add('gamma', value=peak_width['gamma'], min=0, vary=True)
        params.add('wav_0', value=0, vary=True)
        params.add('wav_1', value=0, vary=True)
        params.add('wav_2', value=0, vary=True)

        for index, wav in enumerate(peaks):
            composite_model += model(prefix=f'peak_{index}_')

            mask = (wavelengths > (wav-3*total_width)) & (wavelengths < (wav+3*total_width))
            if np.sum(mask) == 0:
                closest = np.argmin(np.abs(wavelengths-peaks[0]))
                mask[closest] = True
            amplitude = np.max(intensities[mask])
            if peak_type.lower() == 'gauss':
                amplitude = amplitude*np.sqrt(2*np.pi)*peak_width['sigma']
            elif peak_type.lower() == 'lorentz':
                amplitude = amplitude*np.pi*peak_width['sigma']
            elif peak_type.lower() == 'voigt':
                amplitude = amplitude*np.sqrt(2*np.pi)*peak_width['sigma']*peak_width['gamma']

            # params.add(f'peak_{index}_center', value=wav, min=wav-3*total_width, max=wav+3*total_width, vary=True)
            params.add(f'peak_{index}_center', expr=f'{wav} + wav_0 + wav_1*{wav:.4f} + wav_2*{wav**2:.4f}')
            params.add(f'peak_{index}_amplitude', value=amplitude, min=0, vary=True)
            params.add(f'peak_{index}_sigma', expr='sigma')
            if peak_type.lower() == 'voigt':
                params.add(f'peak_{index}_gamma', expr='gamma')

        for index, wav in enumerate(WavelengthCalibration.broad_peaks):
            composite_model += lmfit.models.GaussianModel(prefix=f'peak_b{index}_')
            params.update(lmfit.models.GaussianModel(prefix=f'peak_b{index}_').make_params())
            params[f'peak_b{index}_center'].set(value=wav, min=wav-25, max=wav+25, vary=False)
            params[f'peak_b{index}_amplitude'].set(value=100, min=0, vary=True)
            params[f'peak_b{index}_sigma'].set(value=30, min=5, max=100, vary=True)

        return composite_model.fit(intensities, params, x=wavelengths)

    @staticmethod
    def wavelength_calibration2(wavelengths, intensities, peak_width: tuple[float, ...] = None, wavelength_range: tuple[float, float] = None, peak_type='Gauss'):
        if peak_type.lower() == 'gauss':
            model = lmfit.models.GaussianModel
        elif peak_type.lower() == 'lorentz':
            model = lmfit.models.LorentzianModel
        elif peak_type.lower() == 'voigt':
            model = lmfit.models.VoigtModel
        else:
            raise ValueError(f'Unknown peak type `{peak_type}`, allowed values are: `Gauss`, `Lorentz`, or `Voigt`')

        if peak_type.lower() == 'voigt' and len(peak_width) > 2:
            raise ValueError(f'Voigt model only accepts one or two peak widths, not {len(peak_width)}')
        if peak_type.lower() in ('gauss', 'lorentz') and len(peak_width) > 1:
            raise ValueError(f'Gauss and Lorentz models only accept one peak width, not {len(peak_width)}')

        if peak_width is not None:
            if peak_type.lower() == 'voigt':
                if len(peak_width) == 1:
                    peak_width = {'sigma': peak_width[0], 'gamma': peak_width[0]}
                else:
                    peak_width = {'sigma': peak_width[0], 'gamma': peak_width[1]}
            if peak_type.lower() in ('gauss', 'lorentz'):
                peak_width = {'sigma': peak_width[0]}

        if wavelength_range is None:
            wavelength_range = (wavelengths[0], wavelengths[-1])

        total_width = peak_width['sigma'] + peak_width.get('gamma', 0)
        peaks = [p for p in WavelengthCalibration.avantes_peaks if (wavelength_range[0] < p < wavelength_range[1])]

        centers = np.zeros(len(peaks))
        centers_std = np.zeros(len(peaks))
        sigma = np.zeros(len(peaks))
        sigma_std = np.zeros(len(peaks))
        intercept = np.zeros(len(peaks))
        intercept_std = np.zeros(len(peaks))
        slope = np.zeros(len(peaks))
        slope_std = np.zeros(len(peaks))
        amplitude = np.zeros(len(peaks))
        amplitude_std = np.zeros(len(peaks))

        for index, wav in enumerate(peaks):
            model = lmfit.models.LinearModel() + lmfit.models.GaussianModel()
            mask = (wavelengths > (wav-3*total_width)) & (wavelengths < (wav+3*total_width))

            wav, inten = wavelengths[mask], intensities[mask]
            params = lmfit.Parameters()
            params.add('slope', value=0, vary=True)
            params.add('intercept', value=inten[0], vary=True)
            peak_model = lmfit.models.GaussianModel().guess(inten, x=wav)
            params.update(peak_model)
            params['center'].set(min=wav[0], max=wav[-1])
            params['amplitude'].set(min=0)

            close_peaks = [p for p in peaks if ((wav[0] < (p+3*total_width)) and ((p-3*total_width) < wav[-1]))]
            for jindex, p in enumerate(close_peaks):
                p_model = lmfit.models.GaussianModel(prefix=f'peak_{jindex}_')
                params.add(f'peak_{jindex}_center', value=p, vary=False)
                params.add(f'peak_{jindex}_sigma', expr='sigma')
                wav_loc = np.argmin(np.abs(wavelengths-p))
                params.add(f'peak_{jindex}_amplitude', value=intensities[wav_loc]/(total_width*2.5), vary=True, min=0)
                model += p_model

            result = model.fit(inten, params, x=wav)
            centers[index] = result.best_values['center']
            centers_std[index] = result.params['center'].stderr
            sigma[index] = result.best_values['sigma']
            sigma_std[index] = result.params['sigma'].stderr
            intercept[index] = result.best_values['intercept']
            intercept_std[index] = result.params['intercept'].stderr
            slope[index] = result.best_values['slope']
            slope_std[index] = result.params['slope'].stderr
            amplitude[index] = result.best_values['amplitude']
            amplitude_std[index] = result.params['amplitude'].stderr
        
        return pd.DataFrame({
            'center': centers, 'center_std': centers_std,
            'sigma': sigma, 'sigma_std': sigma_std,
            'intercept': intercept, 'intercept_std': intercept_std,
            'slope': slope, 'slope_std': slope_std,
            'amplitude': amplitude, 'amplitude_std': amplitude_std
        }, index=peaks)

    @staticmethod
    def wavelength_calibration3(wavelengths, intensities, wavelength_range: tuple[float, float] = None, min_intensity: float = 10.0):
        if wavelength_range is None:
            wavelength_range = (wavelengths[0], wavelengths[-1])

        peaks = [p for p in WavelengthCalibration.peaks if (wavelength_range[0] < p < wavelength_range[1])]
        centers = np.zeros(len(peaks))
        for index, wav in enumerate(peaks):
            mask = (wavelengths > (wav-3)) & (wavelengths < (wav+3))
            wavs, inten = wavelengths[mask], intensities[mask]
            changes = np.argwhere(np.diff((np.diff(inten) > 0).astype(int)) == -1).flatten()
            changes = changes[inten[changes+1] > min_intensity]
            if len(changes) == 0:
                centers[index] = np.nan
                warnings.warn(f'No peak found for wavelength {wav}')
                continue
            best_index = changes[np.argmin(np.abs(wavs[changes+1]-wav))]+1
            peak_wav = WavelengthCalibration.quadratic_peak(wavs[best_index-1:best_index+2], inten[best_index-1:best_index+2])
            centers[index] = peak_wav

        return np.array(peaks), centers

    @staticmethod
    def quadratic_peak(x, y):
        """
        Find the x-coordinate peak of a quadratic function given three points.
        """
        x_1, x_2, x_3 = x
        y_1, y_2, y_3 = y
        return -((x_1**2)*(y_2 - y_3) + (x_2**2)*(-y_1 + y_3) + (x_3**2)*(y_1 - y_2))/(2*(x_1*(-y_2 + y_3) + x_2*(y_1 - y_3) + x_3*(-y_1 + y_2)))

    @staticmethod
    def quadratic_peak_xy(x, y):
        """
        Find the x and y coordinate peak of a quadratic function given three points.
        """
        a, b, c = WavelengthCalibration.quadratic(x, y)
        x_peak = -b / (2 * a)
        y_peak = a * (x_peak ** 2) + b * x_peak + c
        return x_peak, y_peak

    @staticmethod
    def quadratic(x, y):
        x_1, x_2, x_3 = x
        y_1, y_2, y_3 = y
        denominator = x_1 ** 2 * x_2 - x_1 ** 2 * x_3 - x_1 * x_2 ** 2 + x_1 * x_3 ** 2 + x_2 ** 2 * x_3 - x_2 * x_3 ** 2
        a = (-x_1 * y_2 + x_1 * y_3 + x_2 * y_1 - x_2 * y_3 - x_3 * y_1 + x_3 * y_2) / denominator
        b = (x_1 ** 2 * y_2 - x_1 ** 2 * y_3 - x_2 ** 2 * y_1 + x_2 ** 2 * y_3 + x_3 ** 2 * y_1 - x_3 ** 2 * y_2) / denominator
        c = (x_1 ** 2 * x_2 * y_3 - x_1 ** 2 * x_3 * y_2 - x_1 * x_2 ** 2 * y_3 + x_1 * x_3 ** 2 * y_2 + x_2 ** 2 * x_3 * y_1 - x_2 * x_3 ** 2 * y_1) / denominator
        return a, b, c