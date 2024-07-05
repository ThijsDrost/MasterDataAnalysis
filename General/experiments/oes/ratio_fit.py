import numpy as np
import scipy

from General.simulation.specair.specair import SpecAirSimulations, Spectrum
from General.plotting import plot

N2_RANGES = ((321, 338.0), (274.0, 282.0), (286.0, 298.0), (302.0, 317), (343.0, 357.0), (363.0, 379.0), (385.0, 398))
N2_RANGES_SEL = ((321, 338.0), (286.0, 298.0), (302.0, 317))


class RatioFit:
    def __init__(self, spectrum_loc, fwhm, output_wavelengths, ranges=N2_RANGES):
        self._ranges = ranges
        wavs = np.linspace(-4 * fwhm, 4 * fwhm, 100)
        peak = Spectrum(wavs, scipy.stats.norm.pdf(wavs, 0, fwhm))
        spectrum = SpecAirSimulations.from_hdf5(spectrum_loc, peak, output_wavelengths)

        t_rot = spectrum.rot_energy.reshape(-1)
        t_vib = spectrum.vib_energy.reshape(-1)
        self._t_vib_vals = np.unique(t_vib)
        spec = spectrum.datas.reshape(-1, spectrum.datas.shape[-1])

        results = np.zeros((len(t_rot), len(ranges) - 1))

        for index, (t_r, t_v, spec) in enumerate(zip(t_rot, t_vib, spec, strict=True)):
            results[index] = rel_ranges_intensity(spectrum.wavelengths, spec, ranges)

        res = []
        for index, t_v in enumerate(self._t_vib_vals):
            mask = t_vib == t_v
            res.append((np.min(results[mask], axis=0), np.max(results[mask], axis=0),
                        results[mask].mean(axis=0), results[mask].std(axis=0)))

        self._v_min, self._v_max, self._values, self._stds = np.transpose(res, axes=(1, 0, 2))

        self._min_interps = [scipy.interpolate.interp1d(min_vals, self._t_vib_vals, kind='cubic', fill_value=None,
                                                        bounds_error=False) for min_vals in self._v_min.T]
        self._max_interps = [scipy.interpolate.interp1d(max_vals, self._t_vib_vals, kind='cubic', fill_value=None,
                                                        bounds_error=False) for max_vals in self._v_max.T]
        self._mean_interps = [scipy.interpolate.interp1d(mean_vals, self._t_vib_vals, kind='cubic', fill_value=None,
                                                         bounds_error=False) for mean_vals in self._values.T]

    def plot(self, *, plot_kwargs, labels=None, legend_kwargs=None, **kwargs):
        plt_kwargs = {'xlabel': 'Vibrational temperature [K]', 'ylabel': 'Ratio [A.U.]'}
        plot_kwargs = plot.set_defaults(plot_kwargs, **plt_kwargs)
        if labels is None:
            labels = [str(index) for index in range(len(self._ranges[1:]))]
        legend_kwargs = plot.set_defaults(legend_kwargs, title='Range')
        return plot.errorrange(self._t_vib_vals, self._values, yerr=self._stds, plot_kwargs=plot_kwargs, labels=labels,
                               legend_kwargs=legend_kwargs, **kwargs)

    def fit(self, wavelength, intensity):
        ratios = rel_ranges_intensity(wavelength, intensity, self._ranges)
        results = np.zeros((len(self._max_interps), 2))
        for index, (min_interp, max_interp, mean_interp) in enumerate(zip(self._min_interps, self._max_interps, self._mean_interps)):
            results[index] = (min_interp(ratios[index]), max_interp(ratios[index]))
        return results


def ranges_intensity(wav, intensity, ranges: tuple[tuple[float, float], ...]):
    results = []
    for r in ranges:
        mask = (wav > r[0]) & (wav < r[1])
        results.append(intensity[mask].sum())
    return results


def rel_ranges_intensity(wav, intensity, ranges: tuple[tuple[float, float], ...]):
    results = np.array(ranges_intensity(wav, intensity, ranges))
    return results[1:]/results[0]


if __name__ == '__main__':
    loc = r'E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\N2_fps_rot_500-5000_vib_1000-11000_elec_12000.hdf5'
    fwhm = 1
    ratio_fitter = RatioFit(loc, fwhm, np.linspace(270, 450, 200))
