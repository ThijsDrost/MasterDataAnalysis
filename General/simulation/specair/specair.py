import warnings

import numpy as np
import scipy
import lmfit

from General.experiments.spectrum import Spectrum
from General.simulation.specair.hdf5 import SpecAirData, read_hdf5
from General.plotting import plot


class SpecAirSimulations:
    def __init__(self, specair_data, slit_function, output_wavelengths, *, convolve_before=True, resolution=None,
                 interpolator_kwargs=None):
        """
        Class to interpolate the simulated spectra.

        Parameters
        ----------
        specair_data: SpecAirData
            The Specair data
        slit_function: Spectrum
            The slit function of the spectrometer
        output_wavelengths: np.ndarray
            The wavelengths at which the spectra should be evaluated
        convolve_before: bool
            If True the spectra will be convolved before interpolation, otherwise the convolution will be done after interpolation
        resolution: float
            The resolution of the spectra. If None the resolution will be one tenth of the median of the differences between the
            output_wavelengths
        interpolator_kwargs:
            The kwargs for the RegularGridInterpolator

        Notes
        -----
        The files in the simulation_loc should be named 'rot_{rot_energy}K_{}_{}_vib_{vib_energy}K.txt' where {rot_energy} and {vib_energy} are the
        rotational and vibrational energies of the molecule respectively. The files should be tab separated with the first column the
        wavelengths and the second the intensities.

        The spectra will be interpolated with a RegularGridInterpolator. The interpolation is done on the rotational and vibrational
        energies.

        The slit function should be higher in resolution than the spectra.

        If convolve_before is False the spectra will be convolved after interpolation. This is done by convolving the interpolated
        spectra with the slit function. The resolution of the spectra will be lowered to the given resolution before interpolation.

        It checks whether every value for rotational temperature has the same values for vibrational temperature and vice versa.
        If not, it will try to find the intersection of the values and use only those, it will give a warning if it has to remove
        data.
        """
        self.convolve_before = convolve_before
        self.resolution = resolution

        self.rot_energy = specair_data.rot_values
        self.vib_energy = specair_data.vib_values
        self.wavelengths = specair_data.wavelengths
        self.intensities = specair_data.intensities
        if resolution is None:
            resolution = np.median(np.diff(output_wavelengths))/10

        if convolve_before:
            self.intensities = np.array([convolve_spectrum(Spectrum(wav, inten), slit_function, output_wavelengths, resolution).intensities
                                         for wav, inten in zip(self.wavelengths, self.intensities, strict=True)])
            self.wavelengths = output_wavelengths

        rot_values = np.unique(self.rot_energy)
        vib_values = np.unique(self.vib_energy)
        if len(rot_values)*len(vib_values) > len(self.intensities):
            # If there are more unique combinations than data, some data is missing
            # Try to find the intersection of the values, to construct a regular grid
            vib_set = set(vib_values[rot_values == rot_values[0]])
            for rot_value in rot_values[1:]:
                vib_set = vib_set.intersection(set(vib_values[rot_values == rot_value]))
            if len(vib_set) < 2:
                raise ValueError("Not enough data to interpolate")

            rot_set = set(rot_values[vib_values == vib_values[0]])
            for vib_value in vib_values[1:]:
                rot_set = rot_set.intersection(set(rot_values[vib_values == vib_value]))
            if len(rot_set) < 2:
                raise ValueError("Not enough data to interpolate")

            mask = np.array([rot in rot_set and vib in vib_set for rot, vib in zip(self.rot_energy, self.vib_energy)])

            self.intensities = self.intensities[mask]
            self.rot_energy = self.rot_energy[mask]
            self.vib_energy = self.vib_energy[mask]

            warnings.warn("Not enough data to interpolate, some data is removed")
        elif len(rot_values)*len(vib_values) < len(self.intensities):
            # If there is more data than unique combinations, there is duplicate data, which is not allowed
            raise ValueError("Duplicate data found, cannot interpolate")

        self.datas = self.intensities/np.max(self.intensities, axis=-1)[:, None]

        self.slit_function = slit_function
        self.output_wavelengths = output_wavelengths

        # reshape datas to (rot_energy, vib_energy, wavelengths) and make sure they are sorted
        sorter = np.lexsort((self.vib_energy, self.rot_energy))
        rot_axis_len = len(np.unique(self.rot_energy))
        vib_axis_len = len(np.unique(self.vib_energy))
        self.rot_energy = self.rot_energy[sorter].reshape(rot_axis_len, vib_axis_len)
        self.vib_energy = self.vib_energy[sorter].reshape(rot_axis_len, vib_axis_len)
        self.datas = self.datas[sorter].reshape(rot_axis_len, vib_axis_len, -1)

        interpolator_kwargs = interpolator_kwargs or {}
        self._interpolator = scipy.interpolate.RegularGridInterpolator((self.rot_energy[:, 0], self.vib_energy[0]),
                                                                       self.datas, **interpolator_kwargs)

    @staticmethod
    def from_hdf5(hdf5_loc: str, slit_function: Spectrum, output_wavelengths: np.ndarray, *, convolve_before=True, resolution=None,
                  interpolator_kwargs=None):
        return SpecAirSimulations(read_hdf5(hdf5_loc), slit_function, output_wavelengths, convolve_before=convolve_before,
                                  resolution=resolution, interpolator_kwargs=interpolator_kwargs)

    def __call__(self, rot_energy, vib_energy):
        return self._interpolate(rot_energy, vib_energy)

    def interpolate(self, rot_energy, vib_energy):
        return self._interpolate(rot_energy, vib_energy)

    def _interpolate(self, rot_energy, vib_energy):
        interpolation = self._interpolator((rot_energy, vib_energy))
        if self.convolve_before:
            return interpolation
        return convolve_spectrum(Spectrum(self.wavelengths, interpolation), self.slit_function, self.output_wavelengths,
                                 self.resolution)

    def model(self, prefix=''):
        def interp_model(amplitude, rot_energy, vib_energy):
            return amplitude*self.interpolate(rot_energy, vib_energy)

        model = lmfit.Model(interp_model, independent_vars=[], param_names=['amplitude', 'rot_energy', 'vib_energy'], prefix=prefix)
        model.set_param_hint(f'{prefix}rot_energy', value=self.rot_energy.mean(), min=self.rot_energy.min(), max=self.rot_energy.max())
        model.set_param_hint(f'{prefix}vib_energy', value=self.vib_energy.mean(), min=self.vib_energy.min(), max=self.vib_energy.max())
        model.set_param_hint(f'{prefix}amplitude', value=1, min=0)
        return model

    def first_global_fit_result_plot(self, data):
        global_fit = self._global_fit_res(data)
        results = np.average((self.datas - global_fit)**2, axis=-1)
        t_vibs = self.vib_energy[0]
        t_rots = self.rot_energy[:, 0]
        plot_kwargs = {'xlabel': 'T$_{vib}$ [K]', 'ylabel': 'Squared difference [A.U.]'}
        legend_kwargs = {'title': 'T$_{rot}$ [K]'}
        return plot.lines(t_vibs, results, plot_kwargs=plot_kwargs, labels=t_rots, legend_kwargs=legend_kwargs)

    def best_global_fit_result_plot(self, data, plot_kwargs=None, legend_kwargs=None):
        global_fit = self._global_fit_res(data)
        results = np.average((global_fit - data)**2, axis=-1)
        min_idx = np.unravel_index(np.argmin(results), results.shape)
        t_vib, t_rot = self.vib_energy[*min_idx], self.rot_energy[*min_idx]
        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Wavelength [nm]', ylabel='Intensity [A.U.]')
        legend_kwargs = plot.set_defaults(legend_kwargs, title='T$_{rot}$ [K]')

        return (t_vib, t_rot), plot.lines(self.output_wavelengths, [data, global_fit[min_idx]],
                          plot_kwargs=plot_kwargs, labels=['data', 'best_fit'], legend_kwargs=legend_kwargs)

    def _global_fit_res(self, data):
        def model(a, iten):
            return a*iten

        model = lmfit.Model(model, independent_vars=['iten'], param_names=['a'])
        params = model.make_params()
        params['a'].value = 1
        result = np.empty_like(self.datas)
        for i, row in enumerate(self.datas):
            for j, intensity in enumerate(row):
                res = model.fit(intensity, params, iten=data)
                result[i, j] = intensity/res.params['a'].value
        return result


class N2SpecAirSimulations(SpecAirSimulations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranges = [(274.0, 282.0), (286.0, 298.0), (302.0, 318), (321, 338.5), (345.0, 359.0), (365.0, 381.5), (385.0, 402)]
        self.masks = [
            (self.output_wavelengths > r[0]) & (self.output_wavelengths < r[1]) for r in self.ranges
        ]
        self.total_mask = np.any(self.masks, axis=0)

    def model(self, prefix=''):
        def interp_model(rot_energy, vib_energy, amplitude1, amplitude2, amplitude3, amplitude4, amplitude5, amplitude6, amplitude7):
            interp = self.interpolate(rot_energy, vib_energy)
            result = np.zeros_like(interp)
            amplitudes = [amplitude1, amplitude2, amplitude3, amplitude4, amplitude5, amplitude6, amplitude7]
            for amplitude, mask in zip(amplitudes, self.masks):
                result[mask] = amplitude*interp[mask]
            # result[~self.total_mask] = inten[~self.total_mask]
            return result

        param_names = ['rot_energy', 'vib_energy'] + [f'amplitude{i+1}' for i in range(len(self.ranges))]
        model = lmfit.Model(interp_model, independent_vars=[], param_names=param_names, prefix=prefix)
        model.set_param_hint(f'{prefix}rot_energy', value=self.rot_energy.mean(), min=self.rot_energy.min(), max=self.rot_energy.max())
        model.set_param_hint(f'{prefix}vib_energy', value=self.vib_energy.mean(), min=self.vib_energy.min(), max=self.vib_energy.max())
        for i in range(len(self.ranges)):
            model.set_param_hint(f'{prefix}amplitude{i+1}', value=1, min=0)
        return model

    def _global_fit_res(self, data):
        def model(a, iten):
            return a * iten

        model = lmfit.Model(model, independent_vars=['iten'], param_names=['a'])
        params = model.make_params()
        params['a'].value = 1
        result = np.empty_like(self.datas)
        for i, row in enumerate(self.datas):
            for j, intensity in enumerate(row):
                res = model.fit(intensity[self.total_mask], params, iten=data[self.total_mask])
                result[i, j] = intensity / res.params['a'].value
        return result

    @staticmethod
    def from_hdf5(hdf5_loc: str, slit_function: Spectrum, output_wavelengths: np.ndarray, *, convolve_before=True, resolution=None,
                  interpolator_kwargs=None):
        return N2SpecAirSimulations(read_hdf5(hdf5_loc), slit_function, output_wavelengths, convolve_before=convolve_before,
                                    resolution=resolution, interpolator_kwargs=interpolator_kwargs)

def convolve_spectrum(spectrum_lines: Spectrum, slit_function: Spectrum, output_wavelengths: np.ndarray, resolution=1e-1) -> Spectrum:
    """
    Convolve the spectrum with the slit function

    Parameters
    ----------
    spectrum_lines: Spectrum
        The intensities of the spectrum at each wavelength
    slit_function: Spectrum
        The slit function. The resolution of the slit function should be (quite a bit) higher than the resolution of the spectrum
    output_wavelengths: np.ndarray
        The wavelengths at which the spectrum is to be evaluated
    resolution: float
        The spectral lines within a distance of `resolution` (in nm) will be convolved together

    Returns
    -------
    Spectrum
        The convolved spectrum
    """
    return Spectrum(output_wavelengths,
                    _convolve_spectrum((spectrum_lines.wavelengths, spectrum_lines.intensities),
                                       (slit_function.wavelengths, slit_function.intensities), output_wavelengths, resolution))


def _convolve_spectrum(spectrum_lines: tuple[np.ndarray, np.ndarray], slit_function: tuple[np.ndarray, np.ndarray],
                       output_wavelengths: np.ndarray, resolution: float):
    result_intensity = np.zeros_like(output_wavelengths)
    pixel_bounds = _pixel_bounds(output_wavelengths)
    if np.min(np.diff(spectrum_lines[0])) < 0.9*resolution:
        spectrum_lines = _lower_resolution(spectrum_lines, resolution)
    for wavelength, intensity in zip(*spectrum_lines):
        if intensity == 0:
            continue

        wavelengths = wavelength + slit_function[0]
        boundaries = np.searchsorted(wavelengths, pixel_bounds)
        for index, boundary in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if boundary[0] == boundary[1]:
                continue
            result_intensity[index] += intensity*np.sum(slit_function[1][boundary[0]:boundary[1]])
    return result_intensity


def _pixel_bounds(pixel_locs: np.ndarray):
    values = np.zeros(len(pixel_locs) + 1)
    values[1:-1] = (pixel_locs[1:] + pixel_locs[:-1]) / 2
    values[0] = pixel_locs[0] - (pixel_locs[1] - pixel_locs[0]) / 2
    values[-1] = pixel_locs[-1] + (pixel_locs[-1] - pixel_locs[-2]) / 2
    return values


def lower_resolution(spectrum: Spectrum, resolution: float):
    """
    Lower the resolution of the spectrum to the given resolution

    Parameters
    ----------
    spectrum: Spectrum
        The spectrum to be lowered in resolution
    resolution: float
        The resolution to which the spectrum should be lowered. It gives the width of the bins in nm.

    Returns
    -------
    Spectrum
        The spectrum with the lower resolution
    """
    return Spectrum(*_lower_resolution((spectrum.wavelengths, spectrum.intensities), resolution))


def _lower_resolution(spectrum: tuple[np.ndarray, np.ndarray], resolution: float):
    bounds = np.arange(spectrum[0].min(), spectrum[0].max(), resolution)
    new_wavelengths = (bounds[:-1] + bounds[1:]) / 2
    new_intensities = np.empty_like(new_wavelengths)
    for i, bound in enumerate(bounds[:-1]):
        mask = (spectrum[0] >= bound) & (spectrum[0] < bounds[i+1])
        new_intensities[i] = np.sum(spectrum[1][mask])
    return new_wavelengths, new_intensities


if __name__ == '__main__':
    loc = r"E:\OneDrive - TU Eindhoven\Master thesis\SpecAir"
    wavs = np.linspace(-6, 6, 120)
    intensity = scipy.stats.norm.pdf(wavs, 0, 0.5)
    result = SpecAirSimulations(loc, Spectrum(wavs, intensity), np.arange(300, 320, 0.1))
