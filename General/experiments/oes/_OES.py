import warnings

from attrs import define
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import lmfit

from General.plotting import plot, linestyles
from General.plotting import cbar as mcbar
from General import numpy_funcs as npf
from General.experiments.spectrum import TemporalSpectrum
from General.experiments import WavelengthCalibration
from General.itertools import argmax

DEAD_PIXEL = {
    '2201415U1': [562, 336]
}

RANGES = {
    'ar': ((694.575, 698.575), (704.775, 708.775), (725.356, 729.356), (736.403, 740.403), (748.520, 752.520),
           (761.504, 765.504), (770.433, 774.433), (775.220, 779.220), (792.782, 796.782), (799.292, 803.292),
           (809.485, 813.485), (824.425, 828.425), (840.346, 844.346), (850.103, 854.103), (910.275, 914.275))
}

PEAKS = {
    'ar': (696.575, 706.775, 727.356, 738.403, 750.520, 763.504, 772.433,
           794.782, 801.292, 811.485, 826.425, 842.346, 852.103, 912.275)
}

ENERGIES = {
    'ar': (13.33, 13.30, 13.33, 13.30, 13.48, 13.17, 13.15, 13.28, 13.09, 13.08, 13.33, 13.09, 13.28, 12.91)
}

PEAKS['argon'] = PEAKS['ar']


@define(frozen=True)
class OESData:
    spectrum: TemporalSpectrum
    spectrometer_settings: dict = None

    @staticmethod
    def new(wavelengths, intensities, relative_time_s, spectrometer_settings=None):
        spectrum = TemporalSpectrum(wavelengths, intensities, relative_time_s)
        return OESData(spectrum, spectrometer_settings)

    def __getitem__(self, item):
        return OESData(self.spectrum[item], self.spectrometer_settings)

    @property
    def wavelengths(self):
        return self.spectrum.wavelengths

    @property
    def intensities(self):
        return self.spectrum.intensities

    def remove_dead_pixels(self):
        spectrometer = self.spectrometer_settings.get('serial_number', None)
        if spectrometer is None:
            warnings.warn('No spectrometer settings found, cannot remove dead pixels.')
            return self
        if spectrometer not in DEAD_PIXEL:
            warnings.warn(f'No dead pixels found for spectrometer {spectrometer}.')
            return self
        dead_pixels = DEAD_PIXEL[spectrometer]

        # interpolate dead pixels from the surrounding pixels
        intensities = self.spectrum.intensities.copy()
        for dead_pixel in dead_pixels:
            intensities[:, dead_pixel] = (intensities[:, dead_pixel - 1] + intensities[:, dead_pixel + 1]) / 2

        spectrum = TemporalSpectrum(self.spectrum.wavelengths, intensities, self.spectrum.times)
        return OESData(spectrum, self.spectrometer_settings)

    def remove_background(self, values: np.ndarray):
        spectrum = self.spectrum.remove_background(values)
        return OESData(spectrum, self.spectrometer_settings)

    def remove_background_index(self, background_index: int):
        spectrum = self.spectrum.remove_background_index(background_index)
        return OESData(spectrum, self.spectrometer_settings)

    def remove_background_interp(self, start_indexes: int | tuple[int | None, int], end_indexes: int | tuple[int, int | None]):
        spectrum = self.spectrum.remove_background_interp(start_indexes, end_indexes)
        return OESData(spectrum, self.spectrometer_settings)

    def remove_background_interp_off(self, is_on_kwargs=None):
        values = self.is_on(**is_on_kwargs)
        diff = np.diff(values.astype(int))
        try:
            up = np.where(diff == 1)[0][0]
            down = np.where(diff == -1)[0][0]
        except IndexError:
            warnings.warn('No background found.')
            return self

        up = up - 3
        down = down + 4
        if up < 0 or down >= len(values):
            warnings.warn('No background found.')
            return self

        return self.remove_background_interp((None, up), (down, None))


    def remove_baseline(self, wavelength_range):
        spectrum = self.spectrum.remove_baseline(wavelength_range)
        return OESData(spectrum, self.spectrometer_settings)

    def moving_average(self, num: int):
        spectrum = self.spectrum.moving_average(num)
        return OESData(spectrum, self.spectrometer_settings)

    def block_average(self, num: int):
        spectrum = self.spectrum.block_average(num)
        return OESData(spectrum, self.spectrometer_settings)

    def intensity_vs_wavelength_with_time(self, *, plot_kwargs=None, cbar_kwargs=None, cbar='turbo', block_average: int = None,
                                          moving_average: int = None, background_index=None, **kwargs):
        x_values = self.spectrum.wavelengths
        y_values = self.spectrum.intensities
        t_values = self.spectrum.times

        y_values = npf.averaging(y_values, block_average_num=block_average, moving_average_num=moving_average)
        t_values = npf.averaging(t_values, block_average_num=block_average, moving_average_num=moving_average)

        if background_index is not None:
            y_values = y_values - y_values[background_index]

        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Wavelength [nm]', ylabel='Intensity [A.U.]')

        colors, mappable = mcbar.cbar_norm_colors(t_values, cbar=cbar)
        cbar_kwargs = plot.set_defaults(cbar_kwargs, label='Time [s]', mappable=mappable)

        return plot.lines(x_values, y_values, colors=colors, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, **kwargs)

    def total_intensity_vs_time(self, *, plot_kwargs=None, block_average: int = None, moving_average: int = None, **kwargs):
        intensities = npf.averaging(self.spectrum.intensities, block_average_num=block_average, moving_average_num=moving_average)
        t_values = npf.averaging(self.spectrum.times, block_average_num=block_average, moving_average_num=moving_average)

        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='time [s]', ylabel='Intensity [A.U.]')

        return plot.lines(t_values, np.sum(intensities, axis=1), plot_kwargs=plot_kwargs, **kwargs)

    def total_intensity_vs_index(self, *, plot_kwargs=None, block_average: int = None, moving_average: int = None, **kwargs):
        intensities = npf.averaging(self.spectrum.intensities, block_average_num=block_average, moving_average_num=moving_average)
        t_values = npf.averaging(self.spectrum.times, block_average_num=block_average, moving_average_num=moving_average)

        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Index', ylabel='Intensity [A.U.]')

        return plot.lines(np.arange(len(t_values)), np.sum(intensities, axis=1), plot_kwargs=plot_kwargs, **kwargs)

    def ranged_intensities(self, ranges: tuple[tuple[float, float], ...], *, block_average=None, moving_average=None):
        intensities = npf.averaging(self.spectrum.intensities, block_average_num=block_average, moving_average_num=moving_average).T
        t_values = npf.averaging(self.spectrum.times, block_average_num=block_average, moving_average_num=moving_average)

        ranged_intensities = np.empty((len(ranges), len(t_values)))
        for index, (start, end) in enumerate(ranges):
            start_idx = np.searchsorted(self.spectrum.wavelengths, start)
            end_idx = np.searchsorted(self.spectrum.wavelengths, end)
            ranged_intensities[index] = np.average(intensities[start_idx:end_idx], axis=0)
        return t_values, ranged_intensities

    def ranged_intensity(self, wavelength_range: tuple[float, float] | tuple[tuple[float, float], ...] = None):
        if wavelength_range:
            mask = np.full(self.spectrum.intensities.shape[1], False)
            if isinstance(wavelength_range[0], float | int):
                wavelength_range = [wavelength_range]

            for start, end in wavelength_range:
                start_idx = np.searchsorted(self.spectrum.wavelengths, start)
                end_idx = np.searchsorted(self.spectrum.wavelengths, end)
                mask[start_idx:end_idx] = True

            return np.sum(self.spectrum.intensities[:, mask], axis=1)
        else:
            return np.sum(self.spectrum.intensities, axis=1)

    def peak_intensity(self, peaks: tuple[float | int, ...] | str, *, block_average=None, moving_average=None,
                       is_on_kwargs=None):
        if isinstance(peaks, str):
            try:
                peaks = PEAKS[peaks.lower()]
            except KeyError:
                raise ValueError(f'No peaks found for {peaks}.')

        if is_on_kwargs:
            mask = self.is_on(**is_on_kwargs)
            inten_vals = self.spectrum.intensities[mask]
            time_vals = self.spectrum.times[mask]
        else:
            inten_vals = self.spectrum.intensities
            time_vals = self.spectrum.times

        intensities = npf.averaging(inten_vals, block_average_num=block_average, moving_average_num=moving_average).T
        t_values = npf.averaging(time_vals, block_average_num=block_average, moving_average_num=moving_average)

        ranged_intensities = np.zeros((len(peaks), len(t_values)))
        peak_locs = np.zeros((len(peaks), len(t_values)))
        for index, peak in enumerate(peaks):
            peak_idx = np.searchsorted(self.spectrum.wavelengths, peak)
            max_idx = peak_idx - 1 + np.argmax(intensities[peak_idx - 1:peak_idx + 2], axis=0)
            indexer = [(idx-1, idx, idx+1) for idx in max_idx]
            sel_intensities = np.empty((3, len(max_idx)))
            for i in (-1, 0, 1):
                sel_intensities[i+1] = self.spectrum.intensities[np.arange(len(max_idx)), max_idx+i]

            mask = (sel_intensities[0] < sel_intensities[1]) & (sel_intensities[2] < sel_intensities[1])

            x, y = WavelengthCalibration.quadratic_peak_xy(self.spectrum.wavelengths[indexer].T[:, mask], sel_intensities[:, mask])

            ranged_intensities[index][mask] = y
            peak_locs[index][mask] = x
            ranged_intensities[index][~mask] = np.nan
            peak_locs[index][~mask] = np.nan
        return t_values, ranged_intensities, peak_locs

    def ranged_intensity_vs_wavelength_with_time(self, ranges: tuple[tuple[float, float], ...], *, plot_kwargs=None,
                                                 block_average: int = None, moving_average: int = None, line_kwargs=None,
                                                 legend_kwargs=None, labels=None, **kwargs):
        t_values, ranged_intensities = self.ranged_intensities(ranges, block_average=block_average, moving_average=moving_average)

        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='time [s]', ylabel='Intensity [A.U.]')
        line_kwargs = plot.set_defaults(line_kwargs, linestyle='', marker='.')
        legend_kwargs = plot.set_defaults(legend_kwargs, title='wav [nm]')

        if labels is None:
            labels = [f'{(start + end)/2}' for start, end in ranges]

        return plot.lines(t_values, ranged_intensities, plot_kwargs=plot_kwargs, labels=labels, line_kwargs=line_kwargs,
                          legend_kwargs=legend_kwargs, **kwargs)

    def peak_intensity_vs_wavelength_with_time(self, peaks: tuple[float | int, ...] | str, *, norm=False, cbar: str | bool = True, cbar_type='blocked',
                                               plot_kwargs=None, block_average=None, moving_average=None, line_kwargs=None, legend_kwargs=None,
                                               cbar_kwargs=None, labels=None, **kwargs):
        t_values, ranged_intensities, peak_locs = self.peak_intensity(peaks, block_average=block_average, moving_average=moving_average)
        if isinstance(peaks, str):
            peaks = self.peaks(peaks)

        # ranged_intensities[ranged_intensities < 0] = 0
        if norm:
            ranged_intensities = ranged_intensities / np.max(ranged_intensities, axis=1)[:, None]

        t_values = (t_values - t_values[0])/60

        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='time [min]', ylabel='Intensity [A.U.]')

        if cbar:
            if cbar is True:
                cbar = 'turbo'
            cmap = plt.get_cmap(cbar)
            if cbar_type == 'blocked':
                boundaries = [1.5*peaks[0] - 0.5*peaks[1]] + [(peaks[i] + peaks[i + 1]) / 2 for i in range(len(peaks) - 1)] + [
                    1.5*peaks[-1] - 0.5*peaks[1]]
                norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
                colors = cmap(norm(peaks))

                ticker = mpl.ticker.FixedFormatter([f'{int(x)}' for x in peaks])
                tick_loc = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
                t_cbar_kwargs = {'mappable': plt.cm.ScalarMappable(norm=norm, cmap=cmap), 'label': 'Peak wavelength [nm]',
                               'ticks': tick_loc, 'format': ticker}
            else:
                mappable, colors = mcbar.cbar_norm_colors(peaks, cbar=cbar)
                t_cbar_kwargs = {'mappable': mappable, 'label': 'Peak wavelength [nm]'}
            cbar_kwargs = plot.set_defaults(cbar_kwargs, **t_cbar_kwargs)

            return plot.lines(t_values, ranged_intensities, plot_kwargs=plot_kwargs, line_kwargs=line_kwargs, cbar_kwargs=cbar_kwargs,
                              colors=colors, **kwargs)
        else:
            legend_kwargs = plot.set_defaults(legend_kwargs, title='wav [nm]')

            if labels is None:
                labels = [f'{peak}' for peak in peaks]

            return plot.lines(t_values, ranged_intensities, plot_kwargs=plot_kwargs, labels=labels, line_kwargs=line_kwargs,
                              legend_kwargs=legend_kwargs, **kwargs)

    def peak_loc_intensities_with_time(self, peaks: tuple[float | int, ...] | str, is_on_kwargs=None):
        if isinstance(peaks, str):
            try:
                peaks = PEAKS[peaks.lower()]
            except KeyError:
                raise ValueError(f'No peaks found for {peaks}.')

        is_on_kwargs = is_on_kwargs or {'relative_threshold': 0.5}
        is_on = self.is_on(**is_on_kwargs)


        inten_vals = self.spectrum.intensities[is_on]
        time_vals = self.spectrum.times[is_on]

        peak_relative_intensities = np.empty((len(peaks), len(inten_vals)))
        peak_relative_locs = np.empty((len(peaks), len(inten_vals)))
        for index, peak in enumerate(peaks):
            peak_idx = np.searchsorted(self.spectrum.wavelengths, peak)
            max_idx = peak_idx - 1 + np.argmax(inten_vals[:, peak_idx - 1:peak_idx + 2], axis=1)
            x, y = WavelengthCalibration.quadratic_peak_xy(self.spectrum.wavelengths[(max_idx-1, max_idx, max_idx+1),],
                                                           inten_vals[np.arange(len(max_idx)), (max_idx-1, max_idx, max_idx+1)])
            wavs = (0.5*(self.spectrum.wavelengths[max_idx] + self.spectrum.wavelengths[max_idx - 1]),
                    0.5*(self.spectrum.wavelengths[max_idx] + self.spectrum.wavelengths[max_idx + 1]))
            mask = (x < wavs[0]) | (wavs[1] < x)  # if the wavelength is outside the pixel, the three pixels do not form an peak together
            y[mask] = np.nan
            peak_relative_intensities[index] = y
            x[mask] = np.nan
            peak_relative_locs[index] = x
        return time_vals, peak_relative_locs, peak_relative_intensities

    def is_on_fit(self, wavelength_range: tuple[float, float] | tuple[tuple[float, float], ...] = None):
        intensities = self.ranged_intensity(wavelength_range)

        times = self.spectrum.times - self.spectrum.times[0]

        model = lmfit.models.RectangleModel(form='linear') + lmfit.models.LinearModel()
        params = model.make_params()
        params['slope'].set(value=0)
        params['intercept'].set(value=0)
        params['amplitude'].set(value=0.9*np.max(intensities), min=0)
        params['center1'].set(value=0.1*times[-1])
        params['center2'].set(expr='center1 + width')
        params.add('width', value=0.8*times[-1], min=0)

        result = model.fit(intensities, params, x=times)
        on_mask = ((result.params['center1'].value + result.params['sigma1']/2) < times) & (times < result.params['center2'].value - result.params['sigma2']/2)

        # diff = np.diff(on_mask.astype(int))
        # up = np.where(diff == 1)[0][0]
        # down = np.where(diff == -1)[0][0]
        #
        # new_intensity = new_data.ranged_intensity(wavelength_range)
        #
        # result2 = model.fit(new_intensity, params, x=times)

        return on_mask

    def is_on(self, *, threshold=None, relative_threshold=None, wavelength_range: tuple[float, float] | tuple[tuple[float, float], ...] = None,
              use_max = True, fix_offset=5):
        """
        Returns a boolean array indicating whether the intensity is above a certain threshold or a certain relative threshold.
        If neither threshold nor relative_threshold is set, the relative threshold is set to 0.5.

        Parameters
        ----------
        threshold: float
            The absolute threshold (in counts) above which the average intensity is considered on.
        relative_threshold: float
            The relative threshold above which the intensity is considered on.
        wavelength_range: tuple[float, float] | tuple[tuple[float, float], ...]
            The wavelength range over which the intensity should be considered.

        Returns
        -------
        np.ndarray
            A boolean array indicating whether the intensity is above the threshold.

        Notes
        -------
        For the relative threshold, the minimum and maximum intensity are calculated and the threshold is set to the minimum plus
        the relative threshold times the difference between the maximum and minimum.
        """
        if fix_offset:
            if not isinstance(fix_offset, int):
                fix_offset = 5
            data = self.remove_background_interp((None, fix_offset), (len(self.spectrum.times) - fix_offset, None))
        else:
            data = self

        if threshold is not None and relative_threshold is not None:
            raise ValueError('Only one of threshold and relative_threshold can be set.')
        if threshold is None and relative_threshold is None:
            raise ValueError('Either threshold or relative_threshold should be set.')

        if wavelength_range is not None:
            if isinstance(wavelength_range[0], float | int):
                wavelength_range = [wavelength_range]
            mask = np.full(len(data.spectrum.wavelengths), False)
            for start, end in wavelength_range:
                start_idx = np.searchsorted(data.spectrum.wavelengths, start)
                end_idx = np.searchsorted(data.spectrum.wavelengths, end)
                mask[start_idx:end_idx] = True
            intensities = data.spectrum.intensities[:, mask]
        else:
            intensities = data.spectrum.intensities

        if use_max:
            intensities = np.max(intensities, axis=1)
        else:
            intensities = np.mean(intensities, axis=1)

        if threshold is not None:
            return intensities > threshold
        else:
            inten_min = np.min(intensities)
            inten_max = np.max(intensities)
            return intensities > inten_min + relative_threshold * (inten_max - inten_min)

    @staticmethod
    def peaks(string: str):
        try:
            return PEAKS[string.lower()]
        except KeyError:
            raise ValueError(f'No peaks found for {string}.')

    @staticmethod
    def energies(string: str):
        try:
            return ENERGIES[string.lower()]
        except KeyError:
            raise ValueError(f'No energies found for {string}.')