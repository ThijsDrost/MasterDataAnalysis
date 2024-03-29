# TODO: add docstrings
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import lmfit

from General.Data_handling import DataSet, import_hdf5
from General.Plotting import Plot
from General.Numpy_funcs import block_averages


class MeasurementsAnalyzer:
    def __init__(self, dataset: DataSet, cmap='turbo'):
        if not isinstance(dataset, DataSet):
            raise ValueError(f'dataset should be an instance of DataSet, not {type(dataset)}')
        self.data_set = dataset
        self.cmap = cmap

    @staticmethod
    def read_hdf5(loc, *, dependent='timestamp_s', cmap='turbo', wavelength_range=(200, 400)):
        dataset = import_hdf5(loc, dependent)
        return MeasurementsAnalyzer(DataSet.from_simple(dataset, wavelength_range=wavelength_range), cmap)

    @staticmethod
    def _timescale(timescale):
        if timescale.lower() in ('s', 'sec', 'second', 'seconds'):
            return 1
        elif timescale.lower() in ('m', 'min', 'minute', 'minutes'):
            return 60
        elif timescale.lower() in ('h', 'hour', 'hours'):
            return 3600
        else:
            raise ValueError(f'timescale should `sec`, `min` or `hour`, not {timescale}')

    @staticmethod
    def _averages(values, n_averages: int, average_over: int):
        if n_averages * average_over > len(values):
            warnings.warn(f'`average_over`*`n_averages` > `len(values)`, the averaging blocks will overlap')
        averaging_len = n_averages * average_over
        num_left = len(values) - averaging_len
        num_left_per = num_left / n_averages

        results = []
        for i in range(n_averages):
            start = round(i * (average_over + num_left_per))
            end = round((i + 1) * (average_over + num_left_per))
            results.append(np.mean(values[start:end], axis=0))
        return results

    @staticmethod
    def _boundaries(middle_values: np.ndarray):
        """
        Calculates the boundaries of the bins for a histogram with `middle_values` as the bin centers. Uses linear inter- and extrapolation.
        """
        boundaries = np.zeros(len(middle_values) + 1)
        boundaries[1:-1] = (middle_values[1:] + middle_values[:-1]) / 2
        boundaries[0] = 2 * middle_values[0] - boundaries[1]
        boundaries[-1] = 2 * middle_values[-1] - boundaries[-2]
        return boundaries

    @staticmethod
    def _avg(x, y, num, average_num):
        """
        Averages the data in `x` and `y` over `num` of length `average_num`.
        If `num` is None, the data is averaged with blocks of length `average_num`.
        If `average_num` is None, the data is averaged with `num` blocks.
        If both are None, the data is returned as is.
        """
        if isinstance(num, int):
            if average_num is None:
                average_num = len(y)//num
                x = block_averages(x, average_num)
                y = block_averages(y, average_num)
            elif isinstance(average_num, int):
                x = MeasurementsAnalyzer._averages(x, num, average_num)
                y = MeasurementsAnalyzer._averages(y, num, average_num)
            else:
                raise ValueError('`average_num` should be an integer or None')
        elif num is None:
            if average_num is None:
                return x, y
            num_val = len(y)//average_num
            x = block_averages(x, num_val)
            y = block_averages(y, num_val)
        else:
            raise ValueError('`num` should be an integer or None')
        return x, y

    def absorbance_with_wavelength_over_time(self, num, timescale='min', average_num: int = None, *, corrected=True, masked=True, plot_kwargs=None, save_loc=None, save_suffix='', show=True, save_kwargs=None,
                                             cbar_kwargs=None, line_kwargs=None, **kwargs):
        timestamps_s = self.data_set.variable
        timestamps_s = timestamps_s - timestamps_s[0]
        timestamps = timestamps_s/self._timescale(timescale)

        xs = self.data_set.get_wavelength(masked=masked)
        ys = self.data_set.get_absorbances(corrected=corrected, masked=masked)

        timestamps, ys = self._avg(timestamps, ys, num, average_num)

        cmap = plt.get_cmap(self.cmap)
        norm = plt.Normalize(vmin=0, vmax=timestamps[-1])
        colors = cmap(norm(timestamps))

        plot_kwargs = Plot.set_defaults(plot_kwargs, xlabel=f'Wavelength [nm]', ylabel='Absorbance [A.U.]', xlim=(xs[0], xs[-1]))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar_kwargs = Plot.set_defaults(cbar_kwargs, label=f'Time [{timescale}]', mappable=sm)
        line_kwargs = Plot.set_defaults(line_kwargs)

        if save_loc is not None:
            if '.' not in save_suffix:
                save_suffix += '.pdf'
            save_loc = os.path.join(save_loc, f'absorbance with wavelength over time{save_suffix}')
        return Plot.lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                          colors=colors, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs, save_kwargs=save_kwargs, **kwargs)

    def absorbance_over_time_with_wavelength(self, num, timescale='min', average_num: int = None, time_num: int = None, norm=True, *,
                                             min_absorbance=0.001, corrected=True, masked=True, plot_kwargs=None, save_loc=None, save_suffix='', show=True,
                                             save_kwargs=None, cbar_kwargs=None, line_kwargs=None, **kwargs):
        timestamps_s = self.data_set.variable
        timestamps_s = timestamps_s - timestamps_s[0]
        xs = timestamps_s/self._timescale(timescale)

        wavelengths = self.data_set.get_wavelength(masked=masked)
        ys = self.data_set.get_absorbances(corrected=corrected, masked=masked).T

        wavelengths, ys = self._avg(wavelengths, ys, num, average_num)
        mask = np.max(ys, axis=1) > min_absorbance
        ys = ys[mask]
        wavelengths = wavelengths[mask]

        xs, ys = self._avg(xs, ys.T, time_num, None)
        ys = ys.T

        if norm:
            ys = ys/np.max(ys, axis=1)[:, None]

        cmap = plt.get_cmap(self.cmap)
        norm = plt.Normalize(vmin=wavelengths[0], vmax=wavelengths[-1])
        colors = cmap(norm(wavelengths))
        norm = BoundaryNorm(self._boundaries(wavelengths), cmap.N)
        ticks = plt.MaxNLocator(7).tick_values(wavelengths[0], wavelengths[-1])

        if norm:
            ylabel = 'Normalized absorbance [A.U.]'
        else:
            ylabel = 'Absorbance [A.U.]'

        plot_kwargs = Plot.set_defaults(plot_kwargs, xlabel=f'Time [{timescale}]', ylabel=ylabel, ylim=(-0.05, 1.05), xlim=(xs[0], xs[-1]))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar_kwargs = Plot.set_defaults(cbar_kwargs, label=f'Wavelength [nm]', mappable=sm, ticks=ticks)
        line_kwargs = Plot.set_defaults(line_kwargs)

        if save_loc is not None:
            if '.' not in save_suffix:
                save_suffix += '.pdf'
            save_loc = os.path.join(save_loc, f'absorbance over time with wavelength{save_suffix}')
        return Plot.lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                          colors=colors, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs, save_kwargs=save_kwargs, **kwargs)

    def fit(self, model: lmfit.CompositeModel, timescale='min', num=None, average_num=None, use_previous=True, wavelength_range: tuple[float, float] = None, *,
            corrected=True, masked=True, plot_kwargs=None, save_loc=None, save_suffix='', show=True, save_kwargs=None, cbar_kwargs=None, line_kwargs=None,
            legend_kwargs=None, export_fit_loc=None, **kwargs):
        timestamps_s = self.data_set.variable
        timestamps_s = timestamps_s - timestamps_s[0]
        timestamps = timestamps_s/self._timescale(timescale)

        xs = self.data_set.get_wavelength(masked=masked)
        ys = self.data_set.get_absorbances(corrected=corrected, masked=masked)

        timestamps, ys = self._avg(timestamps, ys, num, average_num)

        if wavelength_range is not None:
            wav_mask = (xs > wavelength_range[0]) & (xs < wavelength_range[1])
            xs = xs[wav_mask]
            ys = ys[:, wav_mask]

        results = []
        params = model.make_params()
        for i, y in enumerate(ys):
            result = model.fit(y, x=xs, params=params)
            results.append(result)
            if use_previous:
                params = result.params.copy()
            if export_fit_loc is not None:
                plt.figure()
                result.plot()
                plt.savefig(os.path.join(export_fit_loc, f'fit_{i}.png'))
                plt.close()

        res = []
        std = []
        names = []
        for param in params:
            res.append(np.array([result.params[param].value for result in results]))
            std.append(np.array([result.params[param].stderr for result in results]))
            names.append(param)

        plot_kwargs = Plot.set_defaults(plot_kwargs, xlabel=f'Time [{timescale}]', ylabel='Conc [mmol]')
        legend_kwargs = Plot.set_defaults(legend_kwargs)
        if save_loc is not None:
            if '.' not in save_suffix:
                save_suffix += '.pdf'
            save_loc = os.path.join(save_loc, f'absorbance with wavelength over time{save_suffix}')
        return Plot.errorbar(timestamps, res, yerr=std, labels=names, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show, save_kwargs=save_kwargs,
                             cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs, legend_kwargs=legend_kwargs, **kwargs)
