import os
import warnings
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy
import lmfit

from General.experiments.absorption.DataSets import DataSet
from General.plotting import plot


# TODO: add docstrings
class CalibrationAnalyzer:
    _defaults = {
        'cmap': 'turbo',
    }

    def __init__(self, variable_display_name: str, *args, variable_factor: float = 1, cmap='turbo', **kwargs):
        """
        Class for analyzing calibration data

        Parameters
        ----------
        variable_display_name: str
            name of the variable to be displayed in plots
        args
            from :py:class:`DataSet<Data_handling.DataSets.DataSet>`
        variable_factor
            factor to multiply the variable (x-axis) with before plotting
        cmap
            colormap to use in plots
        kwargs
            from :py:class:`DataSet<Data_handling.DataSets.DataSet>`
        """
        self.data_set = DataSet(*args, **kwargs)
        self.variable_factor = variable_factor
        self.variable_display_name = variable_display_name
        self.cmap = cmap

    @staticmethod
    def from_DataSet(dataset: DataSet, variable_factor, variable_display_name, cmap=None):
        """
        Create an Analyzer from a DataSet

        For parameter descriptions see :py:meth:`__init__`
        """
        cmap = cmap or CalibrationAnalyzer._defaults['cmap']
        return CalibrationAnalyzer(variable_display_name, dataset.wavelength, dataset.absorbances, dataset.variable,
                                   dataset.measurement_num, dataset.variable_name, dataset.wavelength_range,
                                   dataset._selected_num, dataset.baseline_correction, variable_factor=variable_factor,
                                   cmap=cmap)

    @staticmethod
    def standard(loc: str, species, display_name, *, variable_factor: float = None, **kwargs):
        """
        Create an Analyzer with default settings from a DataSet

        For parameter descriptions see :py:meth:`__init__`
        For kwargs see :py:meth:`DataSet.standard<Data_handling.DataSets.DataSet.standard>`

        Returns
        -------
        CalibrationAnalyzer
        """
        factor = variable_factor or CalibrationAnalyzer._default_factor(display_name)
        return CalibrationAnalyzer.from_DataSet(DataSet.standard(loc, species, **kwargs), factor, display_name)

    _match_units = ['L', 'M', 'mol', 's', 'g', 'K']

    @staticmethod
    def _default_factor(display_name):
        matcher = f"({'|'.join((f'({v})' for v in CalibrationAnalyzer._match_units))})"
        for key, value in {'m': 1000, 'u': 1_000_000, 'n': 1_000_000_000}.items():
            if re.search(key+matcher, display_name):
                return value
        return 1

    def absorbance_vs_wavelength_with_num(self, *, corrected=True, masked=True, save_loc=None, show=False,
                                          save_suffix='', plot_kwargs=None, legend_kwargs=None, line_kwargs=None,
                                          save_kwargs=None, **kwargs):
        """
        Plot absorbance vs wavelength for each measurement number

        `Corrected` and `masked` are passed to :py:meth:`DataSet.get_absorbances<Data_handling.DataSets.DataSet.get_absorbances>`
        Other parameters and kwargs are passed to :py:meth:`plot.lines<Plotting.plot.lines>`
        """

        for value in np.unique(self.data_set.variable):
            ys = [self.data_set.get_absorbances(corrected=corrected, masked=masked, num=var, var_value=value).T for var in self.data_set.measurement_num_at_value(value)]
            labels = list(range(len(ys)))
            plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Absorbance')
            legend_kwargs = plot.set_defaults(legend_kwargs, title='Measurement number')
            if save_loc is not None:
                save_loc = os.path.join(save_loc, f'absorbance vs wavelength at {value} {self.data_set.variable_name}{save_suffix}.pdf')
            plot.lines(self.data_set.get_wavelength(masked), ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                       save_loc=save_loc, show=show, labels=labels, line_kwargs=line_kwargs, save_kwargs=save_kwargs, **kwargs)
    
    def absorbance_vs_measurement_num_with_wavelength(self, *, corrected=True, masked=True, save_loc=None, num='plot',
                                                      wavelength_plot_every=5, min_absorbance=0.02, show=False,
                                                      save_suffix='', plot_kwargs=None, cbar_kwargs=None, line_kwargs=None,
                                                      save_kwargs=None, **kwargs):
        """
        Plot absorbance vs measurement number for each wavelength

        `Corrected`, `masked`, `num`, and `var_value` are passed to :py:meth:`DataSet.get_absorbances<Data_handling.DataSets.DataSet.get_absorbances>`

        Parameters
        ----------
        wavelength_plot_every: int
            Plot every nth wavelength
        min_absorbance: float
            All wavelengths with an absorbance below this value are ignored
        kwargs:
            All additional keyword arguments for :py:meth:`plot.lines<Plotting.plot.lines>`
        """
        cmap = plt.get_cmap(self.cmap)
        for value in np.unique(self.data_set.variable):
            wav_abs_mask = self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=value)[-1, :] > min_absorbance
            plot_num = len(self.data_set.get_wavelength(masked)[wav_abs_mask][::wavelength_plot_every])
            ys = [y/y[-1] for y in (self.data_set.get_absorbances(corrected=corrected, masked=masked, num='all', var_value=value).T[::wavelength_plot_every][index]
                                    for index in range(plot_num))]
            colors = [cmap(index / (plot_num // wavelength_plot_every)) for index in range(plot_num)]

            plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Measurement number', ylabel='Relative absorbance',
                                            xticks=self.data_set.measurement_num_at_value(value))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.data_set.get_wavelength(masked)[wav_abs_mask][0],
                                                                     vmax=self.data_set.get_wavelength(masked)[wav_abs_mask][-1]))
            cbar_kwargs = plot.set_defaults(cbar_kwargs, label='Wavelength (nm)', mappable=sm)
            line_kwargs = plot.set_defaults(line_kwargs, marker='o', linestyle='-')

            if save_loc is not None:
                save_loc = os.path.join(save_loc, f'absorbance vs measurement num at {value} {self.data_set.variable_name}{save_suffix}.pdf')
            plot.lines(self.data_set.measurement_num_at_value(value), ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                       colors=colors, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs, save_kwargs=save_kwargs, **kwargs)

    def total_absorbance_vs_variable(self, wav_range=(200, 400), *, corrected=True, masked=True, save_loc=None, num='plot', show=False,
                                    save_suffix='', plot_kwargs=None, legend_kwargs=None, line_kwargs=None,
                                    save_kwargs=None, **kwargs):
        xs = self.variable_factor * self.data_set.variable_num
        mask = (self.data_set.get_wavelength(masked) > wav_range[0]) & (self.data_set.get_wavelength(masked) < wav_range[1])
        ys = [np.sum(self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=var)[:, mask]) for var in np.unique(self.data_set.variable)]
        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel=self.variable_display_name, ylabel='Total absorbance')
        if save_loc is not None:
            if '.' not in save_suffix:
                save_suffix += '.pdf'
            save_loc = os.path.join(save_loc, f'total absorbance vs variable{save_suffix}')
        return plot.lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show, line_kwargs=line_kwargs, legend_kwargs=legend_kwargs,
                          save_kwargs=save_kwargs, **kwargs)

    # plot absorbance vs variable
    def absorbance_vs_wavelength_with_variable(self, *, corrected=True, masked=True, save_loc=None, num='plot',
                                               save_suffix='', plot_kwargs=None, legend_kwargs=None, show=False,
                                               line_kwargs: dict = None, save_kwargs: dict = None, variable_range=None,
                                               relative=False, **kwargs):
        """
        Plot absorbance vs wavelength for each variable
        """
        ys = [self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=var).T for var in np.unique(self.data_set.variable)]
        labels = [self.variable_factor * var for var in np.unique(self.data_set.variable)]
        if variable_range is not None:
            ys = [y for y, x in zip(ys, labels) if variable_range[0] < x < variable_range[1]]
            labels = [x for x in labels if variable_range[0] < x < variable_range[1]]
        labels = [f'{x:.2f}' for x in labels]
        ys = np.array(ys)
        if relative:
            ys = ys/np.max(ys)
        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Absorbance')
        legend_kwargs = plot.set_defaults(legend_kwargs, title=self.variable_display_name)
        if save_loc is not None:
            if '.' not in save_suffix:
                save_suffix += '.pdf'  # TODO: add this to other save_suffixes
            save_loc = os.path.join(save_loc, f'absorbance vs wavelength with variable{save_suffix}')
        plot.lines(self.data_set.get_wavelength(masked), ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                   save_loc=save_loc, show=show, labels=labels, line_kwargs=line_kwargs, save_kwargs=save_kwargs,
                   **kwargs)

    def absorbance_vs_variable_with_wavelength(self, *, corrected=True, masked=True, save_loc=None, num='plot',
                                               wavelength_plot_every=5, save_suffix='', plot_kwargs=None,
                                               cbar_kwargs=None, show=False, **kwargs):
        cmap = plt.get_cmap(self.cmap)
        xs = self.variable_factor * self.data_set.variable_num
        y_len = len(self.data_set.get_wavelength(masked)[::wavelength_plot_every])
        ys = [self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None).T[::wavelength_plot_every][index]
              for index in range(y_len)]
        colors = [cmap(index/(y_len-1)) for index in range(y_len)]
        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel=self.data_set.variable_name, ylabel='Absorbance',
                                        xticks=self.variable_factor * self.data_set.variable_num)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.data_set.get_wavelength(masked)[0],
                                                                 vmax=self.data_set.get_wavelength(masked)[-1]))
        cbar_kwargs = plot.set_defaults(cbar_kwargs, label='Wavelength (nm)', mappable=sm)
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'absorbance vs {self.data_set.variable_name} with wavelength{save_suffix}.pdf')
        plot.lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                   colors=colors, cbar_kwargs=cbar_kwargs, **kwargs)

    def relative_absorbance_vs_variable_with_wavelength(self, *, corrected=True, masked=True, num='plot', save_loc=None,
                                                        wavelength_plot_every=5, min_absorbance=0.02, save_suffix='',
                                                        plot_kwargs=None, cbar_kwargs=None, show=False, **kwargs):
        cmap = plt.get_cmap(self.cmap)
        xs = self.variable_factor * self.data_set.variable_num
        wav_abs_mask = self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None)[-1, :] > min_absorbance
        y_len = len(self.data_set.get_wavelength(masked)[wav_abs_mask][::wavelength_plot_every])
        ys = [y / y[-1] for y in (self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None).T[::wavelength_plot_every][index]
                                  for index in range(y_len))]

        colors = [cmap(index / (y_len - 1)) for index in range(y_len)]
        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel=self.data_set.variable_name, ylabel='Absorbance',
                                        xticks=self.variable_factor * self.data_set.variable_num)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.data_set.get_wavelength(masked)[0],
                                                                 vmax=self.data_set.get_wavelength(masked)[-1]))
        cbar_kwargs = plot.set_defaults(cbar_kwargs, label='Wavelength (nm)', mappable=sm)
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'relative absorbance vs {self.data_set.variable_name} with wavelength{save_suffix}.pdf')
        plot.lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                   colors=colors, cbar_kwargs=cbar_kwargs, **kwargs)

    def pearson_r_vs_wavelength_with_methods(self, *, save_loc=None, r2_values=None, masked=True, num='plot',
                                             save_suffix='', plot_kwargs=None, legend_kwargs=None, **kwargs):
        if r2_values is None:
            r2_values = [0, 1.025]
        if num == 'all' or num == 'best':
            warnings.warn(f'num = "{num}" is not logical for pearson_r_vs_wavelength_with_methods, should be "plot" or'
                          f' an integer, "plot" is used instead')
            num = 'plot'

        linearity = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            linearity[i] = scipy.stats.pearsonr(self.data_set.variable, self.data_set.get_absorbances(corrected=True, masked=masked)[:, i])[0]

        linearity_corrected_num = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            linearity_corrected_num[i] = scipy.stats.pearsonr(self.data_set.variable_num, self.data_set.get_absorbances(corrected=True, masked=masked, num=num)[:, i])[0]

        linearity_best_num = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            linearity_best_num[i] = scipy.stats.pearsonr(self.data_set.variable_best_num, self.data_set.get_absorbances(corrected=True, masked=masked, num='best')[:, i])[0]

        linearity_uncorrected = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            linearity_uncorrected[i] = scipy.stats.pearsonr(self.data_set.variable, self.data_set.get_absorbances(corrected=False, masked=masked)[:, i])[0]

        linearity_uncorrected_num = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            linearity_uncorrected_num[i] = scipy.stats.pearsonr(self.data_set.variable_num, self.data_set.get_absorbances(corrected=False, masked=masked, num=num)[:, i])[0]

        linearity_uncorrected_best_num = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            linearity_uncorrected_best_num[i] = \
            scipy.stats.pearsonr(self.data_set.variable_best_num, self.data_set.get_absorbances(corrected=False, masked=masked, num='best')[:, i])[0]

        r2_mask = (r2_values[0] < linearity ** 2) & (linearity ** 2 < r2_values[1])
        lins = [linearity_corrected_num, linearity_best_num, linearity_uncorrected, linearity_uncorrected_num,
                linearity_uncorrected_best_num]
        for lin in lins:
            r2_mask = r2_mask | ((r2_values[0] < lin ** 2) & (lin ** 2 < r2_values[1]))

        wavs = self.data_set.get_wavelength(masked)[r2_mask]
        dw = np.diff(wavs)
        index = np.nonzero(dw > 10)[0]
        if len(index) > 0:
            r2_mask[index[0] + 1:] = False

        xs = [self.data_set.get_wavelength(masked)[r2_mask]] * len(lins)
        ys = [linearity[r2_mask] ** 2, linearity_corrected_num[r2_mask] ** 2, linearity_best_num[r2_mask] ** 2,
              linearity_uncorrected[r2_mask] ** 2, linearity_uncorrected_num[r2_mask] ** 2,
              linearity_uncorrected_best_num[r2_mask] ** 2]
        labels = ['corrected', 'corrected num', 'corrected best num', 'uncorrected', 'uncorrected num',
                  'uncorrected best num']
        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Linearity coefficient',
                                        ylim=r2_values)
        legend_kwargs = plot.set_defaults(legend_kwargs, title='Method')
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'pearson r^2 vs wavelength method comparison{save_suffix}.pdf')

        plot.lines(xs, ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs, labels=labels, save_loc=save_loc,
                   **kwargs)

    def pearson_r_vs_wavelength(self, add_zero=True, *, corrected=True, masked=True, num='plot', save_loc=None, save_suffix='',
                                plot_kwargs=None, show=False, line_kwargs=None, save_kwargs=None, **kwargs):
        linearity = self._linearity(add_zero, corrected=corrected, masked=masked, num=num)

        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='r$^2$')
        if save_loc is not None:
            if '.' not in save_suffix:
                save_suffix += '.pdf'
            save_loc = os.path.join(save_loc, f'absorbance vs wavelength with variable{save_suffix}')
        return plot.lines(self.data_set.get_wavelength(masked), linearity**2, plot_kwargs=plot_kwargs,
                   save_loc=save_loc, show=show, line_kwargs=line_kwargs, save_kwargs=save_kwargs,
                   **kwargs)  # TODO: add return for all plot functions

    def _linearity(self, add_zero, corrected=True, masked=True, num='plot'):
        linearity = np.zeros(len(self.data_set.get_wavelength(masked)))
        absorbances = self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num)
        if add_zero:
            inten_values = np.zeros(absorbances.shape[0] + 1)
            conc = np.zeros(absorbances.shape[0] + 1)
            conc[1:] = self.data_set.variable_num
        else:
            conc = self.data_set.variable_num

        for i in range(len(self.data_set.get_wavelength(masked))):
            if add_zero:
                inten_values[1:] = absorbances[:, i]
            else:
                inten_values = absorbances[:, i]
            linearity[i] = scipy.stats.pearsonr(conc, inten_values)[0]
        return linearity

    def one_minus_pearson_r_vs_wavelength(self, add_zero, corrected=True, masked=True, num='plot', save_loc=None, save_suffix='',
                                plot_kwargs=None, show=False, line_kwargs=None, save_kwargs=None, **kwargs):
        linearity = self._linearity(add_zero, corrected=corrected, masked=masked, num=num)
        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='1 - r$^2$')
        if save_loc is not None:
            if '.' not in save_suffix:
                save_suffix += '.pdf'
            save_loc = os.path.join(save_loc, f'absorbance vs wavelength with variable{save_suffix}')
        return plot.lines(self.data_set.get_wavelength(masked), 1-linearity**2, plot_kwargs=plot_kwargs,
                          save_loc=save_loc, show=show, line_kwargs=line_kwargs, save_kwargs=save_kwargs,
                          **kwargs)  # TODO: add return for all plot functions

    def linear_fit_vs_wavelength_with_methods(self, *, save_loc=None, masked=True, num='plot', show=True,
                                              save_suffix='', plot_kwargs=None, legend_kwargs=None, min_intensity=0.05,
                                              **kwargs):
        # linear fit for each wavelength
        lin_model = lmfit.models.LinearModel()
        params = lin_model.make_params()
        params['intercept'].value = 0
        params['intercept'].vary = False

        slope = np.zeros(len(self.data_set.get_wavelength(masked)))
        slope_std = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            result = lin_model.fit(self.data_set.get_absorbances(corrected=True, masked=masked)[:, i], params, x=self.data_set.variable)
            slope[i] = result.params['slope'].value
            slope_std[i] = result.params['slope'].stderr

        slope_corrected_num = np.zeros(len(self.data_set.get_wavelength(masked)))
        slope_corrected_num_std = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            result = lin_model.fit(self.data_set.get_absorbances(corrected=True, masked=masked, num=num)[:, i], params, x=self.data_set.variable_num)
            slope_corrected_num[i] = result.params['slope'].value
            slope_corrected_num_std[i] = result.params['slope'].stderr

        slope_best_num = np.zeros(len(self.data_set.get_wavelength(masked)))
        slope_best_num_std = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            result = lin_model.fit(self.data_set.get_absorbances(corrected=True, masked=masked, num='best')[:, i], params, x=self.data_set.variable_best_num)
            slope_best_num[i] = result.params['slope'].value
            slope_best_num_std[i] = result.params['slope'].stderr

        slope_uncorrected = np.zeros(len(self.data_set.get_wavelength(masked)))
        slope_uncorrected_std = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            result = lin_model.fit(self.data_set.get_absorbances(corrected=False, masked=masked)[:, i], params, x=self.data_set.variable)
            slope_uncorrected[i] = result.params['slope'].value
            slope_uncorrected_std[i] = result.params['slope'].stderr

        slope_uncorrected_num = np.zeros(len(self.data_set.get_wavelength(masked)))
        slope_uncorrected_num_std = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            result = lin_model.fit(self.data_set.get_absorbances(corrected=False, masked=masked, num=num)[:, i], params, x=self.data_set.variable_num)
            slope_uncorrected_num[i] = result.params['slope'].value
            slope_uncorrected_num_std[i] = result.params['slope'].stderr

        slope_uncorrected_best_num = np.zeros(len(self.data_set.get_wavelength(masked)))
        slope_uncorrected_best_num_std = np.zeros(len(self.data_set.get_wavelength(masked)))
        for i in range(len(self.data_set.get_wavelength(masked))):
            result = lin_model.fit(self.data_set.get_absorbances(corrected=False, masked=masked, num='best')[:, i], params, x=self.data_set.variable_best_num)
            slope_uncorrected_best_num[i] = result.params['slope'].value
            slope_uncorrected_best_num_std[i] = result.params['slope'].stderr

        ys = [slope, slope_corrected_num, slope_best_num, slope_uncorrected, slope_uncorrected_num,
              slope_uncorrected_best_num]
        labels = ['corrected', 'corrected num', 'corrected best num', 'uncorrected', 'uncorrected num',
                  'uncorrected best num']
        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Slope')
        legend_kwargs = plot.set_defaults(legend_kwargs, title='Method')
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'slope vs wavelength method comparison {save_suffix}.pdf')
        plot.lines(self.data_set.get_wavelength(masked), ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                   labels=labels, save_loc=save_loc, show=show, **kwargs)

        # plot relative slope
        intensity = self.data_set.get_absorbances(corrected=True, masked=True, num='plot', var_value=np.max(self.data_set.variable))
        mask = intensity > min_intensity * np.max(intensity)

        y_min, y_max = np.min(slope_uncorrected[mask] / self.data_set.absorbances_masked[-1][mask]), np.max(
            slope_uncorrected[mask] / self.data_set.absorbances_masked[-1][mask])
        dy = y_max - y_min
        y_min -= 0.1 * dy
        y_max += 0.1 * dy

        ys = [slope / self.data_set.absorbances_masked_corrected[-1],
              slope_corrected_num / self.data_set.absorbances_masked_corrected_num[-1],
              slope_best_num / self.data_set.absorbances_masked_best_num[-1],
              slope_uncorrected / self.data_set.absorbances_masked[-1],
              slope_uncorrected_num / self.data_set.absorbances_masked_num[-1],
              slope_uncorrected_best_num / self.data_set.absorbances_masked_best_num[-1]]
        plot_kwargs = plot.set_defaults(plot_kwargs, y_lim=(y_min, y_max))
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'relative slope vs wavelength method comparison{save_suffix}.pdf')
        plot.lines(self.data_set.get_wavelength(masked), ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                   labels=labels, save_loc=save_loc, show=show, **kwargs)

    def relative_intensity_fit_vs_variable(self, *, corrected=True, masked=True, save_loc=None, num='plot', show=True,
                                           reference_line, save_suffix='', plot_kwargs={}, legend_kwargs={}):
        # TODO: change to use plot._1d_lines
        def residual(pars, x, reference):
            a = pars['a'].value
            return x - a * reference

        params = lmfit.Parameters()
        params.add('a', value=1, vary=True)
        ratio = []
        ratio_std = []
        max_var = np.max(self.data_set.variable)
        for i in self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None):
            result = lmfit.minimize(residual, params, args=(i,),
                                    kws={'reference': self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=max_var)})
            ratio.append(result.params['a'].value)
            ratio_std.append(result.params['a'].stderr)
        ratio = np.array(ratio)
        ratio_std = np.array(ratio_std)

        fig, ax = plt.subplots()
        plt.errorbar(self.data_set.variable, ratio, yerr=ratio_std, capsize=2, fmt='.', label='measured intensity')
        if reference_line is not None:
            plt.plot(reference_line['x'], reference_line['y'], label=reference_line['label'])
        plt.xlabel(self.data_set.variable_name)
        plt.ylabel('Ratio')
        plt.grid()
        plot.setting_setter(ax, **plot_kwargs)
        plt.tight_layout()
        if save_loc is not None:
            plt.savefig(os.path.join(save_loc, f'Relative intensity vs {self.data_set.variable_name} method comparison{save_suffix}.pdf'))
        if show:
            plt.show()
        else:
            plt.close()

        lines, labels = [], []
        plt.figure()
        for index, var in enumerate(np.unique(self.data_set.variable)):
            plt.plot(self.data_set.get_wavelength(True), self.data_set.absorbances_masked[self.data_set.variable == var].T / ratio[self.data_set.variable == var], f'C{index}', label=var)
            lines.append(plt.Line2D([0], [0], color=f'C{index}'))
            labels.append(f'{var}')
        # plt.plot(self.wavelength_masked, self.data_set.absorbances_masked.T/ratio, label=self.data_set.variable)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Relative absorbance (A.U.)')
        plt.legend(lines, labels, title=self.variable_display_name, **legend_kwargs)
        plt.grid()
        plot.setting_setter(ax, **plot_kwargs)
        plt.tight_layout()
        if save_loc is not None:
            plt.savefig(os.path.join(save_loc, f'Relative absorbance vs wavelength method comparison{save_suffix}.pdf'))
        if show:
            plt.show()
        else:
            plt.close()

    def absorbances_wavelength_ranges_vs_variable(self, ranges: list[tuple[int, int]], *, corrected=True, masked=True,
                                                  save_loc=None, num='plot', show=True, save_suffix='',
                                                  plot_kwargs=None, legend_kwargs=None, line_kwargs=None,
                                                  xtick_formatter='.1f', wav_range_formatter='.0f', **kwargs):
        """
        Plot average intensity in each wavelength range vs variable
        """
        xs = self.variable_factor * self.data_set.variable_num
        ys = self.data_set.get_average_absorbance_ranges(ranges, corrected=corrected, masked=masked, num=num)
        labels = [f'{r[0]:^{wav_range_formatter}}-{r[1]:^{wav_range_formatter}} nm' for r in ranges]

        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel=self.variable_display_name, ylabel='Absorbance',
                                        xticks=self.variable_factor * self.data_set.variable_num,
                                        xticklabels=[f'{x:^{xtick_formatter}}' for x in self.variable_factor * self.data_set.variable_num])
        legend_kwargs = plot.set_defaults(legend_kwargs, title='Wavelength range')
        line_kwargs = plot.set_defaults(line_kwargs, marker='o', linestyle='-')
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'Average absorbance in ranges vs {self.data_set.variable_name} with{save_suffix}.pdf')
        plot.lines(xs, ys, labels=labels, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs, save_loc=save_loc,
                   show=show, line_kwargs=line_kwargs, **kwargs)

    def wavelength_range_ratio_vs_variable(self, ranges1: list[tuple[int, int]], ranges2: list[tuple[int, int]], *,
                                           corrected=True, masked=True, save_loc=None, num='plot', show=True,
                                           save_suffix='', plot_kwargs=None, line_kwargs=None, xtick_formatter='.1f',
                                           variable_val_ticks=True, **kwargs):
        """
        Plot the ratio of the average absorbance in ranges1 to the average absorbance in ranges2 vs the variable
        """
        xs = self.variable_factor * self.data_set.variable_num
        mask1 = np.full(self.data_set.get_wavelength(masked).shape, False)
        for range_val in ranges1:
            mask1 = mask1 | ((range_val[0] < self.data_set.get_wavelength(masked)) & (self.data_set.get_wavelength(masked) < range_val[1]))
        mask2 = np.full(self.data_set.get_wavelength(masked).shape, False)
        for range_val in ranges2:
            mask2 = mask2 | ((range_val[0] < self.data_set.get_wavelength(masked)) & (self.data_set.get_wavelength(masked) < range_val[1]))

        y1 = np.average(self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None)[:, mask1], axis=1)
        y2 = np.average(self.data_set.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None)[:, mask2], axis=1)
        ys = [y1/y2]
        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel=self.data_set.variable_name, ylabel='Ratio')
        if variable_val_ticks:
            if (plot_kwargs is not None) and ('xticks' in plot_kwargs):
                plot_kwargs['xticklabels'] = [f'{x:^{xtick_formatter}}' for x in plot_kwargs['xticks']]
            else:
                plot_kwargs = plot.set_defaults(plot_kwargs, xticks=self.variable_factor * self.data_set.variable_num,
                                                xticklabels=[f'{x:^{xtick_formatter}}' for x in self.variable_factor * self.data_set.variable_num])
        line_kwargs = plot.set_defaults(line_kwargs, marker='o', linestyle='')
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'absorbance vs {self.data_set.variable_name} with in ranges{save_suffix}.pdf')
        plot.lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show, line_kwargs=line_kwargs, **kwargs)

    def full_spectrum_fit_with_methods(self):
        def residual(pars, x, concentration):
            a = pars['a'].value
            return x - a * concentration * x[-1]

        params = lmfit.Parameters()
        params.add('a', value=1, vary=True)

        result = lmfit.minimize(residual, params, args=(self.data_set.get_absorbances(corrected=True, masked=True),),
                                kws={'concentration': self.data_set.variable[:, np.newaxis]})
        result_num = lmfit.minimize(residual, params, args=(self.data_set.get_absorbances(corrected=True, masked=True, num='plot'),),
                                    kws={'concentration': self.data_set.variable_num[:, np.newaxis]})
        result_best_num = lmfit.minimize(residual, params, args=(self.data_set.get_absorbances(corrected=True, masked=True, num='best'),),
                                         kws={'concentration': self.data_set.variable_best_num[:, np.newaxis]})
        result_uncorr = lmfit.minimize(residual, params, args=(self.data_set.get_absorbances(corrected=False, masked=True),),
                                       kws={'concentration': self.data_set.variable[:, np.newaxis]})
        result_uncorr_num = lmfit.minimize(residual, params, args=(self.data_set.get_absorbances(corrected=False, masked=True, num='plot'),),
                                           kws={'concentration': self.data_set.variable_num[:, np.newaxis]})
        result_uncorr_best_num = lmfit.minimize(residual, params, args=(self.data_set.get_absorbances(corrected=False, masked=True, num='best'),),
                                                kws={'concentration': self.data_set.variable_best_num[:, np.newaxis]})

        print(
            f"""
            # Fit report
            corrected: {result.params['a'].value:.3f} ± {result.params['a'].stderr:.3f}
            corrected num: {result_num.params['a'].value:.3f} ± {result_num.params['a'].stderr:.3f}
            corrected best num: {result_best_num.params['a'].value:.3f} ± {result_best_num.params['a'].stderr:.3f}
            uncorrected: {result_uncorr.params['a'].value:.3f} ± {result_uncorr.params['a'].stderr:.3f}
            uncorrected num: {result_uncorr_num.params['a'].value:.3f} ± {result_uncorr_num.params['a'].stderr:.3f}
            uncorrected best num: {result_uncorr_best_num.params['a'].value:.3f} ± {result_uncorr_best_num.params['a'].stderr:.3f}
            """)
