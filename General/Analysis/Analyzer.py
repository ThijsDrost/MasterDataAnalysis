import os
import warnings
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy
import lmfit

from General.Data_handling.Import import DataSet
from General.Plotting.Plot import Plot


class Analyzer(DataSet, Plot):
    def __init__(self, variable_display_name, *args, variable_factor=1, cmap='turbo', **kwargs):
        super().__init__(*args, **kwargs)
        self.variable_factor = variable_factor
        self.variable_display_name = variable_display_name
        self.cmap = cmap

    @staticmethod
    def from_DataSet(dataset, variable_factor, variable_display_name, cmap='turbo'):
        return Analyzer(variable_display_name, dataset.wavelength, dataset.absorbances, dataset.variable,
                        dataset.measurement_num, dataset.variable_name, dataset.wavelength_range,
                        dataset._selected_num, dataset.baseline_correction, variable_factor=variable_factor,
                        cmap=cmap)

    @staticmethod
    def standard(loc, species, display_name, *, variable_factor=None, **kwargs):
        """
        Create an Analyzer with default settings from a DataSet

        Parameters
        ----------
        loc: str or os.PathLike
        species: str
        display_name: str
        variable_factor: float
        kwargs
            Values for DataSet.standard

            wavelength_range: list[float], default=[180, 450]
            selected_num: int, default=1
            baseline_correction: list[float], default=[450, 500]


        Returns
        -------
        Analyzer
        """
        factor = variable_factor or Analyzer._default_factor(display_name)
        return Analyzer.from_DataSet(DataSet.standard(loc, species, **kwargs), factor, display_name)

    _match_units = ['L', 'M', 'mol', 's', 'g', 'K']

    @staticmethod
    def _default_factor(display_name):
        matcher = f"({'|'.join((f'({v})' for v in Analyzer._match_units))})"
        for key, value in {'m': 1000, 'u': 1_000_000, 'n': 1_000_000_000}.items():
            if re.search(key+matcher, display_name):
                return value
        return 1

    def absorbance_vs_wavelength_with_num(self, *, corrected=True, masked=True, save_loc=None, show=False,
                                          save_suffix='', plot_kwargs=None, legend_kwargs=None, line_kwargs=None,
                                          save_kwargs=None, **kwargs):
        """
        Plot absorbance vs wavelength for each measurement number
        """

        for value in np.unique(self.variable):
            ys = [self.get_absorbances(corrected=corrected, masked=masked, num=var, var_value=value).T for var in self.measurement_num_at_value(value)]
            labels = list(range(len(ys)))
            plot_kwargs = Analyzer._set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Absorbance')
            legend_kwargs = Analyzer._set_defaults(legend_kwargs, title='Measurement number')
            if save_loc is not None:
                save_loc = os.path.join(save_loc, f'absorbance vs wavelength at {value} {self.variable_name}{save_suffix}.pdf')
            self._1d_lines(self.get_wavelength(masked), ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                           save_loc=save_loc, show=show, labels=labels, line_kwargs=line_kwargs, save_kwargs=save_kwargs, **kwargs)
    
    def absorbance_vs_measurement_num_with_wavelength(self, *, corrected=True, masked=True, save_loc=None, num='plot',
                                                      wavelength_plot_every=5, min_absorbance=0.02, show=False,
                                                      save_suffix='', plot_kwargs=None, cbar_kwargs=None, line_kwargs=None,
                                                      save_kwargs=None, **kwargs):
        """
        Plot absorbance vs measurement number for each wavelength
        """
        cmap = plt.get_cmap(self.cmap)
        for value in np.unique(self.variable):
            wav_abs_mask = self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=value)[-1, :] > min_absorbance
            plot_num = len(self.get_wavelength(masked)[wav_abs_mask][::wavelength_plot_every])
            ys = [y/y[-1] for y in (self.get_absorbances(corrected=corrected, masked=masked, num='all', var_value=value).T[::wavelength_plot_every][index]
                                    for index in range(plot_num))]
            colors = [cmap(index / (plot_num // wavelength_plot_every)) for index in range(plot_num)]

            plot_kwargs = Analyzer._set_defaults(plot_kwargs, xlabel='Measurement number', ylabel='Relative absorbance',
                                                 xticks=self.measurement_num_at_value(value))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.get_wavelength(masked)[wav_abs_mask][0],
                                                                     vmax=self.get_wavelength(masked)[wav_abs_mask][-1]))
            cbar_kwargs = Analyzer._set_defaults(cbar_kwargs, label='Wavelength (nm)', mappable=sm)
            line_kwargs = Analyzer._set_defaults(line_kwargs, marker='o', linestyle='-')

            if save_loc is not None:
                save_loc = os.path.join(save_loc, f'absorbance vs measurement num at {value} {self.variable_name}{save_suffix}.pdf')
            self._1d_lines(self.measurement_num_at_value(value), ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                           colors=colors, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs, save_kwargs=save_kwargs, **kwargs)

    # plot absorbance vs variable
    def absorbance_vs_wavelength_with_variable(self, *, corrected=True, masked=True, save_loc=None, num='plot',
                                               save_suffix='', plot_kwargs=None, legend_kwargs=None, show=False,
                                               line_kwargs: dict = None, save_kwargs: dict = None, **kwargs):
        """
        Plot absorbance vs wavelength for each variable
        """
        ys = [self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=var).T for var in np.unique(self.variable)]
        labels = [self.variable_factor * var for var in np.unique(self.variable)]
        plot_kwargs = Analyzer._set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Absorbance')
        legend_kwargs = Analyzer._set_defaults(legend_kwargs, title=self.variable_display_name)
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'absorbance vs wavelength with variable{save_suffix}.pdf')
        self._1d_lines(self.get_wavelength(masked), ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                       save_loc=save_loc, show=show, labels=labels, line_kwargs=line_kwargs, save_kwargs=save_kwargs,
                       **kwargs)

    def absorbance_vs_variable_with_wavelength(self, *, corrected=True, masked=True, save_loc=None, num='plot',
                                               wavelength_plot_every=5, save_suffix='', plot_kwargs=None,
                                               cbar_kwargs=None, show=False, **kwargs):
        cmap = plt.get_cmap(self.cmap)
        xs = self.variable_factor * self.variable_num
        y_len = len(self.get_wavelength(masked)[::wavelength_plot_every])
        ys = [self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None).T[::wavelength_plot_every][index]
              for index in range(y_len)]
        colors = [cmap(index/(y_len-1)) for index in range(y_len)]
        plot_kwargs = Analyzer._set_defaults(plot_kwargs, xlabel=self.variable_name, ylabel='Absorbance',
                                             xticks=self.variable_factor * self.variable_num)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.get_wavelength(masked)[0],
                                                                 vmax=self.get_wavelength(masked)[-1]))
        cbar_kwargs = Analyzer._set_defaults(cbar_kwargs, label='Wavelength (nm)', mappable=sm)
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'absorbance vs {self.variable_name} with wavelength{save_suffix}.pdf')
        self._1d_lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                       colors=colors, cbar_kwargs=cbar_kwargs, **kwargs)

    def relative_absorbance_vs_variable_with_wavelength(self, *, corrected=True, masked=True, num='plot', save_loc=None,
                                                        wavelength_plot_every=5, min_absorbance=0.02, save_suffix='',
                                                        plot_kwargs=None, cbar_kwargs=None, show=False, **kwargs):
        cmap = plt.get_cmap(self.cmap)
        xs = self.variable_factor * self.variable_num
        wav_abs_mask = self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None)[-1, :] > min_absorbance
        y_len = len(self.get_wavelength(masked)[wav_abs_mask][::wavelength_plot_every])
        ys = [y / y[-1] for y in (self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None).T[::wavelength_plot_every][index]
                                  for index in range(y_len))]

        colors = [cmap(index / (y_len - 1)) for index in range(y_len)]
        plot_kwargs = Analyzer._set_defaults(plot_kwargs, xlabel=self.variable_name, ylabel='Absorbance',
                                             xticks=self.variable_factor * self.variable_num)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.get_wavelength(masked)[0],
                                                                 vmax=self.get_wavelength(masked)[-1]))
        cbar_kwargs = Analyzer._set_defaults(cbar_kwargs, label='Wavelength (nm)', mappable=sm)
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'relative absorbance vs {self.variable_name} with wavelength{save_suffix}.pdf')
        self._1d_lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                       colors=colors, cbar_kwargs=cbar_kwargs, **kwargs)

    def pearson_r_vs_wavelength_with_methods(self, *, save_loc=None, r2_values=None, masked=True, num='plot',
                                             save_suffix='', plot_kwargs=None, legend_kwargs=None, **kwargs):
        if r2_values is None:
            r2_values = [0, 1.025]
        if num == 'all' or num == 'best':
            warnings.warn(f'num = "{num}" is not logical for pearson_r_vs_wavelength_with_methods, should be "plot" or'
                          f' an integer, "plot" is used instead')
            num = 'plot'

        linearity = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            linearity[i] = scipy.stats.pearsonr(self.variable, self.get_absorbances(corrected=True, masked=masked)[:, i])[0]

        linearity_corrected_num = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            linearity_corrected_num[i] = scipy.stats.pearsonr(self.variable_num, self.get_absorbances(corrected=True, masked=masked, num=num)[:, i])[0]

        linearity_best_num = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            linearity_best_num[i] = scipy.stats.pearsonr(self.variable_best_num, self.get_absorbances(corrected=True, masked=masked, num='best')[:, i])[0]

        linearity_uncorrected = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            linearity_uncorrected[i] = scipy.stats.pearsonr(self.variable, self.get_absorbances(corrected=False, masked=masked)[:, i])[0]

        linearity_uncorrected_num = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            linearity_uncorrected_num[i] = scipy.stats.pearsonr(self.variable_num, self.get_absorbances(corrected=False, masked=masked, num=num)[:, i])[0]

        linearity_uncorrected_best_num = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            linearity_uncorrected_best_num[i] = \
            scipy.stats.pearsonr(self.variable_best_num, self.get_absorbances(corrected=False, masked=masked, num='best')[:, i])[0]

        r2_mask = (r2_values[0] < linearity ** 2) & (linearity ** 2 < r2_values[1])
        lins = [linearity_corrected_num, linearity_best_num, linearity_uncorrected, linearity_uncorrected_num,
                linearity_uncorrected_best_num]
        for lin in lins:
            r2_mask = r2_mask | ((r2_values[0] < lin ** 2) & (lin ** 2 < r2_values[1]))

        wavs = self.get_wavelength(masked)[r2_mask]
        dw = np.diff(wavs)
        index = np.nonzero(dw > 10)[0]
        if len(index) > 0:
            r2_mask[index[0] + 1:] = False

        xs = [self.get_wavelength(masked)[r2_mask]] * len(lins)
        ys = [linearity[r2_mask] ** 2, linearity_corrected_num[r2_mask] ** 2, linearity_best_num[r2_mask] ** 2,
              linearity_uncorrected[r2_mask] ** 2, linearity_uncorrected_num[r2_mask] ** 2,
              linearity_uncorrected_best_num[r2_mask] ** 2]
        labels = ['corrected', 'corrected num', 'corrected best num', 'uncorrected', 'uncorrected num',
                  'uncorrected best num']
        plot_kwargs = Analyzer._set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Linearity coefficient',
                                             ylim=r2_values)
        legend_kwargs = Analyzer._set_defaults(legend_kwargs, title='Method')
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'pearson r^2 vs wavelength method comparison{save_suffix}.pdf')

        self._1d_lines(xs, ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs, labels=labels, save_loc=save_loc,
                       **kwargs)

    def linear_fit_vs_wavelength_with_methods(self, *, save_loc=None, masked=True, num='plot', show=True,
                                              save_suffix='', plot_kwargs=None, legend_kwargs=None, min_intensity=0.05,
                                              **kwargs):
        # linear fit for each wavelength
        lin_model = lmfit.models.LinearModel()
        params = lin_model.make_params()
        params['intercept'].value = 0
        params['intercept'].vary = False

        slope = np.zeros(len(self.get_wavelength(masked)))
        slope_std = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            result = lin_model.fit(self.get_absorbances(corrected=True, masked=masked)[:, i], params, x=self.variable)
            slope[i] = result.params['slope'].value
            slope_std[i] = result.params['slope'].stderr

        slope_corrected_num = np.zeros(len(self.get_wavelength(masked)))
        slope_corrected_num_std = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            result = lin_model.fit(self.get_absorbances(corrected=True, masked=masked, num=num)[:, i], params, x=self.variable_num)
            slope_corrected_num[i] = result.params['slope'].value
            slope_corrected_num_std[i] = result.params['slope'].stderr

        slope_best_num = np.zeros(len(self.get_wavelength(masked)))
        slope_best_num_std = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            result = lin_model.fit(self.get_absorbances(corrected=True, masked=masked, num='best')[:, i], params, x=self.variable_best_num)
            slope_best_num[i] = result.params['slope'].value
            slope_best_num_std[i] = result.params['slope'].stderr

        slope_uncorrected = np.zeros(len(self.get_wavelength(masked)))
        slope_uncorrected_std = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            result = lin_model.fit(self.get_absorbances(corrected=False, masked=masked)[:, i], params, x=self.variable)
            slope_uncorrected[i] = result.params['slope'].value
            slope_uncorrected_std[i] = result.params['slope'].stderr

        slope_uncorrected_num = np.zeros(len(self.get_wavelength(masked)))
        slope_uncorrected_num_std = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            result = lin_model.fit(self.get_absorbances(corrected=False, masked=masked, num=num)[:, i], params, x=self.variable_num)
            slope_uncorrected_num[i] = result.params['slope'].value
            slope_uncorrected_num_std[i] = result.params['slope'].stderr

        slope_uncorrected_best_num = np.zeros(len(self.get_wavelength(masked)))
        slope_uncorrected_best_num_std = np.zeros(len(self.get_wavelength(masked)))
        for i in range(len(self.get_wavelength(masked))):
            result = lin_model.fit(self.get_absorbances(corrected=False, masked=masked, num='best')[:, i], params, x=self.variable_best_num)
            slope_uncorrected_best_num[i] = result.params['slope'].value
            slope_uncorrected_best_num_std[i] = result.params['slope'].stderr

        ys = [slope, slope_corrected_num, slope_best_num, slope_uncorrected, slope_uncorrected_num,
              slope_uncorrected_best_num]
        labels = ['corrected', 'corrected num', 'corrected best num', 'uncorrected', 'uncorrected num',
                  'uncorrected best num']
        plot_kwargs = self._set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Slope')
        legend_kwargs = self._set_defaults(legend_kwargs, title='Method')
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'slope vs wavelength method comparison {save_suffix}.pdf')
        self._1d_lines(self.get_wavelength(masked), ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                       labels=labels, save_loc=save_loc, show=show, **kwargs)

        # plot relative slope
        intensity = self.get_absorbances(corrected=True, masked=True, num='plot', var_value=np.max(self.variable))
        mask = intensity > min_intensity * np.max(intensity)

        y_min, y_max = np.min(slope_uncorrected[mask] / self.absorbances_masked[-1][mask]), np.max(
            slope_uncorrected[mask] / self.absorbances_masked[-1][mask])
        dy = y_max - y_min
        y_min -= 0.1 * dy
        y_max += 0.1 * dy

        ys = [slope / self.absorbances_masked_corrected[-1],
              slope_corrected_num / self.absorbances_masked_corrected_num[-1],
              slope_best_num / self.absorbances_masked_best_num[-1],
              slope_uncorrected / self.absorbances_masked[-1],
              slope_uncorrected_num / self.absorbances_masked_num[-1],
              slope_uncorrected_best_num / self.absorbances_masked_best_num[-1]]
        plot_kwargs = self._set_defaults(plot_kwargs, y_lim=(y_min, y_max))
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'relative slope vs wavelength method comparison{save_suffix}.pdf')
        self._1d_lines(self.get_wavelength(masked), ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                       labels=labels, save_loc=save_loc, show=show, **kwargs)

    def relative_intensity_fit_vs_variable(self, *, corrected=True, masked=True, save_loc=None, num='plot', show=True,
                                           reference_line, save_suffix='', plot_kwargs={}, legend_kwargs={}):
        def residual(pars, x, reference):
            a = pars['a'].value
            return x - a * reference

        params = lmfit.Parameters()
        params.add('a', value=1, vary=True)
        ratio = []
        ratio_std = []
        max_var = np.max(self.variable)
        for i in self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None):
            result = lmfit.minimize(residual, params, args=(i,),
                                    kws={'reference': self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=max_var)})
            ratio.append(result.params['a'].value)
            ratio_std.append(result.params['a'].stderr)
        ratio = np.array(ratio)
        ratio_std = np.array(ratio_std)

        fig, ax = plt.subplots()
        plt.errorbar(self.variable, ratio, yerr=ratio_std, capsize=2, fmt='.', label='measured intensity')
        if reference_line is not None:
            plt.plot(reference_line['x'], reference_line['y'], label=reference_line['label'])
        plt.xlabel(self.variable_name)
        plt.ylabel('Ratio')
        plt.grid()
        self._setting_setter(ax, **plot_kwargs)
        plt.tight_layout()
        if save_loc is not None:
            plt.savefig(os.path.join(save_loc, f'Relative intensity vs {self.variable_name} method comparison{save_suffix}.pdf'))
        if show:
            plt.show()
        else:
            plt.close()

        lines, labels = [], []
        plt.figure()
        for index, var in enumerate(np.unique(self.variable)):
            plt.plot(self.get_wavelength(True), self.absorbances_masked[self.variable == var].T / ratio[self.variable == var], f'C{index}', label=var)
            lines.append(plt.Line2D([0], [0], color=f'C{index}'))
            labels.append(f'{var}')
        # plt.plot(self.wavelength_masked, self.absorbances_masked.T/ratio, label=self.variable)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Relative absorbance (A.U.)')
        plt.legend(lines, labels, title=self.variable_display_name, **legend_kwargs)
        plt.grid()
        self._setting_setter(ax, **plot_kwargs)
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
        xs = self.variable_factor * self.variable_num
        ys = self.get_absorbance_ranges(ranges, corrected=corrected, masked=masked, num=num)
        labels = [f'{r[0]:^{wav_range_formatter}}-{r[1]:^{wav_range_formatter}} nm' for r in ranges]

        plot_kwargs = Analyzer._set_defaults(plot_kwargs, xlabel=self.variable_display_name, ylabel='Absorbance',
                                             xticks=self.variable_factor * self.variable_num,
                                             xticklabels=[f'{x:^{xtick_formatter}}' for x in self.variable_factor * self.variable_num])
        legend_kwargs = Analyzer._set_defaults(legend_kwargs, title='Wavelength range')
        line_kwargs = Analyzer._set_defaults(line_kwargs, marker='o', linestyle='-')
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'Average absorbance in ranges vs {self.variable_name} with{save_suffix}.pdf')
        self._1d_lines(xs, ys, labels=labels, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs, save_loc=save_loc,
                       show=show, line_kwargs=line_kwargs, **kwargs)

    def wavelength_range_ratio_vs_variable(self, ranges1: list[tuple[int, int]], ranges2: list[tuple[int, int]], *,
                                           corrected=True, masked=True, save_loc=None, num='plot', show=True,
                                           save_suffix='', plot_kwargs=None, line_kwargs=None, xtick_formatter='.1f',
                                           **kwargs):
        """
        Plot the ratio of the average absorbance in ranges1 to the average absorbance in ranges2 vs the variable
        """
        xs = self.variable_factor * self.variable_num
        mask1 = np.full(self.get_wavelength(masked).shape, False)
        for range in ranges1:
            mask1 = mask1 | ((range[0] < self.get_wavelength(masked)) & (self.get_wavelength(masked) < range[1]))
        mask2 = np.full(self.get_wavelength(masked).shape, False)
        for range in ranges2:
            mask2 = mask2 | ((range[0] < self.get_wavelength(masked)) & (self.get_wavelength(masked) < range[1]))

        y1 = np.average(self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None).T[mask1], axis=0)
        y2 = np.average(self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=None).T[mask2], axis=0)
        ys = [y1/y2]
        plot_kwargs = Analyzer._set_defaults(plot_kwargs, xlabel=self.variable_name, ylabel='Ratio',
                                             xticks=self.variable_factor * self.variable_num,
                                             xticklabels=[f'{x:^{xtick_formatter}}' for x in self.variable_factor * self.variable_num])
        line_kwargs = Analyzer._set_defaults(line_kwargs, marker='o', linestyle='')
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'absorbance vs {self.variable_name} with in ranges{save_suffix}.pdf')
        self._1d_lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show, line_kwargs=line_kwargs, **kwargs)

    def full_spectrum_fit_with_methods(self):
        def residual(pars, x, concentration):
            a = pars['a'].value
            return x - a * concentration * x[-1]

        params = lmfit.Parameters()
        params.add('a', value=1, vary=True)

        result = lmfit.minimize(residual, params, args=(self.get_absorbances(corrected=True, masked=True),),
                                kws={'concentration': self.variable[:, np.newaxis]})
        result_num = lmfit.minimize(residual, params, args=(self.get_absorbances(corrected=True, masked=True, num='plot'),),
                                    kws={'concentration': self.variable_num[:, np.newaxis]})
        result_best_num = lmfit.minimize(residual, params, args=(self.get_absorbances(corrected=True, masked=True, num='best'),),
                                         kws={'concentration': self.variable_best_num[:, np.newaxis]})
        result_uncorr = lmfit.minimize(residual, params, args=(self.get_absorbances(corrected=False, masked=True),),
                                       kws={'concentration': self.variable[:, np.newaxis]})
        result_uncorr_num = lmfit.minimize(residual, params, args=(self.get_absorbances(corrected=False, masked=True, num='plot'),),
                                           kws={'concentration': self.variable_num[:, np.newaxis]})
        result_uncorr_best_num = lmfit.minimize(residual, params, args=(self.get_absorbances(corrected=False, masked=True, num='best'),),
                                                kws={'concentration': self.variable_best_num[:, np.newaxis]})

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
