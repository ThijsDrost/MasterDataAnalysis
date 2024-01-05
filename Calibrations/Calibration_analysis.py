import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy
import lmfit

from Data_handling.Import import DataSet


plt.rcParams.update({'font.size': 14})


class Analyzer(DataSet):
    def __init__(self, variable_display_name, *args, variable_factor=1, cmap='turbo', **kwargs):
        super().__init__(*args, **kwargs)
        self.variable_factor = variable_factor
        self.variable_display_name = variable_display_name
        self.cmap = cmap

    @staticmethod
    def from_DataSet(dataset, variable_factor, variable_display_name, cmap):
        return Analyzer(variable_display_name, dataset.wavelength, dataset.absorbances, dataset.variable,
                        dataset.measurement_num, dataset.variable_name, dataset.wavelength_range,
                        dataset._selected_num, dataset.baseline_correction, variable_factor=variable_factor,
                        cmap=cmap)

    def get_wavelength(self, masked=True):
        return self.wavelength_masked if masked else self.wavelength

    @staticmethod
    def _setting_setter(ax, *, xlabel='', ylabel='', title='', grid=True, xlim=None, ylim=None, xticks=None,
                        yticks=None, xscale=None, yscale=None):
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if grid:
            ax.grid(grid)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        if xticks:
            ax.set_xticks(xticks)
        if yticks:
            ax.set_yticks(yticks)
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)
        # ax.tight_layout()

    @staticmethod
    def _1d_lines(xs, ys, *, colors=None, labels=None, legend_kwargs: dict = None, save_loc: str = None, show: bool = False,
                  plot_kwargs: dict = None, cbar_kwargs: dict = None, line_kwargs: dict = None, save_kwargs: dict = None):
        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if labels is None:
            labels = [None] * len(xs)

        if not isinstance(xs[0], (list, np.ndarray)):
            if len(xs) == len(ys[0]):
                xs = [xs] * len(ys)
            else:
                raise ValueError('xs and ys must have the same shape or xs must have the same length lists in ys')

        fig, ax = plt.subplots()
        line_kwargs = line_kwargs or {}
        for x, y, color, lab in zip(xs, ys, colors, labels):
            plt.plot(x, y, color, label=lab, **line_kwargs)

        if legend_kwargs is not None:
            plt.legend(**legend_kwargs)
        if cbar_kwargs is not None:
            plt.colorbar(**cbar_kwargs, ax=plt.gca())

        plot_kwargs = plot_kwargs or {}
        Analyzer._setting_setter(ax, **plot_kwargs)

        plt.tight_layout()
        if save_loc is not None:
            save_kwargs = save_kwargs or {}
            plt.savefig(save_loc, **save_kwargs)
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def set_defaults(kwargs_dict: dict | None, **kwargs) -> dict:
        kwargs_dict = kwargs_dict or {}
        for key, value in kwargs.items():
            if key not in kwargs_dict.keys():
                kwargs_dict[key] = value
        return kwargs_dict

    def absorbance_vs_wavelength_with_num(self, *, corrected=True, masked=True, save_loc=None, show=False,
                                          save_suffix='', plot_kwargs=None, legend_kwargs=None, line_kwargs=None,
                                          save_kwargs=None, **kwargs):
        for value in np.unique(self.variable):
            ys = [self.get_absorbances(corrected, masked, var, value).T for var in self.measurement_num_at_value(value)]
            labels = list(range(len(ys)))
            plot_kwargs = Analyzer.set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Absorbance')
            legend_kwargs = Analyzer.set_defaults(legend_kwargs, title='Measurement number')
            if save_loc is not None:
                save_loc = os.path.join(save_loc, f'absorbance vs wavelength at {value} {self.variable_name}{save_suffix}.pdf')
            self._1d_lines(self.get_wavelength(masked), ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                           save_loc=save_loc, show=show, labels=labels, line_kwargs=line_kwargs, save_kwargs=save_kwargs, **kwargs)
    
    def absorbance_vs_measurement_num_with_wavelength(self, *, corrected=True, masked=True, save_loc=None, num='plot',
                                                      wavelength_plot_every=5, min_absorbance=0.02, show=False,
                                                      save_suffix='', plot_kwargs=None, cbar_kwargs=None, line_kwargs=None,
                                                      save_kwargs=None, **kwargs):
        cmap = plt.get_cmap(self.cmap)
        for value in np.unique(self.variable):
            wav_abs_mask = self.get_absorbances(corrected, masked, num, value)[-1, :] > min_absorbance
            plot_num = len(self.wavelength_masked[wav_abs_mask][::wavelength_plot_every])
            ys = [y/y[-1] for y in (self.get_absorbances(corrected, masked, 'all', value).T[::wavelength_plot_every][index]
                                    for index in range(plot_num))]
            colors = [cmap(index / (plot_num // wavelength_plot_every)) for index in range(plot_num)]

            plot_kwargs = Analyzer.set_defaults(plot_kwargs, xlabel='Measurement number', ylabel='Relative absorbance',
                                                xticks=self.measurement_num_at_value(value))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.wavelength_masked[wav_abs_mask][0],
                                                                     vmax=self.wavelength_masked[wav_abs_mask][-1]))
            cbar_kwargs = Analyzer.set_defaults(cbar_kwargs, label='Wavelength (nm)', mappable=sm)
            line_kwargs = Analyzer.set_defaults(line_kwargs, marker='o', linestyle='-')

            if save_loc is not None:
                save_loc = os.path.join(save_loc, f'absorbance vs measurement num at {value} {self.variable_name}{save_suffix}.pdf')
            self._1d_lines(self.measurement_num_at_value(value), ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                           colors=colors, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs, save_kwargs=save_kwargs, **kwargs)

    # plot absorbance vs variable
    def absorbance_vs_wavelength_with_variable(self, *, corrected=True, masked=True, save_loc=None, num='plot',
                                               save_suffix='', plot_kwargs=None, legend_kwargs=None, show=False,
                                               line_kwargs: dict = None, save_kwargs: dict = None, **kwargs):
        ys = [self.get_absorbances(corrected, masked, num, var).T for var in np.unique(self.variable)]
        labels = [self.variable_factor * var for var in np.unique(self.variable)]
        plot_kwargs = Analyzer.set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Absorbance')
        legend_kwargs = Analyzer.set_defaults(legend_kwargs, title=self.variable_display_name)
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'absorbance vs wavelength with variable{save_suffix}.pdf')
        self._1d_lines(self.get_wavelength(masked), ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                       save_loc=save_loc, show=show, labels=labels, line_kwargs=line_kwargs, save_kwargs=save_kwargs,
                       **kwargs)

    def absorbance_vs_variable_with_wavelength(self, norm=False, *, corrected=True, masked=True, save_loc=None, num='plot',
                                               wavelength_plot_every=5, save_suffix='', plot_kwargs=None,
                                               cbar_kwargs=None, show=False, min_absorbance=0.02, **kwargs):
        cmap = plt.get_cmap(self.cmap)
        xs = self.variable_factor * self.variable_num
        y_len = len(self.wavelength_masked[::wavelength_plot_every])
        ys = [self.get_absorbances(corrected, masked, num, None).T[::wavelength_plot_every][index]
              for index in range(y_len)]
        if norm:
            ys = [y / y[-1] for y in ys if y[-1] > min_absorbance]
        colors = [cmap(index/(y_len-1)) for index in range(y_len)]
        plot_kwargs = Analyzer.set_defaults(plot_kwargs, xlabel=self.variable_name, ylabel='Absorbance',
                                            xticks=self.variable_factor * self.variable_num)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.get_wavelength(masked)[0],
                                                                 vmax=self.get_wavelength(masked)[-1]))
        cbar_kwargs = Analyzer.set_defaults(cbar_kwargs, label='Wavelength (nm)', mappable=sm)
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'absorbance vs {self.variable_name} with wavelength{save_suffix}.pdf')
        self._1d_lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                       colors=colors, cbar_kwargs=cbar_kwargs, **kwargs)

    def relative_absorbance_vs_variable_with_wavelength(self, *, corrected=True, masked=True, num='plot', save_loc=None,
                                                        wavelength_plot_every=5, min_absorbance=0.02, save_suffix='',
                                                        plot_kwargs=None, cbar_kwargs=None, show=False, **kwargs):
        cmap = plt.get_cmap(self.cmap)
        xs = self.variable_factor * self.variable_num
        wav_abs_mask = self.get_absorbances(corrected, masked, num, None)[-1, :] > min_absorbance
        y_len = len(self.wavelength_masked[wav_abs_mask][::wavelength_plot_every])
        ys = [y / y[-1] for y in (self.get_absorbances(corrected, masked, num, None).T[::wavelength_plot_every][index]
                                  for index in range(y_len))]

        colors = [cmap(index / (y_len - 1)) for index in range(y_len)]
        plot_kwargs = Analyzer.set_defaults(plot_kwargs, xlabel=self.variable_name, ylabel='Absorbance',
                                            xticks=self.variable_factor * self.variable_num)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.get_wavelength(masked)[0],
                                                                 vmax=self.get_wavelength(masked)[-1]))
        cbar_kwargs = Analyzer.set_defaults(cbar_kwargs, label='Wavelength (nm)', mappable=sm)
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'relative absorbance vs {self.variable_name} with wavelength{save_suffix}.pdf')
        self._1d_lines(xs, ys, plot_kwargs=plot_kwargs, save_loc=save_loc, show=show,
                       colors=colors, cbar_kwargs=cbar_kwargs, **kwargs)

    def absorbance_vs_variable_with_bunched_wavelength(self, bound_width, norm=False, *, corrected=True, masked=True, num='plot', save_loc=None,
                                                                save_suffix='', plot_kwargs={}, min_absorbance=0.02):
        cmap = plt.get_cmap(self.cmap)
        fig, ax = plt.subplots()
        vals = self.get_absorbances(corrected, masked, num, None)
        a_mask = np.max(vals, axis=0) > min_absorbance
        wav_m, vals_m = self.wavelength_masked[a_mask], vals[:, a_mask]
        data_len = len(wav_m)
        sorter = np.argsort(self.variable_num)
        for ndx in range(0, data_len, bound_width):
            i_min, i_max = ndx, min(ndx + bound_width, data_len)
            if i_max - i_min < 2:
                continue
            vals_selected = np.sum(vals_m[:, i_min: i_max], axis=1)
            if np.max(vals_selected) < min_absorbance:
                continue
            if norm:
                vals_selected = vals_selected[sorter] / vals_selected[sorter][-1]
            else:
                vals_selected = vals_selected[sorter]
            plt.plot(self.variable_factor * self.variable_num[sorter], vals_selected, '.', color=cmap(ndx / data_len))
        plt.xlabel(self.variable_name)
        plt.ylabel('Absorbance')
        plt.xticks(self.variable_factor * self.variable_num)
        # make a cmap for the plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=wav_m[0], vmax=wav_m[-1]))
        plt.colorbar(sm, label='Wavelength (nm)', ax=ax)
        self._setting_setter(ax, **plot_kwargs)
        plt.tight_layout()
        if save_loc is not None:
            plt.savefig(os.path.join(save_loc, f'relative absorbance vs {self.variable_name} with wavelength{save_suffix}.pdf'))

# pearson r for each wavelength
    def pearson_r_vs_wavelength_with_methods(self, *, save_loc=None, r2_values=None, masked=True, num='plot',
                                             save_suffix='', plot_kwargs=None, legend_kwargs=None, **kwargs):
        if r2_values is None:
            r2_values = [0, 1.025]
        if num == 'all' or num == 'best':
            warnings.warn(f'num = "{num}" is not logical for pearson_r_vs_wavelength_with_methods, should be "plot" or'
                          f' an integer, "plot" is used instead')
            num = 'plot'

        linearity = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            linearity[i] = scipy.stats.pearsonr(self.variable, self.get_absorbances(True, masked)[:, i])[0]

        linearity_corrected_num = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            linearity_corrected_num[i] = scipy.stats.pearsonr(self.variable_num, self.get_absorbances(True, masked, num)[:, i])[0]

        linearity_best_num = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            linearity_best_num[i] = scipy.stats.pearsonr(self.variable_best_num, self.get_absorbances(True, masked, 'best')[:, i])[0]

        linearity_uncorrected = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            linearity_uncorrected[i] = scipy.stats.pearsonr(self.variable, self.get_absorbances(False, masked)[:, i])[0]

        linearity_uncorrected_num = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            linearity_uncorrected_num[i] = scipy.stats.pearsonr(self.variable_num, self.get_absorbances(False, masked, num)[:, i])[0]

        linearity_uncorrected_best_num = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            linearity_uncorrected_best_num[i] = \
            scipy.stats.pearsonr(self.variable_best_num, self.get_absorbances(False, masked, 'best')[:, i])[0]

        r2_mask = (r2_values[0] < linearity ** 2) & (linearity ** 2 < r2_values[1])
        lins = [linearity_corrected_num, linearity_best_num, linearity_uncorrected, linearity_uncorrected_num,
                linearity_uncorrected_best_num]
        for lin in lins:
            r2_mask = r2_mask | ((r2_values[0] < lin ** 2) & (lin ** 2 < r2_values[1]))

        wavs = self.wavelength_masked[r2_mask]
        dw = np.diff(wavs)
        index = np.nonzero(dw > 10)[0]
        if len(index) > 0:
            r2_mask[index[0] + 1:] = False

        xs = [self.wavelength_masked[r2_mask]] * len(lins)
        ys = [linearity[r2_mask] ** 2, linearity_corrected_num[r2_mask] ** 2, linearity_best_num[r2_mask] ** 2,
              linearity_uncorrected[r2_mask] ** 2, linearity_uncorrected_num[r2_mask] ** 2,
              linearity_uncorrected_best_num[r2_mask] ** 2]
        labels = ['corrected', 'corrected num', 'corrected best num', 'uncorrected', 'uncorrected num',
                  'uncorrected best num']
        plot_kwargs = Analyzer.set_defaults(plot_kwargs, xlabel='Wavelength (nm)', ylabel='Linearity coefficient',
                                            ylim=r2_values)
        legend_kwargs = Analyzer.set_defaults(legend_kwargs, title='Method')
        if save_loc is not None:
            save_loc = os.path.join(save_loc, f'pearson r^2 vs wavelength method comparison{save_suffix}.pdf')

        self._1d_lines(xs, ys, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs, labels=labels, save_loc=save_loc,
                       **kwargs)

    def linear_fit_vs_wavelength_with_methods(self, *, save_loc=None, masked=True, num='plot', show=True,
                                              save_suffix='', plot_kwargs={}, legend_kwargs={}):
        # linear fit for each wavelength
        lin_model = lmfit.models.LinearModel()
        params = lin_model.make_params()
        params['intercept'].value = 0
        params['intercept'].vary = False

        slope = np.zeros(len(self.wavelength_masked))
        slope_std = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            result = lin_model.fit(self.get_absorbances(True, masked)[:, i], params, x=self.variable)
            slope[i] = result.params['slope'].value
            slope_std[i] = result.params['slope'].stderr

        slope_corrected_num = np.zeros(len(self.wavelength_masked))
        slope_corrected_num_std = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            result = lin_model.fit(self.get_absorbances(True, masked, num)[:, i], params, x=self.variable_num)
            slope_corrected_num[i] = result.params['slope'].value
            slope_corrected_num_std[i] = result.params['slope'].stderr

        slope_best_num = np.zeros(len(self.wavelength_masked))
        slope_best_num_std = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            result = lin_model.fit(self.get_absorbances(True, masked, 'best')[:, i], params, x=self.variable_best_num)
            slope_best_num[i] = result.params['slope'].value
            slope_best_num_std[i] = result.params['slope'].stderr

        slope_uncorrected = np.zeros(len(self.wavelength_masked))
        slope_uncorrected_std = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            result = lin_model.fit(self.get_absorbances(False, masked)[:, i], params, x=self.variable)
            slope_uncorrected[i] = result.params['slope'].value
            slope_uncorrected_std[i] = result.params['slope'].stderr

        slope_uncorrected_num = np.zeros(len(self.wavelength_masked))
        slope_uncorrected_num_std = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            result = lin_model.fit(self.get_absorbances(False, masked, num)[:, i], params, x=self.variable_num)
            slope_uncorrected_num[i] = result.params['slope'].value
            slope_uncorrected_num_std[i] = result.params['slope'].stderr

        slope_uncorrected_best_num = np.zeros(len(self.wavelength_masked))
        slope_uncorrected_best_num_std = np.zeros(len(self.wavelength_masked))
        for i in range(len(self.wavelength_masked)):
            result = lin_model.fit(self.get_absorbances(False, masked, 'best')[:, i], params, x=self.variable_best_num)
            slope_uncorrected_best_num[i] = result.params['slope'].value
            slope_uncorrected_best_num_std[i] = result.params['slope'].stderr

        fig, ax = plt.subplots()
        plt.errorbar(self.wavelength_masked, slope, yerr=slope_std, label='corrected')
        plt.errorbar(self.wavelength_masked, slope_corrected_num, yerr=slope_corrected_num_std, label='corrected num')
        plt.errorbar(self.wavelength_masked, slope_best_num, yerr=slope_best_num_std, label='corrected best num')
        plt.errorbar(self.wavelength_masked, slope_uncorrected, yerr=slope_uncorrected_std, label='uncorrected')
        plt.errorbar(self.wavelength_masked, slope_uncorrected_num, yerr=slope_uncorrected_num_std, label='uncorrected num')
        plt.errorbar(self.wavelength_masked, slope_uncorrected_best_num, yerr=slope_uncorrected_best_num_std, label='uncorrected best num')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Slope')
        plt.legend(**legend_kwargs)
        plt.grid()
        self._setting_setter(ax, **plot_kwargs)
        plt.tight_layout()
        if save_loc is not None:
            plt.savefig(os.path.join(save_loc, f'slope vs wavelength method comparison {save_suffix}.pdf'))
        if show:
            plt.show()
        else:
            plt.close()

        # plot relative slope
        intensity = self.get_absorbances(True, True, 'plot', np.max(self.variable))
        mask = intensity > 0.05 * np.max(intensity)

        y_min, y_max = np.min(slope_uncorrected[mask] / self.absorbances_masked[-1][mask]), np.max(
            slope_uncorrected[mask] / self.absorbances_masked[-1][mask])
        dy = y_max - y_min
        y_min -= 0.1 * dy
        y_max += 0.1 * dy


        fig, ax = plt.subplots()
        plt.plot(self.wavelength_masked, slope / self.absorbances_masked_corrected[-1], label='corrected')
        plt.plot(self.wavelength_masked, slope_corrected_num / self.absorbances_masked_corrected_num[-1], label='corrected num')
        plt.plot(self.wavelength_masked, slope_best_num / self.absorbances_masked_best_num[-1], label='corrected best num')
        plt.plot(self.wavelength_masked, slope_uncorrected / self.absorbances_masked[-1], label='uncorrected')
        plt.plot(self.wavelength_masked, slope_uncorrected_num / self.absorbances_masked_num[-1], label='uncorrected num')
        plt.plot(self.wavelength_masked, slope_uncorrected_best_num / self.absorbances_masked_best_num[-1], label='uncorrected best num')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Relative slope')
        plt.legend(**legend_kwargs)
        plt.grid()
        plt.ylim(y_min, y_max)
        self._setting_setter(ax, **plot_kwargs)
        plt.tight_layout()
        if save_loc is not None:
            plt.savefig(os.path.join(save_loc, f'relative slope vs wavelength method comparison{save_suffix}.pdf'))
        if show:
            plt.show()
        else:
            plt.close()

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
        for i in self.get_absorbances(corrected, masked, num, None):
            result = lmfit.minimize(residual, params, args=(i,),
                                    kws={'reference': self.get_absorbances(corrected, masked, num, max_var)})
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
            plt.plot(self.wavelength_masked, self.absorbances_masked[self.variable == var].T / ratio[self.variable == var], f'C{index}', label=var)
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

    def full_spectrum_fit_with_methods(self):
        def residual(pars, x, concentration):
            a = pars['a'].value
            return x - a * concentration * x[-1]


        params = lmfit.Parameters()
        params.add('a', value=1, vary=True)

        result = lmfit.minimize(residual, params, args=(self.get_absorbances(True, True),),
                                kws={'concentration': self.variable[:, np.newaxis]})
        result_num = lmfit.minimize(residual, params, args=(self.get_absorbances(True, True, 'plot'),),
                                    kws={'concentration': self.variable_num[:, np.newaxis]})
        result_best_num = lmfit.minimize(residual, params, args=(self.get_absorbances(True, True, 'best'),),
                                         kws={'concentration': self.variable_best_num[:, np.newaxis]})
        result_uncorr = lmfit.minimize(residual, params, args=(self.get_absorbances(False, True),),
                                       kws={'concentration': self.variable[:, np.newaxis]})
        result_uncorr_num = lmfit.minimize(residual, params, args=(self.get_absorbances(False, True, 'plot'),),
                                           kws={'concentration': self.variable_num[:, np.newaxis]})
        result_uncorr_best_num = lmfit.minimize(residual, params, args=(self.get_absorbances(False, True, 'best'),),
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

# plt.figure()
# plt.plot(self.wavelength_masked, (self.absorbances_masked_num / (result.params['a'].value * self.variable_num[:, np.newaxis])).T)
# plt.close()