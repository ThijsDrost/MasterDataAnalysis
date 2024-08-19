import os
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
import lmfit

from General.experiments.absorption import MeasurementsAnalyzer
from General.experiments.hdf5.readHDF5 import read_hdf5, DataSet
from General.experiments.oes import OESData
from General.experiments.absorption.Models import multi_species_model
from General.experiments.waveforms import Waveforms, MeasuredWaveforms
from General.plotting import plot, linestyles, cbar
from General.plotting.linestyles import linelooks_by, legend_linelooks, legend_linelooks_by, legend_linelooks_combines
from General.simulation.specair.specair import N2SpecAirSimulations, Spectrum, SpecAirSimulations
from General.itertools import argmax, argmin, flatten_2D, transpose, sort_by
import General.numpy_funcs as npf

width_colors = {
    '0.3': 'C0',
    '0.5': 'C1',
    '1': 'C2',
    '1.5': 'C7',
    '2': 'C3',
    '3': 'C4',
    '4': 'C6',
    '5': 'C5',
}

species_style = {
    r'N$_{2}$+OH': '-',
    r'N$_{2}$': '-',
    r'OH': '-',
    r'H$_{\alpha}$': '-.',
    'Ar': '--',
    'O': ':'
}


def analyse_directory_absorption(data_loc, voltages, pulse_lengths, save_loc=None, save_kwargs=None, lines_kwargs=None, add_slope=True,
                                 model_loc=r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Cuvette4.hdf5',
                                 show=False, show2=False, save_loc2=None, save_loc3=None, block_average_time=None, block_average_wav=None):
    if block_average_wav is not None:
        for voltage in voltages:
            for pulse in pulse_lengths:
                loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
                if not os.path.exists(loc):
                    continue
                data: DataSet = read_hdf5(loc)['absorbance'].remove_index(-1)
                break
            break
        wav_bounds = npf.block_average_bounds(data.wavelength, block_average_wav)
        model = multi_species_model(model_loc, add_zero=True, add_constant=True, add_slope=add_slope, wavelength_bounds=wav_bounds)
    else:
        model = multi_species_model(model_loc, add_zero=True, add_constant=True, add_slope=add_slope)

    voltages_ = []
    pulses_ = []
    results_ = []

    for voltage in voltages:
        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
            if not os.path.exists(loc):
                continue

            data: DataSet = read_hdf5(loc)['absorbance'].remove_index(-1)
            if block_average_time is not None:
                absorbances = npf.block_average(data.absorbances, block_average_time)
                variable = npf.block_average(data.variable, block_average_time)
                wavelengths = npf.block_average(data.wavelength, block_average_time)
                measurement_num = npf.block_average(data.measurement_num, block_average_time)
                data = DataSet(wavelengths, absorbances, variable, measurement_num, data.variable_name, data.wavelength_range,
                               data._selected_num)
            if block_average_wav is not None:
                absorbances = [npf.block_average(abs, block_average_wav) for abs in data.absorbances]
                data = DataSet(data.wavelength, absorbances, data.variable, data.measurement_num, data.variable_name, data.wavelength_range,
                               data._selected_num)

            analyzer = MeasurementsAnalyzer(data)

            result, _ = analyzer.fit(model, wavelength_range=(250, 400), show=show2, save_loc=save_loc2, export_fit_loc=save_loc3)
            results_.append(result)
            pulses_.append(pulse)
            voltages_.append(voltage)

    times_ = [result[1] for result in results_]
    h2o2_ = [result[2][0] for result in results_]
    no2 = [result[2][1] for result in results_]
    no3 = [result[2][2] for result in results_]

    lines_kwargs = lines_kwargs or {}
    v = [v.replace('kV', '') for v in voltages_]
    p = [p.replace('us', '') for p in pulses_]
    line_kwargs = linelooks_by(color_values=p, linestyle_values=v, colors=width_colors)
    legend_kwargs = legend_linelooks(line_kwargs, color_labels=p, linestyle_labels=v, color_title='W [us]', linestyle_title='H [kV]')
    plot_kwargs = {'ylabel': 'Concentration [mM]', 'xlabel': 'Time [min]'}

    if 'legend_kwargs' in lines_kwargs:
        legend_kwargs = plot.set_defaults(lines_kwargs['legend_kwargs'], **legend_kwargs)
        del lines_kwargs['legend_kwargs']
    if 'plot_kwargs' in lines_kwargs:
        plot_kwargs = plot.set_defaults(lines_kwargs['plot_kwargs'], **plot_kwargs)
        del lines_kwargs['plot_kwargs']
    h2o2_loc = save_loc + '_h2o2.pdf' if save_loc else None
    plot.lines(times_, h2o2_, line_kwargs_iter=line_kwargs, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs,
               save_loc=h2o2_loc, save_kwargs=save_kwargs, show=show, **lines_kwargs)
    no2_loc = save_loc + '_no2.pdf' if save_loc else None
    plot.lines(times_, no2, line_kwargs_iter=line_kwargs, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs,
               save_loc=no2_loc, save_kwargs=save_kwargs, show=show, **lines_kwargs)
    no3_loc = save_loc + '_no3.pdf' if save_loc else None
    return plot.lines(times_, no3, line_kwargs_iter=line_kwargs, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs,
                      save_loc=no3_loc, save_kwargs=save_kwargs, show=show, **lines_kwargs)


def analyse_fit(loc, plot_every_nth_fit, save_loc, model, plot_kwargs=None, wavelength_range=(250, 400),
                corrected=True):
    data: DataSet = read_hdf5(loc)['absorbance'].remove_index(-1)

    xs = data.get_wavelength(masked=True)
    ys = data.get_absorbances(corrected=corrected, masked=True)
    wav_mask = (xs > wavelength_range[0]) & (xs < wavelength_range[1])
    xs = xs[wav_mask]
    ys = ys[:, wav_mask]

    for index in range(0, len(data), plot_every_nth_fit):
        absorbance = ys[index]
        params = model.make_params()
        fit_result = model.fit(absorbance, x=xs, params=params)

        h2o2_name = [x for x in fit_result.params.keys() if 'h2o2' in x.lower()][0]
        no3_name = [x for x in fit_result.params.keys() if 'no3' in x.lower()][0]
        no2_name = [x for x in fit_result.params.keys() if 'no2' in x.lower()][0]

        h2o2_conc = fit_result.params[h2o2_name].value
        no3_conc = fit_result.params[no3_name].value
        no2_conc = fit_result.params[no2_name].value

        h2o2_std = fit_result.params[h2o2_name].stderr
        no3_std = fit_result.params[no3_name].stderr
        no2_std = fit_result.params[no2_name].stderr

        test_params = fit_result.params.copy()
        result = model.eval(x=xs, params=test_params)

        if h2o2_std is not None:
            test_params[h2o2_name].value = h2o2_conc - h2o2_std
            test_params[no3_name].value = no3_conc - no3_std
            test_params[no2_name].value = no2_conc - no2_std
            lower_bound = model.eval(x=xs, params=test_params)
            test_params[h2o2_name].value = h2o2_conc + h2o2_std
            test_params[no3_name].value = no3_conc + no3_std
            test_params[no2_name].value = no2_conc + no2_std
            upper_bound = model.eval(x=xs, params=test_params)
        else:
            print(fit_result.fit_report())

        plot_kwargs = plot_kwargs or {}

        fig, ax = plt.subplots()
        if h2o2_std is not None:
            ax.fill_between(xs, lower_bound, upper_bound, alpha=0.5)
        this_save_loc = save_loc + f'_{index}.pdf' if save_loc else None
        plot.lines(xs, [result, absorbance], fig_ax=(fig, ax), labels=['Fit', 'Data'], save_loc=this_save_loc, **plot_kwargs)


def analyse_directory_argon_emission(data_loc, voltages, pulse_lengths, *, save_loc=None, save_loc2=None, save_kwargs=None, show1=False,
                                     show2=False, save_kwargs2=None):
    argon_results = {}
    argon_results_std = {}
    argon_results_std2 = {}
    argon_wavs = {}
    argon_ratios = {}
    argon_ratios_times = {}

    peak_intensities = []
    voltage_vals = []
    std_vals = []

    peak_wav = OESData.peaks('argon')
    peak_index = peak_wav.index(763.504)

    for index, voltage in enumerate(voltages):
        argon_results[voltage] = {}
        argon_wavs[voltage] = {}
        argon_results_std[voltage] = {}
        argon_results_std2[voltage] = {}
        argon_ratios[voltage] = {}
        argon_ratios_times[voltage] = {}

        peak_intensity = []
        meas_num = []
        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
            if not os.path.exists(loc):
                continue

            data: OESData = read_hdf5(loc)['emission']
            data.remove_dead_pixels()
            data.remove_baseline((880, 900))
            plot_kwargs = {'title': f'{voltage} {pulse}'}
            t_save_loc = rf'{save_loc2}{voltage}_{pulse}.pdf' if save_loc2 else None
            data.peak_intensity_vs_wavelength_with_time('argon', plot_kwargs=plot_kwargs, show=show2, close=not show2,
                                                        save_loc=t_save_loc, save_kwargs=save_kwargs2)

            plot_kwargs = {'title': f'{voltage} {pulse}'}
            t_save_loc = rf'{save_loc2}{voltage}_{pulse}_rel.pdf' if save_loc2 else None
            data.peak_intensity_vs_wavelength_with_time('argon', plot_kwargs=plot_kwargs, norm=True, show=show2, close=not show2,
                                                        save_loc=t_save_loc, save_kwargs=save_kwargs2)

            time_vals, _, peak_intensities_with_time = data.peak_loc_intensities_with_time('argon', {'relative_threshold': 0.5, 'wavelength_range': (690, 870), 'use_max': False})

            colors, cbar_kwargs = cbar.bounded_cbar(peak_wav)
            plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Ratio', 'ylim': (0, 1.05)}
            cbar_kwargs['label'] = 'Peak wavelength [nm]'
            time_vals = (time_vals - time_vals[0])/60
            t_save_loc = rf'{save_loc2}_{voltage}_{pulse}_ratio_time.pdf' if save_loc else None
            ratios = peak_intensities_with_time/peak_intensities_with_time[5]
            mask = np.nanmax(ratios, axis=0) < 1.5
            if np.sum(mask) == 0:
                continue
            argon_ratios[voltage][pulse] = ratios
            argon_ratios_times[voltage][pulse] = time_vals
            time_vals, ratios = time_vals[mask], ratios[:, mask]
            plot.lines(time_vals, ratios, colors=colors, cbar_kwargs=cbar_kwargs,
                       plot_kwargs=plot_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs2, show=show2, close=not show2)
            plot_kwargs['ylim'] = (0.6, 1.025)
            t_save_loc = rf'{save_loc2}{voltage}_{pulse}_ratio_time_zoom1.pdf' if save_loc else None
            plot.lines(time_vals, ratios, colors=colors, cbar_kwargs=cbar_kwargs,
                       plot_kwargs=plot_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs2, show=show2, close=not show2)
            plot_kwargs['ylim'] = (0, 0.38)
            t_save_loc = rf'{save_loc2}{voltage}_{pulse}_ratio_time_zoom2.pdf' if save_loc else None
            plot.lines(time_vals, ratios, colors=colors, cbar_kwargs=cbar_kwargs,
                       plot_kwargs=plot_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs2, show=show2, close=not show2)

            peak_intensity.append(np.nanmean(peak_intensities_with_time[:, mask], axis=1))

            argon_results[voltage][pulse] = np.nanmean(peak_intensities_with_time[:, mask], axis=1)
            argon_results_std[voltage][pulse] = np.nanstd(peak_intensities_with_time[:, mask], axis=1)
            argon_results_std2[voltage][pulse] = np.nanstd((peak_intensities_with_time/peak_intensities_with_time[5])[:, mask], axis=1)
            argon_wavs[voltage][pulse] = peak_wav
            meas_num.append(len(time_vals))

        pulses = np.array(list(argon_results[voltage].keys()))
        pulse_vals = np.array([x.replace('us', '') for x in argon_results[voltage].keys()])
        peak_intensity = np.array(peak_intensity).T
        peak_intensity_norm = peak_intensity / peak_intensity[peak_index]
        peak_std_norm = np.array([argon_results_std[voltage][pulse] for pulse in pulses]).T / peak_intensity[peak_index]
        std_val_2 = np.array([argon_results_std2[voltage][pulse] for pulse in pulses]).T
        std_ref = peak_std_norm[peak_index]*peak_intensity_norm
        std_val = np.sqrt(std_ref**2 + peak_std_norm**2)
        std_val[peak_index] = 0
        std_val = std_val / np.sqrt(np.array(meas_num))

        line_kwargs = {'linestyle': '--'}

        cmap = plt.get_cmap('turbo')
        boundaries = [1.5 * peak_wav[0] - 0.5 * peak_wav[1]] + [(peak_wav[i] + peak_wav[i + 1]) / 2 for i in range(len(peak_wav) - 1)] + [
            1.5 * peak_wav[-1] - 0.5 * peak_wav[1]]
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
        colors = cmap(norm(peak_wav))

        ticker = mpl.ticker.FixedFormatter([f'{int(x)}' for x in peak_wav])
        tick_loc = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
        t_cbar_kwargs = {'mappable': plt.cm.ScalarMappable(norm=norm, cmap=cmap), 'label': 'Peak wavelength [nm]',
                         'ticks': tick_loc, 'format': ticker}

        plot_kwargs = {'ylabel': 'Relative intensity', 'ylim': (0., 1.05), 'xlabel': 'Pulse width [us]'}

        t_save_loc = rf'{save_loc}{voltage}.pdf' if save_loc else None
        plot.errorbar(pulse_vals, peak_intensity_norm, yerr=std_val_2, line_kwargs=line_kwargs, colors=colors, cbar_kwargs=t_cbar_kwargs,
                      save_loc=t_save_loc, save_kwargs=save_kwargs, plot_kwargs=plot_kwargs, show=show1, close=not show1)

        plot_kwargs['ylim'] = (0.6, 0.9)
        t_save_loc = rf'{save_loc}{voltage}_zoom1.pdf' if save_loc else None
        plot.errorbar(pulse_vals, peak_intensity_norm, yerr=std_val_2, line_kwargs=line_kwargs, colors=colors, close=not show1,
                      cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs, plot_kwargs=plot_kwargs, show=show1)

        plot_kwargs['ylim'] = (0, 0.38)
        t_save_loc = rf'{save_loc}{voltage}_zoom2.pdf' if save_loc else None
        plot.errorbar(pulse_vals, peak_intensity_norm, yerr=std_val_2, line_kwargs=line_kwargs, colors=colors, close=not show1,
                      cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs, plot_kwargs=plot_kwargs, show=show1)

        ratio_vals = [[npf.block_average(argon_ratios[voltage][pulse][x], 10) for pulse in argon_ratios[voltage]] for x in
                      range(len(peak_wav))]
        ratio_times = [[npf.block_average(argon_ratios_times[voltage][pulse], 10) for pulse in argon_ratios[voltage]] for x in
                       range(len(peak_wav))]
        pulse_length = [[pulse.replace('us', '') for pulse in argon_ratios[voltage]] for _ in range(len(peak_wav))]
        wavelength = [[peak_wav[x] for _ in range(len(argon_ratios[voltage]))] for x in range(len(peak_wav))]
        # color_vals = [colors.copy() for _ in range(len(peak_wav))]

        ratio_vals_f = flatten_2D(ratio_vals)
        ratio_times_f = flatten_2D(ratio_times)
        pulse_length_f = flatten_2D(pulse_length)
        wavelength_f = flatten_2D(wavelength)
        colors_f = cmap(norm(wavelength_f))

        line_kwargs_iter = linestyles.linelook_by(pulse_length_f, linestyles=True)
        legend_kwargs = legend_linelooks(line_kwargs_iter, linestyle_labels=pulse_length_f, linestyle_title='W [us]')
        for index in range(len(line_kwargs_iter)):
            line_kwargs_iter[index]['color'] = colors_f[index]
        t_save_loc = rf'{save_loc}{voltage}_ratio_time.pdf' if save_loc else None
        plot.lines(ratio_times_f, ratio_vals_f, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs,
                   cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc, show=show1)

        ratio_vals_s = ratio_vals[4] + ratio_vals[9]
        ratio_vals_change_s = [x/x[0] for x in ratio_vals_s]
        ratio_times_s = ratio_times[4] + ratio_times[9]
        pulse_length_s = pulse_length[4] + pulse_length[9]
        wavelength_s = [peak_wav[4]] * len(ratio_vals[4]) + [peak_wav[9]] * len(ratio_vals[9])

        line_kwargs_iter = linestyles.linelooks_by(color_values=pulse_length_s, linestyle_values=wavelength_s, colors=width_colors)
        legend_kwargs = linestyles.legend_linelooks(line_kwargs_iter, color_labels=pulse_length_s, color_title='W [us]',
                                                    linestyle_labels=wavelength_s, linestyle_title='Wav [nm]')
        plot_kwargs = {'ylabel': 'Relative intensity', 'xlabel': 'Time [min]'}
        t_save_loc = rf'{save_loc}{voltage}_ratio_time_s.pdf' if save_loc else None
        plot.lines(ratio_times_s, ratio_vals_s, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs,
                   save_loc=t_save_loc, show=show1, plot_kwargs=plot_kwargs, close=True)
        t_save_loc = rf'{save_loc}{voltage}_ratio_time_s_l.pdf' if save_loc else None
        plot.lines(ratio_times_s, ratio_vals_s, line_kwargs_iter=line_kwargs_iter, save_loc=t_save_loc, show=False,
                   plot_kwargs=plot_kwargs, close=True)

        plot_kwargs = {'ylabel': 'Relative intensity change', 'xlabel': 'Time [min]'}
        t_save_loc = rf'{save_loc}{voltage}_ratio_time_change_s.pdf' if save_loc else None
        plot.lines(ratio_times_s, ratio_vals_change_s, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs,
                   save_loc=t_save_loc, show=show1, plot_kwargs=plot_kwargs, close=True)
        t_save_loc = rf'{save_loc}{voltage}_ratio_time_change_s_l.pdf' if save_loc else None
        plot.lines(ratio_times_s, ratio_vals_change_s, line_kwargs_iter=line_kwargs_iter, save_loc=t_save_loc, show=False,
                   plot_kwargs=plot_kwargs, close=True)

        t_p = [pulse.replace('us', '') for pulse in pulse_lengths]*2
        t_w = [peak_wav[4], peak_wav[9]]*len(pulse_lengths)
        line_kwargs_iter = linestyles.linelooks_by(color_values=t_p, linestyle_values=t_w, colors=width_colors)
        legend_kwargs = linestyles.legend_linelooks(line_kwargs_iter, color_labels=t_p, color_title='W [us]',
                                                    linestyle_labels=t_w, linestyle_title='Wav [nm]')
        t_save_loc = rf'{save_loc}ratio_time_s_legend.pdf' if save_loc else None
        plot.export_legend(legend_kwargs, save_loc=t_save_loc)

        peak_intensities.append(peak_intensity)
        voltage_vals.append([voltage]*len(peak_wav))
        std_vals.append(std_val_2)

    energies = OESData.energies('ar')
    dE = max(energies[5] - min(energies), max(energies) - energies[5])
    centered_norm = mpl.colors.CenteredNorm(energies[0], halfrange=dE)
    cmap2 = plt.get_cmap('coolwarm')
    mappable2 = plt.cm.ScalarMappable(centered_norm, cmap2)
    colors2 = cmap2(centered_norm(energies))

    line_styles = linestyles.linelook_by(pulse_lengths, linestyles=True)
    for index, (voltage, inner_dict) in enumerate(argon_results.items()):
        line_kwargs = line_styles[index]

        pulses = list(inner_dict.keys())
        peak_intensity = [inner_dict[pulse] for pulse in pulses]
        peak_intensity_start_norm = [np.array(inner_dict[pulse]) for pulse in pulses]
        peak_intensity_norm = [np.array(inner_dict[pulse])/max(inner_dict[pulse]) for pulse in pulses]
        peak_std_norm = [np.array(argon_results_std[voltage][pulse])/max(inner_dict[pulse]) for pulse in pulses]
        pulses = list(map(list, zip(*[[p]*len(peak_intensity[0]) for p in pulses])))
        peak_intensity_norm_t = list(map(list, zip(*peak_intensity_norm)))
        peak_std_norm_t = list(map(list, zip(*peak_std_norm)))
        wavs = [x[0] for x in map(list, zip(*argon_wavs[voltage].values()))]
        std_val = std_vals[index]
        colors = cmap(norm(wavs))

        peak_intensity_start_norm = list(map(list, zip(*peak_intensity_start_norm)))
        peak_intensity_start_norm = [x/x[0] for x in peak_intensity_start_norm]

        peak_intensity_t = list(map(list, zip(*peak_intensity)))
        peak_intensity_t_norm = [np.array(inten)/max(inten) for inten in peak_intensity_t]

        peak_intensity_max_peak_norm = [np.array(inten)/inten[peak_index] for inten in peak_intensity]

        peak_intensity_norm_t_norm = [np.array(inten)/inten[0] for inten in peak_intensity_norm_t]

        t_save_loc = rf'{save_loc}{voltage}_total_energies.pdf' if save_loc else None
        plot.lines(pulses, peak_intensity_norm_t_norm, colors=colors2, show=show1, close=not show1, save_loc=t_save_loc,
                   cbar_kwargs={'mappable': mappable2, 'label': 'Energy [eV]'})

        if index + 1 == len(argon_results):
            plot_kwargs = {'ylabel': 'Relative intensity', 'xlabel': 'Pulse width [us]', 'ylim': (0, 1.05)}
            t_save_loc = rf'{save_loc}total.pdf' if save_loc else None
            plot.errorbar(pulses, peak_intensity_start_norm, yerr=std_val, line_kwargs=line_kwargs, colors=colors, fig_ax=rel_intensity_total,
                          save_kwargs=save_kwargs, cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
                          show=show1, close=not show1)
            plot_kwargs = {'ylabel': 'Relative intensity', 'xlabel': 'Pulse width [us]', 'ylim': (0.6, 0.9)}
            t_save_loc = rf'{save_loc}total_zoom1.pdf' if save_loc else None
            plot.errorbar(pulses, peak_intensity_norm_t, yerr=std_val, line_kwargs=line_kwargs, colors=colors,
                          fig_ax=rel_intensity_total_1, plot_kwargs=plot_kwargs, show=show1, close=not show1,
                          save_kwargs=save_kwargs, cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc)
            plot_kwargs = {'ylabel': 'Relative intensity', 'xlabel': 'Pulse width [us]', 'ylim': (0, 0.38)}
            t_save_loc = rf'{save_loc}total_zoom2.pdf' if save_loc else None
            plot.errorbar(pulses, peak_intensity_norm_t, yerr=std_val, line_kwargs=line_kwargs, colors=colors,
                          fig_ax=rel_intensity_total_2, plot_kwargs=plot_kwargs, show=show1, close=not show1,
                          save_kwargs=save_kwargs, cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc)
            t_save_loc = rf'{save_loc}total_energies.pdf' if save_loc else None
            plot.lines(pulses, peak_intensity_norm_t_norm, colors=colors2, line_kwargs=line_kwargs,
                       show=show1, close=not show1, fig_ax=peak_intensity_norm_plot, save_loc=t_save_loc,
                       cbar_kwargs={'mappable': mappable2, 'label': 'Energy [eV]'})
            plt.show()
        elif index == 0:
            rel_intensity_total = plot.errorbar(pulses, peak_intensity_norm_t, yerr=std_val, line_kwargs=line_kwargs, colors=colors, show=False, close=False)
            rel_intensity_total_1 = plot.errorbar(pulses, peak_intensity_norm_t, yerr=std_val, line_kwargs=line_kwargs,
                                                colors=colors, show=False, close=False)
            rel_intensity_total_2 = plot.errorbar(pulses, peak_intensity_norm_t, yerr=std_val, line_kwargs=line_kwargs,
                                                colors=colors, show=False, close=False)
            peak_intensity_norm_plot = plot.lines(pulses, peak_intensity_norm_t_norm, colors=colors2, line_kwargs=line_kwargs,
                                             show=False, close=False)
        else:
            rel_intensity_total = plot.errorbar(pulses, peak_intensity_norm_t, yerr=std_val, line_kwargs=line_kwargs, colors=colors, fig_ax=rel_intensity_total, show=False, close=False)
            rel_intensity_total_1 = plot.errorbar(pulses, peak_intensity_norm_t, yerr=std_val, line_kwargs=line_kwargs,
                                                  colors=colors, show=False, close=False, fig_ax=rel_intensity_total_1)
            rel_intensity_total_2 = plot.errorbar(pulses, peak_intensity_norm_t, yerr=std_val, line_kwargs=line_kwargs,
                                                  colors=colors, show=False, close=False, fig_ax=rel_intensity_total_2)
            peak_intensity_norm_plot = plot.lines(pulses, peak_intensity_norm_t_norm, colors=colors2, line_kwargs=line_kwargs,
                                             show=False, close=False, fig_ax=peak_intensity_norm_plot)

    # all_intensities = np.transpose(peak_intensities, axes=(1, 0, 2)).reshape(-1, len(pulse_lengths))
    # peak_intensities = np.array(peak_intensities)
    # all_intensities_norm = np.transpose(peak_intensities/peak_intensities.max(axis=1)[:, None, :], axes=(1, 0, 2)).reshape(-1, len(pulse_lengths))

    # all_colors = np.transpose(color_values, axes=(1, 0, 2)).reshape(-1, 4)
    # all_voltages = np.array(voltage_vals).T.flatten()
    # line_kwargs_iter = linelooks_by(linestyle_values=all_voltages)
    # legend_kwargs = legend_linelooks(line_kwargs_iter, linestyle_labels=all_voltages)

    # pd_frame = pd.DataFrame(argon_results)
    # for voltage in voltages:
    #     pd_series = pd_frame[voltage]
    #     for pulse in pd_series.index:
    #         intensities = pd_series[pulse]
    #         if isinstance(intensities, float):
    #             continue
    #
    #         c = 299_792_458
    #         f_763 = c/763.51e-9
    #         f_811 = c/811.53e-9
    #         f_750 = c/750.39e-9
    #         i_763 = intensities[763.504]/(0.274*f_763)
    #         i_811 = intensities[811.485]/(0.331*f_811)
    #         i_750 = intensities[750.52]/(0.472*f_750)
    #         e_763 = 13.171670  # eV
    #         e_811 = 13.075609
    #         e_750 = 13.479776
    #
    #         Te_1 = (e_763 - e_811)/(math.log(i_811/i_763))
    #         Te_2 = (e_763 - e_750)/(math.log(i_750/i_763))
    #         print(Te_1, Te_2)  # Gives negative electron temperatures, so not useful
    #
    # pulse_lengths = [list(pd_frame[name].index) for name in pd_frame]
    # values = [pd_frame[series].values for series in pd_frame if not pd.isna(pd_frame[series])]
    # intensities = [[pd_frame[series][s].values for s in pd_frame[series].index if not isinstance(pd_frame[series][s], float)] for series in pd_frame]
    # wavelengths = [[list(pd_frame[series][s].index) for s in pd_frame[series].index if not isinstance(pd_frame[series][s], float)] for series in pd_frame]
    #
    # intensity2 = [list(map(list, zip(*l))) for l in intensities]
    # wavelengths2 = [x[0] for l in wavelengths for x in list(map(list, zip(*l)))]
    # voltages2 = [y for x in list(map(list, zip(*voltage_vals))) for y in x]
    #
    #
    #
    # plot.lines(pulse_lengths, all_intensities_norm, colors=all_colors, cbar_kwargs=t_cbar_kwargs,
    #            save_loc=save_loc, save_kwargs=save_kwargs, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs)
    #
    # plot.lines(pulse_lengths, all_intensities, colors=all_colors, cbar_kwargs=t_cbar_kwargs,
    #            save_loc=save_loc, save_kwargs=save_kwargs, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs)
    #
    # plot.lines(pulse_lengths, all_intensities/all_intensities[:, 0][:, None], colors=all_colors, cbar_kwargs=t_cbar_kwargs,
    #            save_loc=save_loc, save_kwargs=save_kwargs, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs)


def analyse_directory_pulse(data_loc, voltages, pulse_lengths, channels, save_loc=None, save_loc2=None, show=False, show2=False):
    def zoom(times, currents, dts: tuple[int|float, int|float], find_func: callable):
        max_idx = find_func(currents)
        delta_t = np.average(np.diff(times))
        start = max_idx - int(dts[0]/delta_t)
        end = max_idx + int(dts[1]/delta_t)
        return times[start:end]-times[start], currents[start:end]

    def model(x, amplitude, decay, phase, length, offset):
        return amplitude * np.exp(-(x - x[0]) / decay) * np.sin((2 * np.pi * x) / length + phase) + offset

    lmfit_model = lmfit.Model(model)
    lmfit_model.set_param_hint('amplitude', value=0.25, min=0)
    lmfit_model.set_param_hint('decay', value=2e-6)
    lmfit_model.set_param_hint('phase', value=-1.5)
    lmfit_model.set_param_hint('length', value=1.9e-7)
    lmfit_model.set_param_hint('offset', value=0.8)

    first_pulse = {}
    middle_pulse = {}
    end_pulse = {}

    pulse_height = {}
    pulse_width = {}
    rise_times = {}
    raw_pulse_height = {}
    raw_pulse_width = {}
    raw_rise_times = {}
    raw_timestamps = {}

    background_current = {}
    background_current_std = {}
    background_current_time = {}

    background_current2 = {}
    background_current_std2 = {}
    background_current_time2 = {}

    background_current3 = {}
    background_current_std3 = {}
    background_current_time3 = {}

    background_current4 = {}
    background_current_std4 = {}

    power_val = {}
    power_time = {}
    power_avg_val = {}
    power_avg_time = {}

    highest_current = {}
    lowest_current = {}
    time_current = {}

    for voltage in voltages:
        first_pulse[voltage] = {}
        middle_pulse[voltage] = {}
        end_pulse[voltage] = {}
        pulse_height[voltage] = {}
        pulse_width[voltage] = {}
        rise_times[voltage] = {}
        raw_pulse_height[voltage] = {}
        raw_pulse_width[voltage] = {}
        raw_rise_times[voltage] = {}
        raw_timestamps[voltage] = {}
        background_current[voltage] = {}
        background_current_std[voltage] = {}
        background_current_time[voltage] = {}
        background_current2[voltage] = {}
        background_current_std2[voltage] = {}
        background_current_time2[voltage] = {}
        background_current3[voltage] = {}
        background_current_std3[voltage] = {}
        background_current_time3[voltage] = {}
        background_current4[voltage] = {}
        background_current_std4[voltage] = {}
        highest_current[voltage] = {}
        lowest_current[voltage] = {}
        time_current[voltage] = {}
        power_val[voltage] = {}
        power_avg_val[voltage] = {}
        power_time[voltage] = {}
        power_avg_time[voltage] = {}

        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
            if not os.path.exists(loc):
                continue

            print(rf'{voltage}_{pulse}')
            data: Waveforms = read_hdf5(loc)['waveforms']

            wavs = MeasuredWaveforms.from_waveforms(data, channels)
            t_save_loc = save_loc + f'{voltage}_{pulse}_' if save_loc else None
            wavs.plot(save_loc=t_save_loc, close=True)

            is_on = wavs.is_on()
            time = wavs.time
            current = wavs.currents
            voltage_vals = wavs.voltages
            time_offsets = wavs.time_offset

            if len(current) == 0:
                continue

            n = 20
            avg_current = npf.block_average(current, n)
            avg_voltage = npf.block_average(voltage_vals, n)
            avg_time_offsets = npf.block_average(time_offsets, n)
            avg_time = npf.block_average(time, n)
            avg_power = avg_current*avg_voltage
            avg_pulse_power = np.trapz(avg_time, avg_power, axis=1)

            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Energy [J]'}
            t_save_loc = save_loc2 + f'{voltage}_{pulse}_avg_energy.pdf' if save_loc2 else None
            plot.lines(avg_time_offsets-avg_time_offsets[0], avg_pulse_power, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show2, close=True)

            plot_kwargs = {'xlabel': r'Time [us]', 'ylabel': 'Power [W]'}
            colors, mappable = cbar.cbar_norm_colors(avg_time_offsets/60, 'turbo')
            cbar_kwargs = {'label': 'Time [min]', 'mappable': mappable}
            t_save_loc = save_loc + f'{voltage}_{pulse}_avg_power.pdf' if save_loc else None
            plot.lines((avg_time-avg_time[0])*1e6, avg_power, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show2, close=True, colors=colors,
                       cbar_kwargs=cbar_kwargs)

            plot_kwargs = {'xlabel': r'Time [us]', 'ylabel': 'Current change [A]', 'xlim': (0, float(pulse.replace('us', '')) + 1)}
            t_save_loc = save_loc2 + f'{voltage}_{pulse}_avg_current_change.pdf' if save_loc2 else None
            t_offsets = npf.block_average(time_offsets[is_on][1:], 60)
            colors, mappable = cbar.cbar_norm_colors((t_offsets-t_offsets[0])/60, 'turbo')
            cbar_kwargs = {'label': 'Time [min]', 'mappable': mappable}
            index = 0
            t_vals = npf.block_average(time[is_on][1:], 60)*1e6
            t_vals = t_vals - t_vals[:, index][:, None]
            c_vals = npf.block_average(current[is_on][1:], 60)
            c_vals = c_vals-c_vals[index]
            plot.lines(t_vals[::-1], c_vals[::-1], save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show2,
                       close=True, colors=colors[::-1], cbar_kwargs=cbar_kwargs)

            power_time[voltage][pulse] = (time_offsets[is_on] - time_offsets[is_on][0])/60
            power_val[voltage][pulse] = np.trapz(time[is_on], current[is_on]*voltage_vals[is_on], axis=1)
            power_avg_time[voltage][pulse] = npf.block_average((time_offsets[is_on] - time_offsets[is_on][0])/60, n)
            power_avg_val[voltage][pulse] = np.trapz(npf.block_average(time[is_on], n), npf.block_average(current[is_on]*voltage_vals[is_on], n), axis=1)

            on_current = current[is_on]
            middle = on_current.shape[0]//2
            first_pulse[voltage][pulse] = (np.nanmean(time[is_on][5:15], axis=0), np.nanmean(on_current[5:15], axis=0))
            middle_pulse[voltage][pulse] = (np.nanmean(time[is_on][(middle-5):(middle+5)], axis=0), np.nanmean(on_current[(middle-5):(middle+5)], axis=0))
            end_pulse[voltage][pulse] = (np.nanmean(time[is_on][-15:-5], axis=0), np.nanmean(on_current[-15:-5], axis=0))

            rise_time, pulse_length, height = wavs.fit_voltage()
            rise_times[voltage][pulse] = np.mean(rise_time[is_on]), np.std(rise_time[is_on])
            mask = np.isfinite(rise_time[is_on])
            pulse_height[voltage][pulse] = (np.average(height[is_on][mask][10:-5]), np.std(height[is_on][mask][10:-5]))
            pulse_width[voltage][pulse] = (np.average(pulse_length[is_on][mask][10:-5]), np.std(pulse_length[is_on][mask][10:-5]))

            raw_pulse_height[voltage][pulse] = height[is_on]
            raw_pulse_width[voltage][pulse] = pulse_length[is_on]
            raw_rise_times[voltage][pulse] = rise_time[is_on]
            raw_timestamps[voltage][pulse] = time_offsets[is_on]

            avg_num = 20
            # avg, std = wavs.background_current_averaging(on_mask=is_on)
            time_vals = npf.block_average(wavs.time_offset[is_on], avg_num)
            # background_current[voltage][pulse] = avg
            # background_current_std[voltage][pulse] = std
            if pulse.replace('us', '').strip() == '0.3':
                start_offset = 0.125
                end_offset = 0.125
            else:
                start_offset = 0.25
                end_offset = 0.15

            avg, std = wavs.background_current_averaging(start_offset=start_offset, end_offset=end_offset, on_mask=is_on, block_average=avg_num)
            background_current[voltage][pulse] = avg
            background_current_std[voltage][pulse] = std
            background_current_time[voltage][pulse] = time_vals - time_vals[0]

            avg, std = wavs.background_current_averaging(on_mask=is_on, block_average=avg_num)
            time_vals = wavs.time_offset[is_on]
            time_vals = npf.block_average(time_vals, avg_num)
            background_current3[voltage][pulse] = avg
            background_current_std3[voltage][pulse] = std
            background_current_time3[voltage][pulse] = time_vals - time_vals[0]

            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Background current [A]'}
            t_save_loc = save_loc + f'{voltage}_{pulse}_background.pdf' if save_loc else None
            plot.errorrange(background_current_time[voltage][pulse], background_current[voltage][pulse], yerr=background_current_std[voltage][pulse], save_loc=t_save_loc,
                            plot_kwargs=plot_kwargs, show=show2, close=True)

            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Background current [A]'}
            t_save_loc = save_loc + f'{voltage}_{pulse}_background2.pdf' if save_loc else None
            plot.errorrange(background_current_time3[voltage][pulse], background_current3[voltage][pulse],
                            yerr=background_current_std3[voltage][pulse], save_loc=t_save_loc,
                            plot_kwargs=plot_kwargs, show=show2, close=True)

            t_save_loc = save_loc2 + rf'{voltage}_{pulse}_bacground_fit' if save_loc2 else None
            avg_offset, back_curr_results, back_curr_std = wavs.background_current_fitting(float(pulse.replace('us', '')),
                                                                                           save_loc=t_save_loc)
            background_current2[voltage][pulse] = back_curr_results
            background_current_std2[voltage][pulse] = back_curr_std
            background_current_time2[voltage][pulse] = avg_offset

            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Background current [A]'}
            t_save_loc = save_loc + f'{voltage}_{pulse}_background_fit.pdf' if save_loc else None
            plot.errorrange(background_current_time2[voltage][pulse], background_current2[voltage][pulse],
                            yerr=background_current_std2[voltage][pulse], save_loc=t_save_loc,
                            plot_kwargs=plot_kwargs, show=show2, close=True)

            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Background current [A]'}
            t_save_loc = save_loc + f'{voltage}_{pulse}_background_comp.pdf' if save_loc else None
            fig_ax = plot.errorrange(background_current_time[voltage][pulse], background_current[voltage][pulse],
                                     yerr=background_current_std[voltage][pulse], show=False, close=False)
            plot.errorrange(background_current_time2[voltage][pulse], background_current2[voltage][pulse],
                            yerr=background_current_std2[voltage][pulse], save_loc=t_save_loc,
                            plot_kwargs=plot_kwargs, show=show2, close=True, fig_ax=fig_ax)

            highest_current[voltage][pulse] = np.nanmax(on_current, axis=1)
            lowest_current[voltage][pulse] = np.nanmin(on_current, axis=1)
            time_current[voltage][pulse] = wavs.time_offset[is_on]

            t_save_loc = save_loc + f'{voltage}_{pulse}_highest_current.pdf' if save_loc else None
            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Maximum current [A]'}
            plot.lines(time_current[voltage][pulse], highest_current[voltage][pulse], show=show2, close=True,
                       save_loc=t_save_loc, plot_kwargs=plot_kwargs)
            t_save_loc = save_loc + f'{voltage}_{pulse}_lowest_current.pdf' if save_loc else None
            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Minimum current [A]'}
            plot.lines(time_current[voltage][pulse], lowest_current[voltage][pulse], show=show2, close=True,
                          save_loc=t_save_loc, plot_kwargs=plot_kwargs)

        pulses = list(first_pulse[voltage].keys())
        if len(pulses) == 0:
            warnings.warn(f"No pulses for voltage: {voltage}")
            continue
        first_pulse_vals = [first_pulse[voltage][pulse] for pulse in pulses]
        middle_pulse_vals = [middle_pulse[voltage][pulse] for pulse in pulses]
        end_pulse_vals = [end_pulse[voltage][pulse] for pulse in pulses]

        first_pulse_time = [x[0] for x in first_pulse_vals]
        first_pulse_current = [x[1] for x in first_pulse_vals]
        middle_pulse_time = [x[0] for x in middle_pulse_vals]
        middle_pulse_current = [x[1] for x in middle_pulse_vals]
        end_pulse_time = [x[0] for x in end_pulse_vals]
        end_pulse_current = [x[1] for x in end_pulse_vals]

        plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Current [A]'}
        t_save_loc = save_loc + f'{voltage}_first_pulse.pdf' if save_loc else None
        plot.lines(first_pulse_time, first_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show, close=not show)
        t_save_loc = save_loc + f'{voltage}_middle_pulse.pdf' if save_loc else None
        plot.lines(middle_pulse_time, middle_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show, close=not show)
        t_save_loc = save_loc + f'{voltage}_end_pulse.pdf' if save_loc else None
        plot.lines(end_pulse_time, end_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show, close=not show)

        for name, func, dts in zip(('min', 'max'), (argmin, argmax), ((1e-7, 2e-7), (1e-7, 2e-7))):
            first_pulse_zoom_max = [zoom(time, current, dts, func) for time, current in zip(first_pulse_time, first_pulse_current)]
            middle_pulse_zoom_max = [zoom(time, current, dts, func) for time, current in zip(middle_pulse_time, middle_pulse_current)]
            end_pulse_zoom_max = [zoom(time, current, dts, func) for time, current in zip(end_pulse_time, end_pulse_current)]

            first_pulse_time_zoom_max = [val[0]*1e6 for val in first_pulse_zoom_max]
            first_pulse_vals_zoom_max = [val[1] for val in first_pulse_zoom_max]
            middle_pulse_time_zoom_max = [val[0]*1e6 for val in middle_pulse_zoom_max]
            middle_pulse_vals_zoom_max = [val[1] for val in middle_pulse_zoom_max]
            end_pulse_time_zoom_max = [val[0]*1e6 for val in end_pulse_zoom_max]
            end_pulse_vals_zoom_max = [val[1] for val in end_pulse_zoom_max]

            t_save_loc = save_loc + f'{voltage}_first_pulse_zoom_{name}.pdf' if save_loc else None
            plot.lines(first_pulse_time_zoom_max, first_pulse_vals_zoom_max, save_loc=t_save_loc, plot_kwargs=plot_kwargs, labels=pulses,
                       show=show, close=not show)
            t_save_loc = save_loc + f'{voltage}_middle_pulse_zoom_{name}.pdf' if save_loc else None
            plot.lines(middle_pulse_time_zoom_max, middle_pulse_vals_zoom_max, save_loc=t_save_loc, plot_kwargs=plot_kwargs, labels=pulses,
                       show=show, close=not show)
            t_save_loc = save_loc + f'{voltage}_end_pulse_zoom_{name}.pdf' if save_loc else None
            plot.lines(end_pulse_time_zoom_max, end_pulse_vals_zoom_max, save_loc=t_save_loc, plot_kwargs=plot_kwargs, labels=pulses,
                       show=show, close=not show)

        pulse_height_vals = [pulse_height[voltage][pulse] for pulse in pulses]
        pulse_width_vals = [pulse_width[voltage][pulse] for pulse in pulses]
        rise_time_vals = [rise_times[voltage][pulse] for pulse in pulses]

        pulse_height_val = [x[0] for x in pulse_height_vals]
        pulse_height_std = [x[1] for x in pulse_height_vals]
        rise_time_val = [1e9*x[0] for x in rise_time_vals]
        rise_time_std = [1e9*x[1] for x in rise_time_vals]
        pulse_width_val = [x[0]/float(val.replace('us', '')) for x, val in zip(pulse_width_vals, pulses)]
        pulse_width_std = [x[1]/float(val.replace('us', '')) for x, val in zip(pulse_width_vals, pulses)]

        p_val = [p.removesuffix('us') for p in pulses]
        plot_kwargs = {'xlabel': 'Pulse width', 'ylabel': 'Height [V]'}
        t_save_loc = save_loc + f'{voltage}_pulse_height.pdf' if save_loc else None
        plot.errorbar(p_val, pulse_height_val, yerr=pulse_height_std, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show, close=not show)
        plot_kwargs = {'xlabel': 'Pulse width', 'ylabel': 'Width accuracy'}
        t_save_loc = save_loc + f'{voltage}_pulse_width.pdf' if save_loc else None
        plot.errorbar(p_val, pulse_width_val, yerr=pulse_width_std, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show, close=not show)
        plot_kwargs = {'xlabel': 'Pulse width', 'ylabel': 'Rise time [ns]'}
        t_save_loc = save_loc + f'{voltage}_rise_time.pdf' if save_loc else None
        plot.errorbar(p_val, rise_time_val, yerr=rise_time_std, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show, close=not show)

        for background, background_std, background_time, name in zip((background_current, background_current2, background_current3),
                                                                     (background_current_std, background_current_std2, background_current_std3),
                                                                     (background_current_time, background_current_time2, background_current_time3),
                                                                     ('', '_fit', '2'), strict=True):
            background_current_vals = [background[voltage][pulse] for pulse in pulses]
            background_current_std_vals = [background_std[voltage][pulse] for pulse in pulses]
            background_current_time_vals = [background_time[voltage][pulse] for pulse in pulses]

            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Background current [A]'}
            t_save_loc = save_loc + f'{voltage}_background{name}.pdf' if save_loc else None
            plot.errorrange(background_current_time_vals, background_current_vals, yerr=background_current_std_vals,
                            save_loc=t_save_loc, plot_kwargs=plot_kwargs, labels=pulses, show=show, close=not show)

        b_currents_rel = [background_current[voltage][pulse] - np.average(background_current[voltage][pulse][:10]) for pulse in pulses]
        bt_currents = [background_current_time[voltage][pulse] - background_current_time[voltage][pulse][0] for pulse in pulses]

        h_currents = [highest_current[voltage][pulse] for pulse in pulses]
        l_currents = [lowest_current[voltage][pulse] for pulse in pulses]
        h_currents_rel = [highest_current[voltage][pulse] - np.average(highest_current[voltage][pulse][:10]) for pulse in pulses]
        l_currents_rel = [lowest_current[voltage][pulse] - np.average(lowest_current[voltage][pulse][:10]) for pulse in pulses]
        h_currents_norm = [highest_current[voltage][pulse]/np.average(highest_current[voltage][pulse][:10]) for pulse in pulses]
        l_currents_norm = [lowest_current[voltage][pulse]/np.average(lowest_current[voltage][pulse][:10]) for pulse in pulses]
        t_currents = [time_current[voltage][pulse] - time_current[voltage][pulse][0] for pulse in pulses]

        for name, currents in zip(('highest', 'lowest', 'highest_rel', 'lowest_rel', 'highest_norm', 'lowest_norm'),
                                  (h_currents, l_currents, h_currents_rel, l_currents_rel, h_currents_norm, l_currents_norm)):
            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Current [A]'}
            t_save_loc = save_loc + f'{voltage}_peak_current_{name}.pdf' if save_loc else None
            plot.lines(t_currents, currents, save_loc=t_save_loc, plot_kwargs=plot_kwargs, labels=pulses, show=show, close=not show)

        temp_times = bt_currents + t_currents + t_currents
        temp_currents = b_currents_rel + h_currents_rel + l_currents_rel
        labels = ['Background']*len(bt_currents) + ['Highest']*len(h_currents) + ['Lowest']*len(l_currents)
        pulse_vals = p_val*3
        line_kwargs_iter = linestyles.linelooks_by(color_values=pulse_vals, linestyle_values=labels, linestyles=[':', '--', '-.'],
                                                   colors=width_colors)
        legend_kwargs = linestyles.legend_linelooks(line_kwargs_iter, linestyle_labels=labels, color_labels=pulse_vals,
                                                    linestyle_title='Type', color_title='Pulse [us]')
        plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Current change [A]'}
        t_save_loc = save_loc + f'{voltage}_all_currents.pdf' if save_loc else None
        plot.lines(temp_times, temp_currents, save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                   legend_kwargs=legend_kwargs, show=show, close=not show)

    pulses = [list(first_pulse[voltage].keys()) for voltage in voltages]
    pulse_vals = flatten_2D(pulses)
    pulse_vals = [p.removesuffix('us') for p in pulse_vals]
    voltage_vals = flatten_2D([[v]*len(pulse) for v, pulse in zip(voltages, pulses)])
    voltage_vals = [v.removesuffix('kV') for v in voltage_vals]

    first_pulse_vals = [[first_pulse[voltage][pulse] for pulse in pulse_list] for voltage, pulse_list in zip(voltages, pulses)]
    middle_pulse_vals = [[middle_pulse[voltage][pulse] for pulse in pulse_list] for voltage, pulse_list in zip(voltages, pulses)]
    end_pulse_vals = [[end_pulse[voltage][pulse] for pulse in pulse_list] for voltage, pulse_list in zip(voltages, pulses)]

    first_pulse_time = [x[0]*1e6 for pulse_list in first_pulse_vals for x in pulse_list]
    first_pulse_current = [x[1] for pulse_list in first_pulse_vals for x in pulse_list]
    middle_pulse_time = [x[0]*1e6 for pulse_list in middle_pulse_vals for x in pulse_list]
    middle_pulse_current = [x[1] for pulse_list in middle_pulse_vals for x in pulse_list]
    end_pulse_time = [x[0]*1e6 for pulse_list in end_pulse_vals for x in pulse_list]
    end_pulse_current = [x[1] for pulse_list in end_pulse_vals for x in pulse_list]

    # p_vals = [p.removesuffix('us') for p in pulse_vals]
    line_kwargs_iter = linestyles.linelooks_by(color_values=pulse_vals, linestyle_values=voltage_vals, colors=width_colors)
    legend_kwargs = linestyles.legend_linelooks(line_kwargs_iter, linestyle_labels=voltage_vals, color_labels=pulse_vals,
                                                linestyle_title='H [kV]', color_title='W [us]')

    plot_kwargs = {'xlabel': 'Time [us]', 'ylabel': 'Current [A]'}
    t_save_loc = save_loc + f'first_pulse.pdf' if save_loc else None
    plot.lines(first_pulse_time, first_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
               line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show, close=not show)
    t_save_loc = save_loc + f'middle_pulse.pdf' if save_loc else None
    plot.lines(middle_pulse_time, middle_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
               line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show, close=not show)
    t_save_loc = save_loc + f'end_pulse.pdf' if save_loc else None
    plot.lines(end_pulse_time, end_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
               line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show, close=not show)

    for name, func, dts in zip(('min', 'max'), (argmin, argmax), ((1e-7, 2e-7), (1e-7, 2e-7))):
        first_pulse_vals_zoom = [zoom(x[0], x[1], dts, func) for pulse_list in first_pulse_vals for x in pulse_list]
        middle_pulse_vals_zoom = [zoom(x[0], x[1], dts, func) for pulse_list in middle_pulse_vals for x in pulse_list]
        end_pulse_vals_zoom = [zoom(x[0], x[1], dts, func) for pulse_list in end_pulse_vals for x in pulse_list]
        first_pulse_time_zoom = [x[0]*1e6 for x in first_pulse_vals_zoom]
        first_pulse_vals_zoom = [x[1] for x in first_pulse_vals_zoom]
        middle_pulse_time_zoom = [x[0]*1e6 for x in middle_pulse_vals_zoom]
        middle_pulse_vals_zoom = [x[1] for x in middle_pulse_vals_zoom]
        end_pulse_time_zoom = [x[0]*1e6 for x in end_pulse_vals_zoom]
        end_pulse_vals_zoom = [x[1] for x in end_pulse_vals_zoom]

        t_save_loc = save_loc + f'first_pulse_zoom_{name}.pdf' if save_loc else None
        plot.lines(first_pulse_time_zoom, first_pulse_vals_zoom, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
                   line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show, close=not show)
        t_save_loc = save_loc + f'middle_pulse_zoom_{name}.pdf' if save_loc else None
        plot.lines(middle_pulse_time_zoom, middle_pulse_vals_zoom, save_loc=t_save_loc,
                   plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show, close=not show)
        t_save_loc = save_loc + f'end_pulse_zoom_{name}.pdf' if save_loc else None
        plot.lines(end_pulse_time_zoom, end_pulse_vals_zoom, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
                   line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show, close=not show)

    for background, background_std, background_time, name in zip((background_current, background_current2, background_current3),
                                                                 (background_current_std, background_current_std2, background_current_std3),
                                                                 (background_current_time, background_current_time2, background_current_time3),
                                                                 ('', '_fit', '2'), strict=True):
        background_current_total = [background[voltage][pulse] for voltage in voltages for pulse in background[voltage]]
        background_current_total_change = [x - x[0] for x in background_current_total]
        background_current_std_total = [background_std[voltage][pulse] for voltage in voltages for pulse in background_std[voltage]]
        background_current_time_total = [background_time[voltage][pulse]/60 for voltage in voltages for pulse in background_time[voltage]]

        plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Background current [A]'}
        t_save_loc = save_loc + f'background_e{name}.pdf' if save_loc else None
        plot.errorrange(background_current_time_total, background_current_total, yerr=background_current_std_total,
                        save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                        legend_kwargs=legend_kwargs, show=show, close=not show)
        t_save_loc = save_loc + f'background{name}.pdf' if save_loc else None
        plot.lines(background_current_time_total, background_current_total,
                        save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                        legend_kwargs=legend_kwargs, show=show, close=not show)
        plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Background current change [A]'}
        t_save_loc = save_loc + f'background_change_e{name}.pdf' if save_loc else None
        plot.errorrange(background_current_time_total, background_current_total_change, yerr=background_current_std_total,
                        save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                        legend_kwargs=legend_kwargs, show=show, close=not show)
        t_save_loc = save_loc + f'background_change{name}.pdf' if save_loc else None
        plot.lines(background_current_time_total, background_current_total_change,
                        save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                        legend_kwargs=legend_kwargs, show=show, close=not show)

    h_currents = [highest_current[voltage][pulse] for voltage in voltages for pulse in highest_current[voltage]]
    l_currents = [lowest_current[voltage][pulse] for voltage in voltages for pulse in lowest_current[voltage]]
    h_currents_rel = [x - np.average(x[:10]) for x in h_currents]
    l_currents_rel = [x - np.average(x[:10]) for x in l_currents]
    h_currents_norm = [x/np.average(x[:10]) for x in h_currents]
    l_currents_norm = [x/np.average(x[:10]) for x in l_currents]
    t_currents = [time_current[voltage][pulse]/60 for voltage in voltages for pulse in time_current[voltage]]

    plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Current [A]'}
    line_kwargs_iter = linestyles.linelooks_by(color_values=pulse_vals, linestyle_values=voltage_vals, colors=width_colors)
    legend_kwargs = linestyles.legend_linelooks(line_kwargs_iter, linestyle_labels=voltage_vals, color_labels=pulse_vals,
                                                linestyle_title='H [kV]', color_title='W [us]')

    for name, currents in zip(('highest', 'lowest', 'highest_rel', 'lowest_rel', 'highest_norm', 'lowest_norm'),
                                (h_currents, l_currents, h_currents_rel, l_currents_rel, h_currents_norm, l_currents_norm)):
        t_save_loc = save_loc + f'peak_current_{name}.pdf' if save_loc else None
        plot.lines(t_currents, currents, save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                   legend_kwargs=legend_kwargs, show=show, close=not show)

    for name, time, value in zip(('', 'avg'), (power_time, power_avg_time), (power_val, power_avg_val)):
        power_total = [1000*value[voltage][pulse] for voltage in voltages for pulse in value[voltage]]
        time_total = [time[voltage][pulse] for voltage in voltages for pulse in time[voltage]]
        plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Pulse energy [mJ]'}
        t_save_loc = save_loc + f'power_{name}.pdf' if save_loc else None
        plot.lines(time_total, power_total, save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                   legend_kwargs=legend_kwargs, show=show, close=not show)

    pulse_height_vals = [np.array([pulse_height[voltage][pulse] for pulse in pulse_height[voltage]]) for voltage in voltages]
    pulse_width_vals = [np.array([pulse_width[voltage][pulse] for pulse in pulse_width[voltage]]) for voltage in voltages]
    rise_time_vals = [np.array([rise_times[voltage][pulse] for pulse in rise_times[voltage]]) for voltage in voltages]
    p_vals = [[pulse.removesuffix('us') for pulse in rise_times[voltage]] for voltage in voltages]
    voltages = [v.removesuffix('kV') for v in voltages]

    pulse_height_val = [x[:, 0] for x in pulse_height_vals]
    pulse_height_std = [x[:, 1] for x in pulse_height_vals]
    pulse_height_val_acc = [1e-3*x[:, 0] / np.array(val, dtype=float) for x, val in zip(pulse_height_vals, voltages)]
    pulse_height_std_acc = [1e-3*x[:, 1] / np.array(val, dtype=float) for x, val in zip(pulse_height_vals, voltages)]
    pulse_width_val = [1e6*x[:, 0] for x in pulse_width_vals]
    pulse_width_std = [1e6*x[:, 1] for x in pulse_width_vals]
    pulse_width_val_acc = [1e6*x[:, 0] / np.array(val, dtype=float) for x, val in zip(pulse_width_vals, p_vals)]
    pulse_width_std_acc = [1e6*x[:, 1] / np.array(val, dtype=float) for x, val in zip(pulse_width_vals, p_vals)]
    rise_time_val = [1e9*x[:, 0] for x in rise_time_vals]
    rise_time_std = [1e9*x[:, 1] for x in rise_time_vals]
    colors = ['k' for _ in range(len(pulse_height_val))]

    # line_kwargs_iter = linestyles.linelooks_by(marker_values=voltages, color_values=voltages)
    # legend_kwargs = linestyles.legend_linelooks(line_kwargs_iter, marker_labels=voltages, marker_title='H [kV]', no_linestyle=None)

    line_kwargs_iter, legend_kwargs = legend_linelooks_combines(voltages, 'H [kV]', colors=True, markers=True)

    plot_kwargs = {'xlabel': 'Pulse width [us]', 'ylabel': 'Pulse height [kV]'}
    t_save_loc = save_loc + f'pulse_height.pdf' if save_loc else None
    plot.errorbar(p_vals, pulse_height_val, yerr=pulse_height_std, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show,
                  colors=colors, close=not show, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs)
    plot_kwargs = {'xlabel': 'Pulse width [us]', 'ylabel': 'Relative height'}
    t_save_loc = save_loc + f'pulse_height_acc.pdf' if save_loc else None
    plot.errorbar(p_vals, pulse_height_val_acc, yerr=pulse_height_std_acc, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
                  colors=colors, show=show, close=not show, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs)

    plot_kwargs = {'xlabel': 'Pulse width [us]', 'ylabel': 'Pulse width [us]'}
    t_save_loc = save_loc + f'pulse_width.pdf' if save_loc else None
    plot.errorbar(p_vals, pulse_width_val, yerr=pulse_width_std, line_kwargs_iter=line_kwargs_iter, close=not show,
                  legend_kwargs=legend_kwargs, show=show, plot_kwargs=plot_kwargs, save_loc=t_save_loc)
    plot_kwargs = {'xlabel': 'Pulse width [us]', 'ylabel': 'Relative width'}
    t_save_loc = save_loc + f'pulse_width_acc.pdf' if save_loc else None
    plot.errorbar(p_vals, pulse_width_val_acc, yerr=pulse_width_std_acc, line_kwargs_iter=line_kwargs_iter, close=not show,
                  legend_kwargs=legend_kwargs, show=show, plot_kwargs=plot_kwargs, save_loc=t_save_loc)

    plot_kwargs = {'xlabel': 'Pulse width [us]', 'ylabel': 'Rise time [ns]'}
    t_save_loc = save_loc + f'rise_time.pdf' if save_loc else None
    plot.errorbar(p_vals, rise_time_val, yerr=rise_time_std, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs,
                  show=show, plot_kwargs=plot_kwargs, save_loc=t_save_loc, close=not show)

    raw_timestamp = [(raw_timestamps[voltage][pulse]-raw_timestamps[voltage][pulse][0])/60 for voltage in raw_timestamps for pulse in raw_timestamps[voltage]]
    raw_pulse_width = [1e6*raw_pulse_width[voltage][pulse] for voltage in raw_pulse_width for pulse in raw_pulse_width[voltage]]
    raw_pulse_height = [raw_pulse_height[voltage][pulse] for voltage in raw_pulse_height for pulse in raw_pulse_height[voltage]]
    raw_rise_time = [1e9*raw_rise_times[voltage][pulse] for voltage in raw_rise_times for pulse in raw_rise_times[voltage]]
    relative_raw_pulse_width = [x/float(p) for x, p in zip(raw_pulse_width, pulse_vals)]
    relative_raw_pulse_height = [1e-3*x/float(v) for x, v in zip(raw_pulse_height, voltage_vals)]

    line_kwargs_iter = linestyles.linelooks_by(color_values=pulse_vals, colors=width_colors, linestyle_values=voltage_vals)
    legend_kwargs = linestyles.legend_linelooks(line_kwargs_iter, marker_labels=voltage_vals, color_labels=pulse_vals,
                                                marker_title='H [kV]', color_title='W [us]')
    plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Relative width'}
    t_save_loc = save_loc + f'raw_pulse_width.pdf' if save_loc else None
    plot.lines(raw_timestamp, relative_raw_pulse_width, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs,
               plot_kwargs=plot_kwargs, show=show, save_loc=t_save_loc, close=not show)
    plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Relative height'}
    t_save_loc = save_loc + f'raw_pulse_height.pdf' if save_loc else None
    plot.lines(raw_timestamp, relative_raw_pulse_height, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs,
               plot_kwargs=plot_kwargs, show=show, save_loc=t_save_loc, close=not show)
    plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Rise time [ns]'}
    t_save_loc = save_loc + f'raw_rise_time.pdf' if save_loc else None
    plot.lines(raw_timestamp, raw_rise_time, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs,
               plot_kwargs=plot_kwargs, show=show, save_loc=t_save_loc, close=not show)


def emission_ranges(data_loc, voltages, pulse_lengths, labels, wavelength_ranges=((270, 450), (654, 659), (690, 860)), *,
                    save_loc1=None, save_loc2=None, save_loc3=None, save_kwargs1=None, save_kwargs2=None, save_kwargs3=None,
                    show1=False, show2=False, show3=False, peaks:tuple[tuple[float, ...], ...]=None, distance=3):
    if len(labels) != len(wavelength_ranges):
        raise ValueError("`labels` must have the same length as `wavelength_ranges`")
    intensities_total = {}
    times_total = {}
    intensities_total_avg = {}
    intensities_total_std = {}
    intensities_total_avg2 = {}
    intensities_total_std2 = {}
    intensity_change = {}
    intensity_change_std = {}
    for voltage in voltages:
        intensities_total[voltage] = {}
        intensities_total_avg[voltage] = {}
        intensities_total_std[voltage] = {}
        intensities_total_avg2[voltage] = {}
        intensities_total_std2[voltage] = {}
        intensity_change[voltage] = {}
        intensity_change_std[voltage] = {}
        times_total[voltage] = {}
        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
            if not os.path.exists(loc):
                continue

            data: OESData = read_hdf5(loc)['emission']
            data = data.remove_dead_pixels()
            is_on_kwargs = {'wavelength_range': wavelength_ranges, 'relative_threshold': 0.3}
            data = data.remove_background_interp_off(is_on_kwargs)
            is_on = data.is_on(**is_on_kwargs)
            # last_idx = len(is_on) - np.argwhere(np.diff(is_on[::-1].astype(int)) == 1)[0][0] - 1
            # is_on[last_idx] = False
            intensities = []
            if peaks is not None:
                for peak in peaks:
                    _, _, intens = data.peak_loc_intensities_with_time(peak, is_on_kwargs=is_on_kwargs, distance=distance)
                    mask = intens > 0
                    intensities.append(np.nanmean(intens, axis=0, where=mask))
            else:
                for wavelength_range in wavelength_ranges:
                    if isinstance(wavelength_range[0], tuple):
                        new_data = data.spectrum.wavelength_ranges(*wavelength_range)
                    else:
                        new_data = data.spectrum.wavelength_ranges(wavelength_range)
                    intensity = new_data.intensities[is_on]
                    intensities.append(np.average(intensity, axis=1))
                # mask = (data.spectrum.wavelengths > wavelength_range[0]) & (data.spectrum.wavelengths < wavelength_range[1])
                # intensity = data.spectrum.intensities[is_on][:, mask]
                # intensities.append(np.average(intensity, axis=1))

            intensities = np.array(intensities)
            intensities_total[voltage][pulse] = intensities
            times_total[voltage][pulse] = (data.spectrum.times[is_on] - data.spectrum.times[is_on][0])/60
            intensities_total_avg[voltage][pulse] = np.average(intensities, axis=1)
            intensities_total_std[voltage][pulse] = np.std(intensities, axis=1)
            intensities_total_avg2[voltage][pulse] = np.average(intensities[:, -30:-5], axis=1)
            intensities_total_std2[voltage][pulse] = np.std(intensities[:, -30:-5], axis=1)
            start_inten = np.average(intensities[:, 5:30], axis=1)
            end_inten = np.average(intensities[:, -30:-5], axis=1)
            start_std = np.std(intensities[:, 5:30], axis=1)
            end_std = np.std(intensities[:, -30:-5], axis=1)
            intensity_change[voltage][pulse] = (end_inten - start_inten)/start_inten
            intensity_change_std[voltage][pulse] = abs(intensity_change[voltage][pulse])*np.sqrt((end_std/end_inten)**2 + (start_std/start_inten)**2)

            line_kwargs = linestyles.linelook_by(labels, colors=True, linestyles=True)
            plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Time [min]'}
            # labels = [f'{wavelength_range[0]}-{wavelength_range[1]}' for wavelength_range in wavelength_ranges]

            t_save_loc = f'{save_loc3}{voltage}_{pulse}.pdf' if save_loc3 else None
            plot.lines(times_total[voltage][pulse], intensities_total[voltage][pulse], line_kwargs_iter=line_kwargs,
                       plot_kwargs=plot_kwargs, save_loc=t_save_loc, labels=labels, show=show3, close=not show3,
                       save_kwargs=save_kwargs3)
            t_save_loc = f'{save_loc3}{voltage}_{pulse}_norm.pdf' if save_loc3 else None
            values = intensities_total[voltage][pulse]/np.max(intensities_total[voltage][pulse], axis=1)[:, None]
            plot.lines(times_total[voltage][pulse], values, line_kwargs_iter=line_kwargs, save_kwargs=save_kwargs3,
                       plot_kwargs=plot_kwargs, save_loc=t_save_loc, labels=labels, show=show3, close=not show3)

        pulse_vals = list(intensities_total[voltage].keys())
        pulses = np.repeat(list(intensities_total[voltage].keys()), len(wavelength_ranges))
        pulses = [pulse.removesuffix('us') for pulse in pulses]
        wav_ranges = np.tile([f'{wavelength_range[0]}-{wavelength_range[1]}' for wavelength_range in wavelength_ranges], len(pulse_vals))
        intensities = [x for values in intensities_total[voltage].values() for x in values]
        intensities_norm = [intensity/max(intensity) for intensity in intensities]
        times = [[x for x in values] for values in times_total[voltage].values() for _ in range(len(wavelength_ranges))]

        tiled_labels = np.tile(labels, len(pulse_vals))
        line_kwargs_iter = linelooks_by(color_values=pulses, linestyle_values=tiled_labels, colors=width_colors, linestyles=species_style)
        legend_kwargs = legend_linelooks(line_kwargs_iter, color_labels=pulses, linestyle_labels=tiled_labels,
                                         color_title='W [us]', linestyle_title='Species')

        mask = [np.average(val) > 50 for val in intensities]
        intensities_norm_ma = [val for val, ma in zip(intensities_norm, mask) if ma]
        times_mask = [val for val, ma in zip(times, mask) if ma]
        line_kwargs_iter_ma = [val for val, ma in zip(line_kwargs_iter, mask) if ma]

        plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Time [min]'}
        t_save_loc = f'{save_loc2}{voltage}.pdf' if save_loc2 else None
        plot.lines(times, intensities, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm.pdf' if save_loc2 else None
        plot.lines(times, intensities_norm, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs,
                   save_loc=t_save_loc, legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm_masked.pdf' if save_loc2 else None
        plot.lines(times_mask, intensities_norm_ma, line_kwargs_iter=line_kwargs_iter_ma, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)

        t_save_loc = f'{save_loc2}{voltage}_l.pdf' if save_loc2 else None
        plot.lines(times, intensities, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm_l.pdf' if save_loc2 else None
        plot.lines(times, intensities_norm, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs,
                   save_loc=t_save_loc, show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm_masked_l.pdf' if save_loc2 else None
        plot.lines(times_mask, intensities_norm_ma, line_kwargs_iter=line_kwargs_iter_ma, plot_kwargs=plot_kwargs,
                   save_loc=t_save_loc, show=show2, close=not show2, save_kwargs=save_kwargs2)

        plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Time [min]'}
        times_avg = [npf.block_average(np.array(t), 10) for t in times]
        times_avg_ma = [npf.block_average(np.array(t), 10) for t in times_mask]
        intensities_avg = [npf.block_average(np.array(intensity), 10) for intensity in intensities]
        intensities_norm_avg = [npf.block_average(np.array(t), 10) for t in intensities_norm]
        intensities_norm_avg_ma = [npf.block_average(np.array(t), 10) for t in intensities_norm_ma]

        t_save_loc = f'{save_loc2}{voltage}_avg.pdf' if save_loc2 else None
        plot.lines(times_avg, intensities_avg, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm_avg.pdf' if save_loc2 else None
        plot.lines(times_avg, intensities_norm_avg, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm_avg_masked.pdf' if save_loc2 else None
        plot.lines(times_avg_ma, intensities_norm_avg_ma, line_kwargs_iter=line_kwargs_iter_ma, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)

        t_save_loc = f'{save_loc2}{voltage}_avg_l.pdf' if save_loc2 else None
        plot.lines(times_avg, intensities_avg, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm_avg_l.pdf' if save_loc2 else None
        plot.lines(times_avg, intensities_norm_avg, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs,
                   save_loc=t_save_loc, show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm_avg_masked_l.pdf' if save_loc2 else None
        plot.lines(times_avg_ma, intensities_norm_avg_ma, line_kwargs_iter=line_kwargs_iter_ma, plot_kwargs=plot_kwargs,
                   save_loc=t_save_loc, show=show2, close=not show2, save_kwargs=save_kwargs2)

    p_vals = [pulse.removesuffix('us') for pulse in pulse_lengths]
    _, legend_kwargs = legend_linelooks_by(color_labels=p_vals, linestyle_labels=labels, color_title='W [us]', linestyle_values=species_style,
                                           linestyle_title='Species', sort=False, color_values=width_colors)
    plot.export_legend(legend_kwargs, save_loc=f'{save_loc2}legend.pdf')

    intensities_avg = [[intensities_total_avg[voltage][pulse] for pulse in intensities_total_avg[voltage]] for voltage in voltages]
    intensities_std = [[intensities_total_std[voltage][pulse] for pulse in intensities_total_std[voltage]] for voltage in voltages]
    intensities_avg_t = [list(map(list, zip(*values))) for values in intensities_avg]
    intensities_std_t = [list(map(list, zip(*values))) for values in intensities_std]

    intensities_norm = [[np.array(intensity)/max(intensity) for intensity in intensities] for intensities in intensities_avg_t]
    intensities_norm_std = [[np.array(s)/max(v) for v, s in zip(val, std)] for val, std in zip(intensities_avg_t, intensities_std_t)]

    intensities_avg2 = [[intensities_total_avg2[voltage][pulse] for pulse in intensities_total_avg2[voltage]] for voltage in
                       voltages]
    intensities_std2 = [[intensities_total_std2[voltage][pulse] for pulse in intensities_total_std2[voltage]] for voltage in
                       voltages]
    intensities_avg_t2 = [list(map(list, zip(*values))) for values in intensities_avg2]
    intensities_std_t2 = [list(map(list, zip(*values))) for values in intensities_std2]

    intensities_norm2 = [[np.array(intensity) / max(intensity) for intensity in intensities] for intensities in intensities_avg_t2]
    intensities_norm_std2 = [[np.array(s) / max(v) for v, s in zip(val, std)] for val, std in
                            zip(intensities_avg_t2, intensities_std_t2)]

    ratio = [[intensity_change[voltage][pulse] for pulse in intensity_change[voltage]] for voltage in voltages]
    ratio_std = [[intensity_change_std[voltage][pulse] for pulse in intensity_change_std[voltage]] for voltage in voltages]
    ratio_t = [list(map(list, zip(*values))) for values in ratio]
    ratio_std_t = [list(map(list, zip(*values))) for values in ratio_std]

    pulses = [[pulse.removesuffix('us') for pulse in intensities_total_avg[voltage].keys()] for voltage in voltages]

    voltages = [v.removesuffix('kV') for v in voltages]
    plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Pulse length [us]'}
    line_styles = linestyles.linelooks_by(linestyle_values=voltages, linestyles=True)
    line_styles = [l['linestyle'] for l in line_styles]

    colors = linestyles.linelooks_by(color_values=labels, colors=True)
    colors = [c['color'] for c in colors]

    _, legend_kwargs = legend_linelooks_by(color_labels=labels, linestyle_labels=voltages, color_title='Emission', linestyle_title='H [kV]',
                                        color_values=colors, linestyle_values=line_styles, sort=False)

    present_pulses = sorted(list(set((p for pulse in pulses for p in pulse))), key=lambda x: float(x))
    mask = [[intensities_total_avg[voltage][pulse] > 50 for pulse in intensities_total_avg[voltage]] for voltage in intensities_total_avg]
    mask_t = [list(map(list, zip(*values))) for values in mask]

    def apply_mask_2D(value, mask):
        return [[val for val, ma in zip(v, m, strict=True) if ma] for v, m in zip(value, mask, strict=True)]

    for name, data_vals, data_std, ylabel, lims in zip(
            ('total', 'total_norm', 'total_norm2', 'total_ratio'),
            (intensities_avg_t, intensities_norm, intensities_norm2, ratio_t),
            (intensities_std_t, intensities_norm_std, intensities_norm_std2, ratio_std_t),
            ['Intensity [A.U.]'] + ['Relative intensity [A.U.]']*2 + ['Relative intensity change [A.U.]'],
            [(None, None)] + [(0, 1.1)]*2 + [(None, None)], strict=True):
        t_save_loc = f'{save_loc1}{name}.pdf' if save_loc1 else None
        plot1 = plot.lines(present_pulses, np.full(len(present_pulses), np.nan), show=False, close=False)
        plot2 = plot.lines(present_pulses, np.full(len(present_pulses), np.nan), show=False, close=False)
        for pulse, value, value_std, linestyle, m in zip(pulses, data_vals, data_std, line_styles, mask_t, strict=True):
            line_kwargs = {'linestyle': linestyle}
            plot1 = plot.errorbar(pulse, value, yerr=value_std, colors=colors, close=False, show=False, fig_ax=plot1,
                                  line_kwargs=line_kwargs)
            pulses_vals = [pulse.copy() for _ in range(len(value))]
            plot2 = plot.errorbar(apply_mask_2D(pulses_vals, m), apply_mask_2D(value, m), yerr=apply_mask_2D(value_std, m),
                                  colors=colors, close=False, show=False, fig_ax=plot2, line_kwargs=line_kwargs)
        plot_kwargs['ylabel'] = ylabel
        plot_kwargs['ylim'] = lims
        plot.lines([], [], plot_kwargs=plot_kwargs, save_loc=t_save_loc.replace('.pdf', '_l.pdf'),
                   save_kwargs=save_kwargs1, show=False, close=False, fig_ax=plot1)
        plot.lines([], [], plot_kwargs=plot_kwargs, save_loc=t_save_loc.replace('.pdf', '_masked_l.pdf'),
                   save_kwargs=save_kwargs1, show=False, close=False, fig_ax=plot2)
        plot.lines([], [], plot_kwargs=plot_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs1, show=show1,
                   close=not show1, fig_ax=plot1, legend_kwargs=legend_kwargs)
        plot.lines([], [], plot_kwargs=plot_kwargs, save_loc=t_save_loc.replace('.pdf', '_masked.pdf'),
                   save_kwargs=save_kwargs1, show=show1, close=not show1, fig_ax=plot2, legend_kwargs=legend_kwargs)
    plot.export_legend(legend_kwargs, save_loc=f'{save_loc1}legend.pdf' if save_loc1 else None)

    values = [[[intensities_total_avg[voltage][pulse][x] for pulse in intensity_change[voltage]] for voltage in intensity_change] for x in range(len(wavelength_ranges))]
    stds = [[[intensities_total_std[voltage][pulse][x] for pulse in intensity_change[voltage]] for voltage in intensity_change] for x in range(len(wavelength_ranges))]
    voltages = [[[voltage for _ in intensity_change[voltage]] for voltage in intensity_change] for _ in range(len(wavelength_ranges))]
    pulses = [[[pulse.removesuffix('us') for pulse in intensity_change[voltage]] for voltage in intensity_change] for _ in range(len(wavelength_ranges))]

    def extract(data, along):
        indexer = {}
        out = []
        for al, dat in zip(along, data):
            for i, (a, d) in enumerate(zip(al, dat)):
                if a not in indexer:
                    indexer[a] = len(indexer)
                    out.append(list())
                out[indexer[a]].append(d)
        return out

    wavs = [labels[i] for i, (v, p) in enumerate(zip(values, pulses)) for _ in extract(v, p)]
    pulses_t = [x[0] for v in pulses for x in extract(v, v)]
    values_t = [x for v, p in zip(values, pulses) for x in extract(v, p)]
    values_t_norm = [np.array(x)/max(x) for x in values_t]
    stds_t = [x for v, p in zip(stds, pulses) for x in extract(v, p)]
    stds_t_norm = [np.array(x)/max(v) for v, x in zip(values_t, stds_t)]
    voltages_t = [[y.removesuffix('kV') for y in x] for v, p in zip(voltages, pulses) for x in extract(v, p)]

    line_kwargs_iter = linelooks_by(color_values=pulses_t, linestyle_values=wavs, colors=width_colors, linestyles=True)
    legend_kwargs = legend_linelooks(line_kwargs_iter, color_labels=pulses_t, linestyle_labels=wavs, color_title='W [us]', linestyle_title='Species')

    plot_kwargs = {'xlabel': 'Pulse height [kV]', 'ylabel': 'Relative intensity [A.U.]'}
    t_save_loc = f'{save_loc1}emission_with_voltage.pdf' if save_loc1 else None
    plot.lines(voltages_t, values_t_norm, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs,
               save_loc=t_save_loc, show=show1, close=not show1, save_kwargs=save_kwargs1)
    t_save_loc = f'{save_loc1}emission_with_voltage_l.pdf' if save_loc1 else None
    plot.lines(voltages_t, values_t_norm, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs,
               save_loc=t_save_loc, show=False, close=True, save_kwargs=save_kwargs1)
    t_save_loc = f'{save_loc1}emission_with_voltage_legend.pdf' if save_loc1 else None
    plot.export_legend(legend_kwargs, save_loc=t_save_loc)



def emission_peaks(data_loc, voltages, pulse_lengths, wavelength_peaks={'Ha': 656.3, 'Ar': 763.5, 'OH': 309, 'N2': 337, 'O': 777.5}, *,
                   save_loc1=None, save_loc2=None, save_loc3=None, save_kwargs1=None, save_kwargs2=None, save_kwargs3=None,
                   show1=True, show2=True, show3=False):
    labels, peak_wavelengths = zip(*wavelength_peaks.items())

    intensities_total = {}
    times_total = {}
    intensities_total_avg = {}
    intensities_total_std = {}
    intensities_total_avg2 = {}
    intensities_total_std2 = {}
    intensity_change = {}
    intensity_change_std = {}
    for voltage in voltages:
        intensities_total[voltage] = {}
        intensities_total_avg[voltage] = {}
        intensities_total_std[voltage] = {}
        intensities_total_avg2[voltage] = {}
        intensities_total_std2[voltage] = {}
        intensity_change[voltage] = {}
        intensity_change_std[voltage] = {}
        times_total[voltage] = {}
        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
            if not os.path.exists(loc):
                continue

            data: OESData = read_hdf5(loc)['emission']
            data = data.remove_dead_pixels()
            is_on_kwargs = {'wavelength_range': ((270, 450), (654, 659), (690, 860)), 'relative_threshold': 0.3}
            data = data.remove_background_interp_off(is_on_kwargs)
            is_on = data.is_on(**is_on_kwargs)

            time, locs, intensities = data.peak_loc_intensities_with_time(peak_wavelengths, is_on_kwargs=is_on_kwargs)

            intensities_total[voltage][pulse] = np.array(intensities)
            times_total[voltage][pulse] = (data.spectrum.times[is_on] - data.spectrum.times[is_on][0])/60
            intensities_total_avg[voltage][pulse] = np.average(intensities, axis=1)
            intensities_total_std[voltage][pulse] = np.std(intensities, axis=1)
            intensities_total_avg2[voltage][pulse] = np.average(intensities[:, -30:-5], axis=1)
            intensities_total_std2[voltage][pulse] = np.std(intensities[:, -30:-5], axis=1)
            start_inten = np.average(intensities[:, 5:30], axis=1)
            end_inten = np.average(intensities[:, -30:-5], axis=1)
            start_std = np.std(intensities[:, 5:30], axis=1)
            end_std = np.std(intensities[:, -30:-5], axis=1)
            intensity_change[voltage][pulse] = (end_inten - start_inten)/start_inten
            intensity_change_std[voltage][pulse] = abs(intensity_change[voltage][pulse])*np.sqrt((end_std/end_inten)**2 + (start_std/start_inten)**2)

            line_kwargs = linestyles.linelook_by(peak_wavelengths, colors=True, linestyles=True)
            plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Time [min]'}

            t_save_loc = f'{save_loc3}{voltage}_{pulse}.pdf' if save_loc3 else None
            plot.lines(times_total[voltage][pulse], intensities_total[voltage][pulse], line_kwargs_iter=line_kwargs,
                       plot_kwargs=plot_kwargs, save_loc=t_save_loc, labels=labels, show=show3, close=not show3,
                       save_kwargs=save_kwargs3)
            t_save_loc = f'{save_loc3}{voltage}_{pulse}_norm.pdf' if save_loc3 else None
            values = intensities_total[voltage][pulse]/np.nanmax(intensities_total[voltage][pulse], axis=1)[:, None]
            plot.lines(times_total[voltage][pulse], values, line_kwargs_iter=line_kwargs, save_kwargs=save_kwargs3,
                       plot_kwargs=plot_kwargs, save_loc=t_save_loc, labels=labels, show=show3, close=not show3)

        pulse_vals = list(intensities_total[voltage].keys())
        pulses = np.repeat(list(intensities_total[voltage].keys()), len(peak_wavelengths))
        label_vals = np.tile(labels, len(pulse_vals))
        intensities = [x for values in intensities_total[voltage].values() for x in values]
        intensities_norm = [intensity/np.nanmax(intensity) for intensity in intensities]

        max_vals = [np.nanargmax([np.nanmax(x) for x in values]) for values in intensities_total[voltage].values()]
        intens = [np.array([values[max_vals[index]]]) for index, values in enumerate(intensities_total[voltage].values())]
        intensities_norm2 = [(x/intens[index])[0] for index, values in enumerate(intensities_total[voltage].values()) for x in values]
        intensities_norm3 = [x/np.nanmax(x) for x in intensities_norm2]
        times = [[x for x in values] for values in times_total[voltage].values() for _ in range(len(peak_wavelengths))]

        line_kwargs_iter = linelooks_by(color_values=pulses, linestyle_values=label_vals, colors=width_colors)
        legend_kwargs = legend_linelooks(line_kwargs_iter, color_labels=pulses, linestyle_labels=label_vals)

        plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Time [min]'}
        t_save_loc = f'{save_loc2}{voltage}.pdf' if save_loc2 else None
        plot.lines(times, intensities, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm.pdf' if save_loc2 else None
        plot.lines(times, intensities_norm, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm2.pdf' if save_loc2 else None
        plot.lines(times, intensities_norm2, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm3.pdf' if save_loc2 else None
        plot.lines(times, intensities_norm3, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)

    # intensities_avg = [intensities_total_avg[voltage][pulse] for voltage in voltages for pulse in intensities_total_avg[voltage]]
    # intensities_std = [intensities_total_std[voltage][pulse] for voltage in voltages for pulse in intensities_total_std[voltage]]
    # intensities_avg_t = list(map(list, zip(*intensities_avg)))
    # intensities_std_t = list(map(list, zip(*intensities_std)))

    intensities_avg = [[intensities_total_avg[voltage][pulse] for pulse in intensities_total_avg[voltage]] for voltage in voltages]
    intensities_std = [[intensities_total_std[voltage][pulse] for pulse in intensities_total_std[voltage]] for voltage in voltages]
    intensities_avg_t = [list(map(list, zip(*values))) for values in intensities_avg]
    intensities_std_t = [list(map(list, zip(*values))) for values in intensities_std]

    intensities_norm = [[np.array(intensity)/np.nanmax(intensity) for intensity in intensities] for intensities in intensities_avg_t]
    intensities_norm_std = [[np.array(s)/np.nanmax(v) for v, s in zip(val, std)] for val, std in zip(intensities_avg_t, intensities_std_t)]

    intensities_avg2 = [[intensities_total_avg2[voltage][pulse] for pulse in intensities_total_avg2[voltage]] for voltage in
                       voltages]
    intensities_std2 = [[intensities_total_std2[voltage][pulse] for pulse in intensities_total_std2[voltage]] for voltage in
                       voltages]
    intensities_avg_t2 = [list(map(list, zip(*values))) for values in intensities_avg2]
    intensities_std_t2 = [list(map(list, zip(*values))) for values in intensities_std2]

    intensities_norm2 = [[np.array(intensity) / max(intensity) for intensity in intensities] for intensities in intensities_avg_t2]
    intensities_norm_std2 = [[np.array(s) / max(v) for v, s in zip(val, std)] for val, std in
                            zip(intensities_avg_t2, intensities_std_t2)]

    ratio = [[intensity_change[voltage][pulse] for pulse in intensity_change[voltage]] for voltage in voltages]
    ratio_std = [[intensity_change_std[voltage][pulse] for pulse in intensity_change_std[voltage]] for voltage in voltages]
    ratio_t = [list(map(list, zip(*values))) for values in ratio]
    ratio_std_t = [list(map(list, zip(*values))) for values in ratio_std]

    # pulses = [pulse for voltage in voltages for pulse in intensities_total_avg[voltage].keys()]
    pulses = [[pulse.removesuffix('us') for pulse in intensities_total_avg[voltage].keys()] for voltage in voltages]

    plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Pulse length [us]'}
    line_styles = linestyles.linelook_by(voltages, linestyles=True)
    line_styles = [l['linestyle'] for l in line_styles]

    colors = linestyles.linelook_by(labels, colors=width_colors)
    colors = [c['color'] for c in colors]

    _, legend_kwargs = legend_linelooks_by(color_labels=labels, linestyle_labels=voltages, color_values=colors,
                                           linestyle_values=line_styles, sort=False)

    print(pulse_lengths)

    t_save_loc = f'{save_loc1}total.pdf' if save_loc1 else None
    plot1 = plot.lines(pulse_lengths, [None for _ in pulse_lengths], show=False, close=False)
    for pulse, value, value_std, linestyle in zip(pulses, intensities_avg_t, intensities_std_t, line_styles, strict=True):
        line_kwargs = {'linestyle': linestyle}
        plot1 = plot.errorbar(pulse, value, yerr=value_std, colors=colors, close=False, show=False, fig_ax=plot1, line_kwargs=line_kwargs)
    plot.lines([], [], plot_kwargs=plot_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs1, show=show1,
               close=not show1, fig_ax=plot1, legend_kwargs=legend_kwargs)

    t_save_loc = f'{save_loc1}total_norm.pdf' if save_loc1 else None
    plot2 = plot.lines(pulse_lengths, np.full(len(pulse_lengths), np.nan), show=False, close=False)
    for pulse, value, value_std, linestyle in zip(pulses, intensities_norm, intensities_norm_std, line_styles, strict=True):
        line_kwargs = {'linestyle': linestyle}
        plot2 = plot.errorbar(pulse, value, yerr=value_std, colors=colors, close=False, show=False, fig_ax=plot2, line_kwargs=line_kwargs)
    plot.lines([], [], plot_kwargs=plot_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs1, show=show1,
               close=not show1, fig_ax=plot2, legend_kwargs=legend_kwargs)

    t_save_loc = f'{save_loc1}total_norm2.pdf' if save_loc1 else None
    plot3 = plot.lines(pulse_lengths, np.full(len(pulse_lengths), np.nan), show=False, close=False)
    for pulse, value, value_std, linestyle in zip(pulses, intensities_norm2, intensities_norm_std2, line_styles, strict=True):
        line_kwargs = {'linestyle': linestyle}
        plot3 = plot.errorbar(pulse, value, yerr=value_std, colors=colors, close=False, show=False, fig_ax=plot3,
                              line_kwargs=line_kwargs)
    plot.lines([], [], plot_kwargs=plot_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs1, show=show1,
               close=not show1, fig_ax=plot3, legend_kwargs=legend_kwargs)

    plot_kwargs['ylabel'] = 'Relative intensity change'
    t_save_loc = f'{save_loc1}total_ratio.pdf' if save_loc1 else None
    plot3 = plot.lines(pulse_lengths, np.full(len(pulse_lengths), np.nan), show=False, close=False)
    for pulse, value, value_std, linestyle in zip(pulses, ratio_t, ratio_std_t, line_styles, strict=True):
        line_kwargs = {'linestyle': linestyle}
        plot3 = plot.errorbar(pulse, value, yerr=value_std, colors=colors, close=False, show=False, fig_ax=plot3,
                              line_kwargs=line_kwargs)
    plot.lines([], [], plot_kwargs=plot_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs1, show=show1,
               close=not show1, fig_ax=plot3, legend_kwargs=legend_kwargs)


def analyse_directory_nitrogen_oh_emission(data_loc, voltages, pulse_lengths, interp_loc_N2=None, interp_loc_OH=None,
                                           *, wav_range=(270, 450), save_loc=None, save_loc2=None, save_loc3=None, save_kwargs=None,
                                           show1=True, show2=False, save_kwargs2=None, save_kwargs3=None):
    fwhm = 0.5
    peak_wavs = np.linspace(-3, 3, 100)
    peak = scipy.stats.norm.pdf(peak_wavs, 0, fwhm)
    sample: OESData = read_hdf5(f'{data_loc}_{voltages[0]}_{pulse_lengths[0]}.hdf5')['emission']
    wav = sample.spectrum.wavelengths
    mask = (wav > wav_range[0]) & (wav < wav_range[1])
    if interp_loc_N2 and interp_loc_OH:
        spectrum_fitter_N2 = N2SpecAirSimulations.from_hdf5(interp_loc_N2, Spectrum(peak_wavs, peak), wav[mask])
        spectrum_fitter_OH = SpecAirSimulations.from_hdf5(interp_loc_OH, Spectrum(peak_wavs, peak), wav[mask])
        model = spectrum_fitter_N2.model(prefix='N2_') + spectrum_fitter_OH.model(prefix='OH_')
    elif interp_loc_N2:
        spectrum_fitter_N2 = N2SpecAirSimulations.from_hdf5(interp_loc_N2, Spectrum(peak_wavs, peak), wav[mask])
        model = spectrum_fitter_N2.model(prefix='N2_')
    elif interp_loc_OH:
        spectrum_fitter_OH = SpecAirSimulations.from_hdf5(interp_loc_OH, Spectrum(peak_wavs, peak), wav[mask])
        model = spectrum_fitter_OH.model(prefix='OH_')
    else:
        raise ValueError('No interpolation location given')

    params = model.make_params()

    if interp_loc_N2:
        N2_total_vib = {}
        N2_total_rot = {}
        N2_total_vib_std = {}
        N2_total_rot_std = {}

    if interp_loc_OH:
        OH_total_vib = {}
        OH_total_rot = {}
        OH_total_vib_std = {}
        OH_total_rot_std = {}

    for voltage in voltages:
        if interp_loc_N2:
            N2_total_vib[voltage] = {}
            N2_total_rot[voltage] = {}
            N2_total_vib_std[voltage] = {}
            N2_total_rot_std[voltage] = {}
        if interp_loc_OH:
            OH_total_vib[voltage] = {}
            OH_total_rot[voltage] = {}
            OH_total_vib_std[voltage] = {}
            OH_total_rot_std[voltage] = {}

        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
            data: OESData = read_hdf5(loc)['emission']
            data = data.remove_dead_pixels()
            data = data.remove_background_interp_off({'wavelength_range': (300, 400),
                                                      'relative_threshold': 0.25})
            intensity = data.spectrum.intensities[:, mask]
            results = []
            on_mask = data.is_on(wavelength_range=(300, 400), relative_threshold=0.25)
            for index, inten in enumerate(intensity[on_mask]):
                result = model.fit(inten, params=params)
                results.append(result.params)
                if save_loc3:
                    line_kwargs_iter = [{'linestyle': '-'}, {'linestyle': '--'}]
                    t_save_loc = rf'{save_loc3}_{voltage}_{pulse}_{index}.pdf'
                    plot.lines(wav[mask], [inten, result.best_fit], labels=['measured', 'fit'], show=False, close=True,
                               save_loc=t_save_loc, save_kwargs=save_kwargs3, line_kwargs_iter=line_kwargs_iter)

            if interp_loc_N2:
                N2_vib = [result['N2_vib_energy'].value for result in results]
                N2_vib_std = [(result['N2_vib_energy'].stderr if result['N2_vib_energy'].stderr else 1000) for result in results]
                N2_rot = [result['N2_rot_energy'].value for result in results]
                N2_rot_std = [(result['N2_rot_energy'].stderr if result['N2_rot_energy'].stderr else 1000) for result in results]

            if interp_loc_OH:
                OH_vib = [result['OH_vib_energy'].value for result in results]
                OH_vib_std = [(result['OH_vib_energy'].stderr if result['OH_vib_energy'].stderr else 1000) for result in results]
                OH_rot = [result['OH_rot_energy'].value for result in results]
                OH_rot_std = [(result['OH_rot_energy'].stderr if result['OH_rot_energy'].stderr else 1000) for result in results]

            time_vals = data.spectrum.times[on_mask]
            plot_kwargs = {'ylabel': 'T$_{vib}$ [K]', 'xlabel': 'Time [min]'}
            plot_kwargs2 = {'ylabel': 'T$_{rot}$ [K]', 'xlabel': 'Time [min]'}

            if interp_loc_N2:
                t_save_loc = rf'{save_loc2}_{voltage}_{pulse}_N2_vib.pdf' if save_loc else None
                plot.errorrange(time_vals, N2_vib, yerr=N2_vib_std, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs)

                t_save_loc = rf'{save_loc2}_{voltage}_{pulse}_N2_rot.pdf' if save_loc else None
                plot.errorrange(time_vals, N2_rot, yerr=N2_rot_std, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs2)

            if interp_loc_OH:
                t_save_loc = rf'{save_loc2}_{voltage}_{pulse}_OH_vib.pdf' if save_loc else None
                plot.errorrange(time_vals, OH_vib, yerr=OH_vib_std, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs)

                t_save_loc = rf'{save_loc2}_{voltage}_{pulse}_OH_rot.pdf' if save_loc else None
                plot.errorrange(time_vals, OH_rot, yerr=OH_rot_std, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs2)

            if interp_loc_N2:
                N2_total_vib[voltage][pulse] = np.average(N2_vib, weights=1/np.array(N2_vib_std)**2)
                N2_total_rot[voltage][pulse] = np.average(N2_rot, weights=1/np.array(N2_rot_std)**2)
                N2_total_vib_std[voltage][pulse] = np.std(N2_total_rot[voltage][pulse])
                N2_total_rot_std[voltage][pulse] = np.std(N2_total_vib[voltage][pulse])

            if interp_loc_OH:
                OH_total_vib[voltage][pulse] = np.average(OH_vib, weights=1/np.array(OH_vib_std)**2)
                OH_total_rot[voltage][pulse] = np.average(OH_rot, weights=1/np.array(OH_rot_std)**2)
                OH_total_vib_std[voltage][pulse] = np.std(OH_total_rot[voltage][pulse])
                OH_total_rot_std[voltage][pulse] = np.std(OH_total_vib[voltage][pulse])

        pulses = np.array(list(N2_total_vib[voltage].keys()))

        plot_kwargs = {'ylabel': 'T$_{vib}$ [K]', 'xlabel': 'Pulse length [us]'}
        plot_kwargs2 = {'ylabel': 'T$_{rot}$ [K]', 'xlabel': 'Pulse length [us]'}

        if interp_loc_N2:
            N2_vib = np.array([N2_total_vib[voltage][pulse] for pulse in pulses])
            N2_rot = np.array([N2_total_rot[voltage][pulse] for pulse in pulses])
            N2_vib_std = np.array([N2_total_vib_std[voltage][pulse] for pulse in pulses])
            N2_rot_std = np.array([N2_total_rot_std[voltage][pulse] for pulse in pulses])

            t_save_loc = rf'{save_loc2}_{voltage}_N2_vib.pdf' if save_loc else None
            plot.errorrange(pulses, N2_vib, yerr=N2_vib_std, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs,
                            show=show2, close=not show2)
            t_save_loc = rf'{save_loc2}_{voltage}_N2_rot.pdf' if save_loc else None
            plot.errorrange(pulses, N2_rot, yerr=N2_rot_std, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs2,
                            show=show2, close=not show2)

        if interp_loc_OH:
            OH_vib = np.array([OH_total_vib[voltage][pulse] for pulse in pulses])
            OH_rot = np.array([OH_total_rot[voltage][pulse] for pulse in pulses])
            OH_vib_std = np.array([OH_total_vib_std[voltage][pulse] for pulse in pulses])
            OH_rot_std = np.array([OH_total_rot_std[voltage][pulse] for pulse in pulses])

            t_save_loc = rf'{save_loc2}_{voltage}_OH_vib.pdf' if save_loc else None
            plot.errorrange(pulses, OH_vib, yerr=OH_vib_std, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs,
                            show=show2, close=not show2)
            t_save_loc = rf'{save_loc2}_{voltage}_OH_rot.pdf' if save_loc else None
            plot.errorrange(pulses, OH_rot, yerr=OH_rot_std, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs2,
                            show=show2, close=not show2)

    pulses_total = [[p.removesuffix('us') for p in N2_total_vib[voltage].keys()] for voltage in N2_total_vib.keys()]
    plot_kwargs = {'ylabel': 'T$_{vib}$ [K]', 'xlabel': 'Pulse length [us]'}
    plot_kwargs2 = {'ylabel': 'T$_{rot}$ [K]', 'xlabel': 'Pulse length [us]'}

    line_kwargs_iter = linestyles.linelook_by(voltages, linestyles=True)

    if interp_loc_N2:
        N2_total_vib_l = [[N2_total_vib[voltage][pulse] for pulse in N2_total_vib[voltage].keys()] for voltage in N2_total_vib.keys()]
        N2_total_rot_l = [[N2_total_rot[voltage][pulse] for pulse in N2_total_rot[voltage].keys()] for voltage in N2_total_rot.keys()]
        N2_total_vib_std_l = [[N2_total_vib_std[voltage][pulse] for pulse in N2_total_vib_std[voltage].keys()] for voltage in N2_total_vib_std.keys()]
        N2_total_rot_std_l = [[N2_total_rot_std[voltage][pulse] for pulse in N2_total_rot_std[voltage].keys()] for voltage in N2_total_rot_std.keys()]

        t_save_loc = rf'{save_loc}_N2_vib.pdf' if save_loc else None
        plot.errorrange(pulses_total, N2_total_vib_l, yerr=N2_total_vib_std_l, save_loc=t_save_loc, save_kwargs=save_kwargs, plot_kwargs=plot_kwargs,
                        show=show1, close=not show1, line_kwargs_iter=line_kwargs_iter)
        t_save_loc = rf'{save_loc}_N2_rot.pdf' if save_loc else None
        plot.errorrange(pulses_total, N2_total_rot_l, yerr=N2_total_rot_std_l, save_loc=t_save_loc, save_kwargs=save_kwargs, plot_kwargs=plot_kwargs2,
                        show=show1, close=not show1, line_kwargs_iter=line_kwargs_iter)

    if interp_loc_OH:
        OH_total_vib_l = [[OH_total_vib[voltage][pulse] for pulse in OH_total_vib[voltage].keys()] for voltage in OH_total_vib.keys()]
        OH_total_rot_l = [[OH_total_rot[voltage][pulse] for pulse in OH_total_rot[voltage].keys()] for voltage in OH_total_rot.keys()]
        OH_total_vib_std_l = [[OH_total_vib_std[voltage][pulse] for pulse in OH_total_vib_std[voltage].keys()] for voltage in OH_total_vib_std.keys()]
        OH_total_rot_std_l = [[OH_total_rot_std[voltage][pulse] for pulse in OH_total_rot_std[voltage].keys()] for voltage in OH_total_rot_std.keys()]

        t_save_loc = rf'{save_loc}_OH_vib.pdf' if save_loc else None
        plot.errorrange(pulses_total, OH_total_vib_l, yerr=OH_total_vib_std_l, save_loc=t_save_loc, save_kwargs=save_kwargs, plot_kwargs=plot_kwargs,
                        show=show1, close=not show1, line_kwargs_iter=line_kwargs_iter)
        t_save_loc = rf'{save_loc}_OH_rot.pdf' if save_loc else None
        plot.errorrange(pulses_total, OH_total_rot_l, yerr=OH_total_rot_std_l, save_loc=t_save_loc, save_kwargs=save_kwargs, plot_kwargs=plot_kwargs2,
                        show=show1, close=not show1, line_kwargs_iter=line_kwargs_iter)


def analyse_directory_H_alpha(data_loc, voltages, pulse_lengths, *, save_loc1=None, save_loc2=None, save_loc3=None,
                              save_kwargs1=None, save_kwargs2=None, save_kwargs3=None, show1=True, show2=False):
    model = lmfit.models.GaussianModel()
    params = lmfit.Parameters()
    params['center'] = lmfit.Parameter('center', value=656.28, min=655.28, max=657.28)
    params['amplitude'] = lmfit.Parameter('amplitude', value=1000, min=0)
    params['sigma'] = lmfit.Parameter('sigma', value=0.5, min=0)

    sigma = {}
    sigma_std = {}
    peak_intensity_vals = {}
    peak_intensity_vals_std = {}

    for voltage in voltages:
        sigma[voltage] = {}
        sigma_std[voltage] = {}
        peak_intensity_vals[voltage] = {}
        peak_intensity_vals_std[voltage] = {}

        intensity_vals = []

        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
            if not os.path.exists(loc):
                continue

            data: OESData = read_hdf5(loc)['emission']
            data = data.remove_dead_pixels()
            data = data.remove_background_interp_off({'wavelength_range': ((300, 400), (650, 860)),
                                                      'relative_threshold': 0.25})
            mask = data.is_on(wavelength_range=((300, 400), (650, 860)), relative_threshold=0.25)
            intensity = data.spectrum.intensities[mask]

            peak_heights = data.peak_intensity((656.28,), is_on_kwargs={'wavelength_range': ((300, 400), (650, 860)),
                                                      'relative_threshold': 0.25})[1][0]

            wavelengths = data.spectrum.wavelengths
            wav_mask = (wavelengths > 650) & (wavelengths < 660)

            results = []
            for index, inten in enumerate(intensity):
                result = model.fit(inten[wav_mask], x=data.spectrum.wavelengths[wav_mask], params=params)
                results.append(result)

                if save_loc3:
                    line_kwargs_iter = [{'linestyle': '-'}, {'linestyle': '--'}]
                    t_save_loc = rf'{save_loc3}{voltage}_{pulse}_{index}.pdf'
                    plot.lines(data.spectrum.wavelengths[wav_mask], [inten[wav_mask], result.best_fit], labels=['measured', 'fit'], show=False, close=True,
                               save_loc=t_save_loc, save_kwargs=save_kwargs3, line_kwargs_iter=line_kwargs_iter)

            s_std = np.array([(result.params['sigma'].stderr if result.params['sigma'].stderr else 5) for result in results])
            s_vals = [result.params['sigma'].value for result in results]
            sigma[voltage][pulse] = np.average(s_vals, weights=1/s_std**2)
            sigma_std[voltage][pulse] = np.std(s_vals)
            p_vals = [result.params['amplitude'].value for result in results]
            p_std = np.array([(result.params['amplitude'].stderr if result.params['amplitude'].stderr else 1000) for result in results])
            peak_intensity_vals[voltage][pulse] = np.average(p_vals, weights=1/p_std**2)
            peak_intensity_vals_std[voltage][pulse] = np.std(p_vals[10:20])

            plot_kwargs = {'ylabel': 'Sigma [nm]', 'xlabel': 'Time [min]', 'ylim': (0, 2)}
            y_min = min(p_vals)
            y_max = max(p_vals)
            dy = y_max - y_min
            plot_kwargs3 = {'ylabel': 'Intensity [A.U]', 'xlabel': 'Time [min]', 'ylim': (y_min-0.05*dy, y_max+0.05*dy)}

            time_vals = data.spectrum.times[mask]
            time_vals = (time_vals - time_vals[0])/60

            t_save_loc = rf'{save_loc2}{voltage}_{pulse}_sigma.pdf' if save_loc2 else None
            plot.errorrange(time_vals, s_vals, yerr=s_std, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs)
            t_save_loc = rf'{save_loc2}{voltage}_{pulse}_intensity.pdf' if save_loc2 else None
            plot.errorrange(time_vals, p_vals, yerr=p_std, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs3)

            intensity_vals.append((time_vals, p_vals, p_std))

        pulses = list(sigma[voltage].keys())
        sigma_vals = [sigma[voltage][pulse] for pulse in pulses]
        sigma_std_vals = [sigma_std[voltage][pulse] for pulse in pulses]
        amplitude_vals = [val[1] for val in intensity_vals]
        amplitude_vals_norm = [np.array(val)/np.average(val[:10]) for val in amplitude_vals]
        amplitude_std_vals = [val[1] for val in intensity_vals]
        amplitude_std_vals_norm = [np.array(val)/np.average(val[:10]) for val in amplitude_std_vals]
        time_vals = [val[0] for val in intensity_vals]

        plot_kwargs = {'ylabel': 'Sigma [nm]', 'xlabel': ' [us]', 'ylim': (0, 1)}
        plot_kwargs3 = {'ylabel': 'Intensity [A.U]', 'xlabel': 'Time [min]'}

        pulses_val = [pulse.removesuffix('us') for pulse in pulses]
        t_save_loc = rf'{save_loc2}{voltage}_sigma.pdf' if save_loc2 else None
        plot.errorrange(pulses_val, sigma_vals, yerr=sigma_std_vals, save_loc=t_save_loc, save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs,
                        show=show2, close=not show2)
        t_save_loc = rf'{save_loc2}{voltage}_intensity.pdf' if save_loc2 else None
        plot.errorrange(time_vals, amplitude_vals, yerr=amplitude_std_vals, labels=pulses, save_loc=t_save_loc, save_kwargs=save_kwargs2,
                        plot_kwargs=plot_kwargs3, show=show2, close=not show2)
        t_save_loc = rf'{save_loc2}{voltage}_intensity_no_err.pdf' if save_loc2 else None
        plot.lines(time_vals, amplitude_vals, labels=pulses, save_loc=t_save_loc,
                   save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs3, show=show2, close=not show2)
        t_save_loc = rf'{save_loc2}{voltage}_intensity_norm.pdf' if save_loc2 else None
        plot.errorrange(time_vals, amplitude_vals_norm, yerr=amplitude_std_vals_norm, labels=pulses, save_loc=t_save_loc,
                        save_kwargs=save_kwargs2,
                        plot_kwargs=plot_kwargs3, show=show2, close=not show2)
        t_save_loc = rf'{save_loc2}{voltage}_intensity_no_err_norm.pdf' if save_loc2 else None
        plot.lines(time_vals, amplitude_vals_norm, labels=pulses, save_loc=t_save_loc,
                   save_kwargs=save_kwargs2, plot_kwargs=plot_kwargs3, show=show2, close=not show2)

    pulses_total = [[x.removesuffix('us') for x in sigma[voltage].keys()] for voltage in sigma.keys()]

    sigma_total = [[sigma[voltage][pulse] for pulse in sigma[voltage].keys()] for voltage in sigma.keys()]
    sigma_total_std = [[sigma_std[voltage][pulse] for pulse in sigma[voltage].keys()] for voltage in sigma.keys()]
    intensity_total = [[peak_intensity_vals[voltage][pulse] for pulse in peak_intensity_vals[voltage].keys()] for voltage in peak_intensity_vals.keys()]
    intensity_total_std = [[peak_intensity_vals_std[voltage][pulse] for pulse in peak_intensity_vals[voltage].keys()] for voltage in peak_intensity_vals.keys()]

    plot_kwargs = {'ylabel': 'Sigma [nm]', 'xlabel': 'Pulse length [us]', 'ylim': (0, 1)}
    plot_kwargs3 = {'ylabel': 'Intensity [A.U]', 'xlabel': 'Pulse length [us]'}

    line_kwargs_iter = linestyles.linelook_by(voltages, linestyles=True)

    t_save_loc = rf'{save_loc1}sigma.pdf' if save_loc1 else None
    plot.errorrange(pulses_total, sigma_total, yerr=sigma_total_std, save_loc=t_save_loc, save_kwargs=save_kwargs1, plot_kwargs=plot_kwargs,
                    line_kwargs_iter=line_kwargs_iter, labels=voltages, show=show1, close=not show1)
    t_save_loc = rf'{save_loc1}intensity.pdf' if save_loc1 else None
    plot.errorrange(pulses_total, intensity_total, labels=voltages, yerr=intensity_total_std, save_loc=t_save_loc, save_kwargs=save_kwargs1, plot_kwargs=plot_kwargs3,
                    show=show1, close=not show1, line_kwargs_iter=line_kwargs_iter)


def analyse_directory_conductivity(data_loc, voltages, pulse_lengths, *, save_loc=None, show=False, align='start',
                                   end_time=2000, start_time=120):
    time_vals = {}
    conductivity_vals = {}
    temp_vals = {}

    for voltage in voltages:
        time_vals[voltage] = {}
        conductivity_vals[voltage] = {}
        temp_vals[voltage] = {}

        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
            if not os.path.exists(loc):
                continue

            data = read_hdf5(loc)

            try:
                conductivity = data['conductivity']
            except KeyError:
                continue

            if len(conductivity[0]) == 0:
                continue

            if align == 'start':
                time_vals[voltage][pulse] = (conductivity[0] - conductivity[0][0] - start_time)/60
            elif align == 'end':
                time_vals[voltage][pulse] = (conductivity[0] - conductivity[0][-1] + end_time)/60
                if time_vals[voltage][pulse][0] > 0:
                    time_vals[voltage][pulse] = time_vals[voltage][pulse] - time_vals[voltage][pulse][0]
            else:
                raise ValueError(rf'Invalid alignment: {align}; must be either "start" or "end"')
            conductivity_vals[voltage][pulse] = conductivity[1]
            temp_vals[voltage][pulse] = conductivity[2]

    voltages_total = [voltage.removesuffix('kV') for voltage in voltages for _ in time_vals[voltage].keys()]
    pulses_total = [p.removesuffix('us') for voltage in voltages for p in time_vals[voltage].keys()]

    time_total = [time_vals[voltage][pulse] for voltage in voltages for pulse in time_vals[voltage].keys()]
    arg_sorter = [np.argsort(x) for x in time_total]
    time_total = [x[y] for x, y in zip(time_total, arg_sorter)]
    conductivity_total = [conductivity_vals[voltage][pulse] for voltage in voltages for pulse in time_vals[voltage].keys()]
    conductivity_total = [x[y] for x, y in zip(conductivity_total, arg_sorter)]
    temperature_total = [temp_vals[voltage][pulse] for voltage in voltages for pulse in time_vals[voltage].keys()]
    temperature_total = [x[y] for x, y in zip(temperature_total, arg_sorter)]

    line_kwargs_iter = linestyles.linelooks_by(color_values=pulses_total, linestyle_values=voltages_total, colors=width_colors, linestyles=True)
    legend_kwargs = legend_linelooks(line_kwargs_iter, color_labels=pulses_total, linestyle_labels=voltages_total,
                                     color_title='W [us]', linestyle_title='H [kV]')

    plot_kwargs = {'ylabel': r'Conductivity [uS/cm]', 'xlabel': 'Time [min]', 'xlim': (0, None)}
    t_save_loc = f'{save_loc}conductivity.pdf' if save_loc else None
    plot.lines(time_total, conductivity_total, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
               legend_kwargs=legend_kwargs, show=show, close=True)

    plot_kwargs = {'ylabel': r'Temperature [$^o$C]', 'xlabel': 'Time [min]', 'xlim': (0, None)}
    t_save_loc = f'{save_loc}temperature.pdf' if save_loc else None
    plot.lines(time_total, temperature_total, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
               legend_kwargs=legend_kwargs, show=show, close=True)

    plot_kwargs = {'ylabel': r'Temperature change [$\Delta ^{\circ}$C]', 'xlabel': 'Time [min]', 'xlim': (0, None)}
    t_save_loc = f'{save_loc}temperature_change.pdf' if save_loc else None
    temperature_change = [temps - temps[np.argmin(t**2)] for temps, t in zip(temperature_total, time_total)]
    plot.lines(time_total, temperature_change, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
               legend_kwargs=legend_kwargs, show=show, close=True)

    temperature_change = [[temp_vals[voltage][pulse][np.argmin((time_vals[voltage][pulse]-30)**2)]-
                           temp_vals[voltage][pulse][np.argmin(time_vals[voltage][pulse]**2)]
                           for pulse in time_vals[voltage].keys()] for voltage in voltages]
    pulses = [[pulse.removesuffix('us') for pulse in time_vals[voltage].keys()] for voltage in voltages]
    voltage_vals = [voltage.removesuffix('kV') for voltage in voltages]

    plot_kwargs = {'ylabel': r'Temperature change [$\Delta ^{\circ}$C]', 'xlabel': 'Pulse length [us]'}
    t_save_loc = f'{save_loc}temperature_change2.pdf' if save_loc else None
    line_kwargs_iter, legend_kwargs = legend_linelooks_combines(voltage_vals, colors=True, linestyles=True)
    plot.lines(pulses, temperature_change, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
               legend_kwargs=legend_kwargs, show=show, close=True)


def spectra_over_time(data_loc, voltages, pulse_lengths, wavelength_ranges: dict[str, tuple[float, float]], *, save_loc=None,
                      plot_num=10, plot_num2=4):
    wavelength = {}
    intensity = {}
    times = {}

    for voltage in voltages:
        wavelength[voltage] = {}
        intensity[voltage] = {}
        times[voltage] = {}

        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
            if not os.path.exists(loc):
                continue

            data: OESData = read_hdf5(loc)['emission']
            data = data.remove_dead_pixels()
            wavelength_range_vals = [wavelength_ranges[key] for key in wavelength_ranges.keys()]
            data = data.remove_background_interp_off({'wavelength_range': wavelength_range_vals, 'relative_threshold': 0.25})

            wavelength[voltage][pulse] = data.spectrum.wavelengths
            intensity[voltage][pulse] = data.spectrum.intensities
            times[voltage][pulse] = (data.spectrum.times - data.spectrum.times[0])/60

    def mk_mask(wavelengths, range_val):
        return (wavelengths > range_val[0]) & (wavelengths < range_val[1])

    def mk_average(a, num):
        values = np.linspace(0, len(a), num + 1, dtype=int)
        return np.array([np.mean(a[x:y], axis=0) for x, y in zip(values[:-1], values[1:])])

    for name, range_val in wavelength_ranges.items():
        wavelengths_total = [wavelength[voltage][pulse][mk_mask(wavelength[voltage][pulse], range_val)] for voltage in wavelength for pulse in wavelength[voltage]]
        intensities_flat = [intensity[voltage][pulse][:, mk_mask(wavelength[voltage][pulse], range_val)] for voltage in wavelength for pulse in wavelength[voltage]]
        times_flat = [times[voltage][pulse] for voltage in wavelength for pulse in wavelength[voltage]]

        intensities_total = [mk_average(val, plot_num) for val in intensities_flat]
        times_total = [mk_average(val, plot_num) for val in times_flat]

        voltage = [voltage.removesuffix('kV') for voltage in wavelength for _ in wavelength[voltage]]
        pulse = [pulse.removesuffix('us') for voltage in wavelength for pulse in wavelength[voltage]]
        for v, p, wav, inten, tims in zip(voltage, pulse, wavelengths_total, intensities_total, times_total, strict=True):
            colors, sm = cbar.cbar_norm_colors(tims)
            cbar_kwargs = {'label': 'Time [min]', 'mappable': sm}
            plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Wavelength [nm]'}
            t_save_loc = f'{save_loc}emission_{name}_{v}_{p}.pdf' if save_loc else None
            plot.lines(wav, inten[1:-1], colors=colors[1:-1], plot_kwargs=plot_kwargs, save_loc=t_save_loc, cbar_kwargs=cbar_kwargs, show=False, close=True)
            inten_normed = [i/np.max(i) for i in inten]
            t_save_loc = f'{save_loc}emission_{name}_{v}_{p}_norm.pdf' if save_loc else None
            plot.lines(wav, inten_normed[1:-1], colors=colors[1:-1], plot_kwargs=plot_kwargs, save_loc=t_save_loc, cbar_kwargs=cbar_kwargs, show=False, close=True)
            index = len(inten_normed)//2
            inten_normed_rel = [i - inten_normed[index] for i in inten_normed]
            t_save_loc = f'{save_loc}emission_{name}_{v}_{p}_norm_change.pdf' if save_loc else None
            plot.lines(wav, inten_normed_rel[1:-1], colors=colors[1:-1], plot_kwargs=plot_kwargs, save_loc=t_save_loc, cbar_kwargs=cbar_kwargs,
                       show=False, close=True)

        intensities_total = [mk_average(val, plot_num2) for val in intensities_flat]

        for i in range(plot_num2):
            wavelengths_sel = [wavelengths_total[j] for j in range(len(wavelengths_total))]
            intensities_sel = [intensities_total[j][i] for j in range(len(intensities_total))]
            max_intensity = [np.max(i) for i in intensities_sel]
            intensities_sel = sort_by(max_intensity, intensities_sel)[0]
            mask = [(np.max(i) - np.min(i)) > 100 for i in intensities_sel]

            wavelengths_selc = [wavelengths_sel[j] for j in range(len(wavelengths_sel)) if mask[j]]
            intensities_selc = [intensities_sel[j] for j in range(len(intensities_sel)) if mask[j]]
            times_selc = [times_total[j][i] for j in range(len(times_total)) if mask[j]]
            pulse_selc = [pulse[j] for j in range(len(pulse)) if mask[j]]
            voltage_selc = [voltage[j] for j in range(len(voltage)) if mask[j]]

            line_kwargs_iter = linestyles.linelooks_by(color_values=pulse, linestyle_values=voltage, colors=width_colors)
            legend_kwargs = legend_linelooks(line_kwargs_iter, color_labels=pulse, linestyle_labels=voltage,
                                             color_title='W [us]', linestyle_title='H [kV]')
            plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Wavelength [nm]'}
            t_save_loc = f'{save_loc}emission_{name}_{i}.pdf' if save_loc else None
            plot.lines(wavelengths_sel, intensities_sel, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                       legend_kwargs=legend_kwargs, show=False, close=True)

            plot_kwargs['ylim'] = (0, 1.05)
            intensities_sel_normed = [i/np.max(i) for i in intensities_sel]
            t_save_loc = f'{save_loc}emission_{name}_{i}_norm.pdf' if save_loc else None
            plot.lines(wavelengths_sel, intensities_sel_normed, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                       legend_kwargs=legend_kwargs, show=False, close=True)

            line_kwargs_iter = linestyles.linelooks_by(color_values=pulse_selc, linestyle_values=voltage_selc, colors=width_colors)
            legend_kwargs = legend_linelooks(line_kwargs_iter, color_labels=pulse_selc, linestyle_labels=voltage_selc,
                                             color_title='W [us]', linestyle_title='H [kV]')
            plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Wavelength [nm]'}
            t_save_loc = f'{save_loc}emission_mask_{name}_{i}.pdf' if save_loc else None
            plot.lines(wavelengths_selc, intensities_selc, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs,
                       save_loc=t_save_loc, legend_kwargs=legend_kwargs, show=False, close=True)

            plot_kwargs['ylim'] = (0, 1.05)
            intensities_selc_normed = [i / np.max(i) for i in intensities_selc]
            t_save_loc = f'{save_loc}emission_mask_{name}_{i}_norm.pdf' if save_loc else None
            plot.lines(wavelengths_selc, intensities_selc_normed, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs,
                       save_loc=t_save_loc, legend_kwargs=legend_kwargs, show=False, close=True)