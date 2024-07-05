import os
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
import lmfit
import bottleneck as bn

from General.experiments.absorption import MeasurementsAnalyzer
from General.experiments.hdf5.readHDF5 import read_hdf5, DataSet
from General.experiments.oes import OESData
from General.experiments.absorption.Models import multi_species_model
from General.experiments.waveforms import Waveforms, MeasuredWaveforms
from General.plotting import plot, linestyles, cbar
from General.plotting.linestyles import linelooks_by, legend_linelooks, legend_linelooks_by
from General.simulation.specair.specair import N2SpecAirSimulations, Spectrum, SpecAirSimulations
from General.itertools import argmax, argmin, flatten_2D
import General.numpy_funcs as npf


def analyse_directory_absorption(data_loc, voltages, pulse_lengths, save_loc=None, save_kwargs=None, lines_kwargs=None,
                                 model_loc=r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Cuvette4.hdf5'):
    model = multi_species_model(model_loc, add_zero=True, add_constant=True, add_slope=True)

    voltages_ = []
    pulses_ = []
    results_ = []

    for voltage in voltages:
        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'

            data: DataSet = read_hdf5(loc)['absorbance'].remove_index(-1)
            analyzer = MeasurementsAnalyzer(data)

            result, _ = analyzer.fit(model, wavelength_range=(250, 400))
            results_.append(result)
            pulses_.append(pulse)
            voltages_.append(voltage)

    times_ = [result[1] for result in results_]
    h2o2_ = [result[2][0] for result in results_]
    no2 = [result[2][1] for result in results_]
    no3 = [result[2][2] for result in results_]

    lines_kwargs = lines_kwargs or {}

    line_kwargs = linelooks_by(color_values=voltages_, linestyle_values=pulses_)
    legend_kwargs = legend_linelooks(line_kwargs, color_labels=voltages_, linestyle_labels=pulses_)
    plot_kwargs = {'ylabel': 'Concentration [mM]', 'xlabel': 'Time [min]'}

    if 'legend_kwargs' in lines_kwargs:
        legend_kwargs = plot.set_defaults(lines_kwargs['legend_kwargs'], **legend_kwargs)
        del lines_kwargs['legend_kwargs']
    if 'plot_kwargs' in lines_kwargs:
        plot_kwargs = plot.set_defaults(lines_kwargs['plot_kwargs'], **plot_kwargs)
        del lines_kwargs['plot_kwargs']
    h2o2_loc = save_loc + '_h2o2.pdf' if save_loc else None
    plot.lines(times_, h2o2_, line_kwargs_iter=line_kwargs, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs,
               save_loc=h2o2_loc, save_kwargs=save_kwargs, **lines_kwargs)
    no2_loc = save_loc + '_no2.pdf' if save_loc else None
    plot.lines(times_, no2, line_kwargs_iter=line_kwargs, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs,
               save_loc=no2_loc, save_kwargs=save_kwargs, **lines_kwargs)
    no3_loc = save_loc + '_no3.pdf' if save_loc else None
    return plot.lines(times_, no3, line_kwargs_iter=line_kwargs, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs,
                      save_loc=no3_loc, save_kwargs=save_kwargs, **lines_kwargs)


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
            t_save_loc = rf'{save_loc2}_{voltage}_{pulse}.pdf' if save_loc2 else None
            data.peak_intensity_vs_wavelength_with_time('argon', plot_kwargs=plot_kwargs, show=show2, close=not show2,
                                                        save_loc=t_save_loc, save_kwargs=save_kwargs2)

            plot_kwargs = {'title': f'{voltage} {pulse}'}
            t_save_loc = rf'{save_loc2}_{voltage}_{pulse}_rel.pdf' if save_loc2 else None
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
            time_vals, ratios = time_vals[mask], ratios[:, mask]
            plot.lines(time_vals, ratios, colors=colors, cbar_kwargs=cbar_kwargs,
                       plot_kwargs=plot_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs2, show=show2, close=not show2)
            plot_kwargs['ylim'] = (0.6, 1.025)
            t_save_loc = rf'{save_loc2}_{voltage}_{pulse}_ratio_time_zoom1.pdf' if save_loc else None
            plot.lines(time_vals, ratios, colors=colors, cbar_kwargs=cbar_kwargs,
                       plot_kwargs=plot_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs2, show=show2, close=not show2)
            plot_kwargs['ylim'] = (0, 0.38)
            t_save_loc = rf'{save_loc2}_{voltage}_{pulse}_ratio_time_zoom2.pdf' if save_loc else None
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

        plot_kwargs = {'ylabel': 'Relative intensity', 'ylim': (0, 1.05), 'xlabel': 'Pulse width [us]'}

        t_save_loc = rf'{save_loc}_{voltage}.pdf' if save_loc else None
        plot.errorbar(pulse_vals, peak_intensity_norm, yerr=std_val_2, line_kwargs=line_kwargs, colors=colors, cbar_kwargs=t_cbar_kwargs,
                      save_loc=t_save_loc, save_kwargs=save_kwargs, plot_kwargs=plot_kwargs, show=show1, close=not show1)

        plot_kwargs['ylim'] = (0.6, 0.9)
        t_save_loc = rf'{save_loc}_{voltage}_zoom1.pdf' if save_loc else None
        plot.errorbar(pulse_vals, peak_intensity_norm, yerr=std_val_2, line_kwargs=line_kwargs, colors=colors, close=not show1,
                      cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs, plot_kwargs=plot_kwargs, show=show1)

        plot_kwargs['ylim'] = (0, 0.38)
        t_save_loc = rf'{save_loc}_{voltage}_zoom2.pdf' if save_loc else None
        plot.errorbar(pulse_vals, peak_intensity_norm, yerr=std_val_2, line_kwargs=line_kwargs, colors=colors, close=not show1,
                      cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc, save_kwargs=save_kwargs, plot_kwargs=plot_kwargs, show=show1)

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

        t_save_loc = rf'{save_loc}_{voltage}_total_energies.pdf' if save_loc else None
        plot.lines(pulses, peak_intensity_norm_t_norm, colors=colors2, show=show1, close=not show1, save_loc=t_save_loc,
                   cbar_kwargs={'mappable': mappable2, 'label': 'Energy [eV]'})

        if index + 1 == len(argon_results):
            plot_kwargs = {'ylabel': 'Relative intensity', 'xlabel': 'Pulse width [us]', 'ylim': (0, 1.05)}
            t_save_loc = rf'{save_loc}_total.pdf' if save_loc else None
            plot.errorbar(pulses, peak_intensity_start_norm, yerr=std_val, line_kwargs=line_kwargs, colors=colors, fig_ax=rel_intensity_total,
                          save_kwargs=save_kwargs, cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
                          show=show1, close=not show1)
            plot_kwargs = {'ylabel': 'Relative intensity', 'xlabel': 'Pulse width [us]', 'ylim': (0.6, 0.9)}
            t_save_loc = rf'{save_loc}_total_zoom1.pdf' if save_loc else None
            plot.errorbar(pulses, peak_intensity_norm_t, yerr=std_val, line_kwargs=line_kwargs, colors=colors,
                          fig_ax=peak_intensity_start_norm, plot_kwargs=plot_kwargs, show=show1, close=not show1,
                          save_kwargs=save_kwargs, cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc)
            plot_kwargs = {'ylabel': 'Relative intensity', 'xlabel': 'Pulse width [us]', 'ylim': (0, 0.38)}
            t_save_loc = rf'{save_loc}_total_zoom2.pdf' if save_loc else None
            plot.errorbar(pulses, peak_intensity_norm_t, yerr=std_val, line_kwargs=line_kwargs, colors=colors,
                          fig_ax=peak_intensity_start_norm, plot_kwargs=plot_kwargs, show=show1, close=not show1,
                          save_kwargs=save_kwargs, cbar_kwargs=t_cbar_kwargs, save_loc=t_save_loc)
            t_save_loc = rf'{save_loc}_total_energies.pdf' if save_loc else None
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

    background_current = {}
    background_current_std = {}
    background_current_time = {}

    background_current2 = {}
    background_current_std2 = {}
    background_current_time2 = {}

    highest_current = {}
    lowest_current = {}
    time_current = {}

    for voltage in voltages:
        first_pulse[voltage] = {}
        middle_pulse[voltage] = {}
        end_pulse[voltage] = {}
        pulse_height[voltage] = {}
        pulse_width[voltage] = {}
        background_current[voltage] = {}
        background_current_std[voltage] = {}
        background_current_time[voltage] = {}
        background_current2[voltage] = {}
        background_current_std2[voltage] = {}
        background_current_time2[voltage] = {}
        highest_current[voltage] = {}
        lowest_current[voltage] = {}
        time_current[voltage] = {}
        for pulse in pulse_lengths:
            loc = f'{data_loc}_{voltage}_{pulse}.hdf5'
            if not os.path.exists(loc):
                continue

            data: Waveforms = read_hdf5(loc)['waveforms']

            wavs = MeasuredWaveforms.from_waveforms(data, channels)
            t_save_loc = save_loc + f'{voltage}_{pulse}_' if save_loc else None
            wavs.plot(save_loc=t_save_loc, close=True)

            is_on = wavs.is_on()
            time = wavs.time
            current = wavs.currents
            voltage_vals = wavs.voltages
            time_offsets = wavs.time_offset

            n = 20
            avg_current = npf.block_average(current, n)
            avg_voltage = npf.block_average(voltage_vals, n)
            avg_time_offsets = npf.block_average(time_offsets, n)
            avg_time = npf.block_average(time, n)
            avg_power = avg_current*avg_voltage
            avg_pulse_power = np.trapz(avg_time, avg_power, axis=1)

            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Energy [J]'}
            t_save_loc = save_loc2 + f'{voltage}_{pulse}_avg_energy.pdf' if save_loc2 else None
            plot.lines(avg_time_offsets, avg_pulse_power, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show2, close=True)

            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Power [W]'}
            colors, mappable = cbar.cbar_norm_colors(avg_time_offsets, 'turbo')
            cbar_kwargs = {'label': 'Time [s]', 'mappable': mappable}
            t_save_loc = save_loc + f'{voltage}_{pulse}_avg_power.pdf' if save_loc else None
            plot.lines(avg_time, avg_power, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show2, close=True, colors=colors,
                       cbar_kwargs=cbar_kwargs)

            on_current = current[is_on]
            middle = on_current.shape[0]//2
            first_pulse[voltage][pulse] = (np.nanmean(time[is_on][5:15], axis=0), np.nanmean(on_current[5:15], axis=0))
            middle_pulse[voltage][pulse] = (np.nanmean(time[is_on][(middle-5):(middle+5)], axis=0), np.nanmean(on_current[(middle-5):(middle+5)], axis=0))
            end_pulse[voltage][pulse] = (np.nanmean(time[is_on][-15:-5], axis=0), np.nanmean(on_current[-15:-5], axis=0))

            rise_time, pulse_length, height = wavs.fit_voltage()
            mask = np.isfinite(rise_time)
            pulse_height[voltage][pulse] = (np.average(height[mask][10:-5]), np.std(height[mask][10:-5]))
            pulse_width[voltage][pulse] = (np.average(pulse_length[mask][10:-5]), np.std(pulse_length[mask][10:-5]))

            avg, std = wavs.background_current_averaging(float(pulse.replace('us', '')))
            time_vals = wavs.time_offset[is_on]
            background_current[voltage][pulse] = avg[is_on]
            background_current_std[voltage][pulse] = std[is_on]
            background_current_time[voltage][pulse] = time_vals - time_vals[0]

            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Background current [A]'}
            t_save_loc = save_loc + f'{voltage}_{pulse}_background.pdf' if save_loc else None
            plot.errorrange(background_current_time[voltage][pulse], background_current[voltage][pulse], yerr=background_current_std[voltage][pulse], save_loc=t_save_loc,
                            plot_kwargs=plot_kwargs, show=show2, close=True)

            avg_offset, back_curr_results, back_curr_std = wavs.background_current_fitting(float(pulse.replace('us', '')),
                                                                                           save_loc=save_loc2)
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

            highest_current[voltage][pulse] = np.nanmax(on_current, axis=0)[is_on]
            lowest_current[voltage][pulse] = np.nanmin(on_current, axis=0)[is_on]
            time_current[voltage][pulse] = time[is_on]

            t_save_loc = save_loc + f'{voltage}_{pulse}_highest_current.pdf' if save_loc else None
            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Maximum current [A]'}
            plot.lines(time_current[voltage][pulse], highest_current[voltage][pulse], show=show2, close=True,
                       save_loc=t_save_loc, plot_kwargs=plot_kwargs)
            t_save_loc = save_loc + f'{voltage}_{pulse}_lowest_current.pdf' if save_loc else None
            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Minimum current [A]'}
            plot.lines(time_current[voltage][pulse], lowest_current[voltage][pulse], show=show2, close=True,
                          save_loc=t_save_loc, plot_kwargs=plot_kwargs)

        pulses = list(first_pulse[voltage].keys())
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
        plot.lines(first_pulse_time, first_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show)
        t_save_loc = save_loc + f'{voltage}_middle_pulse.pdf' if save_loc else None
        plot.lines(middle_pulse_time, middle_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show)
        t_save_loc = save_loc + f'{voltage}_end_pulse.pdf' if save_loc else None
        plot.lines(end_pulse_time, end_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show)

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
                       show=show, close=True)
            t_save_loc = save_loc + f'{voltage}_middle_pulse_zoom_{name}.pdf' if save_loc else None
            plot.lines(middle_pulse_time_zoom_max, middle_pulse_vals_zoom_max, save_loc=t_save_loc, plot_kwargs=plot_kwargs, labels=pulses,
                       show=show, close=True)
            t_save_loc = save_loc + f'{voltage}_end_pulse_zoom_{name}.pdf' if save_loc else None
            plot.lines(end_pulse_time_zoom_max, end_pulse_vals_zoom_max, save_loc=t_save_loc, plot_kwargs=plot_kwargs, labels=pulses,
                       show=show, close=True)

        pulse_height_vals = [pulse_height[voltage][pulse] for pulse in pulses]
        pulse_width_vals = [pulse_width[voltage][pulse] for pulse in pulses]

        pulse_height_val = [x[0] for x in pulse_height_vals]
        pulse_height_std = [x[1] for x in pulse_height_vals]
        pulse_width_val = [x[0]/float(val.replace('us', '')) for x, val in zip(pulse_width_vals, pulses)]
        pulse_width_std = [x[1]/float(val.replace('us', '')) for x, val in zip(pulse_width_vals, pulses)]

        p_val = [p.removesuffix('us') for p in pulses]
        plot_kwargs = {'xlabel': 'Pulse width', 'ylabel': 'Height [V]'}
        t_save_loc = save_loc + f'{voltage}_pulse_height.pdf' if save_loc else None
        plot.errorbar(p_val, pulse_height_val, yerr=pulse_height_std, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show)
        plot_kwargs = {'xlabel': 'Pulse width', 'ylabel': 'Width accuracy'}
        t_save_loc = save_loc + f'{voltage}_pulse_width.pdf' if save_loc else None
        plot.errorbar(p_val, pulse_width_val, yerr=pulse_width_std, save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=show)

        for background, background_std, background_time, name in zip((background_current, background_current2),
                                                                     (background_current_std, background_current_std2),
                                                                     (background_current_time, background_current_time2),
                                                                     ('', '_fit')):
            background_current_vals = [background[voltage][pulse] for pulse in pulses]
            background_current_std_vals = [background_std[voltage][pulse] for pulse in pulses]
            background_current_time_vals = [background_time[voltage][pulse] for pulse in pulses]

            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Background current [A]'}
            t_save_loc = save_loc + f'{voltage}_background{name}.pdf' if save_loc else None
            plot.errorrange(background_current_time_vals, background_current_vals, yerr=background_current_std_vals,
                            save_loc=t_save_loc, plot_kwargs=plot_kwargs, labels=pulses, show=show)

        b_currents_rel = [background_current[voltage][pulse] - np.average(background_current[voltage][pulse][:10]) for pulse in pulses]
        bt_currents = [background_current_time[voltage][pulse] for pulse in pulses]

        h_currents = [highest_current[voltage][pulse] for pulse in pulses]
        l_currents = [lowest_current[voltage][pulse] for pulse in pulses]
        h_currents_rel = [highest_current[voltage][pulse] - np.average(highest_current[voltage][pulse][:10]) for pulse in pulses]
        l_currents_rel = [lowest_current[voltage][pulse] - np.average(lowest_current[voltage][pulse][:10]) for pulse in pulses]
        h_currents_norm = [highest_current[voltage][pulse]/np.average(highest_current[voltage][pulse][:10]) for pulse in pulses]
        l_currents_norm = [lowest_current[voltage][pulse]/np.average(lowest_current[voltage][pulse][:10]) for pulse in pulses]
        t_currents = [time_current[voltage][pulse] for pulse in pulses]

        for name, currents in zip(('highest', 'lowest', 'highest_rel', 'lowest_rel', 'highest_norm', 'lowest_norm'),
                                  (h_currents, l_currents, h_currents_rel, l_currents_rel, h_currents_norm, l_currents_norm)):
            plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Current [A]'}
            t_save_loc = save_loc + f'{voltage}_peak_current_{name}.pdf' if save_loc else None
            plot.lines(t_currents, currents, save_loc=t_save_loc, plot_kwargs=plot_kwargs, labels=pulses, show=show)

        temp_times = bt_currents + t_currents + t_currents
        temp_currents = b_currents_rel + h_currents_rel + l_currents_rel
        labels = ['Background']*len(bt_currents) + ['Highest']*len(h_currents) + ['Lowest']*len(l_currents)
        pulse_vals = p_val*3
        line_kwargs_iter = linestyles.linelooks_by(color_values=pulse_vals, linestyle_values=labels, linestyles=[':', '--', '-.'])
        legend_kwargs = linestyles.legend_linelooks(line_kwargs_iter, linestyle_labels=labels, color_labels=pulse_vals,
                                                    linestyle_title='Type', color_title='Pulse [us]')
        plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Current change [A]'}
        t_save_loc = save_loc + f'{voltage}_all_currents.pdf' if save_loc else None
        plot.lines(temp_times, temp_currents, save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                   legend_kwargs=legend_kwargs, show=show)

    pulses = [list(first_pulse[voltage].keys()) for voltage in voltages]
    pulse_vals = flatten_2D(pulses)
    voltage_vals = flatten_2D([[v]*len(pulse) for v, pulse in zip(voltages, pulses)])

    first_pulse_vals = [[first_pulse[voltage][pulse] for pulse in pulse_list] for voltage, pulse_list in zip(voltages, pulses)]
    middle_pulse_vals = [[middle_pulse[voltage][pulse] for pulse in pulse_list] for voltage, pulse_list in zip(voltages, pulses)]
    end_pulse_vals = [[end_pulse[voltage][pulse] for pulse in pulse_list] for voltage, pulse_list in zip(voltages, pulses)]

    first_pulse_time = [x[0]*1e6 for pulse_list in first_pulse_vals for x in pulse_list]
    first_pulse_current = [x[1] for pulse_list in first_pulse_vals for x in pulse_list]
    middle_pulse_time = [x[0]*1e6 for pulse_list in middle_pulse_vals for x in pulse_list]
    middle_pulse_current = [x[1] for pulse_list in middle_pulse_vals for x in pulse_list]
    end_pulse_time = [x[0]*1e6 for pulse_list in end_pulse_vals for x in pulse_list]
    end_pulse_current = [x[1] for pulse_list in end_pulse_vals for x in pulse_list]

    p_vals = [p.removesuffix('us') for p in pulse_vals]
    line_kwargs_iter = linestyles.linelooks_by(color_values=p_vals, linestyle_values=voltage_vals)
    legend_kwargs = linestyles.legend_linelooks(line_kwargs_iter, linestyle_labels=voltage_vals, color_labels=pulse_vals,
                                                linestyle_title='Voltage', color_title='Pulse', show=show)

    plot_kwargs = {'xlabel': 'Time [us]', 'ylabel': 'Current [A]'}
    t_save_loc = save_loc + f'first_pulse.pdf' if save_loc else None
    plot.lines(first_pulse_time, first_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
               line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show)
    t_save_loc = save_loc + f'middle_pulse.pdf' if save_loc else None
    plot.lines(middle_pulse_time, middle_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
               line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show)
    t_save_loc = save_loc + f'end_pulse.pdf' if save_loc else None
    plot.lines(end_pulse_time, end_pulse_current, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
               line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show)

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
                   line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show)
        t_save_loc = save_loc + f'middle_pulse_zoom_{name}.pdf' if save_loc else None
        plot.lines(middle_pulse_time_zoom, middle_pulse_vals_zoom, save_loc=t_save_loc,
                   plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show)
        t_save_loc = save_loc + f'end_pulse_zoom_{name}.pdf' if save_loc else None
        plot.lines(end_pulse_time_zoom, end_pulse_vals_zoom, save_loc=t_save_loc, plot_kwargs=plot_kwargs,
                   line_kwargs_iter=line_kwargs_iter, legend_kwargs=legend_kwargs, show=show)

    for background, background_std, background_time, name in zip((background_current, background_current2),
                                                                 (background_current_std, background_current_std2),
                                                                 (background_current_time, background_current_time2),
                                                                 ('', '_fit')):
        background_current_total = [background[voltage][pulse] for voltage in voltages for pulse in background[voltage]]
        background_current_total_change = [x - x[0] for x in background_current_total]
        background_current_std_total = [background_std[voltage][pulse] for voltage in voltages for pulse in background_std[voltage]]
        background_current_time_total = [background_time[voltage][pulse] for voltage in voltages for pulse in background_time[voltage]]

        plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Background current [A]'}
        t_save_loc = save_loc + f'background{name}.pdf' if save_loc else None
        plot.errorrange(background_current_time_total, background_current_total, yerr=background_current_std_total,
                        save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                        legend_kwargs=legend_kwargs, show=show)
        t_save_loc = save_loc + f'background_change{name}.pdf' if save_loc else None
        plot.errorrange(background_current_time_total, background_current_total_change, yerr=background_current_std_total,
                        save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                        legend_kwargs=legend_kwargs, show=show)

    h_currents = [highest_current[voltage][pulse] for voltage in voltages for pulse in highest_current[voltage]]
    l_currents = [lowest_current[voltage][pulse] for voltage in voltages for pulse in lowest_current[voltage]]
    h_currents_rel = [x - np.average(x[:10]) for x in h_currents]
    l_currents_rel = [x - np.average(x[:10]) for x in l_currents]
    h_currents_norm = [x/np.average(x[:10]) for x in h_currents]
    l_currents_norm = [x/np.average(x[:10]) for x in l_currents]
    t_currents = [time_current[voltage][pulse] for voltage in voltages for pulse in time_current[voltage]]

    plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Current [A]'}
    line_kwargs_iter = linestyles.linelooks_by(color_values=pulse_vals, linestyle_values=voltage_vals)
    legend_kwargs = linestyles.legend_linelooks(line_kwargs_iter, linestyle_labels=voltage_vals, color_labels=pulse_vals,
                                                linestyle_title='Voltage [kV]', color_title='Pulse [us]')

    for name, currents in zip(('highest', 'lowest', 'highest_rel', 'lowest_rel', 'highest_norm', 'lowest_norm'),
                                (h_currents, l_currents, h_currents_rel, l_currents_rel, h_currents_norm, l_currents_norm)):
        t_save_loc = save_loc + f'peak_current_{name}.pdf' if save_loc else None
        plot.lines(t_currents, currents, save_loc=t_save_loc, plot_kwargs=plot_kwargs, line_kwargs_iter=line_kwargs_iter,
                   legend_kwargs=legend_kwargs, show=show)



def emission_ranges(data_loc, voltages, pulse_lengths, wavelength_ranges=((270, 450), (654, 659), (690, 860)), *,
                    save_loc1=None, save_loc2=None, save_loc3=None, save_kwargs1=None, save_kwargs2=None, save_kwargs3=None,
                    show1=True, show2=True, show3=False):
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

            intensities = []
            for wavelength_range in wavelength_ranges:
                mask = (data.spectrum.wavelengths > wavelength_range[0]) & (data.spectrum.wavelengths < wavelength_range[1])
                intensity = data.spectrum.intensities[is_on][:, mask]
                intensities.append(np.average(intensity, axis=1))

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

            line_kwargs = linestyles.linelook_by(wavelength_ranges, colors=True, linestyles=True)
            plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Time [min]'}
            labels = [f'{wavelength_range[0]}-{wavelength_range[1]}' for wavelength_range in wavelength_ranges]

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
        wav_ranges = np.tile([f'{wavelength_range[0]}-{wavelength_range[1]}' for wavelength_range in wavelength_ranges], len(pulse_vals))
        intensities = [x for values in intensities_total[voltage].values() for x in values]
        intensities_norm = [intensity/max(intensity) for intensity in intensities]
        times = [[x for x in values] for values in times_total[voltage].values() for _ in range(len(wavelength_ranges))]

        line_kwargs_iter = linelooks_by(color_values=pulses, linestyle_values=wav_ranges)
        legend_kwargs = legend_linelooks(line_kwargs_iter, color_labels=pulses, linestyle_labels=wav_ranges)

        plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Time [min]'}
        t_save_loc = f'{save_loc2}{voltage}.pdf' if save_loc2 else None
        plot.lines(times, intensities, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)
        t_save_loc = f'{save_loc2}{voltage}_norm.pdf' if save_loc2 else None
        plot.lines(times, intensities_norm, line_kwargs_iter=line_kwargs_iter, plot_kwargs=plot_kwargs, save_loc=t_save_loc,
                   legend_kwargs=legend_kwargs, show=show2, close=not show2, save_kwargs=save_kwargs2)

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

    plot_kwargs = {'ylabel': 'Intensity [A.U.]', 'xlabel': 'Pulse length [us]'}
    line_styles = linestyles.linelook_by(voltages, linestyles=True)
    line_styles = [l['linestyle'] for l in line_styles]

    wav_range_vals = [f'{wavelength_range[0]}-{wavelength_range[1]}' for wavelength_range in wavelength_ranges]
    colors = linestyles.linelook_by(wav_range_vals, colors=True)
    colors = [c['color'] for c in colors]

    legend_kwargs = legend_linelooks_by(color_labels=wav_range_vals, linestyle_labels=voltages,
                                        color_values=colors, linestyle_values=line_styles, sort=False)

    t_save_loc = f'{save_loc1}total.pdf' if save_loc1 else None
    plot1 = plot.lines(pulse_lengths, np.full(len(pulse_lengths), np.nan), show=False, close=False)
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

        line_kwargs_iter = linelooks_by(color_values=pulses, linestyle_values=label_vals)
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

    colors = linestyles.linelook_by(labels, colors=True)
    colors = [c['color'] for c in colors]

    legend_kwargs = legend_linelooks_by(color_labels=labels, linestyle_labels=voltages,
                                        color_values=colors, linestyle_values=line_styles, sort=False)

    t_save_loc = f'{save_loc1}total.pdf' if save_loc1 else None
    plot1 = plot.lines(pulse_lengths, np.full(len(pulse_lengths), np.nan), show=False, close=False)
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
