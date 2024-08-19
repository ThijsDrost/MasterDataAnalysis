import numpy as np
import lmfit

from General.experiments.hdf5.readHDF5 import read_hdf5
from General.experiments.waveforms import MeasuredWaveforms
from General.plotting import plot, cbar
from General.plotting import linestyles
import General.numpy_funcs as npf

out_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Method'
# %%
bloc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results\Air_3slm_Ar_1slm_9kV_'
for t in (0.3, 0.5, 1, 2,):
    loc = rf'{bloc}{t}us.hdf5'
    data = read_hdf5(loc)
    waveform_data = data['waveforms']
    waveforms = MeasuredWaveforms.from_waveforms(waveform_data, channels={1: 'voltage', 2: 'current',
                                                                          3: 'pulse_generator', 4: 'ground_current'})

    is_on = waveforms.is_on()
    num = np.sum(is_on.astype(int))
    values = np.argwhere(np.diff(is_on.astype(int)) == 1)[0][0] + np.linspace(10, num - 52, 10, dtype=int)
    indexes = values + np.arange(20, dtype=int)[:, None]

    times = np.mean(waveforms.time[indexes], axis=0)
    currents = np.mean(waveforms.currents[indexes], axis=0)
    rel_times = waveforms.time_offset[values]
    rel_times = rel_times - rel_times[0]
    colors, _ = cbar.cbar_norm_colors(rel_times)
    labels = [f'{rel_time / 60:.0f}' for rel_time in rel_times]

    plot_kwargs = {'xlabel': 'Time [us]', 'ylabel': 'Current [A]', 'xlim': (0, 0.7 + t)}
    # legend_kwargs = {'title': 'Time [min]', 'loc': 'upper right'}
    save_loc = rf'{out_loc}\currents_{t}.pdf'

    line_kwargs_iter = linestyles.linelook_by(labels, colors=colors)
    l_kwargs = linestyles.legend_linelooks(line_kwargs_iter, color_labels=labels, color_title='Time [min]', no_marker=False)
    l_kwargs['loc'] = 'upper right'

    plot.lines(1e6 * times, currents, colors=colors, plot_kwargs=plot_kwargs, legend_kwargs=l_kwargs, save_loc=save_loc)
    plot.lines(1e6 * times, currents, colors=colors, plot_kwargs=plot_kwargs, save_loc=save_loc.replace('.pdf', '_l.pdf'))

    max_time = 1e6 * times[1][np.argmax(currents[1])]
    min_time = 1e6 * times[1][np.argmin(currents[1])]
    dt = min_time - max_time

    plot_kwargs['ylabel'] = 'Current change [A]'
    save_loc = rf'{out_loc}\currents_{t}2.pdf'
    fig, ax = plot.lines(1e6 * times, currents - currents[0], colors=colors, plot_kwargs=plot_kwargs, show=False)
    ax.axvline(max_time, color='black', linestyle='-')
    ax.axvline(min_time, color='black', linestyle='-')
    if t == 0.3:
        ax.axvline(max_time + 0.125, color='black', linestyle=':')
        ax.axvline(min_time - 0.125, color='black', linestyle=':')
    else:
        ax.axvline(max_time + 0.25, color='black', linestyle=':')
        ax.axvline(min_time - 0.15, color='black', linestyle=':')
    fig.savefig(save_loc.replace('.pdf', '_l.pdf'))
    plot.lines([], [], plot_kwargs=plot_kwargs, legend_kwargs=l_kwargs, fig_ax=(fig, ax), save_loc=save_loc)

    n = 100
    _, f_val, f_std = waveforms.background_current_fitting(t, block_average=n)
    on_mask = waveforms.is_on()
    a_val, a_std = waveforms.background_current_averaging(on_mask=on_mask, block_average=n)
    a_val2, a_std2 = waveforms.background_current_averaging(on_mask=on_mask, block_average=n, start_offset=0.45, end_offset=0.45)
    times = 1e6 * npf.block_average(waveforms.time[on_mask], n)
    currents = npf.block_average(waveforms.currents[on_mask], n)
    start_time = np.average(times, axis=0)[np.argmax(np.average(currents, axis=0))]
    end_time = np.average(times, axis=0)[np.argmin(np.average(currents, axis=0))]
    s_time = start_time + (end_time - start_time) / 3
    e_time = end_time - (end_time - start_time) / 3

    plot_kwargs = {'xlabel': 'Time [us]', 'ylabel': 'Current change [A]', 'xlim': (0.375, 0.19 + t)}
    fig, ax = plot.lines(times, currents, show=False, plot_kwargs=plot_kwargs)
    ax.axvline(s_time, color='black', linestyle=':')
    ax.axvline(e_time, color='black', linestyle=':')
    for i, (value, value2, value3) in enumerate(zip(f_val, a_val, a_val2)):
        ax.axhline(value, color=f'C{i}', linestyle='--')
        ax.axhline(value2, color=f'C{i}', linestyle='-')
        ax.axhline(value3, color=f'C{i}', linestyle=':')

plot.export_legend(l_kwargs, rf'{out_loc}\time_legend.pdf')