from dataclasses import dataclass
import warnings

import numpy as np
import lmfit

from General.plotting import plot, linestyles, cbar
from General.checking import Descriptor
import General.numpy_funcs as npf


@dataclass
class Waveform:
    waveform: np.ndarray = Descriptor.numpy_dim(1)
    time: np.ndarray = Descriptor.numpy_dim(1)
    unit: str = Descriptor.is_str(default='none')

    def peak_peak_height(self):
        return np.max(self.waveform) - np.min(self.waveform)

    def plot(self):
        plot_kwargs = {'xlabel': 'Time [us]', 'ylabel': f'Amplitude [{self.unit}]'}
        return plot.lines(self.time*1e6, self.waveform, plot_kwargs=plot_kwargs)

    def __mul__(self, other):
        if isinstance(other, Waveform):
            if not np.all(self.time == other.time):
                raise ValueError('Time arrays are not equal')
            return Waveform(self.waveform * other.waveform, self.time, self.unit)
        return Waveform(self.waveform * other, self.time, self.unit)


@dataclass
class Waveforms:
    waveforms: list[Waveform]
    time_stamp: np.ndarray
    channel: np.ndarray
    num: np.ndarray = None

    def plot(self):
        line_kwargs_iter = linestyles.linelooks_by(color_values=self.time_stamp, linestyle_values=self.channel)
        legend_kwargs = plot.make_legend(line_kwargs_iter, [str(channel) for channel in self.channel])
        plot_kwargs = {'xlabel': 'Time [us]', 'ylabel': 'Amplitude [kV]'}

        plot.lines(self.time_stamp*1e6, [waveform.waveform for waveform in self.waveforms], line_kwargs_iter=line_kwargs_iter,
                   legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs)

    def sorted(self):
        sorter = self.num if self.num is not None else self.time_stamp

        channel_num = np.array([(t, int(channel[1])) for t, channel in zip(sorter, self.channel)], dtype=[('time', float), ('channel', int)])
        sort_mask = np.argsort(channel_num, order=['time', 'channel'])
        nums = None if self.num is None else self.num[sort_mask]
        return Waveforms([self.waveforms[i] for i in sort_mask], self.time_stamp[sort_mask], self.channel[sort_mask], nums)


@dataclass
class MeasuredWaveform:
    voltage: np.ndarray = Descriptor.numpy_dim(1)
    current: np.ndarray = Descriptor.numpy_dim(1)
    ground_current: np.ndarray = Descriptor.numpy_dim(1)
    pulse_generator: np.ndarray = Descriptor.numpy_dim(1)
    time: np.ndarray = Descriptor.numpy_dim(1)

    def power(self):
        return self.voltage * self.current

    def pulse_energy(self):
        return np.trapz(self.power(), self.time)

    def ground_power(self):
        return self.voltage * self.ground_current

    def ground_pulse_energy(self):
        return np.trapz(self.ground_power(), self.time)


@dataclass
class MeasuredWaveforms:
    voltages: np.ndarray = Descriptor.numpy_dim(2)
    currents: np.ndarray = Descriptor.numpy_dim(2)
    ground_currents: np.ndarray = Descriptor.numpy_dim(2)
    pulse_generators: np.ndarray = Descriptor.numpy_dim(2)
    time: np.ndarray = Descriptor.numpy_dim(2)
    time_offset: np.ndarray = Descriptor.numpy_dim(1)

    def __post_init__(self):
        if self.voltages.shape != self.currents.shape or self.voltages.shape != self.ground_currents.shape or self.voltages.shape != self.pulse_generators.shape:
            raise ValueError('All arrays should have the same shape')

    def power(self):
        return self.voltages * self.currents

    def pulse_energy(self):
        bit_depth_voltage = np.min(np.diff(self.voltages))
        bit_depth_current = np.min(np.diff(self.currents))

        measurements = np.zeros(self.voltages.shape[0])
        for index, measurement in enumerate(self.power()):
            mask = (np.abs(self.voltages[index]) > 1.5*bit_depth_voltage) & (np.abs(self.currents[index]) > 1.5*bit_depth_current)
            measurements[index] = np.trapz(measurement[mask], self.time[index][mask])
        return measurements

    def get_average(self, start_index, end_index):
        return MeasuredWaveform(np.mean(self.voltages[start_index:end_index], axis=0), np.mean(self.currents[start_index:end_index], axis=0),
                                np.mean(self.ground_currents[start_index:end_index], axis=0), np.mean(self.pulse_generators[start_index:end_index], axis=0),
                                np.mean(self.time[start_index:end_index], axis=0))

    @staticmethod
    def _fit_voltage(times, voltage):
        model = lmfit.models.RectangleModel(form='linear')

        rise_time = 0.25e-7

        high = voltage > 0.5 * np.max(voltage)
        up = np.where(np.diff(high.astype(int)) == 1)[0][0]
        down = np.where(np.diff(high.astype(int)) == -1)[0]
        if high[down[0] + 2]:
            down = down[1]
        else:
            down = down[0]
        high_time = times[down] - times[up] - rise_time
        start_offset = times[up] - 0.5 * rise_time

        params = model.guess(voltage, x=times)

        params['width'] = lmfit.Parameter('width', value=high_time + rise_time)

        params['center1'].value = start_offset - 0.5 * rise_time
        params['center1'].min = 0
        params['center2'].expr = 'center1 + width'

        params['amplitude'].value = 0.9 * np.max(voltage)
        params['amplitude'].max = np.max(voltage)

        params['sigma1'].value = rise_time
        params['sigma2'].expr = 'sigma1'

        return model.fit(voltage, x=times, params=params)

    def is_on(self, threshold=0.8):
        return np.any(self.voltages > threshold*np.max(self.voltages), axis=1)

    def fit_voltage(self):
        rise_times = np.zeros(self.voltages.shape[0])
        high_times = np.zeros(self.voltages.shape[0])
        heights = np.zeros(self.voltages.shape[0])

        for index, voltage in enumerate(self.voltages):
            if np.max(voltage) < 0.2*np.max(self.voltages):
                rise_times[index] = np.nan
                high_times[index] = np.nan
                heights[index] = np.nan
                continue

            fit = self._fit_voltage(self.time[index], voltage)
            rise_times[index] = fit.best_values['sigma1']
            high_times[index] = fit.best_values['center2'] - fit.best_values['center1'] - fit.best_values['sigma1']
            heights[index] = fit.best_values['amplitude']
        return rise_times, high_times, heights

    def background_current_averaging(self, pulse_length_us, upper_offset=1e-7):
        avg_time = np.average(self.time, axis=0)
        avg_curr = np.average(self.currents, axis=0)

        upper_bound = avg_time[np.argmin(avg_curr)] - upper_offset
        lower_bound = upper_bound - 1e-6*pulse_length_us / 3
        mask = (avg_time > lower_bound) & (avg_time < upper_bound)

        avg = np.average(self.currents[:, mask], axis=1)
        std = np.std(self.currents[:, mask], axis=1)
        return avg, std

    @staticmethod
    def _background_current_fitting_model(x, amplitude, decay, phase, length, offset):
        return amplitude * np.exp(-(x - x[0]) / decay) * np.sin((2 * np.pi * x) / length + phase) + offset

    def background_current_fitting(self, pulse_length_us, is_on_kwargs=None, save_loc=None, block_average=20):
        model = self._background_current_fitting_model

        lmfit_model = lmfit.Model(model)
        lmfit_model.set_param_hint('amplitude', value=0.25, min=0)
        lmfit_model.set_param_hint('decay', value=2e-6)
        lmfit_model.set_param_hint('phase', value=-1.5)
        lmfit_model.set_param_hint('length', value=1.9e-7)
        lmfit_model.set_param_hint('offset', value=0.8)

        is_on_kwargs = is_on_kwargs or {}
        is_on = self.is_on(**is_on_kwargs)

        n = block_average
        avg_current = npf.block_average(self.currents[is_on], n)
        avg_time = npf.block_average(self.time[is_on], n)
        avg_offset = npf.block_average(self.time_offset[is_on], n)
        avg_offset = avg_offset - avg_offset[0]

        pulse_len = pulse_length_us * 1e-6
        params = lmfit_model.make_params()
        back_curr_results = np.zeros_like(avg_offset)
        back_curr_std = np.zeros_like(avg_offset)

        for index, (tim, cur) in enumerate(zip(avg_time, avg_current)):
            upper_bound = tim[np.argmin(cur)] - 1e-7
            lower_bound = upper_bound - pulse_len / 3
            mask = (tim > lower_bound) & (tim < upper_bound)
            result = lmfit_model.fit(cur[mask], x=tim[mask], params=params)

            if save_loc:
                t_save_loc = save_loc + f'{index}.pdf'
                plot_kwargs = {'xlabel': 'Time [s]', 'ylabel': 'Current [A]'}
                fig_ax = plot.lines(tim[mask], cur[mask], save_loc=t_save_loc, plot_kwargs=plot_kwargs, show=False, close=False)
                plot.lines(tim[mask], result.best_fit, fig_ax=fig_ax, show=False, close=True, save_loc=t_save_loc)
            back_curr_results[index] = result.params['offset'].value
            back_curr_std[index] = result.params['offset'].stderr
            if (back_curr_std[index] is not None) and back_curr_std[index] > 10:
                back_curr_results[index] = np.nan
                back_curr_std[index] = np.nan

            params = result.params

        return avg_offset, back_curr_results, back_curr_std


    def ground_power(self):
        return self.voltages * self.ground_currents

    def ground_pulse_energy(self):
        return np.trapz(self.ground_power(), self.time, axis=1)

    def plot(self, block_average=None, running_average=None, plot_kwargs=None, save_loc=None, close=False):
        times = self.time - self.time[:, 0][:, None]
        times = npf.averaging(times, block_average, running_average)

        voltages = npf.averaging(self.voltages, block_average, running_average)
        currents = npf.averaging(self.currents, block_average, running_average)
        ground_currents = npf.averaging(self.ground_currents, block_average, running_average)
        pulse_generators = npf.averaging(self.pulse_generators, block_average, running_average)

        rise_time, pulse_length, height = self.fit_voltage()

        colors, mappable = cbar.cbar_norm_colors(self.time_offset, 'turbo')
        cbar_kwargs = {'label': 'Time [s]', 'mappable': mappable}
        xmax = max(np.nanmax(2*pulse_length), np.max(times))

        t_save_loc = save_loc + 'voltage.pdf' if save_loc is not None else None
        t_plot_kwargs = plot.set_defaults(plot_kwargs, **{'xlabel': 'Time [s]', 'ylabel': 'Amplitude [V]', 'title': 'voltages',
                                                          'xlim': (0, xmax)})
        plot.lines(times, voltages, plot_kwargs=t_plot_kwargs, cbar_kwargs=cbar_kwargs, colors=colors, save_loc=t_save_loc, close=close, show=not close)

        t_save_loc = save_loc + 'current.pdf' if save_loc is not None else None
        t_plot_kwargs = plot.set_defaults(plot_kwargs, **{'xlabel': 'Time [s]', 'ylabel': 'Amplitude [A]', 'title': 'currents',
                                                          'xlim': (0, xmax)})
        plot.lines(times, currents, plot_kwargs=t_plot_kwargs, cbar_kwargs=cbar_kwargs, colors=colors, save_loc=t_save_loc, close=close, show=not close)

        t_save_loc = save_loc + 'ground_current.pdf' if save_loc is not None else None
        t_plot_kwargs = plot.set_defaults(plot_kwargs, **{'xlabel': 'Time [s]', 'ylabel': 'Amplitude [A]', 'title': 'ground currents',
                                                          'xlim': (0, xmax)})
        plot.lines(times, ground_currents, plot_kwargs=t_plot_kwargs, cbar_kwargs=cbar_kwargs, colors=colors, save_loc=t_save_loc, close=close, show=not close)

        t_save_loc = save_loc + 'pulse_generator.pdf' if save_loc is not None else None
        t_plot_kwargs = plot.set_defaults(plot_kwargs, **{'xlabel': 'Time [s]', 'ylabel': 'Amplitude [V]', 'title': 'pulse generated',
                                                          'xlim': (0, xmax)})
        plot.lines(times, pulse_generators, plot_kwargs=t_plot_kwargs, cbar_kwargs=cbar_kwargs, colors=colors, save_loc=t_save_loc, close=close, show=not close)
        # t_plot_kwargs = plot.set_defaults(plot_kwargs, **{'xlabel': 'Time [s]', 'ylabel': 'Pulse energy [J]', 'title': 'pulse energy'})

        t_save_loc = save_loc + 'power.pdf' if save_loc is not None else None
        t_plot_kwargs = plot.set_defaults(plot_kwargs, **{'xlabel': 'Time [s]', 'ylabel': 'Power [W]'})
        plot.lines(times, self.power(), plot_kwargs=t_plot_kwargs, cbar_kwargs=cbar_kwargs, colors=colors, save_loc=t_save_loc, close=close, show=not close)

        t_save_loc = save_loc + 'pulse_energy.pdf' if save_loc is not None else None
        plot.lines(self.time_offset, self.pulse_energy(), plot_kwargs={'xlabel': 'Time [s]', 'ylabel': 'Pulse energy [J]', 'title': 'pulse energy'}, save_loc=t_save_loc, close=close, show=not close)

        start = 0 if height[0] > 0.9*height[-2] else 1
        end = len(height) if height[-1] > 0.9*height[-2] else -1

        t_save_loc = save_loc + 'rise_time.pdf' if save_loc is not None else None
        plot.lines(self.time_offset[start:end], rise_time[start:end], plot_kwargs={'xlabel': 'Time [s]', 'ylabel': 'Rise time [s]', 'title': 'Rise time'}, save_loc=t_save_loc, close=close, show=not close)
        t_save_loc = save_loc + 'pulse_length.pdf' if save_loc is not None else None
        plot.lines(self.time_offset[start:end], pulse_length[start:end], plot_kwargs={'xlabel': 'Time [s]', 'ylabel': 'Pulse length [s]', 'title': 'Pulse length'}, save_loc=t_save_loc, close=close, show=not close)
        t_save_loc = save_loc + 'height.pdf' if save_loc is not None else None
        plot.lines(self.time_offset[start:end], height[start:end], plot_kwargs={'xlabel': 'Time [s]', 'ylabel': 'Height [V]', 'title': 'Height'}, save_loc=t_save_loc, close=close, show=not close)

    @staticmethod
    def from_waveforms(waveforms: Waveforms, channels: dict[str, str]):
        waveforms = waveforms.sorted()
        num_channels = np.unique(waveforms.channel).size

        channels_template = [f'C{i}' for i in range(1, num_channels+1)]

        # wavs = np.array([wav.waveform for wav in waveforms.waveforms])
        # tims = np.array([wav.time for wav in waveforms.waveforms])
        # chan_vals = np.array([int(chan[1]) for chan in waveforms.channel])
        # time_stamps = waveforms.time_stamp

        index = 0
        while index < len(waveforms.channel):
            # if index+num_channels > len(waveforms.channel):
            #     wavs = wavs[:index]
            #     chan_vals = chan_vals[:index]
            #     time_stamps = time_stamps[:index]
            #     tims = tims[:index]
            #     break

            if np.any(waveforms.channel[index:index + num_channels] != channels_template):
                raise ValueError('Channels are not in the correct order')
            index += 4

        values = np.array([len(wav.waveform) for wav in waveforms.waveforms])
        mask = np.ones(len(values), dtype=bool)
        if not np.all(values[0] == values):
            keep_length = np.argmax(np.bincount(values))
            mask = values == keep_length

            mask = mask.reshape(-1, num_channels)
            mask = np.all(mask, axis=1)
            mask = np.repeat(mask, num_channels)
            warnings.warn(f'Waveforms are not of equal length, removing {np.sum(~mask)} of {len(mask)},'
                          f'keeping length {keep_length}')

        input_channels = np.array([int(chan[1]) for chan in waveforms.channel])[mask].reshape(-1, num_channels)
        time_stamps = waveforms.time_stamp[mask].reshape(-1, num_channels)
        wavs = np.array([wav.waveform for index, wav in enumerate(waveforms.waveforms) if mask[index]])
        tims = np.array([wav.time for index, wav in enumerate(waveforms.waveforms) if mask[index]])

        # length = len(waveforms.waveforms[0].waveform)
        # for i in range(1, len(waveforms.waveforms)):
        #     if len(waveforms.waveforms[i].waveform) != length:
        #         raise ValueError('Waveforms are not of equal length')
        wavs = wavs.reshape(*time_stamps.shape, -1)
        times = tims.reshape(*time_stamps.shape, -1)

        # if np.sum((time_stamps-time_stamps[:, 0][:, None]).sum(axis=1) == 0) > 0:
        #     warnings.warn(f'Time stamps are not equal for all channels, removing offset ones. Removed {np.sum(~mask)} of {len(mask)}')

        voltages = np.empty((wavs.shape[0], wavs.shape[2]))
        currents = np.empty(voltages.shape)
        ground_currents = np.empty(voltages.shape)
        pulse_generators = np.empty(voltages.shape)

        i = 0
        for channel_val in input_channels:
            for j, channel in enumerate(channel_val):
                if channels[channel] == 'voltage':
                    voltages[i] = wavs[i, j]
                elif channels[channel] == 'current':
                    currents[i] = wavs[i, j]
                elif channels[channel] == 'ground_current':
                    ground_currents[i] = wavs[i, j]
                elif channels[channel] == 'pulse_generator':
                    pulse_generators[i] = wavs[i, j]
            i += 1

        return MeasuredWaveforms(voltages=voltages, currents=currents, ground_currents=ground_currents,
                                 pulse_generators=pulse_generators, time=times[:, 0], time_offset=time_stamps[:, 0])


def func_trapezoid(t, start_offset, rise_time, high_time, height):
    mask1 = t < start_offset
    mask2 = (t >= start_offset) & (t < start_offset + rise_time)
    mask3 = (t >= start_offset + rise_time) & (t < start_offset + rise_time + high_time)
    mask4 = (t >= start_offset + rise_time + high_time) & (t < start_offset + rise_time + high_time + rise_time)
    mask5 = t >= start_offset + rise_time + high_time + rise_time
    return np.concatenate([np.zeros(mask1.sum()), np.linspace(0, height, mask2.sum()), np.full(mask3.sum(), height),
                           np.linspace(height, 0, mask4.sum()), np.zeros(mask5.sum())])
