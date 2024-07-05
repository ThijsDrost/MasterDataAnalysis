from dataclasses import dataclass, field
import pathlib
import queue
import time
from datetime import datetime
import threading

import serial.tools.list_ports
from lxml import objectify
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as plt_animation
import h5py


def main():
    global ani
    out_loc = 'Somewhere/conductivity.hdf5'
    measurements = {
        'conductivity': 'ResultOrp',
        # 'pH': 'ResultPh',
    }

    # measurering = Measure(measurements, out_loc)
    q = queue.Queue()

    read_thread = threading.Thread(target=read, args=(q,))
    read_thread.start()

    keys = list(measurements.keys())
    fig, axis = plt.subplots(ncols=len(keys)+1, figsize=(6 * (len(keys)+1), 4))
    lines = [axis[index].plot([], [], f'C{index}', label=key)[0] for index, key in enumerate(keys + ['temperature'])]
    for ax, title in zip(axis, keys + ['temperature']):
        ax.set_title(title)
        ax.set_xlabel('Time [min]')
        ax.grid()

    def update(frame):
        data = q.get()
        if data:
            for index, key in enumerate(keys):
                time_data = data[key].time
                time_data = [(time_dat - time_data[0])/60 for time_dat in time_data]
                value_data = data[key].value

                lines[index].set_xdata(time_data)
                lines[index].set_ydata(value_data)

                if len(time_data) > 1:
                    axis[index].set_xlim(min(time_data), max(time_data))
                    axis[index].set_ylim(min(value_data) - 0.1, max(value_data) + 0.1)

            time_data = data[keys[0]].time
            time_data = [(time_dat - time_data[0])/60 for time_dat in time_data]
            temp_data = data[keys[0]].temperature

            lines[-1].set_xdata(time_data)
            lines[-1].set_ydata(temp_data)
            if len(time_data) > 1:
                axis[-1].set_xlim(min(time_data), max(time_data))
                axis[-1].set_ylim(min(temp_data) - 0.1, max(temp_data) + 0.1)

            return lines

    ani = plt_animation.FuncAnimation(fig, update, frames=1, interval=1_000, blit=False)
    plt.tight_layout()
    fig.show()


def read(q: queue.Queue):
    results = {
        'conductivity': Result(),
        # 'pH': Result(),
    }
    start_time = time.time()
    while True:
        # new_data = measurering.read_serial()

        results['conductivity'].append(np.sin(time.time()/5)+(time.time()-start_time)/10, 20.1, time.time() - start_time)
        q.put(results.copy())
        time.sleep(1)


@dataclass
class Result:
    value: list[float] = field(default_factory=list)
    temperature: list[float] = field(default_factory=list)
    time: list[float] = field(default_factory=list)

    def append(self, value, temperature, time):
        self.value.append(value)
        self.temperature.append(temperature)
        self.time.append(time)

    def extend(self, value, temperature, time):
        if len(value) != len(temperature) != len(time):
            raise ValueError('All lists should have the same length')
        self.value.extend(value)
        self.temperature.extend(temperature)
        self.time.extend(time)

    def to_numpy(self):
        return np.array(self.value), np.array(self.temperature), np.array(self.time)

    def copy(self):
        return Result(self.value.copy(), self.temperature.copy(), self.time.copy())

    def __getitem__(self, item):
        return self.value[item], self.temperature[item], self.time[item]

    def __len__(self):
        return len(self.value)


class Measure:
    def __init__(self, measurements, h5py_loc):
        self.h5py_loc = pathlib.Path(h5py_loc)
        if not self.h5py_loc.suffix == '.hdf5':
            raise ValueError('File should be of type .hdf5')
        if self.h5py_loc.exists():
            raise FileExistsError(f'{h5py_loc} already exists')
        if not self.h5py_loc.parent.exists():
            raise FileNotFoundError(f'Parent directory of {h5py_loc} does not exist')

        with h5py.File(self.h5py_loc, 'w-') as h5py_file:
            for key in measurements:
                group = h5py_file.create_group(key)
                group.create_dataset('value', [0,], maxshape=(None,), dtype=float)
                group.create_dataset('temperature', [0,], maxshape=(None,), dtype=float)
                group.create_dataset('time', [0,], maxshape=(None,), dtype=float)

        self.measurements = measurements
        self.data: dict[str, Result] = {key: Result() for key in measurements.keys()}

        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "SevenExcellence" in p[1]:
                comport = p[0]
                break
        else:
            raise ValueError('SevenExcellence not found')

        try:
            self.serial_port = serial.Serial(
                port=comport,
                baudrate=9600,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=30)
            print("connected to: " + self.serial_port.portstr)
        except Exception as e:
            raise ConnectionError('Unsuccessful connection to SevenExcellence!') from e

    def read_serial(self):
        s = self.serial_port.readline().decode('utf-8')
        if not s:
            print('No data received within timeout')
            return

        xml_object = objectify.fromstring(s)
        result = xml_object.ResultMessage.result

        for key, value in self.measurements.items():
            subresult = getattr(result, value)
            timestamp = datetime.strftime(subresult.timeStamp, '%Y-%m-%dT%H:%M:%S.%f')
            self.data[key].append(float(subresult.resultValue), float(subresult.rawTemperature), timestamp)

        with h5py.File(self.h5py_loc, 'a') as f:
            for key, result in self.data.items():
                group = f[key]
                group['value'].resize((len(result),))
                group['value'][-1] = result.value[-1]

                group['temperature'].resize((len(result),))
                group['temperature'][-1] = result.temperature[-1]

                group['time'].resize((len(result),))
                group['time'][-1] = result.time[-1]

        return self.data


if __name__ == '__main__':
    main()
