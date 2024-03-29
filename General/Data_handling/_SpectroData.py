"""
This module contains the SpectroData class which is used to store and handle spectral data.
"""

import struct
import os
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np
import pandas as pd


@dataclass
class SpectroData:
    wavelength: np.ndarray
    intensity: np.ndarray
    serial_number: str
    integration_time_ms: float
    n_averages: int
    n_smoothing: int
    time_ms: int
    relative_times_ms: np.ndarray = None

    def __post_init__(self):
        if not isinstance(self.wavelength, np.ndarray):
            self.wavelength = np.array(self.wavelength)
        if not isinstance(self.intensity, np.ndarray):
            self.intensity = np.array(self.intensity)

    def __eq__(self, other):
        return self.same_measurement(other)

    def same_measurement(self, measurement):
        if not isinstance(measurement, SpectroData):
            raise TypeError(f'`measurement` should be type SpectroData not {type(measurement)}')
        if not (self.serial_number == measurement.serial_number):
            return False
        if not math.isclose(self.integration_time_ms, measurement.integration_time_ms, rel_tol=1e-3):
            return False
        if not (self.n_smoothing == measurement.n_smoothing):
            return False
        if not np.all(np.isclose(self.wavelength, measurement.wavelength, rtol=1e-3)):
            return False
        return True

    def give_diff(self, other):
        message = ""
        if not math.isclose(self.integration_time_ms, other.integration_time_ms, rel_tol=1e-3):
            message += f"integration time: {self.integration_time_ms} vs {other.integration_time_ms}\n"
        if not np.all(np.isclose(self.wavelength, other.wavelength, rtol=1e-3)):
            message += f"wavelengths are different\n"
        if self.n_smoothing != other.n_smoothing:
            message += f"smoothing: {self.n_smoothing} vs {other.n_smoothing}\n"
        if self.serial_number != other.serial_number:
            message += f"spectrometer: {self.serial_number} vs {other.serial_number}\n"
        return message.removesuffix('\n')

    def get_intensity(self, multi_method='mean'):
        if self.intensity.ndim == 1:
            return self.intensity

        if multi_method in ('mean', 'average'):
            return np.mean(self.intensity, axis=1)
        elif multi_method in ('sum', 'total'):
            return np.sum(self.intensity, axis=1)
        elif multi_method == 'median':
            return np.median(self.intensity, axis=1)
        elif (multi_method is None) or (multi_method in ('none', 'raw', 'all')):
            return self.intensity
        else:
            raise ValueError(f'`multi_method` should be one of `mean`, `sum`, `median`, or `all`, not {multi_method}')

    @staticmethod
    def read_data(filename):
        if filename.lower().endswith('raw8'):
            return SpectroData.read_raw8(filename)
        elif filename.lower().endswith('txt'):
            return SpectroData.read_txt(filename)
        elif os.path.isdir(filename):
            return SpectroData.read_folder(filename)
        else:
            raise NotImplementedError(f'Extension `{filename.split('.')[-1]}` not implemented, only `.txt` and `.raw8` are supported.')

    @staticmethod
    def read_folder(filename):
        read_files = [SpectroData.read_data(file.path) for file in os.scandir(filename)]
        if len(read_files) == 0:
            raise FileNotFoundError(f'Folder {filename} does not contain any files')
        if not all([read_files[0].same_measurement(d) for d in read_files[1:]]):
            raise ValueError(f'Not all files in {filename} contain the same measurement')
        times = np.array([d.time_ms for d in read_files])
        times = times - times[0]
        intensities = np.array([d.get_intensity() for d in read_files]).T
        return SpectroData(read_files[0].wavelength, intensities, read_files[0].serial_number, read_files[0].integration_time_ms, read_files[0].n_averages,
                           read_files[0].n_smoothing, read_files[0].time_ms, times)

    @staticmethod
    def read_txt(filename):
        with open(filename, 'r') as file:
            # Old unknown legacy code
            if len(read_lines := file.readlines()) == 4:
                lines = [x.replace('\n', '') for x in read_lines]
                if len(lines) < 4:
                    return None
                return SpectroData(np.fromstring(lines[0], sep=' '), np.fromstring(lines[1], sep=' '), lines[2], int(lines[3]), 1, int(lines[4]), 0)
        return SpectroData._read_ava_txt(read_lines, filename)

    @staticmethod
    def _read_ava_txt(lines: list[str], filename):
        if lines[2].startswith('Nr. of StoreToRam scans'):  # Multi scan (store to RAM)
            return SpectroData._read_ava_multi_txt(lines, filename)
        elif lines[2].startswith('Averaging Nr.'):  # Single scan
            return SpectroData._read_ava_single_txt(lines, filename)
        else:
            raise ValueError(f'File {filename} does not contain the expected header')

    @staticmethod
    def _read_ava_multi_txt(lines: list[str], filename):
        serial_number = lines[4].split(':')[1].strip()
        integration_time = float(lines[1].split(':')[1].strip().replace(',', '.'))
        n_smoothing = int(lines[2].split(':')[1].strip())
        timestamp_ms = int(os.path.getmtime(filename) * 1000)
        timestamps_ms = np.array([float(x.replace(',', '.').strip())/100 for x in lines[9].split(';')[3:]])

        file = pd.read_csv(filename, sep=';', skiprows=9, decimal=',', index_col=0)
        spectra = file.drop([file.columns[0], file.columns[1]], axis=1)
        return SpectroData(spectra.index, spectra.to_numpy(), serial_number, integration_time, file.shape[1], n_smoothing, timestamp_ms, timestamps_ms)

    @staticmethod
    def _read_ava_single_txt(lines: list[str], filename):
        serial_number = lines[4].split(':')[1].strip()
        integration_time = float(lines[1].split(':')[1].strip().replace(',', '.'))
        n_averages = int(lines[2].split(':')[1].strip())
        n_smoothing = int(lines[3].split(':')[1].strip())
        timestamp_ms = int(os.path.getmtime(filename) * 1000)

        wavelength = np.array([float(line.split(';')[0].replace(',', '.')) for line in lines[8:-1]])
        intensity = np.array([float(line.split(';')[1].replace(',', '.')) for line in lines[8:-1]])

        return SpectroData(wavelength, intensity, serial_number, integration_time, n_averages, n_smoothing, timestamp_ms)

    @staticmethod
    def read_raw8(filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'FileError: File {filename} does not exist')

        h = {}
        data = Raw8(filename)
        h['wavelength'] = data.data['wl']
        h['intensity'] = data.data['scope']
        h['integration_time_ms'] = data.header['IntTime']
        h['serial_number'] = data.header['specID'].decode('utf-8')[:-1]
        h['n_averages'] = data.header['Avg']
        h['n_smoothing'] = data.header['Boxcar']
        # h['pixels'] = data.header['stopPixel'] - data.header['startPixel'] + 1
        # h['dark'] = np.array(data.data['dark'])
        # h['reference'] = np.array(data.data['ref'])
        h['time_ms'] = int(1000*data.header['timestamp'])
        return SpectroData(**h)


class Raw8:
    def __init__(self, filename: str):
        self.header = {}
        with open(filename, "rb") as f:
            for k in self._Raw8_Fields:
                s = struct.Struct(k[1])
                dat = s.unpack(f.read(s.size))
                if len(dat) == 1:
                    dat = dat[0]
                self.header[k[0]] = dat
            dataLength = self.header['stopPixel'] - self.header['startPixel'] + 1
            self.dataLenth = dataLength
            self.data = {
                'wl': struct.unpack(f"<{dataLength}f", f.read(4 * dataLength)),
                'scope': struct.unpack(f"<{dataLength}f", f.read(4 * dataLength)),
                'dark': struct.unpack(f"<{dataLength}f", f.read(4 * dataLength)),
                'ref': struct.unpack(f"<{dataLength}f", f.read(4 * dataLength))
            }
            self.header['timestamp'] = self.time_stamp(f)
        self.header['sec'] = datetime.fromtimestamp(os.path.getmtime(filename)).second
        self.header['micro_sec'] = datetime.fromtimestamp(os.path.getmtime(filename)).microsecond

    @staticmethod
    def time_stamp(file_handle):
        def hexer(val):
            value = hex(val)[2:]
            if len(value) == 1:
                value = '0' + value
            return value

        file_handle.seek(134)
        Y1 = ''.join([hexer(x) for x in struct.unpack('BBBB', file_handle.read(4))])
        file_handle.seek(327)
        Y2 = hex(struct.unpack('B', file_handle.read(1))[0])[2:]
        file_handle.seek(198)
        YY = int(Y1[6] + Y1[7] + Y1[4], 16)
        MM = int(Y1[5], 16)
        DD = int(Y1[2], 16) * 2 + int(Y1[4], 16) // 8
        hh = int(Y1[3], 16) % 8 * 4 + int(Y1[0], 16) // 4
        mm = int(str(int(Y1[0], 16) % 4) + Y1[1], 16)
        ss = int(Y2, 16)
        try:
            date_str = datetime(YY, MM, DD, hh, mm, ss).timestamp()
        except ValueError:
            date_str = os.path.getmtime(file_handle.name)
        return date_str

    _Raw8_Fields = \
        [("version", "5s"),
         ("numSpectra", "B"),
         ("length", "I"),
         ("seqNum", "B"),
         ("measMode", "B",
          {0: "scope", 1: "absorbance", 2: "scope corrected for dark", 3: "transmission", 4: "reflectance",
           5: "irradiance", 6: "relative irradiance", 7: "temperature"}),
         ("bitness", "B"),
         ("SDmarker", "B"),
         ("specID", "10s"),
         ("userfriendlyname", "64s"),
         ("status", "B"),
         ("startPixel", "H"),
         ("stopPixel", "H"),
         ("IntTime", "f"),
         ("integrationdelay", "I"),
         ("Avg", "I"),
         ("enable", "B"),
         ("forgetPercentage", "B"),
         ("Boxcar", "H"),
         ("smoothmodel", "B"),
         ("saturationdetection", "B"),
         ("TrigMode", "B"),
         ("TrigSource", "B"),
         ("TrigSourceType", "B"),
         ("strobeCtrl", "H"),
         ("laserDelay", "I"),
         ("laserWidth", "I"),
         ("laserWavelength", "f"),
         ("store2ram", "H"),
         ("timestamp", "I"),
         ("SPCfiledate", "I"),
         ("detectorTemp", "f"),
         ("boardTemp", "f"),
         ("NTC2volt", "f"),
         ("ColorTemp", "f"),
         ("CalIntTime", "f"),
         ("fitdata", "5d"),
         ("comment", "130s")
         ]


if __name__ == '__main__':
    loc = r'E:\OneDrive - TU Eindhoven\Thesis_spectro\Measurements'
    file = r'2203047U1_0002.Raw8'

    data = SpectroData.read_raw8(f'{loc}/{file}')
