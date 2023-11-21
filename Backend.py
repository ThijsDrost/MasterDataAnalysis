from dataclasses import dataclass
import os
from datetime import datetime

import numpy as np

@dataclass
class ImportedSpectrum:
    wavelength: np.ndarray
    intensity: np.ndarray
    integration_time: float
    averaging: int
    smoothing: int
    spectrometer: str

    def __eq__(self, other):
        return (self.integration_time == other.integration_time and self.smoothing == other.smoothing
                and self.spectrometer == other.spectrometer)

    def give_diff(self, other):
        if self.integration_time != other.integration_time:
            return f"integration time: {self.integration_time} vs {other.integration_time}"
        if self.smoothing != other.smoothing:
            return f"smoothing: {self.smoothing} vs {other.smoothing}"
        if self.spectrometer != other.spectrometer:
            return f"spectrometer: {self.spectrometer} vs {other.spectrometer}"


def read_txt(filename):
    with open(filename, 'r') as file:
        read_lines = file.readlines()
    wavelength = np.array([float(line.split(';')[0].replace(',', '.')) for line in read_lines[8:-1]])
    intensity = np.array([float(line.split(';')[1].replace(',', '.')) for line in read_lines[8:-1]])
    integration_time = float(read_lines[1].split(':')[1].strip().replace(',', '.'))
    averaging = int(read_lines[2].split(':')[1].strip().replace(',', '.'))
    smoothing = int(read_lines[3].split(':')[1].strip().replace(',', '.'))
    spectrometer = read_lines[4].split(':')[1].strip().replace(',', '.')

    # return {'wavelength': wavelength, 'intensity': intensity, "integration_time": integration_time,
    #         "averaging": averaging, "smoothing": smoothing, "spectrometer": spectrometer}
    return ImportedSpectrum(wavelength, intensity, integration_time, averaging, smoothing, spectrometer)


def get_time(file: os.path):
    hh = int(file.name.split('_')[1][0:2])
    mm = int(file.name.split('_')[1][2:4])
    ss = int(file.name.split('_')[1][4:6])
    m_timestamp = datetime.fromtimestamp(os.path.getmtime(file.path))
    year, month, day = m_timestamp.year, m_timestamp.month, m_timestamp.day
    return datetime(year, month, day, hh, mm, ss).timestamp()

def get_time_rel(file: os.path):
    hh = int(file.name.split('_')[1][0:2])
    mm = int(file.name.split('_')[1][2:4])
    ss = int(file.name.split('_')[1][4:6])
    return hh*3600 + mm*60 + ss
