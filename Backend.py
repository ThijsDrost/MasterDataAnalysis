import numpy as np


def read_txt(filename):
    with open(filename, 'r') as file:
        read_lines = file.readlines()
    wavelength = np.array([float(line.split(';')[0].replace(',', '.')) for line in read_lines[8:-1]])
    intensity = np.array([float(line.split(';')[1].replace(',', '.')) for line in read_lines[8:-1]])
    integration_time = float(read_lines[1].split(':')[1].strip().replace(',', '.'))

    return {'wavelength': wavelength, 'intensity': intensity, "integration_time": integration_time}