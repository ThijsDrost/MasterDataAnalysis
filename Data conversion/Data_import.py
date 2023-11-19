import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import pandas as pd


class Measurements:
    def __init__(self, loc):
        with h5py.File(loc, 'r') as file:
            for measurement in file.keys():
                measurements = []
                for key in file[measurement].keys():
                    measurements.append(file[measurement][key][()])
                measurements = np.array(measurements)

class Measurement:
    def __init__(self, wavelength, data, *, dark=None, reference=None, integration_time=None, smoothing=None, spectrometer=None):
        self.wavelength = wavelength
        self.data = data
        self.dark = dark
        self.reference = reference
        self.integration_time = integration_time
        self.smoothing = smoothing
        self.spectrometer = spectrometer

    @staticmethod
    def from_hdf5(loc):
        with h5py.File(loc, 'r') as file:
            measurements = []
            for measurement in file.keys():
                wavelength = file[measurement].attrs['wavelength']

                data = file[measurement][()]
                measurements = np.array(measurements)




