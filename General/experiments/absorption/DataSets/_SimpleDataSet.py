"""
This module contains the SimpleDataSet class, which is a simple data structure for storing spectral data.
"""

import numpy as np
import h5py


class SimpleDataSet:
    def __init__(self, wavelength, absorbances, variable, measurement_num, variable_name):
        self.wavelength = wavelength
        self.absorbances = absorbances
        self.variable = variable
        self.measurement_num = measurement_num
        self.variable_name = variable_name

        sort_mask = np.argsort(self.variable)
        self.absorbances = self.absorbances[sort_mask]
        self.variable = self.variable[sort_mask]
        self.measurement_num = self.measurement_num[sort_mask]

        if not len(self.wavelength) == self.absorbances.shape[1]:
            raise ValueError("Wavelength and absorbances don't match")
        if not len(self.variable) == self.absorbances.shape[0]:
            raise ValueError("Variable and absorbances don't match")
        if not len(self.measurement_num) == self.absorbances.shape[0]:
            raise ValueError("Measurement number and absorbances don't match")

    def __add__(self, other):
        if not isinstance(other, SimpleDataSet):
            raise TypeError(f"Can't add {type(other)} to SimpleDataSet")
        if not np.all(self.wavelength == other.wavelength):
            raise ValueError("Wavelengths don't match")
        if not self.variable_name == other.variable_name:
            raise ValueError("Variables don't match")
        return SimpleDataSet(self.wavelength, np.concatenate((self.absorbances, other.absorbances)),
                             np.concatenate((self.variable, other.variable)),
                             np.concatenate((self.measurement_num, other.measurement_num)),
                             self.variable_name)

    def __getitem__(self, item):
        return SimpleDataSet(self.wavelength, self.absorbances[item], self.variable[item], self.measurement_num[item],
                             self.variable_name)

    def __len__(self):
        return len(self.variable)


def import_hdf5(loc, dependent):
    with h5py.File(loc, 'r') as file:
        has_group = False
        has_dataset = False
        for key in file.keys():
            if isinstance(file[key], h5py.Group):
                has_group = True
            elif isinstance(file[key], h5py.Dataset):
                has_dataset = True
            else:
                raise TypeError(f'Unknown type: {type(file[key])}')

        if has_group:
            raise NotImplementedError('Groups are not implemented.')
        elif has_dataset:
            return import_datasets(file, dependent)
        else:
            raise ValueError(f'File {loc} contains neither groups nor datasets.')


def import_datasets(h5py_file, variable_name):
    wavelength = h5py_file.attrs['wavelength'][:]
    absorbance = []
    variable = []
    number = []
    for key in h5py_file.keys():
        absorbance.append(h5py_file[key][:])
        if variable_name is not None:
            variable.append(h5py_file[key].attrs[variable_name])
        else:
            variable.append(0)
        try:
            number.append(int(key.split('_')[1].split('.')[0]))
        except IndexError:
            number.append(-1)
    absorbance = np.array(absorbance)
    variable = np.array(variable)
    number = np.array(number)
    return SimpleDataSet(wavelength, absorbance, variable, number, variable_name)