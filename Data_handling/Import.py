import os
import itertools
import warnings
from typing import Union

import numpy as np
import h5py
import scipy
import lmfit


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

        if has_group and has_dataset:
            warnings.warn(f'File {loc} contains both groups and datasets. Only datasets will be imported.')
        elif (not has_group) and (not has_dataset):
            raise ValueError(f'File {loc} contains neither groups nor datasets.')

        if has_dataset:
            return import_datasets(file, dependent)
        else:
            results = []
            for group in file.keys():
                if isinstance(file[group], h5py.Group):
                    raise NotImplementedError('Nested groups are not implemented.')
                results.append(import_datasets(file[group], dependent))
            return results


def import_datasets(h5py_file, variable_name):
    wavelength = h5py_file.attrs['wavelength'][:]
    absorbance = []
    variable = []
    number = []
    for key in h5py_file.keys():
        absorbance.append(h5py_file[key][:])
        variable.append(h5py_file[key].attrs[variable_name])
        number.append(int(key.split('_')[1].split('.')[0]))
    absorbance = np.array(absorbance)
    variable = np.array(variable)
    number = np.array(number)
    return SimpleDataSet(wavelength, absorbance, variable, number, variable_name)


class SimpleDataSet:
    def __init__(self, wavelength, absorbances, variable, measurement_num, variable_name):
        self.wavelength = wavelength
        self.absorbances = absorbances
        self.variable = variable
        self.measurement_num = measurement_num
        self.variable_name = variable_name
        if not len(self.wavelength) == self.absorbances.shape[1]:
            raise ValueError("Wavelength and absorbances don't match")
        if not len(self.variable) == self.absorbances.shape[0]:
            raise ValueError("Variable and absorbances don't match")
        if not len(self.measurement_num) == self.absorbances.shape[0]:
            raise ValueError("Measurement number and absorbances don't match")

    def __add__(self, other):
        if not isinstance(other, SimpleDataSet):
            raise TypeError(f"Can't add {type(other)} to SimpleDataSet")
        if not self.wavelength == other.wavelength:
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


class DataSet(SimpleDataSet):
    def __init__(self, wavelength, absorbances, variable, measurement_num, variable_name, wavelength_range,
                 baseline_correction, selected_num):
        super().__init__(wavelength, absorbances, variable, measurement_num, variable_name)
        self.wavelength_range = wavelength_range
        self.baseline_correction = baseline_correction
        self._selected_num = selected_num
        if wavelength_range is not None:
            self._mask = (wavelength_range[0] < wavelength) & (wavelength < wavelength_range[1])
        if baseline_correction is not None:
            correction_mask = (baseline_correction[0] < wavelength) & (wavelength < baseline_correction[1])
            self._baseline_correction = np.mean(absorbances[:, correction_mask], axis=1)[:, np.newaxis]

        self._absorbance_best_num = np.zeros((len(np.unique(variable)), np.sum(self._mask.astype(int))))
        self._variable_best_num = np.zeros(len(np.unique(variable)))
        for i, v in enumerate(np.unique(variable)):
            # v_absorbances = absorbances[:, self._mask][variable == v]
            v_absorbances = self.absorbances_masked_corrected[variable == v]
            value = []
            pair_values = []
            for pair in itertools.combinations(range(len(v_absorbances)), 2):
                mask = v_absorbances[0] > 0.5 * np.max(v_absorbances[0])
                value.append(np.sum((v_absorbances[pair[0]][mask] - v_absorbances[pair[1]][mask]) ** 2))
                pair_values.append(pair)
            min_num = pair_values[np.argmin(value)]
            # self._absorbance_best_num[i] = (absorbances[:, self._mask][variable == v][min_num[0]] + absorbances[:, self._mask][variable == v][
            #     min_num[1]]) / 2
            self._absorbance_best_num[i] = (self.absorbances[variable == v][min_num[0]]
                                            + self.absorbances[variable == v][min_num[1]]) / 2
            self._variable_best_num[i] = v

    @staticmethod
    def from_simple(simple_data_set, wavelength_range, baseline_correction, selected_num):
        return DataSet(simple_data_set.wavelength, simple_data_set.absorbances, simple_data_set.variable,
                       simple_data_set.measurement_num, simple_data_set.variable_name, wavelength_range,
                       baseline_correction, selected_num)

    def get_absorbances(self, corrected=True, masked=True, num: Union[str, int, None] = 'all', var_value=None):
        correction = 0
        if corrected:
            correction = self.baseline_correction

        wav_mask = np.ones(len(self._mask), dtype=bool)
        if masked:
            wav_mask = self._mask

        if var_value is not None:
            if var_value not in self.variable:
                raise ValueError(f'Variable value {var_value} not in dataset')
            vn_mask = self.variable == var_value
        else:
            vn_mask = np.ones(len(self.measurement_num), dtype=bool)

        absorbances = self.absorbances
        if (num is None) or num == 'all':
            pass
        elif num == 'plot':
            vn_mask = vn_mask & (self.measurement_num == self._selected_num)
        elif num == 'best':
            absorbances = self._absorbance_best_num
            if var_value is not None:
                vn_mask = self.variable_best_num == var_value
        elif isinstance(num, int):
            vn_mask = vn_mask & (self.measurement_num == num)
        else:
            raise ValueError(f'num should be None, "all", "plot", "best" or an integer, not {num}')

        return absorbances[vn_mask][:, wav_mask] - correction

    @property
    def wavelength_masked(self):
        return self.wavelength[self._mask]

    @property
    def variable_num(self):
        return self.variable[self.measurement_num == self._selected_num]

    def variable_at_num(self, num):
        return self.variable[self.measurement_num == num]

    def measurement_num_at_value(self, value):
        return self.measurement_num[self.variable == value]

    @property
    def absorbances_masked(self):
        return self.get_absorbances(False, True, None)

    @property
    def absorbances_corrected(self):
        return self.get_absorbances(True, False, None)

    @property
    def absorbances_masked_corrected(self):
        return self.get_absorbances(True, True, None)

    @property
    def absorbances_masked_corrected_num(self):
        return self.get_absorbances(True, True, 'plot')

    @property
    def absorbances_num(self):
        return self.get_absorbances(False, False, 'plot')

    @property
    def absorbances_masked_num(self):
        return self.get_absorbances(False, True, 'plot')

    @property
    def absorbances_corrected_num(self):
        return self.get_absorbances(True, False, 'plot')

    def absorbances_masked_corrected_at_num(self, num):
        return self.get_absorbances(True, True, num)

    def absorbances_masked_at_num(self, num):
        return self.get_absorbances(False, True, num)

    def absorbances_at_num(self, num):
        return self.get_absorbances(False, False, num)

    def absorbances_corrected_at_num(self, num):
        return self.get_absorbances(True, False, num)

    @property
    def absorbances_masked_best_num(self):
        return self.get_absorbances(False, True, 'best')

    @property
    def absorbances_masked_corrected_best_num(self):
        return self.get_absorbances(True, True, 'best')

    @property
    def absorbances_corrected_best_num(self):
        return self.get_absorbances(True, False, 'best')

    @property
    def absorbances_best_num(self):
        return self.get_absorbances(False, False, 'best')

    @property
    def variable_best_num(self):
        return self._variable_best_num

    @property
    def simple(self):
        return SimpleDataSet(self.wavelength, self.absorbances, self.variable, self.measurement_num, self.variable_name)

    def add(self, other):
        if isinstance(other, (DataSet, SimpleDataSet)):
            if not self.wavelength == other.wavelength:
                raise ValueError("Wavelengths don't match")
            if not self.variable_name == other.variable_name:
                raise ValueError("Variables don't match")
            return DataSet(self.wavelength, np.concatenate((self.absorbances, other.absorbances)),
                           np.concatenate((self.variable, other.variable)),
                           np.concatenate((self.measurement_num, other.measurement_num)),
                           self.variable_name, self.wavelength_range, self.baseline_correction, self._selected_num)
        else:
            raise TypeError(f"Can't add {type(other)} to DataSet")


class InterpolationCalibrationSet(SimpleDataSet):
    def __init__(self, wavelength, absorbances, variable, variable_name, add_zero=True):
        if add_zero:
            absorbances = np.concatenate((np.zeros((1, absorbances.shape[1])), absorbances))
            variable = np.concatenate((np.zeros(1), variable))
        if len(np.unique(variable)) != len(variable):
            raise ValueError('Variable values must be unique')
        sort_index = np.argsort(variable)
        super().__init__(wavelength, absorbances[sort_index], variable[sort_index], np.zeros(len(variable)), variable_name)
        self._interpolator = scipy.interpolate.interp1d(variable, absorbances, axis=0, kind='linear')

    @staticmethod
    def from_simple(simple_data_set, add_zero=True):
        return InterpolationCalibrationSet(simple_data_set.wavelength, simple_data_set.absorbances, simple_data_set.variable,
                                           simple_data_set.variable_name, add_zero)

    def interpolate(self, variable_value):
        if variable_value < np.min(self.variable) or variable_value > np.max(self.variable):
            raise ValueError('Variable value out of range')
        if variable_value in self.variable:
            return self[self.variable == variable_value]
        else:
            return self._interpolate(variable_value)

    def _interpolate(self, variable_value):
        return self._interpolator(variable_value)

    def fit(self, absorbance):
        def residual(params, absorbance):
            variable = params[self.variable_name]
            return self.interpolate(variable) - absorbance

        params = lmfit.Parameters()
        params.add(self.variable_name, value=np.mean(self.variable), min=np.min(self.variable), max=np.max(self.variable))
        result = lmfit.minimize(residual, params, kws={'absorbance': absorbance})
        return result

    def model(self):
        interp_model = lmfit.Model(self.interpolate)
        interp_model.set_param_hint(self.variable_name, min=np.min(self.variable), max=np.max(self.variable), vary=True,
                                    value=np.mean(self.variable))
        return interp_model

