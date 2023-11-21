import os
import itertools
import warnings

import numpy as np
import h5py


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
            self._absorbance_best_num[i] = (self.absorbances_masked_corrected[variable == v][min_num[0]]
                                            + self.absorbances_masked_corrected[variable == v][min_num[1]]) / 2
            self._variable_best_num[i] = v

    @staticmethod
    def from_simple(simple_data_set, wavelength_range, baseline_correction, selected_num):
        return DataSet(simple_data_set.wavelength, simple_data_set.absorbances, simple_data_set.variable,
                       simple_data_set.measurement_num, simple_data_set.variable_name, wavelength_range,
                       baseline_correction, selected_num)

    @property
    def absorbances_masked(self):
        return self.absorbances[:, self._mask]

    @property
    def wavelength_masked(self):
        return self.wavelength[self._mask]

    @property
    def absorbances_corrected(self):
        return self.absorbances - self._baseline_correction

    @property
    def absorbances_masked_corrected(self):
        return self.absorbances_masked - self._baseline_correction

    @property
    def absorbances_masked_corrected_num(self):
        return self.absorbances_masked_corrected[self.measurement_num == self._selected_num]

    @property
    def absorbances_masked_num(self):
        return self.absorbances_masked[self.measurement_num == self._selected_num]

    @property
    def variable_num(self):
        return self.variable[self.measurement_num == self._selected_num]

    def absorbances_masked_corrected_get_num(self, num):
        return self.absorbances_masked_corrected[self.measurement_num == num]

    def absorbances_masked_get_num(self, num):
        return self.absorbances_masked[self.measurement_num == num]

    def variable_get_num(self, num):
        return self.variable[self.measurement_num == num]

    @property
    def absorbances_masked_best_num(self):
        return self._absorbance_best_num

    @property
    def variable_best_num(self):
        return self._variable_best_num

