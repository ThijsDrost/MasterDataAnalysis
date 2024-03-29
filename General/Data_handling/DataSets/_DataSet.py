"""
This submodule contains the DataSet class (child of SimpleDataSet) for storing spectral/absorbance data. The DataSet class can
also do some simple data cleaning.
"""

import itertools

import numpy as np

from ._SimpleDataSet import SimpleDataSet, import_hdf5


class DataSet(SimpleDataSet):
    _default_values = {
        'wavelength_range': (180, 400),
        'selected_num': 1,
        'baseline_correction': (450, 500)
    }

    def __init__(self, wavelength, absorbances, variable, measurement_num, variable_name, wavelength_range,
                 selected_num, baseline_correction=None, calc_best_num=True):
        SimpleDataSet.__init__(self, wavelength, absorbances, variable, measurement_num, variable_name)
        self.wavelength_range = wavelength_range
        self._selected_num = selected_num
        if wavelength_range is not None:
            self._mask = (wavelength_range[0] < wavelength) & (wavelength < wavelength_range[1])

        self.baseline_correction = baseline_correction or [wavelength[0], wavelength[-1]]
        correction_mask = (self.baseline_correction[0] < wavelength) & (wavelength < self.baseline_correction[1])
        self._baseline_correction = np.mean(absorbances[:, correction_mask], axis=1)[:, np.newaxis]
        self._calc_best_num = calc_best_num

        if calc_best_num:
            self._absorbance_best_num = np.zeros((len(np.unique(variable)), len(self.wavelength)))
            self._variable_best_num = np.zeros(len(np.unique(variable)))
            for i, v in enumerate(np.unique(variable)):
                nums = self.measurement_num_at_value(v)
                if len(nums) == 1:
                    num = self.measurement_num_at_value(v)[0]
                    self._absorbance_best_num[i] = self.get_absorbances(corrected=True, masked=False, num=num, var_value=v)
                    self._variable_best_num[i] = v
                elif len(nums) >= 2:
                    value = []
                    pair_values = []
                    for pair in itertools.combinations(nums, 2):
                        intensities = self.get_absorbances(corrected=True, masked=True, num=nums[0], var_value=v)
                        mask = intensities > 0.1 * np.max(intensities)
                        value.append(np.sum((self.get_absorbances(corrected=False, masked=True, num=pair[0], var_value=v)[mask]
                                             - self.get_absorbances(corrected=False, masked=True, num=pair[1], var_value=v)[mask]) ** 2))
                        pair_values.append(pair)

                    min_num = pair_values[np.argmin(value)]
                    self._absorbance_best_num[i] = (self.get_absorbances(corrected=False, masked=False, num=min_num[0], var_value=v)
                                                    + self.get_absorbances(corrected=False, masked=False, num=min_num[1], var_value=v)) / 2
                    self._variable_best_num[i] = v
                else:
                    raise ValueError(f'No absorbances for variable value {v}')
            self._baseline_correction_best_num = np.mean(self._absorbance_best_num[:, correction_mask], axis=1)[:, np.newaxis]

    @staticmethod
    def from_simple(simple_data_set, wavelength_range=_default_values['wavelength_range'],
                    selected_num=_default_values['selected_num'], baseline_correction=_default_values['baseline_correction']):
        return DataSet(simple_data_set.wavelength, simple_data_set.absorbances, simple_data_set.variable,
                       simple_data_set.measurement_num, simple_data_set.variable_name, wavelength_range, selected_num,
                       baseline_correction, calc_best_num=False)

    @staticmethod
    def read_hdf5(loc, species, **kwargs):
        """
        Read a dataset from a hdf5 file.

        Parameters
        ----------
        loc: str | os.PathLike
        species: str
        kwargs:
            wavelength_range: tuple[float, float]
            selected_num: int
            baseline_correction: tuple[float, float]

        Returns
        -------
        DataSet
        """
        simple = import_hdf5(loc, species)
        return DataSet.from_simple(simple, **kwargs)

    @staticmethod
    def standard(loc, species, *, wavelength_range=None, selected_num=1, baseline_correction=None):
        """
        Make a dataset with standard parameters.

        Parameters
        ----------
        loc: str | os.PathLike
        species: str
        wavelength_range: list[float]
            default=[180, 450]
        selected_num: int, default=1
        baseline_correction: list[float]
            default=[450, 500]

        Returns
        -------
        DataSet
        """
        if baseline_correction is None:
            baseline_correction = DataSet._default_values['baseline_correction']
        if wavelength_range is None:
            wavelength_range = DataSet._default_values['wavelength_range']
        simple = import_hdf5(loc, species)
        return DataSet.from_simple(simple, wavelength_range, selected_num, baseline_correction)

    def get_absorbances(self, *, corrected=True, masked=True, num: str | int | None = 'all', var_value=None):
        absorbances = self.absorbances
        if corrected:
            absorbances = self.absorbances - self._baseline_correction

        wav_mask = np.ones(len(self._mask), dtype=bool)
        if masked:
            wav_mask = self._mask

        if var_value is not None:
            if var_value not in self.variable:
                raise ValueError(f'Variable value {var_value} not in dataset')
            vn_mask = self.variable == var_value
            if num == 'best':
                vn_mask = self.variable_best_num == var_value
        else:
            vn_mask = np.ones(len(self.measurement_num), dtype=bool)
            if num == 'best':
                vn_mask = np.ones(len(self.variable_best_num), dtype=bool)

        # absorbances = self.absorbances
        if (num is None) or num == 'all':
            pass
        elif num == 'plot':
            vn_mask = vn_mask & (self.measurement_num == self._selected_num)
        elif num == 'best':
            if not self._calc_best_num:
                raise ValueError('Best num not calculated')
            absorbances = self._absorbance_best_num
            if corrected:
                absorbances = self._absorbance_best_num - self._baseline_correction_best_num
        elif isinstance(num, (int, np.int_)):
            vn_mask = vn_mask & (self.measurement_num == num)
        else:
            raise ValueError(f'num should be None, "all", "plot", "best" or an integer, not {num}')

        return absorbances[vn_mask][:, wav_mask]

    def get_average_absorbance_ranges(self, ranges: list[tuple[float, float]], *, corrected=True, masked=True,
                                      num: str | int | None = 'all', var_value=None):
        absorbances = self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=var_value)
        return np.array([np.average(absorbances[:, ((r[0] <= self.get_wavelength(masked))
                                                    & (self.get_wavelength(masked) <= r[1]))], axis=1) for r in ranges])

    def get_absorbance_range(self, wavelength_range: tuple[float, float], *, corrected=True, masked=True, num: str | int | None = 'all',
                             var_value=None):
        absorbances = self.get_absorbances(corrected=corrected, masked=masked, num=num, var_value=var_value)
        mask = (wavelength_range[0] <= self.get_wavelength(masked)) & (self.get_wavelength(masked) <= wavelength_range[1])
        return absorbances[:, mask]

    def get_wavelength(self, masked=True):
        return self.wavelength[self._mask] if masked else self.wavelength

    def get_wavelength_range(self, wavelength_range: tuple[float, float]):
        mask = (wavelength_range[0] <= self.wavelength) & (self.wavelength <= wavelength_range[1])
        return self.wavelength[mask]

    @property
    def variable_num(self):
        return self.variable[self.measurement_num == self._selected_num]

    def variable_at_num(self, num):
        return self.variable[self.measurement_num == num]

    def measurement_num_at_value(self, value):
        return self.measurement_num[self.variable == value]

    @property
    def absorbances_masked(self):
        return self.get_absorbances(corrected=False, masked=True, num=None)

    @property
    def absorbances_corrected(self):
        return self.get_absorbances(corrected=True, masked=False, num=None)

    @property
    def absorbances_masked_corrected(self):
        return self.get_absorbances(corrected=True, masked=True, num=None)

    @property
    def absorbances_masked_corrected_num(self):
        return self.get_absorbances(corrected=True, masked=True, num='plot')

    @property
    def absorbances_num(self):
        return self.get_absorbances(corrected=False, masked=False, num='plot')

    @property
    def absorbances_masked_num(self):
        return self.get_absorbances(corrected=False, masked=True, num='plot')

    @property
    def absorbances_corrected_num(self):
        return self.get_absorbances(corrected=True, masked=False, num='plot')

    def absorbances_masked_corrected_at_num(self, num):
        return self.get_absorbances(corrected=True, masked=True, num=num)

    def absorbances_masked_at_num(self, num):
        return self.get_absorbances(corrected=False, masked=True, num=num)

    def absorbances_at_num(self, num):
        return self.get_absorbances(corrected=False, masked=False, num=num)

    def absorbances_corrected_at_num(self, num):
        return self.get_absorbances(corrected=True, masked=False, num=num)

    @property
    def absorbances_masked_best_num(self):
        return self.get_absorbances(corrected=False, masked=True, num='best')

    @property
    def absorbances_masked_corrected_best_num(self):
        return self.get_absorbances(corrected=True, masked=True, num='best')

    @property
    def absorbances_corrected_best_num(self):
        return self.get_absorbances(corrected=True, masked=False, num='best')

    @property
    def absorbances_best_num(self):
        return self.get_absorbances(corrected=False, masked=False, num='best')

    @property
    def variable_best_num(self):
        if not self._calc_best_num:
            raise ValueError('Best num not calculated')
        return self._variable_best_num

    @property
    def simple(self):
        return SimpleDataSet(self.wavelength, self.absorbances, self.variable, self.measurement_num, self.variable_name)

    def add(self, other):
        if isinstance(other, (DataSet, SimpleDataSet)):
            if not np.all(self.wavelength == other.wavelength):
                raise ValueError("Wavelengths don't match")
            if not self.variable_name == other.variable_name:
                raise ValueError("Variables don't match")
            return DataSet(self.wavelength, np.concatenate((self.absorbances, other.absorbances)),
                           np.concatenate((self.variable, other.variable)),
                           np.concatenate((self.measurement_num, other.measurement_num)),
                           self.variable_name, self.wavelength_range, self.baseline_correction, self._selected_num)
        else:
            raise TypeError(f"Can't add {type(other)} to DataSet")