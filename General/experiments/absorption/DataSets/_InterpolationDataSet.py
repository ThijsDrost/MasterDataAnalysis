"""
This submodule contains the InterpolationDataSet class (child of SimpleDataSet) for interpolating spectral/absorbance data.
"""


import numpy as np
import scipy
import lmfit

from ._SimpleDataSet import SimpleDataSet
from ._DataSet import DataSet


class InterpolationDataSet(SimpleDataSet):
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
    def from_simple(simple_data_set: SimpleDataSet, add_zero=True):
        return InterpolationDataSet(simple_data_set.wavelength, simple_data_set.absorbances, simple_data_set.variable,
                                    simple_data_set.variable_name, add_zero)

    @staticmethod
    def from_dataset(data_set: DataSet, add_zero=True, corrected=True, num=None):
        return InterpolationDataSet(data_set.get_wavelength(masked=False),
                                    data_set.get_absorbances(corrected=corrected, masked=False, num=num),
                                    data_set.variable_at_num(num), data_set.variable_name, add_zero)

    def closest(self, variable_value):
        if variable_value in self.variable:
            return self[self.variable == variable_value]
        else:
            return self._closest(variable_value)

    def _closest(self, variable_value):
        index = np.argmin(np.abs(self.variable - variable_value))
        return self.absorbances[index]

    def interpolate(self, variable_value):
        if variable_value < np.min(self.variable) or variable_value > np.max(self.variable):
            raise ValueError('Variable value out of range')
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

    def __call__(self, variable_value):
        return self.interpolate(variable_value)