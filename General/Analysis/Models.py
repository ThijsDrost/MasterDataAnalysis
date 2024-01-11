from scipy.interpolate import RegularGridInterpolator
import lmfit
import h5py
import numpy as np

from General.Data_handling.Import import DataSet


def interpolation_model_2D(x: np.ndarray, y: np.ndarray, values: np.ndarray, *, model_name='interp_model', prefix='',
                           default_dict: dict[str, dict[str, int|float]] = None):
    interp2 = RegularGridInterpolator((x, y), values, bounds_error=False, method='linear', fill_value=None)

    def func(wav, conc):
        return interp2((wav, conc))

    func.__name__ = model_name

    model = lmfit.Model(func, prefix=prefix)
    if default_dict is not None:
        for key, value in default_dict.items():
            model.set_param_hint(f'{prefix}{key}', **value)
    return model


def species_model(database_loc, database_path, **kwargs):
    with h5py.File(database_loc) as file:
        calibration = file[database_path]
        wavelengths = calibration.attrs['wavelength']
        intensities = []
        concentrations = []
        for measurement in calibration:
            intensities.append(calibration[measurement][:])
            concentrations.append(calibration[measurement].attrs['concentration_mmol'])
    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities).T
    concentrations = np.array(concentrations)
    indexes = np.argsort(concentrations)
    intensities = intensities[:, indexes]
    concentrations = concentrations[indexes]

    return interpolation_model_2D(wavelengths, concentrations, intensities, **kwargs)


def multi_species_model(database_loc, database_path):
    models = []
    default_dict = {'conc': {"value": 1}}
    with h5py.File(database_loc) as file:
        calibration_group = file[database_path]
        for dataset in calibration_group.keys():
            models.append(species_model(database_loc, rf'{database_path}/{dataset}', model_name=dataset,
                                        prefix=dataset, default_dict=default_dict))
    model = models[0]
    for m in models[1:]:
        model += m
    return model


def make_lines_model(ranges, *, corrected=True, num=None, **kwargs: DataSet):
    individual_results = {}
    for key, value in kwargs.items():
        result = []
        for r in ranges:
            data = value.get_absorbance_ranges([r,], corrected=corrected, masked=False, num=num)[0]
            values = value.variable_at_num(num)
            fit_result = lmfit.models.LinearModel().fit(data, x=values)
            result.append(fit_result)
            if fit_result.aborted:
                raise ValueError(f'Fit aborted for {key} at {values}')
        individual_results[key] = result

    def lines_model(**func_kwargs):
        intensities = np.zeros(len(ranges))
        for key, value in func_kwargs.items():
            for i in range(len(ranges)):
                intensities[i] += individual_results[key][i].best_values['slope'] * value + individual_results[key][i].best_values['intercept']
        return intensities

    string_input = f"def func({', '.join(kwargs.keys())}):\n    return lines_model({', '.join(f'{x}={x}' for x in kwargs.keys())})"
    vals = {'lines_model': lines_model}
    exec(string_input, globals() | vals, vals)

    model = lmfit.Model(vals['func'], independent_vars=[])
    for key in kwargs.keys():
        model.set_param_hint(key, value=0)
    return model
