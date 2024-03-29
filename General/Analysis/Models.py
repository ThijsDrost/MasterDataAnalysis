from scipy.interpolate import RegularGridInterpolator
import lmfit
import h5py
import numpy as np

from General.Data_handling import DataSet


def export_model(wavelength: list|np.ndarray, absorbance: np.ndarray, concentration: list|np.ndarray, loc, species, conc_unit='mmol/L'):
    with h5py.File(loc, 'w') as file:
        group = file.create_group(species)
        _add_model(group, wavelength, absorbance, concentration, conc_unit=conc_unit)


def export_models(models: dict[str, dict[str, np.ndarray]], loc, units: dict[str, str] | str, add_zero_conc=False):
    """
    Export models to a hdf5 file.

    The models should be a dictionary with the species as keys and a dictionary with the keys `wavelength`, `absorbance` and `concentration`.

    Parameters
    ----------
    models
    loc
    units

    Returns
    -------

    """
    with h5py.File(loc, 'w') as file:
        for specie, model in models.items():
            group = file.create_group(specie)
            if isinstance(units, str):
                unit = units
            else:
                unit = units[specie]
            try:
                _add_model(group, **model, conc_unit=unit)
            except ValueError as e:
                raise ValueError(f'Error for {specie}: {e}') from e


def _add_model(hdf5_file, wavelength: list|np.ndarray, absorbance: np.ndarray, concentration: list|np.ndarray, conc_unit='mmol/L'):
    absorbance = np.array(absorbance)
    if absorbance.ndim == 1:
        if len(concentration) != 1:
            raise ValueError(f'`absorbance` has only data for one concentration, but `concentration` has length {len(concentration)}')
        if not absorbance.shape[0] == len(wavelength):
            raise ValueError(f'absorbance should have the same length as wavelength, not {absorbance.shape[0]} and {len(wavelength)}')
    elif absorbance.ndim == 2:
        if not absorbance.shape[0] == len(concentration):
            raise ValueError(f'absorbance 1st dimension should have the same length as concentration, not {absorbance.shape[0]} and {len(concentration)}')
        if not absorbance.shape[1] == len(wavelength):
            raise ValueError(f'absorbance 2nd dimension should have the same length as wavelength, not {absorbance.shape[1]} and {len(wavelength)}')
    else:
        raise ValueError(f'`absorbance` should have at most 2 dimensions, not {absorbance.ndim}')

    hdf5_file.attrs.create('wavelength', data=wavelength)
    if absorbance.ndim == 1:
        dataset = hdf5_file.create_dataset(f'{concentration[0]}', data=absorbance)
        dataset.attrs.create(f'{conc_unit}', data=concentration[0])
    else:
        for i, conc in enumerate(concentration):
            dataset = hdf5_file.create_dataset(f'{conc}', data=absorbance[i])
            dataset.attrs.create(f'{conc_unit}', data=conc)


def interpolation_model_2D(x: np.ndarray, y: np.ndarray, values: np.ndarray, *, model_name='interp_model', prefix='', conc_name='conc',
                           conc_default_values: dict[str, int|float] = None):
    interp2 = RegularGridInterpolator((x, y), values, bounds_error=False, method='linear', fill_value=None)

    def func(x, conc):
        return interp2((x, conc))

    func.__name__ = model_name

    model = lmfit.Model(func, prefix=f'{prefix}{conc_name}')
    if conc_default_values is not None:
        model.set_param_hint(f'{prefix}{conc_name}conc', **conc_default_values)
    else:
        model.set_param_hint(f'{prefix}{conc_name}conc', value=0.01)
    return model


def species_model(database, add_zero=False, **kwargs):
    wavelengths = database.attrs['wavelength']
    intensities = []
    concentrations = []
    for measurement in database:
        intensities.append(database[measurement][:])
        attrs = database[measurement].attrs
        for key in attrs:
            if key.lower() == 'mmol/l':
                concentrations.append(attrs[key])
            elif key.lower() == 'mol/l':
                concentrations.append(attrs[key]*1_000)
            elif key.lower() == 'm':
                concentrations.append(attrs[key]*1_000)
            elif key.lower() == 'umol/l':
                concentrations.append(attrs[key]/1_000)
            else:
                raise ValueError(f'Unknown concentration unit: {key}, should be `mmol/L`, `mol/L`, `M` or `umol/L`')
            conc_name = f'conc_{key.lower().replace('/', '_')}'
    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities).T
    concentrations = np.array(concentrations)
    indexes = np.argsort(concentrations)
    intensities = intensities[:, indexes]
    concentrations = concentrations[indexes]
    if add_zero:
        intensities = np.concatenate((np.zeros((1, intensities.shape[0])), intensities.T), axis=0).T
        concentrations = np.concatenate(([0], concentrations))
    return interpolation_model_2D(wavelengths, concentrations, intensities, conc_name=conc_name, **kwargs)


def multi_species_model(database_loc, add_zero=False, database_path=None, add_constant=False):
    models = []
    # default_dict = {'conc': {"value": 1}}
    with h5py.File(database_loc) as file:
        if database_path is not None:
            calibration_group = file[database_path]
        else:
            calibration_group = file
        for dataset in calibration_group.keys():
            models.append(species_model(file[dataset], prefix=dataset, add_zero=add_zero))
    model = models[0]
    for m in models[1:]:
        model += m
    if add_constant:
        constant_model = lmfit.models.ConstantModel()
        constant_model.set_param_hint('c', value=0)
        model += constant_model
    return model


def make_lines_model(ranges, *, corrected=True, num=None, **kwargs: DataSet):
    individual_results = {}
    for key, value in kwargs.items():
        result = []
        for r in ranges:
            data = value.get_average_absorbance_ranges([r, ], corrected=corrected, masked=False, num=num)[0]
            values = value.variable_at_num(num)
            fit_result = lmfit.models.LinearModel().fit(data, x=values)
            result.append(fit_result)
            if fit_result.aborted:
                raise ValueError(f'Fit aborted for {key} at {values}')
        individual_results[key] = result

    def make_model(a, b, prefix, conc_default_values=None, model_name='lines_model'):
        def func(conc):
            return a * conc + b

        func.__name__ = model_name
        model = lmfit.Model(func, prefix=prefix, independent_vars=[])
        if conc_default_values is not None:
            model.set_param_hint(f'{prefix}conc', **conc_default_values)
        else:
            model.set_param_hint(f'{prefix}conc', value=0)
        return model

    models = []
    for key, value in kwargs.items():
        slopes = np.array([individual_results[key][i].best_values['slope'] for i in range(len(ranges))])
        intercepts = np.array([individual_results[key][i].best_values['intercept'] for i in range(len(ranges))])
        models.append(make_model(slopes, intercepts, f"{key}_"))

    model = models[0]
    for m in models[1:]:
        model += m
    return model


def make_spectra_model(wav_range: list[tuple[int, int]] | tuple[int, int], *, corrected=True, num=None,
                       **kwargs: DataSet):
    models = []
    for key, value in kwargs.items():
        wavelengths = value.get_wavelength(masked=False)
        if isinstance(wav_range, tuple):
            wav_range = [wav_range]

        mask = np.zeros(len(wavelengths), dtype=bool)
        for w_range in wav_range:
            mask |= (wavelengths >= w_range[0]) & (wavelengths <= w_range[1])
        intensities = value.get_absorbances(corrected=corrected, masked=False, num=num)[:, mask]
        models.append(interpolation_model_2D(wavelengths[mask], value.variable_at_num(num), intensities.T,
                                             prefix=f'{key}_'))

    model = models[0]
    for m in models[1:]:
        model += m
    return model

