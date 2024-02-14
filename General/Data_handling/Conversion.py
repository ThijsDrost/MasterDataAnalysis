import os
import warnings
import time
import math

import numpy as np
import h5py

from General.Data_handling.Data_import import SpectroData


def main():
    loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_01_26 H2O2 Conc 2112120U1'
    make_hdf5(loc, FIND_METHODS[0], SPECTRUM_TYPE[1], {'dark': MULTI_METHODS[0], 'reference': MULTI_METHODS[0], 'measurement': MULTI_METHODS[2]},
              name='data.hdf5')


SPECIES = ['NO2-', 'H2O2', 'NO3-']
SKIP_NAMES = ['dark', '.hdf5', 'notes', 'pH', 'conductivity']
SPECTRUM_TYPE = ['spectrum', 'absorbance', 'intensity']
FIND_METHODS = ['ref_interpolate', 'interpolate', 'closest_before', 'closest']
REFERENCE_NAMES = ['ref', 'reference']
MULTI_METHODS = ['mean', 'median', None]


def read_file(loc):
    data_dict = {}
    with open(loc, 'r') as file:
        read_lines = file.readlines()
    for line in read_lines:
        values = line.replace('\n', '').split('\t')
        data_dict[int(values[0])] = float(values[1])
    return data_dict


def check_skip(file):
    for skip_name in SKIP_NAMES:
        if skip_name in file.name:
            return True
    for specie in SPECIES:
        if specie in file.name:
            return True
    for reference_name in REFERENCE_NAMES:
        if reference_name in file.name.lower():
            return True
    return False


def make_hdf5(loc, find_method, spectrum_type, multi_methods: dict, name='data'):
    if os.path.exists(rf"{loc}\pH.txt"):
        pH = read_file(rf"{loc}\pH.txt")
    else:
        pH = None

    if os.path.exists(rf"{loc}\conductivity.txt"):
        conductivity = read_file(rf"{loc}\conductivity.txt")
    else:
        conductivity = None

    if os.path.exists(rf"{loc}\notes.txt"):
        with open(rf"{loc}\notes.txt", 'r') as file:
            notes = file.read()
    else:
        notes = ''

    species_dicts = {}
    for specie in SPECIES:
        if os.path.exists(rf"{loc}\{specie}.txt"):
            specie_dict = read_file(rf"{loc}\{specie}.txt")
            species_dicts[specie] = specie_dict

    dir = False
    file = False
    for item in os.scandir(loc):
        if item.is_dir():
            dir = True
        if item.is_file():
            if check_skip(item):
                continue
            file = True
    if dir and file:
        warnings.warn(f"{loc} contains both files and folders. The files are disregarded")
    elif (not dir) and (not file):
        raise FileNotFoundError(f"{loc} contains no files nor folders")

    if not name.endswith('.hdf5'):
        name += '.hdf5'

    if file:
        with h5py.File(rf"{loc}\{name}", 'w') as hdf5_file:
            read_folder(loc, hdf5_file, notes, pH, conductivity, species_dicts, find_method, spectrum_type, multi_methods)
    else:
        with h5py.File(rf"{loc}\{name}", 'w') as hdf5_file:
            start = time.time()
            folders = [folder for folder in os.scandir(loc) if folder.is_dir()]
            for index, folder in enumerate(folders):
                group = hdf5_file.create_group(folder.name)
                read_folder(folder.path, group, notes, pH, conductivity, species_dicts, find_method, spectrum_type, multi_methods)
                dt = time.time() - start
                time_left = (len(folders) - index - 1) * dt / (index + 1)
                hours, rem = divmod(time_left, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f'\r{100*(index+1)/len(folders):.1f}% done, done in approx {hours:.02d}:{minutes:.02d}:{seconds:.02d}', end='')


def read_folder(loc, hdf5_place, notes, pH=None, conductivity=None, species_dicts=None, find_method='closest_before', spectrum_type='absorbance',
                multi_methods: dict = None):
    if find_method not in FIND_METHODS:
        raise ValueError(f"find_method `{find_method}` not recognized, should be in {FIND_METHODS}")
    if spectrum_type not in SPECTRUM_TYPE:
        raise ValueError(f"spectrum_type `{spectrum_type}` not recognized, should be in {SPECTRUM_TYPE}")
    if multi_methods is None:
        multi_methods = {key: 'mean' for key in ('dark', 'reference', 'measurement')}
    for key in multi_methods:
        if multi_methods[key] not in MULTI_METHODS:
            raise ValueError(f"multi_method `{multi_methods[key]}` for {key} not recognized, should be in {MULTI_METHODS}")

    files = [file for file in os.scandir(loc) if file.is_file()]

    if find_method is not None:
        dark_files = [file for file in files if 'dark' in file.name]
        dark_reads = [SpectroData.read_data(file.path) for file in dark_files]
        dark_times = np.array([x.timestamp_ms for x in dark_reads])
        if len(dark_reads) > 1:
            for i in range(1, len(dark_reads)):
                if not (dark_reads[0] == dark_reads[i]):
                    raise ValueError(f"{dark_files[i].name}: {dark_reads[i].give_diff(dark_reads[0])}")

        reference_files = []
        for file in files:
            for reference_name in REFERENCE_NAMES:
                if reference_name in file.name.lower():
                    reference_files.append(file)

        reference_reads = [SpectroData.read_data(file.path) for file in reference_files]
        reference_times = np.array([x.timestamp_ms for x in reference_reads])
        for i in range(len(reference_files)):
            if not (reference_reads[0] == reference_reads[i]):
                raise ValueError(f"{reference_files[i].name}: {reference_reads[i].give_diff(reference_reads[0])}")

        if not (dark_reads[0] == reference_reads[0]):
            raise ValueError(f"{reference_files[0].name}: {reference_reads[0].give_diff(dark_reads[0])}")

    temp_read = SpectroData.read_data(files[0].path)
    hdf5_place.attrs.create('wavelength', temp_read.wavelength)
    hdf5_place.attrs.create('integration_time_ms', temp_read.integration_time)
    hdf5_place.attrs.create('smoothing', temp_read.smoothing)
    hdf5_place.attrs.create('spectrometer', temp_read.spectrometer)
    hdf5_place.attrs.create('notes', notes)

    extension = None
    for file in os.scandir(loc):
        if check_skip(file):
            continue

        ext = '.' + file.path.split('.')[-1]
        data = SpectroData.read_data(file.path)
        if extension is not None:
            if extension != ext:
                raise ValueError(f"Multiple extensions found: {extension} and {ext}")
        else:
            extension = ext

        if spectrum_type == 'intensity':
            spectrum = data.get_intensity(multi_methods['measurement'])

        if (spectrum_type == 'spectrum') or (spectrum_type == 'absorbance'):
            if not (data == dark_reads[0]):
                raise ValueError(f"{file.name}: {data.give_diff(dark_reads[0])}")

            dark_found = True
            reference_found = True
            dark_weights = np.zeros(len(dark_reads))
            reference_weights = np.zeros(len(reference_reads))

            if find_method == 'closest_before':
                dark_arg = np.argwhere(data.timestamp_ms > dark_times)[0]
                if dark_arg.size == 0:
                    warnings.warn(f"{file.name}: no earlier dark found, closest one used")
                    dark_found = False
                else:
                    dark_weights[dark_arg[0]] = 1

            if (find_method == 'closest') or (not dark_found) or (find_method == 'ref_interpolate'):
                best_dark_index = np.argmin(np.abs(data.timestamp_ms - dark_times))
                dark_weights[best_dark_index] = 1

            if find_method == 'interpolate':
                if len(dark_times) == 1:
                    warnings.warn(f"{file.name}: only one dark found, no interpolation used")
                    dark_weights[0] = 1
                else:
                    dark_weights = interpolate_weights(data.timestamp_ms, dark_times)

            if not math.isclose(np.sum(dark_weights),  1, abs_tol=1e-3):
                warnings.warn(f"{file.name}: dark weights do not sum to 1, {dark_weights} used")

            dark = np.average([dark.get_intensity(multi_methods['dark']) for dark in dark_reads], axis=0, weights=dark_weights)
            spectrum = data.get_intensity(multi_methods['measurement']) - dark

        if spectrum_type == 'absorbance':
            if find_method == 'closest_before':
                reference_arg = np.argwhere(data.timestamp_ms > reference_times)[0]
                if reference_arg.size == 0:
                    warnings.warn(f"{file.name}: no earlier reference found, closest one used")
                    reference_found = False
                else:
                    reference_weights[reference_arg[0]] = 1

            if (find_method == 'closest') or (not reference_found):
                best_reference_index = np.argmin(np.abs(data.timestamp_ms - reference_times))
                reference_weights[best_reference_index] = 1

            if (find_method == 'interpolate') or (find_method == 'ref_interpolate'):
                if len(reference_times) == 1:
                    warnings.warn(f"{file.name}: only one reference found, no interpolation used")
                    reference_weights[0] = 1
                else:
                    reference_weights = interpolate_weights(data.timestamp_ms, reference_times)

            if not math.isclose(np.sum(reference_weights), 1, abs_tol=1e-3):
                warnings.warn(f"{file.name}: reference weights do not sum to 1, {reference_weights} used")

            reference = np.average([reference.get_intensity(multi_methods['reference']) for reference in reference_reads], axis=0, weights=reference_weights)
            spectrum = -np.log10((data.get_intensity(multi_methods['measurement']) - dark) / (reference - dark))

        dataset = hdf5_place.create_dataset(file.name, data=spectrum)
        dataset.attrs.create('averaging', data.averaging)
        dataset.attrs.create('timestamp_s', data.timestamp_ms)
        dataset.attrs.create('spectrum_type', spectrum_type)
        dataset.attrs.create('find_method', find_method)
        dataset.attrs.create('multi_method', multi_methods)

        if pH is not None:
            dataset.attrs.create('pH', pH[int(file.name.split('_')[0])])
        if conductivity is not None:
            dataset.attrs.create('conductivity', conductivity[int(file.name.split('_')[0])])
        for specie, specie_dict in species_dicts.items():
            dataset.attrs.create(specie, specie_dict[int(file.name.split('_')[0])])


def interpolate_weights(measure_time, spectra_times):
    """
    Interpolates the weights for the spectra based on the measure time

    Parameters
    ----------
    measure_time: float
        Time of the measurement
    spectra_times: np.ndarray
        Times of the spectra

    Returns
    -------
    np.ndarray
    """
    weights = np.zeros(len(spectra_times))
    arg_before = np.argmax(measure_time > spectra_times)
    arg_after = arg_before + 1
    times = (spectra_times[arg_before], spectra_times[arg_after])

    rel_values = (measure_time - times[0]) / (times[1] - times[0])
    weights[arg_before] = 1 - rel_values
    weights[arg_after] = rel_values
    return weights


if __name__ == '__main__':
    main()
