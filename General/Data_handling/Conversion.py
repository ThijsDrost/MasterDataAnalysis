"""
This script is used to convert the data from the spectrometer to a hdf5 file.
"""

import os
import warnings
import math

import numpy as np
import h5py

from General.Data_handling._SpectroData import SpectroData


def main():
    loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_20\Argon'
    make_hdf5(loc, FIND_METHODS[3], SPECTRUM_TYPE[1], {'dark': MULTI_METHODS[0], 'reference': MULTI_METHODS[0], 'measurement': MULTI_METHODS[0]},
              out_loc=rf'{loc}\data.hdf5')


# CONSTANTS
# settings values
SPECTRUM_TYPE = ['spectrum', 'absorbance', 'intensity']  # spectrum = measurement-dark, absorbance = -log10((measurement-dark)/(reference-dark)), intensity = measurement
FIND_METHODS = ['ref_interpolate', 'interpolate', 'closest_before', 'closest']
MULTI_METHODS = ['mean', 'median']

# names
SPECIES = ['NO2-', 'H2O2', 'NO3-']
SKIP_NAMES = ['dark', '.hdf5', 'notes', 'ph', 'conductivity', 'skip']
REFERENCE_NAMES = ['ref', 'reference']


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
        if skip_name in file.name.lower():
            return True
    for specie in SPECIES:
        if specie in file.name:
            return True
    for reference_name in REFERENCE_NAMES:
        if reference_name in file.name.lower():
            return True
    return False


def make_hdf5(loc, find_method, spectrum_type, multi_methods: dict, out_loc='data'):
    if not os.path.exists(loc):
        raise FileNotFoundError(f"{loc} does not exist")

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

    if not out_loc.endswith('.hdf5'):
        out_loc += '.hdf5'

    with h5py.File(rf"{out_loc}", 'w') as hdf5_file:
        read_folder(loc, hdf5_file, notes, pH, conductivity, species_dicts, find_method, spectrum_type, multi_methods)


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

    files = [file for file in os.scandir(loc)]

    if not (spectrum_type == 'intensity'):
        dark_files = [file for file in files if 'dark' in file.name.lower()]
        dark_reads = [SpectroData.read_data(file.path) for file in dark_files]
        if len(dark_files) == 0:
            raise FileNotFoundError(f"No dark files found in {loc}")

        for i in range(1, len(dark_reads)):
            if not (dark_reads[0] == dark_reads[i]):
                raise ValueError(f"{dark_files[i].name}: {dark_reads[i].give_diff(dark_reads[0])}")

    if spectrum_type == 'absorbance':
        reference_files = []
        for file in files:
            for reference_name in REFERENCE_NAMES:
                if reference_name in file.name.lower():
                    reference_files.append(file)
                    break
        if len(reference_files) == 0:
            raise FileNotFoundError(f"No reference files found in {loc}")

        reference_reads = [SpectroData.read_data(file.path) for file in reference_files]
        for i in range(len(reference_files)):
            if not (reference_reads[0] == reference_reads[i]):
                raise ValueError(f"{reference_files[i].name}: {reference_reads[i].give_diff(reference_reads[0])}")

        if not (dark_reads[0] == reference_reads[0]):
            raise ValueError(f"{reference_files[0].name}: {reference_reads[0].give_diff(dark_reads[0])}")

    temp_file = [file for file in files if (not check_skip(file))][0]
    temp_read = SpectroData.read_data(temp_file.path)
    hdf5_place.attrs.create('wavelength', temp_read.wavelength)
    hdf5_place.attrs.create('integration_time_ms', temp_read.integration_time_ms)
    hdf5_place.attrs.create('smoothing', temp_read.n_smoothing)
    hdf5_place.attrs.create('spectrometer', temp_read.serial_number)
    hdf5_place.attrs.create('notes', notes)
    hdf5_place.attrs.create('spectrum_type', spectrum_type)
    hdf5_place.attrs.create('find_method', find_method)
    hdf5_place.attrs.create('multi_method', multi_methods['measurement'])

    # extension = None
    for file in os.scandir(loc):
        if check_skip(file):
            continue

        data = SpectroData.read_data(file.path)

        if spectrum_type == 'intensity':
            spectrum = data.get_intensity(multi_methods['measurement'])

        def get_value(timestamp, datas, find_method):
            timestamps = np.array([x.time_ms for x in datas])  # changed from timestamp_ms
            found = True
            weights = np.zeros(len(datas))

            if find_method == 'closest_before':
                dark_arg = np.argwhere(timestamp > timestamps)
                if dark_arg.size == 0:
                    warnings.warn(f"{file.name}: no earlier dark found, closest one used")
                    found = False
                else:
                    weights[dark_arg[0]] = 1

            if (find_method == 'closest') or (not found) or (find_method == 'ref_interpolate'):
                best_dark_index = np.argmin(np.abs(timestamp - timestamps))
                weights[best_dark_index] = 1

            if find_method == 'interpolate':
                if len(timestamps) == 1:
                    warnings.warn(f"{file.name}: only one dark found, no interpolation used")
                    weights[0] = 1
                else:
                    weights = interpolate_weights(timestamp, timestamps)

            if not math.isclose(np.sum(weights), 1, abs_tol=1e-3):
                warnings.warn(f"{file.name}: dark weights do not sum to 1, {weights} used")

            return np.average([dat.get_intensity(multi_methods['dark']) for dat in datas], axis=0, weights=weights)

        if (spectrum_type == 'spectrum') or (spectrum_type == 'absorbance'):
            if not (data == dark_reads[0]):
                raise ValueError(f"file {file.name}: {data.give_diff(dark_reads[0])}")

            dark = get_value(data.time_ms, dark_reads, find_method)
            spectrum = data.get_intensity(multi_methods['measurement']) - dark

        if spectrum_type == 'absorbance':
            if not (data == reference_reads[0]):
                raise ValueError(f"{file.name}: {data.give_diff(reference_reads[0])}")

            reference = get_value(data.time_ms, reference_reads, find_method)
            spectrum = -np.log10((data.get_intensity(multi_methods['measurement']) - dark) / (reference - dark))

        if data.intensity.ndim == 2:
            with h5py.File('.'.join(file.path.split('.')[:-1])+'.hdf5', 'w') as sub_file:
                if spectrum_type == 'intensity':
                    full_spectrum = data.get_intensity(None)
                elif spectrum_type == 'spectrum':
                    full_spectrum = full_spectrum - dark
                elif spectrum_type == 'absorbance':
                    intensity = data.get_intensity(None)
                    if isinstance(intensity, np.ndarray) and intensity.ndim == 2:
                        intensity = intensity.T
                    full_spectrum = -np.log10((intensity - dark) / (reference - dark))
                else:
                    raise ValueError(f"spectrum_type `{spectrum_type}` not recognized, should be in {SPECTRUM_TYPE}")

                for t, s in zip(data.relative_times_ms, full_spectrum, strict=True):
                    index = 0
                    while f'{t:.5f}' in sub_file.keys():
                        t += 0.0001
                        if index > 10_000:
                            raise ValueError(f"Could not find unique timestamp for {file.name}")
                    sub_dataset = sub_file.create_dataset(f'{t:.5f}', data=s)
                    sub_dataset.attrs.create('relative_timestamp_s', t/1000)

                file_num = int(file.name.split('_')[0].split('.')[0])
                if pH is not None:
                    sub_file.attrs.create('pH', pH[file_num])
                if conductivity is not None:
                    sub_file.attrs.create('conductivity', conductivity[file_num])
                for specie, specie_dict in species_dicts.items():
                    sub_file.attrs.create(specie, specie_dict[file_num])

        if '.' in file.name:
            dataset_name = '.'.join(file.name.split('.')[:-1])
        else:
            dataset_name = file.name

        for spec in ('2203047U1', '2112120U1'):
            dataset_name = dataset_name.replace(f'_{spec}', '')
            dataset_name = dataset_name.replace(spec, '')
        if len(dataset_name.split('_')) == 1:
            dataset_name += '_1'

        dataset = hdf5_place.create_dataset(dataset_name, data=spectrum)
        if data.intensity.ndim == 1:
            dataset.attrs.create('averaging', data.n_averages)
        else:
            dataset.attrs.create('averaging', data.intensity.shape[1])
        dataset.attrs.create('timestamp_s', data.time_ms/1000)

        file_num = int(file.name.split('_')[0].split('.')[0])
        if pH is not None:
            dataset.attrs.create('pH', pH[file_num])
        if conductivity is not None:
            dataset.attrs.create('conductivity', conductivity[file_num])
        for specie, specie_dict in species_dicts.items():
            dataset.attrs.create(specie, specie_dict[file_num])


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
