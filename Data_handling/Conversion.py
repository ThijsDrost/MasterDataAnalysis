import os
import warnings
import time

import numpy as np
import h5py

from Backend import read_txt, get_time, get_time_rel


def main():
    loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\H2O2 pH 2'
    time_index = 0
    find_index = 1
    make_hdf5(loc, TIME_FUNCS[time_index], FIND_METHODS[find_index])

# time index 0 uses the modification time of the file (getmtime) to get timestamp, which is conserved when files are
# copied, but this is only precise to the second for the spectrometer files

# time index 1 uses the time in the filename for the hours, minutes and seconds. For the year, month and
# day the modification time is used. It assumes the following file format: '**_hhmmss_**.txt'

# time index 2 uses the time in the filename for the hours, minutes and seconds. The year, month and
# day are not considered, so the resulting timestamps are in second since the start of the day. It also assumes the
# following file format: '**_hhmmss_**.txt'


TIME_FUNCS = [lambda file: os.path.getmtime(file),
              lambda file: get_time(file),
              lambda file: get_time_rel(file)]
SPECIES = ['NO2-', 'H2O2', 'NO3-']
SKIP_NAMES = ['dark', '.hdf5', 'notes', 'pH', 'conductivity']
FIND_METHODS = ['closest_before', 'closest']
REFERENCE_NAMES = ['ref', 'reference']
# DARK_NAMES = ['dark']


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


def make_hdf5(loc, time_func, find_method):
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

    if file:
        with h5py.File(rf"{loc}\data.hdf5", 'w') as hdf5_file:
            read_folder(loc, hdf5_file, notes, pH, conductivity, species_dicts, time_func, find_method)
    else:
        with h5py.File(rf"{loc}\data.hdf5", 'w') as hdf5_file:
            start = time.time()
            folders = [folder for folder in os.scandir(loc) if folder.is_dir()]
            for index, folder in enumerate(folders):
                group = hdf5_file.create_group(folder.name)
                read_folder(folder.path, group, notes, pH, conductivity, species_dicts, time_func, find_method)
                dt = time.time() - start
                time_left = (len(folders) - index - 1) * dt / (index + 1)
                hours, rem = divmod(time_left, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f'\r{100*(index+1)/len(folders):.1f}% done, done in approx {hours:.02d}:{minutes:.02d}:{seconds:.02d}', end='')


def read_folder(loc, hdf5_place, notes, pH=None, conductivity=None, species_dicts=None, time_func=TIME_FUNCS[0],
                find_method='closest_before'):
    if find_method not in FIND_METHODS:
        warnings.warn(f"find_method {find_method} not recognized, should be in {FIND_METHODS}, `closest_before` used instead")
        find_method = 'closest_before'

    files = [file for file in os.scandir(loc) if file.is_file()]

    dark_files = [file for file in files if 'dark' in file.name]
    dark_times = np.array([time_func(file.path) for file in dark_files])
    dark_reads = [read_txt(file.path) for file in dark_files]
    if len(dark_reads) > 1:
        for i in range(1, len(dark_reads)):
            if not (dark_reads[0] == dark_reads[i]):
                raise ValueError(f"{dark_files[i].name}: {dark_reads[i].give_diff(dark_reads[0])}")

    # reference_files = [file for file in files if 'reference' in file.name]
    reference_files = []
    for file in files:
        for reference_name in REFERENCE_NAMES:
            if reference_name in file.name.lower():
                reference_files.append(file)

    reference_times = np.array([time_func(file.path) for file in reference_files])
    reference_reads = [read_txt(file.path) for file in reference_files]
    for i in range(len(reference_files)):
        if not (reference_reads[0] == reference_reads[i]):
            raise ValueError(f"{reference_files[i].name}: {reference_reads[i].give_diff(reference_reads[0])}")

    if not (dark_reads[0] == reference_reads[0]):
        raise ValueError(f"{reference_files[0].name}: {reference_reads[0].give_diff(dark_reads[0])}")

    hdf5_place.attrs.create('wavelength', dark_reads[0].wavelength)
    hdf5_place.attrs.create('integration_time_ms', dark_reads[0].integration_time)
    hdf5_place.attrs.create('smoothing', dark_reads[0].smoothing)
    hdf5_place.attrs.create('spectrometer', dark_reads[0].spectrometer)
    hdf5_place.attrs.create('notes', notes)

    for file in os.scandir(loc):
        if check_skip(file):
            continue

        data = read_txt(file.path)

        if not (data == dark_reads[0]):
            raise ValueError(f"{file.name}: {data.give_diff(dark_reads[0])}")

        best_dark_index, best_reference_index = 0, 0
        dark_found = True
        reference_found = True
        if find_method == 'closest_before':
            dark_arg = np.argwhere(time_func(file.path) > dark_times)[0]
            if dark_arg.size == 0:
                warnings.warn(f"{file.name}: no earlier dark found, closest one used")
                dark_found = False
            else:
                best_dark_index = dark_arg[0]

            reference_arg = np.argwhere(time_func(file.path) > reference_times)[0]
            if reference_arg.size == 0:
                warnings.warn(f"{file.name}: no earlier reference found, closest one used")
                reference_found = False
            else:
                best_reference_index = reference_arg[0]

        if (find_method == 'closest') or not dark_found:
            best_dark_index = np.argmin(np.abs(time_func(file.path) - dark_times))
        if (find_method == 'closest') or not reference_found:
            best_reference_index = np.argmin(np.abs(time_func(file.path) - reference_times))


        absorbance = -np.log10((data.intensity - dark_reads[best_dark_index].intensity) / (
                    reference_reads[best_reference_index].intensity - dark_reads[best_dark_index].intensity))
        dataset = hdf5_place.create_dataset(file.name, data=absorbance)
        dataset.attrs.create('averaging', data.averaging)

        dataset.attrs.create('timestamp_s', time_func(file.path))

        if pH is not None:
            dataset.attrs.create('pH', pH[int(file.name.split('_')[0])])
        if conductivity is not None:
            dataset.attrs.create('conductivity', conductivity[int(file.name.split('_')[0])])
        for specie, specie_dict in species_dicts.items():
            dataset.attrs.create(specie, specie_dict[int(file.name.split('_')[0])])


if __name__ == '__main__':
    main()
