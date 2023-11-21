import os
from datetime import datetime
import warnings

import numpy as np
import h5py

from Backend import read_txt

loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2 pH'
species = ['NO2-', 'H2O2', 'NO3-']


def read_file(loc):
    data_dict = {}
    with open(loc, 'r') as file:
        read_lines = file.readlines()
    for line in read_lines:
        values = line.replace('\n', '').split('\t')
        data_dict[int(values[0])] = float(values[1])
    return data_dict


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
for specie in species:
    if os.path.exists(rf"{loc}\{specie}.txt"):
        specie_dict = read_file(rf"{loc}\{specie}.txt")
        species_dicts[specie] = specie_dict


files = [file for file in os.scandir(loc) if file.is_file()]
with h5py.File(rf"{loc}\data.hdf5", 'w') as hdf5_file:
    dark_files = [file for file in files if 'dark' in file.name]
    dark_times = np.array([datetime.fromtimestamp(os.path.getmtime(file.path)).timestamp() for file in dark_files])
    dark_reads = [read_txt(file.path) for file in dark_files]
    if len(dark_reads) > 1:
        for i in range(1, len(dark_reads)):
            if not (dark_reads[0] == dark_reads[i]):
                warnings.warn(f"{dark_files[i].name}: {dark_reads[i].give_diff(dark_reads[0])}")

    reference_files = [file for file in files if 'reference' in file.name]
    reference_times = np.array([datetime.fromtimestamp(os.path.getmtime(file.path)).timestamp() for file in reference_files])
    reference_reads = [read_txt(file.path) for file in reference_files]
    for i in range(len(reference_files)):
        if not (reference_reads[0] == reference_reads[i]):
            warnings.warn(f"{reference_files[i].name}: {reference_reads[i].give_diff(reference_reads[0])}")

    hdf5_file.attrs.create('wavelength', dark_reads[0].wavelength)
    hdf5_file.attrs.create('integration_time_ms', dark_reads[0].integration_time)
    hdf5_file.attrs.create('smoothing', dark_reads[0].smoothing)
    hdf5_file.attrs.create('spectrometer', dark_reads[0].spectrometer)
    hdf5_file.attrs.create('notes', notes)

    for file in os.scandir(loc):
        print(file)
        if (('dark' in file.name) or ('reference' in file.name) or file.name.endswith('.hdf5') or 'notes' in file.name
            or 'pH' in file.name or 'conductivity' in file.name):
            continue

        f_s = False
        for specie in species:
            if specie in file.name:
                f_s = True
        if f_s:
            continue

        data = read_txt(file.path)

        if not (data == dark_reads[0]):
            warnings.warn(f"{file.name}: {data.give_diff(dark_reads[0])}")

        best_dark_index = np.argwhere(datetime.fromtimestamp(os.path.getmtime(file.path)).timestamp() > dark_times)[0][0]
        best_reference_index = np.argwhere(datetime.fromtimestamp(os.path.getmtime(file.path)).timestamp() > reference_times)[0][0]

        absorbance = -np.log10((data.intensity-dark_reads[best_dark_index].intensity)/(reference_reads[best_reference_index].intensity-dark_reads[best_dark_index].intensity))
        dataset = hdf5_file.create_dataset(file.name, data=absorbance)
        dataset.attrs.create('averaging', data.averaging)

        dataset.attrs.create('timestamp_s', datetime.fromtimestamp(os.path.getmtime(file.path)).timestamp())

        if pH is not None:
            dataset.attrs.create('pH', pH[int(file.name.split('_')[0])])
        if conductivity is not None:
            dataset.attrs.create('conductivity', conductivity[int(file.name.split('_')[0])])
        for specie, specie_dict in species_dicts.items():
            dataset.attrs.create(specie, specie_dict[int(file.name.split('_')[0])])

