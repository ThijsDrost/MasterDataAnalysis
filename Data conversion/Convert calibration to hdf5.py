import os
from datetime import datetime
import warnings

import numpy as np
import h5py

from Backend import read_txt

loc = r'D:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO3 pH'
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

with (h5py.File(rf"{loc}\data.hdf5", 'w') as hdf5_file):
    dark = read_txt(rf"{loc}\dark.txt")
    hdf5_file.attrs.create('dark', dark['intensity'])
    hdf5_file.attrs.create('dark_averaging', dark['averaging'])

    reference = read_txt(rf"{loc}\reference.txt")
    hdf5_file.attrs.create('reference', reference['intensity'])
    hdf5_file.attrs.create('reference_averaging', reference['averaging'])

    hdf5_file.attrs.create('wavelength', dark['wavelength'])
    hdf5_file.attrs.create('integration_time_ms', dark['integration_time'])
    hdf5_file.attrs.create('smoothing', dark['smoothing'])
    hdf5_file.attrs.create('spectrometer', dark['spectrometer'])
    hdf5_file.attrs.create('notes', notes)

    for file in os.scandir(loc):
        print(file)
        if (('dark' in file.name) or ('reference' in file.name) or file.name.endswith('.hdf5') or ('notes' in file.name)
                or ('pH' in file.name) or ('conductivity' in file.name)):
            continue

        f_s = False
        for specie in species:
            if specie in file.name:
                f_s = True
        if f_s:
            continue

        data = read_txt(file.path)

        if not np.all(data['wavelength'] == dark['wavelength']):
            warnings.warn(f"{file.name} has different wavelength than dark")
        if not (data['smoothing'] == dark['smoothing']):
            warnings.warn(f"{file.name} has different smoothing than dark")
        if not (data['integration_time'] == dark['integration_time']):
            warnings.warn(f"{file.name} has different integration time than dark")
        if not (data['spectrometer'] == dark['spectrometer']):
            warnings.warn(f"{file.name} has different spectrometer than dark")

        dataset = hdf5_file.create_dataset(file.name, data=data['intensity'])
        dataset.attrs.create('averaging', data['averaging'])

        timestamp = datetime.fromtimestamp(os.path.getmtime(file.path))
        dataset.attrs.create('timestamp_ms', 1000*timestamp.timestamp())

        if pH is not None:
            dataset.attrs.create('pH', pH[int(file.name.split('_')[0])])
        if conductivity is not None:
            dataset.attrs.create('conductivity', conductivity[int(file.name.split('_')[0])])
        for specie, specie_dict in species_dicts.items():
            dataset.attrs.create(specie, specie_dict[int(file.name.split('_')[0])])

