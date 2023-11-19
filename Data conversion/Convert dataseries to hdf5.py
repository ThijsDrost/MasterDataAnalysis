import os
from datetime import datetime
import warnings
import collections

import numpy as np
import h5py

from Backend import read_txt

loc = r'D:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2_H2O2 cuvette'
folders = os.scandir(loc)

with h5py.File(rf"{loc}\data.hdf5", 'w') as hdf5_file:
    for folder in folders:
        if not folder.is_dir():
            continue
        group = hdf5_file.create_group(folder.name)
        dark = read_txt(rf"{folder.path}\dark.txt")
        group.attrs.create('dark', dark['intensity'])
        group.attrs.create('dark_averaging', dark['averaging'])

        reference = read_txt(rf"{folder.path}\reference.txt")
        group.attrs.create('reference', reference['intensity'])
        group.attrs.create('reference_averaging', reference['averaging'])

        group.attrs.create('wavelength', dark['wavelength'])
        group.attrs.create('integration_time', dark['integration_time'])
        group.attrs.create('smoothing', dark['smoothing'])
        group.attrs.create('spectrometer', dark['spectrometer'])

        files = np.array([x.name for x in os.scandir(folder.path)])
        files = files[('dark.txt' != files) & ('reference.txt' != files)]
        timestamp_number = collections.Counter(map(lambda x: x.split('_')[1], files))
        # timestamp, count = np.unique(np.array(list(map(lambda x: x.split('_')[1], files))), return_counts=True)
        # timestamp_number = dict(zip(timestamp, count))

        for file in os.scandir(folder.path):
            if ('dark' in file.name) or ('reference' in file.name):
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

            dataset = group.create_dataset(file.name, data=data['intensity'])
            dataset.attrs.create('averaging', data['averaging'])

            ms = (int(file.name.split('_')[1].split('.')[0])-1)/timestamp_number[file.name.split('_')[1]]
            timestamp = datetime.fromtimestamp(os.path.getmtime(file.path)+ms)
            dataset.attrs.create('timestamp_ms', 1000*timestamp.timestamp())
