import os
import warnings
from typing import Callable

import h5py
import numpy as np
import attrs

from General.experiments.spectrum import Spectrum
from General.checking.number_line import NumberLine
from General.checking import Descriptor


def to_hdf5(input_loc, output_loc, *, resolution=None, selector: Callable[[str], bool] = None, rel_cutoff=1e-3):
    if selector is None:
        def selector(_): return True

    files = [file for file in os.listdir(input_loc) if selector(file)]
    elec = set()
    vib = set()
    rot = set()

    index = 0
    with h5py.File(output_loc, 'w') as hdf5_file:
        data = Spectrum(*np.genfromtxt(os.path.join(input_loc, files[0]), delimiter='\t', unpack=True))
        wav_range = data.wavelengths[0], data.wavelengths[-1]

        for file in files[1:]:
            data = Spectrum(*np.genfromtxt(os.path.join(input_loc, file), delimiter='\t', unpack=True))
            wav_range = (min(wav_range[0], data.wavelengths[0]), max(wav_range[1], data.wavelengths[-1]))

        for index, file in enumerate(files):
            print(f'\rReading files: {index+1}/{len(files)}', end='')
            split_file = '.'.join(file.split('.')[:-1]).split('_')
            temps = {split_file[2*i]: split_file[2*i+1].lower().replace('k', '') for i in range(len(split_file)//2)}
            elec.add(temps['elec'])
            vib.add(temps['vib'])
            rot.add(temps['rot'])

            data = Spectrum(*np.genfromtxt(os.path.join(input_loc, file), delimiter='\t', unpack=True))
            if resolution is not None:
                rel_cutoff = 1e-3 if rel_cutoff is None else rel_cutoff
                data = data.lower_resolution(resolution, rel_cutoff=rel_cutoff, wav_range=wav_range)

            dataset = hdf5_file.create_dataset(str(index), data=data.intensities)
            dataset.attrs.create('elec', data=float(temps['elec']))
            dataset.attrs.create('vib', data=float(temps['vib']))
            dataset.attrs.create('rot', data=float(temps['rot']))
            dataset.attrs.create('wavelengths', data=data.wavelengths)
    print('\rReading files: Done')

    if len(elec)*len(vib)*len(rot) != index+1:
        warnings.warn('Not all combinations of `elec`, `vib` and `rot` are present in the files')


def read_hdf5(loc, *, rot_range=None, vib_range=None, elec_range=None):
    def make_range(value, name):
        if value is None:
            return NumberLine.empty()
        elif isinstance(value, tuple):
            return NumberLine.include_from_floats(*value)
        elif isinstance(value, (int, float)):
            return NumberLine.include_from_floats(value, value)
        elif not isinstance(value, NumberLine):
            raise ValueError(f'`{name}` should be a `tuple`, `int`, `float` or `NumberLine`, not {type(value)}')

    rot_range = make_range(rot_range, 'rot_range')
    vib_range = make_range(vib_range, 'vib_range')
    elec_range = make_range(elec_range, 'elec_range')

    intensities = []
    wavelengths = []
    rot_values = []
    vib_values = []
    elec_values = []

    with h5py.File(loc, 'r') as hdf5_file:
        for key in hdf5_file.keys():
            dataset = hdf5_file[key]
            wavelength = dataset.attrs['wavelengths']
            elec = dataset.attrs['elec']
            vib = dataset.attrs['vib']
            rot = dataset.attrs['rot']

            if (elec in elec_range) and (vib in vib_range) and (rot in rot_range):
                intensities.append(dataset[()])
                wavelengths.append(wavelength)
                elec_values.append(elec)
                vib_values.append(vib)
                rot_values.append(rot)

    if not all(len(intensities[0]) == len(intensity) for intensity in intensities):
        raise ValueError('Not all intensities have the same length')
    if not all(all(wavelengths[0] == wav) for wav in wavelengths):
        raise ValueError('Not all wavelengths are the same')

    return SpecAirData(np.array(wavelengths[0]), np.array(intensities), np.array(elec_values), np.array(vib_values), np.array(rot_values))


@attrs.frozen()
class SpecAirData:
    wavelengths: Descriptor.numpy_dim(1)
    intensities: Descriptor.numpy_dim(2)
    elec_values: Descriptor.numpy_dim(1)
    vib_values: Descriptor.numpy_dim(1)
    rot_values: Descriptor.numpy_dim(1)

    def __attrs_post_init__(self):
        if not (len(self.intensities) == len(self.elec_values) == len(self.vib_values) == len(self.rot_values)):
            raise ValueError('The `intensities`, `elec_values`, `vib_values`, and `rot_values` should have the '
                             'same length')

    def lower_resolution(self, resolution):
        delta_wav = self.wavelengths[-1] - self.wavelengths[0] - resolution
        n_points = delta_wav // resolution
        offset = (self.wavelengths[-1] - self.wavelengths[0]) - (n_points * resolution)
        new_wavelengths = np.linspace(self.wavelengths[0] + offset/2, self.wavelengths[-1]-offset/2, n_points)
        new_intensities = np.zeros((len(new_wavelengths), self.intensities.shape[1]))
        for index, wav in enumerate(new_wavelengths):
            mask = (self.wavelengths > (wav-resolution/2)) & (self.wavelengths < (wav+resolution/2))
            if np.sum(mask) == 0:
                continue
            new_intensities[:, index] = np.sum(self.intensities[:, mask], axis=1)
        return SpecAirData(new_wavelengths, new_intensities, self.elec_values, self.vib_values, self.rot_values)

