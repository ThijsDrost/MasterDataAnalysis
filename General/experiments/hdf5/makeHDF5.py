import os
import warnings

import h5py
import numpy as np

from General.experiments import SpectroData


REFERENCE_NAMES = ['ref', 'reference']


def make_hdf5(file_loc, absorption_kwargs=None, emission_kwargs=None, conductivity_kwargs=None, waveform_kwargs=None,
              conductivity_photo_kwargs=None):
    """
    Make an HDF5 file from the given data

    Parameters
    ----------
    file_loc: str
        The location of the HDF5 file to be written.
    absorption_kwargs: dict
        The arguments to be passed to :py:func:`hdf5_absorbance` minus the `hdf5_handle` argument
    emission_kwargs: dict
        The arguments to be passed to :py:func:`hdf5_emission` minus the `hdf5_handle` argument
    conductivity_kwargs
        The arguments to be passed to :py:func:`hdf5_conductivity` minus the `hdf5_handle` argument
    """
    if not file_loc.endswith('.hdf5'):
        raise ValueError(f"`file_loc`: {file_loc} must end with '.hdf5'")

    if absorption_kwargs is None and emission_kwargs is None and conductivity_kwargs is None:
        raise ValueError("At least one of `absorption_kwargs`, `emission_kwargs`, or `conductivity_kwargs` must be given")

    with h5py.File(file_loc, 'w') as hdf5_file:
        if absorption_kwargs is not None:
            hdf5_absorbance(hdf5_file.create_group('Absorbance'), **absorption_kwargs)
            print('Absorbance done')
        if emission_kwargs is not None:
            hdf5_emission(hdf5_file.create_group('Emission'), **emission_kwargs)
            print('Emission done')
        if conductivity_kwargs is not None:
            hdf5_conductivity(hdf5_file.create_group('Conductivity'), **conductivity_kwargs)
            print('Conductivity done')
        if conductivity_photo_kwargs is not None:
            hdf5_conductivity(hdf5_file.create_group('Conductivity photo'), **conductivity_kwargs)
            print('Conductivity done')
        if waveform_kwargs is not None:
            hdf5_waveforms(hdf5_file.create_group('Waveforms'), **waveform_kwargs)
            print('Waveforms done')


def hdf5_emission(hdf5_handle, folder_loc):
    if not os.path.exists(folder_loc):
        raise FileNotFoundError(f"{folder_loc} does not exist")

    files = list(os.listdir(folder_loc))

    dark = [file for file in files if 'dark' in file.lower()]
    data_files = [file for file in files if file not in dark]
    if dark:
        dark = SpectroData.read_data(os.path.join(folder_loc, dark[0]))
    else:
        raise ValueError("No dark data found")

    ref_data = SpectroData.read_data(os.path.join(folder_loc, data_files[0]))

    for i, file in enumerate(data_files):
        data = SpectroData.read_data(os.path.join(folder_loc, file))
        if not ref_data.same_measurement(data):
            diff = ref_data.give_diff(data)
            raise ValueError(f"Data {file} is not the same as the reference data:\n{diff}")
        dataset = hdf5_handle.create_dataset(str(i), data=data.spectrum.intensities-dark.spectrum.intensities)
        dataset.attrs['time_ms'] = os.path.getmtime(os.path.join(folder_loc, file))
        dataset.attrs['file_name'] = file
    write_spectrometer_data(hdf5_handle, ref_data)


def hdf5_conductivity(hdf5_handle, conductivity_hdf5_loc, photo=True):
    if not os.path.exists(conductivity_hdf5_loc):
        raise FileNotFoundError(f"{conductivity_hdf5_loc} does not exist")

    with h5py.File(conductivity_hdf5_loc, 'r') as conductivity_hdf5:
        group = conductivity_hdf5['conductivity']
        if 'temperature' in group:
            hdf5_handle.create_dataset('temperature', data=group['temperature'])
        hdf5_handle.create_dataset('conductivity', data=group['value'])
        hdf5_handle.create_dataset('time', data=group['time'])
        hdf5_handle.attrs['photo'] = photo


def hdf5_absorbance(hdf5_handle, folder_loc):
    if not os.path.exists(folder_loc):
        raise FileNotFoundError(f"{folder_loc} does not exist")

    files = list(os.listdir(folder_loc))

    dark = [file for file in files if 'dark' in file.lower()]
    ref = [file for file in files if any(name in file.lower() for name in REFERENCE_NAMES)]

    data_files = [file for file in files if file not in dark + ref]

    if dark:
        dark = SpectroData.read_data(os.path.join(folder_loc, dark[0]))
    else:
        raise ValueError("No dark data found")
    if ref:
        ref = SpectroData.read_data(os.path.join(folder_loc, ref[0]))
    else:
        raise ValueError("No reference data found")

    ref_data = SpectroData.read_data(os.path.join(folder_loc, data_files[0]))
    for i, file in enumerate(data_files):
        data = SpectroData.read_data(os.path.join(folder_loc, file))
        if not ref_data.same_measurement(data):
            diff = ref_data.give_diff(data)
            raise ValueError(f"Data {file} is not the same as the reference data:\n{diff}")
        dataset = hdf5_handle.create_dataset(str(i), data=-1*np.log10((data.spectrum.intensities - dark.spectrum.intensities)
                                                                      / (ref.spectrum.intensities - dark.spectrum.intensities)))
        dataset.attrs['time_ms'] = os.path.getmtime(os.path.join(folder_loc, file))
        dataset.attrs['file_name'] = file

    write_spectrometer_data(hdf5_handle, ref_data)


def write_spectrometer_data(hdf5_handle, data: SpectroData):
    hdf5_handle.attrs['wavelength'] = data.spectrum.wavelengths
    hdf5_handle.attrs['serial_number'] = data.serial_number
    hdf5_handle.attrs['integration_time_ms'] = data.integration_time_ms
    hdf5_handle.attrs['n_averages'] = data.n_averages
    hdf5_handle.attrs['n_smoothing'] = data.n_smoothing


def hdf5_waveforms(hdf5_handle: h5py.Group | h5py.File, folder_loc, genfromtxt_kwargs=None):
    """
    Fill an HDF5 file with the waveform data in the given folder.

    Parameters
    ----------
    loc: str or pathlib.Path
        The location of the folder with the waveform data files. It is assumed that all the files in this folder are waveform
        data files
    hdf5_handle
        The group or file to which the data should be written.
    genfromtxt_kwargs
        The keyword arguments to be passed to `np.genfromtxt <https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html>`_
        to read the data files.
    """
    files = os.listdir(folder_loc)

    channels = np.array([int(file[1]) for file in files])
    num = np.array([int(file.removesuffix('.dat')[-5:]) for file in files])
    between = [file.removesuffix('.dat')[2:-5] for file in files]
    for bet in between[1:]:
        if bet != between[0]:
            raise ValueError(f"Files have different between values: {between}")

    unique_channels = np.unique(channels)
    mask = unique_channels[0] == channels
    nums = set(num[mask])
    for channel in unique_channels[1:]:
        mask = channels == channel
        these_nums = num[mask]
        if len(nums) != len(these_nums):
            warnings.warn(f"Channel {channel} has different numbers, skipping {abs(len(these_nums)-len(nums))} non overlapping value(s)")
            nums = nums.intersection(these_nums)

    for index, num in enumerate(nums):
        for index2, chan in enumerate(unique_channels):
            file = f'C{chan}{between[0]}{num:05}.dat'
            genfromtxt_kwargs = genfromtxt_kwargs or {}
            data = np.genfromtxt(os.path.join(folder_loc, file), unpack=True, **genfromtxt_kwargs)
            try:
                data[1]
            except IndexError as e:
                raise ValueError(f"File {file} seems not to have two columns, skipping it.") from e

            group = hdf5_handle.create_group(str(index*len(unique_channels)+index2))
            group.create_dataset('time', data=data[0])
            group.create_dataset('voltage', data=data[1])
            group.attrs['time_stamp'] = os.path.getmtime(os.path.join(folder_loc, file))
            group.attrs['num'] = num
            group.attrs['file_name'] = file
