import os

import h5py
import numpy as np

from General.experiments.oes import OESData
from General.experiments.absorption.DataSets import DataSet
from General.experiments.waveforms.waveforms import Waveforms, Waveform


def read_hdf5(loc):
    """
    Read an HDF5 file and return the data

    Parameters
    ----------
    loc: str
        The location of the HDF5 file to be read

    Returns
    -------
    dict
        A dictionary containing the data
    """
    if not os.path.exists(loc):
        raise FileNotFoundError(f"{loc} does not exist")

    with h5py.File(loc, 'r') as file:
        keys = list(file.keys())
        data = {}
        for key in keys:
            if isinstance(file[key], h5py.Group):
                try:
                    if key.lower() == 'absorbance':
                        data['absorbance'] = read_absorbance(file[key])
                    elif key.lower() == 'emission':
                        data['emission'] = read_emission(file[key])
                    elif key.lower() == 'conductivity':
                        data['conductivity'] = read_conductivity(file[key])
                    elif key.lower() == 'waveforms':
                        data['waveforms'] = read_waveforms(file[key])
                    else:
                        raise ValueError(f"Unknown group: {key}")
                except Exception as e:
                    raise ValueError(f"Error reading group {key} in file {loc}") from e
            elif isinstance(file[key], h5py.Dataset):
                raise ValueError(f"Dataset {key} is not in a group")
            else:
                raise TypeError(f'Unknown type: {type(file[key])}')
        data['parameters'] = read_parameters(file)
    return data


def read_absorbance(hdf5_handle):
    """
    Read the absorbance data from the HDF5 file

    Parameters
    ----------
    hdf5_handle: h5py.Group | h5py.File
        The HDF5 group (or file) containing the absorbance data

    Returns
    -------
    DataSet
    """

    parameters, wavelengths, times_ms, absorbances = _read_values(hdf5_handle)

    wavelength_range = DataSet._default_values['wavelength_range']
    baseline_correction = DataSet._default_values['baseline_correction']

    return DataSet(wavelength=wavelengths, absorbances=np.asarray(absorbances), variable=np.asarray(times_ms),
                   measurement_num=np.zeros(len(times_ms), dtype=int), variable_name='time [ms]', selected_num=0,
                   wavelength_range=wavelength_range, baseline_correction=baseline_correction, calc_best_num=False)


def read_conductivity(hdf5_handle):
    """
    Read the conductivity data from the HDF5 file

    Parameters
    ----------
    hdf5_handle: h5py.Group | h5py.File
        The HDF5 group (or file) containing the conductivity data

    Returns
    -------
    DataSet
    """
    return hdf5_handle['time'][:], hdf5_handle['conductivity'][:], hdf5_handle['temperature'][:]


def read_emission(hdf5_handle):
    """
    Read the absorbance data from the HDF5 file

    Parameters
    ----------
    hdf5_handle: h5py.Group | h5py.File
        The HDF5 group (or file) containing the absorbance data

    Returns
    -------
    OESData
    """
    parameters, wavelengths, times_s, intensities = _read_values(hdf5_handle)
    times_s = np.array(times_s)
    return OESData.new(wavelengths, np.asarray(intensities), times_s, spectrometer_settings=parameters)


def read_waveforms(hdf5_handle):
    """
    Read the waveform data from the HDF5 file

    Parameters
    ----------
    hdf5_handle: h5py.Group | h5py.File
        The HDF5 group (or file) containing the waveform data

    Returns
    -------
    dict
        A dictionary containing the waveform data
    """
    waveforms = []
    timestamps = []
    channels = []
    nums = []
    for key in hdf5_handle.keys():
        times = hdf5_handle[key]['time'][:]
        voltages = hdf5_handle[key]['voltage'][:]
        waveforms.append(Waveform(voltages, times))

        timestamps.append(hdf5_handle[key].attrs['time_stamp'])
        channels.append(hdf5_handle[key].attrs['file_name'][:2])
        if 'num' in hdf5_handle[key].attrs:
            nums.append(hdf5_handle[key].attrs['num'])
    nums = None if len(nums) == 0 else np.array(nums)
    return Waveforms(waveforms, np.array(timestamps), np.array(channels), nums)


def _read_values(hdf5_handle):
    """
    Read the values from the HDF5 file

    Parameters
    ----------
    hdf5_handle: h5py.Group | h5py.File
        The HDF5 dataset to be read

    Returns
    -------
    np.ndarray
        The values
    """
    parameters = read_parameters(hdf5_handle)
    wavelengths = parameters['wavelength'][:]
    del parameters['wavelength']

    times_ms = []
    intensities = []

    for key in hdf5_handle.keys():
        times_ms.append(hdf5_handle[key].attrs['time_ms'])
        intensities.append(hdf5_handle[key][:])
    return parameters, wavelengths, np.array(times_ms), np.array(intensities)


def read_parameters(hdf5_handle):
    """
    Read the parameters from the HDF5 file

    Parameters
    ----------
    hdf5_handle: h5py.File
        The HDF5 file to be read

    Returns
    -------
    dict
        A dictionary containing the parameters
    """
    parameters = {}
    for key in hdf5_handle.attrs.keys():
        parameters[key] = hdf5_handle.attrs[key]
    return parameters
