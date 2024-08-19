from __future__ import annotations

from typing import Sequence

import attrs
import numpy as np
import math
import pandas as pd

from General import numpy_funcs as npf
from General.checking._validators import Validator
from General.plotting import plot
import General.plotting.cbar as cbar


@attrs.define(frozen=True)
class Spectrum:
    wavelengths: np.ndarray
    """The wavelengths in nm"""
    intensities: np.ndarray
    """The intensities of the spectrum"""

    def __init__(self, wavelength: np.ndarray | list[float | int], intensities: np.ndarray | list[float | int]):
        """
        Class for storing a spectrum with wavelengths and intensities.

        Parameters
        ----------
        wavelength:
            The (1D) wavelengths of the spectrum (assumed to be in nm)
        intensities:
            The (1D) intensities of the spectrum, should have the same length as `wavelength`.

        Raises
        ------
        ValueError
            If the wavelengths and intensities do not have the same length or are not 1D arrays.
        """
        wavelength, intensities = np.array(wavelength), np.array(intensities)
        if isinstance(wavelength, list):
            wavelength = np.array(wavelength)
        if isinstance(intensities, list):
            intensities = np.array(intensities)
        if wavelength.ndim != 1:
            raise ValueError('`wavelengths` must be a 1D array.')
        if intensities.ndim != 1:
            raise ValueError('`intensities` must be a 1D array.')
        if len(wavelength) != len(intensities):
            raise ValueError('`wavelengths` and `intensities` must have the same length.')
        self.__attrs_init__(wavelength, intensities)

    def __attrs_post_init__(self):
        if self.intensities.ndim != 1:
            raise ValueError('Intensities must be a 1D array.')
        if self.wavelengths.ndim != 1:
            raise ValueError('Wavelengths must be a 1D array.')
        if len(self.intensities) != len(self.wavelengths):
            raise ValueError('Intensities and wavelengths must have the same length.')

    def lower_resolution(self, resolution: float, interpolate=True, rel_cutoff=1e-3,
                         wav_range=None) -> Spectrum:
        """
        Returns a new Spectrum with a lower resolution.

        Parameters
        ----------
        resolution: float
            The resolution of the new Spectrum in nm.
        interpolate: bool
            If True, the intensities are interpolated between the original wavelengths.

        Returns
        -------
        Spectrum

        Notes
        -----
        Assumes that the resolution of the original data is higher than the given resolution.
        """
        Validator.positive(False)(resolution, 'resolution')
        Validator.positive(True)(rel_cutoff, 'rel_cutoff')
        Validator.tuple_of_number(wav_range, 'wav_range')

        if np.diff(self.wavelengths).min() > resolution:
            return self

        if wav_range is None:
            num = math.ceil((self.wavelengths[-1] - self.wavelengths[0]) / resolution)
            extra_wav_range = num*resolution - (self.wavelengths[-1] - self.wavelengths[0])
            wav_range = (self.wavelengths[0] - extra_wav_range/2, self.wavelengths[-1] + extra_wav_range/2)
        else:
            num = math.ceil((wav_range[1] - wav_range[0]) / resolution)

        bounds = np.linspace(*wav_range, num)
        new_wavelengths = (bounds[:-1] + bounds[1:]) / 2

        new_intensities = np.empty_like(new_wavelengths)
        if interpolate:
            wav_bounds = np.zeros(len(self.wavelengths) + 1)
            wav_bounds[1:-1] = (self.wavelengths[:-1] + self.wavelengths[1:]) / 2
            wav_bounds[0] = self.wavelengths[0] - (self.wavelengths[1] - self.wavelengths[0]) / 2
            wav_bounds[-1] = self.wavelengths[-1] + (self.wavelengths[-1] - self.wavelengths[-2]) / 2
            wav_widths = np.diff(wav_bounds)

            cutoff = rel_cutoff * self.intensities.max()
            zero_mask = np.zeros(len(self.wavelengths)+1, dtype=bool)
            zero_mask[1:-1] = (self.intensities[:-1] < cutoff) & (self.intensities[1:] < cutoff)
            zero_mask = ~zero_mask

            wav_bounds = wav_bounds[zero_mask]
            intensities = self.intensities[zero_mask[:-1]]
            wav_widths = wav_widths[zero_mask[:-1]]

            for i in range(len(bounds)-1):
                weights = np.zeros_like(wav_widths)
                weights += np.clip((wav_bounds[:-1] - bounds[i])/wav_widths, 0, 1)
                weights -= np.clip((wav_bounds[1:] - bounds[i + 1])/wav_widths, 0, 1)

                new_intensities[i] = np.dot(intensities, weights)
        else:
            for i, bound in enumerate(bounds[:-1]):
                mask = (self.wavelengths >= bound) & (self.wavelengths < bounds[i + 1])
                new_intensities[i] = np.sum(self.intensities[mask])
        return Spectrum(new_wavelengths, new_intensities)


@attrs.define(frozen=True)
class TemporalSpectrum:
    wavelengths: np.ndarray
    """The wavelengths of the spectra"""
    intensities: np.ndarray
    """The intensities of the spectra"""
    times: np.ndarray
    """The acquisition times of the spectra"""

    def __init__(self, wavelengths: np.ndarray | list, intensities: np.ndarray | list, times: np.ndarray | list):
        """
        Class for storing a temporal spectrum with wavelengths, intensities and times.

        Parameters
        ----------
        wavelengths: np.ndarray | list
            The (shared) wavelengths of the spectra, assumed to be in nm. Should have the same length as the second dimension of `intensities`.
        intensities:
            The intensities of the spectra, the first dimension should be the same length as `times`, the second dimension should be the same length as `wavelengths`
        times:
            The times of the spectra, should have the same length as the first dimension of `intensities`

        Raises
        ------
        ValueError
            If the `wavelengths` or `times` are not 1D arrays, the `intensities` is not a 2D array, the first dimension of the
            `intensities` is not equal to  the length of the `times`, or the second dimension of the  `intensities` is not equal
            to the length of the `wavelengths`.
        """

        wavelengths = np.array(wavelengths)
        intensities = np.array(intensities)
        times = np.array(times)

        if intensities.ndim != 2:
            raise ValueError('`intensities` must be a 2D array.')
        if wavelengths.ndim != 1:
            raise ValueError('`wavelengths` must be a 1D array.')
        if times.ndim != 1:
            raise ValueError('`times` must be a 1D array.')
        if intensities.shape[1] != len(wavelengths):
            raise ValueError(f'The second dimension of `intensities` must be equal to the length of `wavelengths`, not '
                             f'{intensities.shape[1]} and {len(wavelengths)}.')
        if len(times) != intensities.shape[0]:
            raise ValueError(f'The length of `times` must be equal to the first dimension of `intensities`, not '
                             f'{len(times)} and {intensities.shape[0]}.')
        if np.any(np.diff(times) < 0):
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            intensities = intensities[sort_idx]
        if np.any(np.diff(wavelengths) <= 0):
            raise ValueError('The wavelengths must be in increasing order.')

        sort_idx = np.argsort(times)
        self.__attrs_init__(wavelengths, intensities[sort_idx], times[sort_idx])

    def wavelength_range(self, start: float, end: float) -> TemporalSpectrum:
        """
        Returns the Spectrum2D with the intensities and wavelengths within the given range.

        Parameters
        ----------
        start: float
            The start of the range
        end: float
            The end of the range

        Returns
        -------
        TemporalSpectrum
        """
        return self.wavelength_ranges([(start, end)])

    def wavelength_ranges(self, *ranges: tuple[float | int, float | int]):
        """
        Returns the Spectrum2D with the intensities and wavelengths within the given ranges.

        Parameters
        ----------
        ranges: Sequence[tuple[float, float]]
            The ranges to keep

        Returns
        -------
        TemporalSpectrum
        """
        mask = np.zeros(self.wavelengths.shape, dtype=bool)
        for start, end in ranges:
            mask |= (self.wavelengths >= start) & (self.wavelengths <= end)
        return TemporalSpectrum(self.wavelengths[mask], self.intensities[:, mask], self.times)

    def block_average(self, n: int) -> TemporalSpectrum:
        """
        Calculates the block average of the intensities with a block size of n.

        Parameters
        ----------
        n: int
            The block size

        Returns
        -------
        TemporalSpectrum
        """
        return TemporalSpectrum(self.wavelengths, npf.block_average(self.intensities, n), npf.block_average(self.times, n))

    def moving_average(self, n: int) -> TemporalSpectrum:
        """
        Calculates the moving average of the intensities with a window of n.

        Parameters
        ----------
        n: int
            The window size

        Returns
        -------
        TemporalSpectrum
        """
        return TemporalSpectrum(self.wavelengths, npf.moving_average(self.intensities, n), npf.moving_average(self.times, n))

    def remove_background(self, background_spectrum: np.ndarray | TemporalSpectrum) -> TemporalSpectrum:
        """
        Removes the background spectrum from the intensities

        Parameters
        ----------
        background_spectrum: TemporalSpectrum
            The background spectrum

        Returns
        -------
        TemporalSpectrum
        """
        if isinstance(background_spectrum, TemporalSpectrum):
            if not np.array_equal(self.wavelengths, background_spectrum.wavelengths):
                raise ValueError('The wavelengths of the background spectrum must be equal to the wavelengths of the spectrum.')
            return TemporalSpectrum(self.wavelengths, self.intensities - background_spectrum.intensities, self.times)
        else:
            return TemporalSpectrum(self.wavelengths, self.intensities - background_spectrum, self.times)

    def remove_background_index(self, index: int, remove=True) -> TemporalSpectrum:
        """
        Removes the spectrum at the given index from the intensities and returns the new Spectrum2D.

        Parameters
        ----------
        index: int
            The index of the background
        remove: float
            If True, the spectrum used as the background is removed from the intensities (it will have all zeros).

        Returns
        -------
        TemporalSpectrum
        """
        if remove:
            indexes = np.arange(self.intensities.shape[0])
            mask = indexes != index
        else:
            mask = np.ones(self.intensities.shape[0], dtype=bool)
        return TemporalSpectrum(self.wavelengths, self.intensities[mask] - self.intensities[index], self.times[mask])

    def remove_baseline(self, wav_range: tuple[float, float]) -> TemporalSpectrum:
        """
        Removes the background in the given wavelength range.

        Parameters
        ----------
        wav_range: Tuple[float, float]
            The wavelength range over which the background should be removed.

        Returns
        -------
        TemporalSpectrum
        """
        mask = (self.wavelengths >= wav_range[0]) & (self.wavelengths <= wav_range[1])
        return TemporalSpectrum(self.wavelengths, self.intensities - np.mean(self.intensities[:, mask], axis=1)[:, None], self.times)

    def remove_background_interp(self, start_indexes: int | tuple[int | None, int], end_indexes: int | tuple[int, int | None]):
        # TODO: int indexes seem not to work?
        def make_background(indexes, where):
            if where not in ('start', 'end'):
                raise ValueError('`where` must be either `start` or `end`.')

            if isinstance(indexes, int):
                return indexes, self.intensities[indexes]
            if isinstance(indexes, tuple):
                if len(indexes) == 1:
                    if where == 'start':
                        indexes = (0, indexes[0])
                    else:
                        indexes = (indexes[0], len(self.intensities))
                elif len(indexes) > 2:
                    raise ValueError('`indexes` must be a tuple of length 1 or 2.')

                if indexes[0] is None:
                    if where == 'start':
                        indexes = (0, indexes[1])
                    else:
                        raise ValueError('The start index cannot be `None` for end.')
                if indexes[1] is None:
                    if where == 'start':
                        raise ValueError('The end index cannot be `None` for start.')
                    else:
                        indexes = (indexes[0], len(self.intensities))

                return (indexes[0]+indexes[1])/2, self.intensities[indexes[0]:indexes[1]].mean(axis=0)

        f_index, f_back = make_background(start_indexes, 'start')
        e_index, e_back = make_background(end_indexes, 'end')

        indexes = np.arange(len(self.intensities))
        rel_value = np.clip((indexes - f_index) / (e_index - f_index), 0, 1)
        values = f_back + rel_value[:, None] * (e_back - f_back)
        return self.remove_background(values)


    def clean(self, *, wavelength_range: tuple[float, float] = None, moving_average: int = None, block_average: int = None,
              background_index: int = None, background_wavelength: tuple[float, float] = None, average_first=True) -> TemporalSpectrum:
        """
        Cleans the spectrum by applying the given operations.

        Parameters
        ----------
        wavelength_range: Tuple[float, float]
            The range of wavelengths to keep, used in `py:func:Spectrum2D.wavelength_range`.
        moving_average: int
            The window size for the moving average, used in `py:func:Spectrum2D.moving_average`.
        block_average
            The window size for the running average, used in `py:func:Spectrum2D.block_average`.
        background_index
            The index of the background spectrum, used in `py:func:Spectrum2D.remove_background_index`.
        background_wavelength
            The wavelength range of the background spectrum, used in `py:func:Spectrum2D.remove_background_wavelengths`.
        average_first: bool
            If True, the running or moving average is applied before the other operations. If False, it is applied after.

        Returns
        -------
        TemporalSpectrum

        Notes
        -------
        Only one of `running_average` or `moving_average` can be set.
        """

        if block_average is not None and moving_average is not None:
            raise ValueError('Only one of `running_average` or `moving_average` can be set.')

        def average(data):
            if block_average is not None:
                return data.block_average(block_average)
            elif moving_average is not None:
                return data.moving_average(moving_average)
            else:
                return data

        def others(data):
            if wavelength_range is not None:
                data = data.wavelength_range(*wavelength_range)
            if background_index is not None:
                data = data.remove_background_index(background_index)
            if background_wavelength is not None:
                data = data.remove_baseline(background_wavelength)
            return data

        if average_first:
            return others(average(self))
        else:
            return average(others(self))

    def __getitem__(self, item):
        """
        Indexes the TemporalSpectrum. It supports simple and advanced indexing from numpy.

        Parameters
        ----------
        item: int, slice, np.ndarray, list, tuple

        Returns
        -------
        TemporalSpectrum
        """
        if isinstance(item, (slice, np.ndarray, list, int)):
            item = (item, slice(None))

        elif isinstance(item, tuple):
            if len(item) == 1:
                item = item + (slice(None),)
            elif len(item) > 2:
                raise ValueError('Cannot have more than two indices.')
        else:
            raise TypeError(f'Can only index with int, slice, array, list or tuple, not {type(item)}.')

        return TemporalSpectrum(self.wavelengths[item[1],], self.intensities[item], self.times[item[0],])

    def plot_intensity_with_time(self, bounded_cbar=False, **lines_kwargs):
        plot_kwargs = {'xlabel': 'Wavelength [nm]', 'ylabel': 'Intensity [A.U]'}
        if 'plot_kwargs' in lines_kwargs:
            plot_kwargs = plot.set_defaults(lines_kwargs['plot_kwargs'], **plot_kwargs)
            del lines_kwargs['plot_kwargs']
        if bounded_cbar:
            colors, cbar_kwargs = cbar.bounded_cbar(self.times)
        else:
            colors, cm = cbar.cbar_norm_colors(self.times)
            cbar_kwargs = {'mappable': cm}
        cbar_kwargs['label'] = 'Time [s]'
        if 'cbar_kwargs' in lines_kwargs:
            cbar_kwargs = plot.set_defaults(lines_kwargs['cbar_kwargs'], **cbar_kwargs)
            del lines_kwargs['cbar_kwargs']
        return plot.lines(self.wavelengths, self.intensities, colors=colors, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs,
                          **lines_kwargs)

    def plot_spectrum(self, index, *, normalize=False, **lines_kwargs):
        plot_kwargs = {'xlabel': 'Wavelength [nm]', 'ylabel': 'Intensity [A.U]'}
        if 'plot_kwargs' in lines_kwargs:
            plot_kwargs = plot.set_defaults(lines_kwargs['plot_kwargs'], **plot_kwargs)
            del lines_kwargs['plot_kwargs']
        if normalize:
            intensities = self.intensities[index] / self.intensities[index].max()
        else:
            intensities = self.intensities[index]
        return plot.lines(self.wavelengths, intensities, plot_kwargs=plot_kwargs, **lines_kwargs)
