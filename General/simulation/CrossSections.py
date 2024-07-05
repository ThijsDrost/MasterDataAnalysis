"""
This subpackage contains the classes :py:class:`CrossSectionData` and :py:class:`CrossSectionCollection` for importing and handling cross-section data
from LXCat.

The `CrossSectionData` class is used to read and store cross-section data for one species.
The `CrossSectionCollection` class is used to store multiple `CrossSectionData` objects and perform operations on them.
"""

from __future__ import annotations
import warnings
import math
import re

import numpy as np
from scipy.interpolate import CubicSpline
import scipy
import matplotlib.pyplot as plt

from General.plotting import plot, linestyles


class CrossSectionData:
    def __init__(self, energy, cross_section, parameter, process, species, database, updated=None, comments=None, full_process=None, all_species=None,
                 units=(None, None)):
        self.process = process
        self.energy = energy
        self.cross_section = cross_section
        self.parameter = parameter
        self.species = species
        self.updated = updated
        self.comments = comments
        self.database = database
        self.all_species = all_species
        self.database_simplified = database.split()[0]
        self.full_process = full_process
        if units is None:
            self.units = (None, None)
        elif isinstance(units, dict):
            self.units = {'energy': units['energy'], 'cross_section': units['cross_section']}
        else:
            self.units = {'energy': units[0], 'cross_section': units[1]}

    def __repr__(self):
        return f"CrossSectionData('{self.database_simplified}': '{self.process}', '{self.species}')"

    def write_comsol(self, loc):
        """
        Does the func show up in the docs?

        Parameters
        ----------
        loc

        Returns
        -------

        """
        with open(loc, 'w') as file:
            self._write_comsol(file)

    def _write_comsol(self, file_handle):
        file_handle.write(f'{self.process.upper()}\n')
        process = re.split(r'(?:\(\d*\.?\d*(?:eV)?\))? ?, ', self.full_process)[0]
        process = (process.replace('->', '=>').replace('*', 's')
                          .replace('eV', '').replace('E ', 'e ')
                          .replace('(', '').replace(')', '')
                   )
        file_handle.write(f'{process}\n')
        file_handle.write(f'{self.parameter}')
        if self.process.lower() == 'excitation':
            warnings.warn("Writing the ratio of 12, which is only correct for Argon")  # TODO: Make this more general
            file_handle.write(' 12 1')
        file_handle.write('\n')
        file_handle.write('1.0 1.0\n')
        self._write_rest(file_handle)

    def _write_rest(self, file_handle):
        file_handle.write('-----------------------------\n')
        for energy, cross_section in zip(self.energy, self.cross_section):
            file_handle.write(f'{energy}\t{cross_section}\n')
        file_handle.write('-----------------------------\n\n\n')

    def write(self, loc):
        with open(loc, 'w') as file:
            self._write(file)

    def _write(self, file_handle):
        file_handle.write(f'{self.process.upper()}\n')
        file_handle.write(f'{self.species}\n')
        if self.parameter is not None:
            file_handle.write(f'{self.parameter}\n')
        if self.all_species is not None:
            file_handle.write(f'SPECIES: {self.all_species}\n')
        if self.process is not None:
            file_handle.write(f'PROCESS: {self.process}\n')
        if (self.comments is not None) and (self.comments != ''):
            file_handle.write(f'COMMENT: {self.comments}\n')
        if self.updated is not None:
            file_handle.write(f'UPDATED: {self.updated}\n')
        if self.units['energy'] is not None:
            file_handle.write(f'COLUMNS: Energy ({self.units["energy"]}) | Cross section ({self.units["cross_section"]})\n')
        self._write_rest(file_handle)

    @staticmethod
    def write_multi_comsol(loc, data: list[CrossSectionData]):
        with open(loc, 'w') as file:
            for dat in data:
                dat._write_comsol(file)

    @staticmethod
    def write_multi(loc, data: list[CrossSectionData]):
        with open(loc, 'w') as file:
            for dat in data:
                dat._write(file)

    @staticmethod
    def read_txt(loc) -> CrossSectionData | CrossSectionCollection:
        data = open(loc, 'r').read().split('\n')
        values = CrossSectionData._read_datasets(data)
        if len(values) == 1:
            return values[0]
        return values

    @staticmethod
    def _read_datasets(data: list[str]) -> CrossSectionCollection:
        start, end = CrossSectionData._find_database(data)
        datasets = []
        while end is not None:
            datasets.extend(CrossSectionData._read_database(data[start:end]))
            data = data[end:]
            start, end = CrossSectionData._find_database(data)
        return CrossSectionCollection(datasets)

    @staticmethod
    def _find_database(data: list[str]) -> (int | None, int | None):
        """
        Find the start and end of a dataset in the data list. The start is the first line of the dataset, the end is the
        first line after the dataset. Returns (None, None) if no dataset is found.
        """
        start, end, value = None, None, None
        for i, line in enumerate(data):
            if line.startswith('xxxxx'):
                if start is None:
                    start = i
                elif value is None:
                    value = i
                else:
                    end = i - 1
                    break
        return start, end

    @staticmethod
    def _read_database(data: list[str]) -> list[CrossSectionData]:
        values = {'comments': ''}
        inside = False
        index = 0
        for index, line in enumerate(data):
            if line.lower().startswith('database'):
                values['database'] = line.split(':')[1].strip()
            if line.startswith('*****'):
                if inside:
                    index += 1
                    break
                inside = True

        datasets = []
        while index < (len(data) - 3):
            while data[index].strip() not in ('EFFECTIVE', 'ELASTIC', 'IONIZATION', 'EXCITATION', 'DISSOCIATION', 'ATTACHMENT', 'DETACHMENT', 'TOTAL'):
                index += 1

            values.update({'process': data[index].strip(),
                           'species': data[index+1].strip(),
                           'parameter': data[index+2].strip()})
            new_index = index+3
            for new_index, line in enumerate(data[(index + 3):], start=index+3):
                if line.lower().startswith('species'):
                    values['all_species'] = line.split(':')[1].strip()
                elif line.lower().startswith('updated'):
                    values['updated'] = line.split(':')[1].strip()
                elif line.lower().startswith('comment'):
                    values['comments'] += line.split(': ')[1]
                elif line.lower().startswith('process'):
                    values['full_process'] = line.split(':')[1].strip()
                elif line.lower().startswith('columns'):
                    try:
                        values['units'] = [val.split('(')[1].split(')')[0].strip() for val in line.split(':')[1].split('|')]
                    except IndexError:
                        values['units'] = (None, None)
                        warnings.warn(f'Could not parse units from: {line}')
                if line.startswith('-----'):
                    break

            energies, cross_sections = [], []
            index = new_index + 1
            for line in data[(new_index + 1):]:
                if line.startswith('-----'):
                    break
                energy, cross_section = [float(i) for i in line.split()]
                energies.append(energy)
                cross_sections.append(cross_section)
                index += 1

            values['energy'] = np.array(energies)
            values['cross_section'] = np.array(cross_sections)
            datasets.append(CrossSectionData(**values))

        if len(datasets) == 0:
            raise ValueError('No datasets found')
        if len(datasets) > 1:
            for i, dataset in enumerate(datasets, 1):
                dataset.database_simplified += f' ({i})'
        return datasets


class CrossSectionCollection:
    def __init__(self, data: list[CrossSectionData]):
        self.data = data

    def check_data(self):
        problems = {}
        for dat in self.data:
            p = ''
            if not self._check_derivative(dat.energy, dat.cross_section):
                p += 'Non-unique derivative found.\n'
            mask = self._select_spread(dat.energy)
            if not np.all(mask):
                p += 'Energy spread found.\n'
            if p != '':
                problems[dat.database_simplified] = p
        return problems

    def clean_data(self):
        removed = []
        cleaned = []
        for dat in self.data:
            if not self._check_derivative(dat.energy, dat.cross_section):
                removed.append(dat.database_simplified)
                self.data.remove(dat)
            mask = self._select_spread(dat.energy)
            dat.energy = dat.energy[mask]
            dat.cross_section = dat.cross_section[mask]
            if not np.all(mask):
                cleaned.append(dat.database_simplified)
        return removed, cleaned

    @staticmethod
    def _check_derivative(energy, cross_section):
        diff = np.diff(cross_section) / np.diff(energy)

        last_value = 0
        num = 0
        for value in diff:
            if math.isclose(last_value, value):
                num += 1
                if num == 2:
                    return False
            else:
                num = 0
                last_value = value
        return True

    @staticmethod
    def _select_spread(energy, max_distance=0.3):
        log_diff = np.diff(np.log(energy))
        mask = log_diff < max_distance
        halfway = len(mask) // 2
        start = halfway - np.argmin(mask[:halfway:-1])
        end = halfway + np.argmin(mask[halfway:])
        mask = np.zeros(len(energy), dtype=bool)
        mask[start:end] = True
        return mask

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[item]
        if isinstance(item, str):
            for i, dat in enumerate(self.data):
                if self._same_name(i, item):
                    return dat
            raise KeyError(f'No dataset with name {item} found.')
        if isinstance(item, slice):
            return CrossSectionCollection(self.data[item])
        raise TypeError(f'CrossSectionCollection indices must be integers, string or slice, not {type(item)}')

    def __add__(self, other):
        if isinstance(other, CrossSectionData):
            return CrossSectionCollection(self.data + [other])
        if isinstance(other, CrossSectionCollection):
            return CrossSectionCollection(self.data + other.data)
        raise TypeError(f'Can only add `CrossSectionData` or `CrossSectionCollection` to `CrossSectionData`, not {type(other)}')

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'CrossSectionCollection({'; '.join([f"{dat.database_simplified}: {dat.process}, {dat.species}" for dat in self.data])})'

    def _same_name(self, index, name):
        if (self.data[index].database_simplified == name) or (self.data[index].database_simplified.lower() == name):
            return True
        if (self.data[index].database == name) or (self.data[index].database.lower() == name):
            return True

    def write(self, loc):
        CrossSectionData.write_multi(loc, self.data)

    def write_comsol(self, loc):
        CrossSectionData.write_multi_comsol(loc, self.data)

    def select_by(self, func):
        return CrossSectionCollection([dat for dat in self.data if func(dat)])

    @staticmethod
    def _selector_range(x_range, y_range):
        def make_range(_range, name):
            if len(_range) != 2:
                raise ValueError(f'{name} should be a tuple with two elements, not {_range}')
            if _range[0] is None:
                _range = (-np.inf, _range[1])
            if _range[1] is None:
                _range = (_range[0], np.inf)
            if (not isinstance(_range[0], (int, float))) or (not isinstance(_range[1], (int, float))):
                raise ValueError(f'{name} should contain only numbers, not {_range}')
            if _range[0] >= _range[1]:
                raise ValueError(f'{name} should be in increasing order, not {_range}')
            return _range

        x_range = make_range(x_range, 'x_range')
        y_range = make_range(y_range, 'y_range')

        def selector(dat):
            mask = (x_range[0] <= dat.energy) & (dat.energy <= x_range[1])
            return np.all(y_range[0] <= dat.cross_section[mask]) & np.all(dat.cross_section[mask] <= y_range[1])
        return selector

    def select_by_values(self, x_range: tuple, y_range: tuple):
        return self.select_by(self._selector_range(x_range, y_range))

    def select_by_name(self, *, starts_with: str | tuple[str, ...] = None, endswith: str | tuple[str, ...] = None, contains: str | tuple[str, ...] = None,
                       inverted=False):
        total = (starts_with is not None) + (endswith is not None) + (contains is not None)
        if total == 0:
            raise ValueError('At least one of starts_with, endswith or contains should be provided')
        if total > 1:
            raise ValueError('Only one of starts_with, endswith or contains should be provided')
        if starts_with is not None:
            def func(dat): return dat.database_simplified.startswith(starts_with)
        elif endswith is not None:
            def func(dat): return dat.database_simplified.endswith(endswith)
        else:
            def func(dat): return contains in dat.database_simplified
        if inverted:
            def real_func(dat): return not func(dat)
        else:
            real_func = func
        return self.select_by(real_func)

    def average(self, energies, transition_size=25, transition_type='erf'):
        max_val = max([dat.energy[-1] for dat in self.data])
        if max(energies) > max_val:
            raise ValueError(f'Provided energy range {max(energies)} is larger than the maximum energy in the data: {max_val}')
        min_val = min([dat.energy[1] for dat in self.data])
        if min(energies) < min_val:
            raise ValueError(f'Provided energy range ({min(energies)}) is smaller than the minimum (non-zero) energy in the data: {min_val}')

        values = []
        total_weights = []
        for dat in self.data:
            mask = (0 < dat.energy) & (0 < dat.cross_section)
            interpolator = CubicSpline(np.log(dat.energy[mask]), np.log(dat.cross_section[mask]))
            interpolated = np.exp(interpolator(np.log(energies), extrapolate=False))
            values.append(interpolated)
            not_nan = np.isfinite(interpolated)
            start = np.argmax(not_nan)
            end = len(not_nan) - np.argmax(not_nan[::-1])
            indexes = np.arange(end-start)

            if transition_type.lower() == 'linear':
                weigth1 = (1/transition_size)*indexes + 1e-5
                weigth2 = (1/transition_size)*(indexes[::-1]) + 1e-5
                weight = np.minimum(np.minimum(weigth1, weigth2), np.ones(end-start))
            elif transition_type.lower() == 'erf':
                def erf(x, trans_size): return 0.5*(1 + scipy.special.erf((4*(x - trans_size/2)/trans_size)))
                weight1 = erf(indexes, transition_size)
                weight2 = erf(indexes[::-1], transition_size)
                weight = np.minimum(np.minimum(weight1, weight2), np.ones(end-start))
            elif transition_type.lower() == 'none' or transition_type is None:
                weight = np.ones(end-start)
            else:
                raise ValueError(f'Unknown transition type: {transition_type}. Should be `linear`, `erf` or `none`.')

            weights = np.zeros(len(energies))
            weights[start:end] = weight
            total_weights.append(weights)
        values = np.array(values)
        weights = np.array(total_weights)
        ma_array = np.ma.masked_array(values, np.isnan(values))
        return np.average(ma_array, axis=0, weights=weights)

    def diff(self, normalize=False):
        new_values = []
        for index, dat in enumerate(self.data):
            energies = (dat.energy[1:] + dat.energy[:-1])/2
            cross_sections = np.diff(dat.cross_section)/np.diff(dat.energy)
            if normalize:
                cross_sections /= max(abs(cross_sections))
            new_values.append(CrossSectionData(energies, cross_sections, dat.parameter, dat.process, dat.species, dat.database, dat.updated, dat.comments,
                                               dat.full_process, dat.all_species, dat.units))
        return CrossSectionCollection(new_values)

    @staticmethod
    def read_txt(loc) -> CrossSectionData | CrossSectionCollection:
        return CrossSectionData.read_txt(loc)


def plot_CrossSections(data: list[CrossSectionData] | CrossSectionCollection, *, name_simplified=True, plot_kwargs=None, legend_kwargs=True, line_kwargs=None,
                       show=True, rotate_markers=False, line_kwargs_iter=None, color_by: str = None, **kwargs):
    x_values = [dat.energy for dat in data]
    y_values = [dat.cross_section for dat in data]
    if name_simplified:
        labels = [dat.database_simplified for dat in data]
    else:
        labels = [dat.database for dat in data]

    plot_kwargs = plot.set_defaults(plot_kwargs, **{'xlabel': 'Energy [eV]', 'ylabel': 'Cross section [m$^2$]', 'yscale': 'log', 'xscale': 'log'})
    if (legend_kwargs is not None) and (legend_kwargs is not False):
        if legend_kwargs is True:
            legend_kwargs = {}
        legend_kwargs = plot.set_defaults(legend_kwargs, **{'title': 'Database'})
    else:
        legend_kwargs = None

    line_kwargs = plot.set_defaults(line_kwargs, **{'linestyle': '--', 'marker': 'o'})
    if line_kwargs_iter is None:
        line_kwargs_iter = [{} for _ in range(len(data))]
    if rotate_markers:
        for index in range(len(data)):
            line_kwargs_iter[index] = plot.set_defaults(line_kwargs_iter[index], **{'marker': linestyles.marker_cycler(index)})
    if color_by == 'process':
        color_by = [dat.process for dat in data]
        color_line_kwargs_iter = linestyles.linelook_by(color_by, colors=True)
        for index in range(len(line_kwargs_iter)):
            line_kwargs_iter[index] = plot.set_defaults(line_kwargs_iter[index], **color_line_kwargs_iter[index])
    elif color_by is not None:
        raise ValueError(f'Color by should be `process` or None, not {color_by}')

    return plot.lines(x_values, y_values, labels=labels, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs,
                      line_kwargs=line_kwargs, show=show, line_kwargs_iter=line_kwargs_iter, **kwargs)


def plot_averageCrossSections(data: CrossSectionCollection, *, energy_kwargs=None, average_kwargs=None, line_kwargs_avg=None,
                              line_kwargs=None, fig_ax=None, plot_kwargs=None, **kwargs):
    line_kwargs = plot.set_defaults(line_kwargs, **{'color': 'C0', 'linestyle': '-', 'linewidth': 1, 'marker': None})
    start = np.log10(min((dat.energy[1] for dat in data)))
    stop = np.log10(max((dat.energy[-1] for dat in data)))

    energy_kwargs = plot.set_defaults(energy_kwargs, **{'start': start, 'stop': stop, 'num': 1000})
    line_kwargs_avg = plot.set_defaults(line_kwargs_avg, **{'color': 'k', 'linestyle': '-', 'linewidth': 1, 'marker': None})
    average_kwargs = average_kwargs or {}

    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()
    energies = np.logspace(**energy_kwargs)
    ax.plot(energies, data.average(energies, **average_kwargs), 'k-', zorder=2.1)
    legend_kwargs = {'handles': [plt.Line2D([0], [0], **line_kwargs), plt.Line2D([0], [0], **line_kwargs_avg)],
                     'labels': ['Data', 'Average']}
    return plot_CrossSections(data, line_kwargs=line_kwargs, fig_ax=(fig, ax), plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs, **kwargs)


def plot_spread(data: list[CrossSectionData] | CrossSectionCollection, bins=10, *, histtype='step', show=True, close=False, log=False):
    if len(data) > 10:
        warnings.warn('More than 10 datasets are being plotted, not enough unique colors are available.')
    fig, ax = plt.subplots()
    for dat in data:
        energies = dat.energy[0 < dat.energy]
        weights = np.full(len(energies), 1/len(energies))
        ax.hist(energies, bins=bins, histtype=histtype, log=log, weights=weights, label=dat.database_simplified)
    ax.set_xscale('log')
    plt.legend()
    if show:
        plt.show()
    elif close:
        ax.close()
    else:
        return fig, ax

