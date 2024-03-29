"""
This subpackage contains the ComsolAnalyzer class, which is used to import and plot 0D/1D txt data exported from Comsol. Also
contains the old Comsol1DAnalyzer class, which can be used to import and plot 1D data exported from Comsol.
"""

from __future__ import annotations

import math
import re
import pathlib
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from General.Plotting._Plot import Plot


class ComsolAnalyzer:
    def __init__(self, model: str, date: str, dimensions: int, variables: list[str], data: dict[str, pd.DataFrame], dimension_variable, variable):
        self.model = model
        self.date = date
        self.dimensions = dimensions
        self.variables = variables
        self.data = data
        self.dimension_variable = dimension_variable
        self.variable = variable

    @staticmethod
    def read_txt(loc, delimiter: str | None = '\t') -> ComsolAnalyzer:
        data = pathlib.Path(loc).read_text().splitlines()
        model = data[0].split(':', 1)[1].strip()
        date = data[2].split(':', 1)[1].strip()
        dimension = int(data[3].split(':')[1].strip())
        if dimension > 1:
            raise NotImplementedError('Only 0D and 1D data is supported')
        nodes = int(data[4].split(':')[1].strip())
        expressions = int(data[5].split(':')[1].strip())
        names = [name.strip() for name in data[6].removeprefix('% Description:').split(',')]
        variables = [name.strip() for name in data[7][1:].split(delimiter) if name.strip()]
        try:
            num_data = np.array([[float(val.strip()) for val in line.split(delimiter) if val.strip()] for line in data[8:]])
        except ValueError as e:
            raise ValueError('Could not convert data to float, most likely wrong delimiter') from e

        if len(variables) == 1:
            warnings.warn('Length of variable names is 1, most likely wrong delimiter')

        if dimension == 1:
            dim_var = variables[0]
            variables = variables[1:]
            dim_data = num_data[:, 0]
            num_data = num_data[:, 1:]
        elif dimension == 0:
            dim_var = None
            dim_data = [0]
        else:
            raise ValueError('Only 0D and 1D data is supported')

        # check for data consistency
        if len(variables) != expressions:
            if len(variables) == 3*expressions:
                raise ValueError('Wrong delimiter, variable is split in three')
            raise ValueError('Number of variables does not match number of expressions')
        if len(num_data) != nodes:
            raise ValueError('Number of data points does not match number of nodes')
        if len(num_data[0]) != expressions:
            raise ValueError('Number of data points does not match number of expressions')

        unique_vars = []
        for var in variables:
            x = var.split('@')[0]
            if x not in unique_vars:
                unique_vars.append(x)
            else:
                break
        extra_var = variables[0].split('@')[1].split('=')[0].strip()
        extra_var_values_all = np.array([float(var.split('@')[1].split('=')[1]) for var in variables])
        extra_var_values_all_T = extra_var_values_all.reshape((-1, len(unique_vars)))

        for var in extra_var_values_all_T:
            if not np.allclose(var, var[0]):
                raise ValueError('Extra variable values are not the same for all variables')

        extra_vars = extra_var_values_all_T[:, 0]

        data_dict = {var: pd.DataFrame(num_data[:, index::len(unique_vars)], columns=extra_vars, index=dim_data) for index, var in enumerate(unique_vars)}

        if len(unique_vars) != len(names):
            raise ValueError('Number of variables does not match number of names')

        return ComsolAnalyzer(model, date, dimension, names, data_dict, dim_var, extra_var)

    def plot_vs_vars(self, x_name, y_name, *, plot_kwargs=None, **kwargs):
        x_keys = [key for val, key in zip(self.variables, self.data.keys()) if ((x_name in key) or (x_name in val))]
        y_keys = [key for val, key in zip(self.variables, self.data.keys()) if ((y_name in key) or (y_name in val))]
        x = [self.data[variable].iloc[0].values for variable in x_keys][0]
        y = [self.data[variable].iloc[0].values for variable in y_keys]
        plot_kwargs = Plot.set_defaults(plot_kwargs, xlabel=self.dimension_variable, ylabel=self.variable)
        return Plot.lines(x, y, plot_kwargs=plot_kwargs, **kwargs)


class Comsol1DAnalyzer(Plot):
    def __init__(self, loc, *, delimiter=';', comments='%'):
        with open(loc, 'r') as f:
            lines = []
            while (line := f.readline()).startswith(comments):
                lines.append(line.replace(comments, '').removesuffix('\n'))
        self.model = lines[0].split(':')[1].strip()
        self.date = lines[2].split(':')[1].strip()
        self.dimension = int(lines[3].split(':')[1].strip())
        if self.dimension != 1:
            raise ValueError('Only 1D data is supported')
        self._nodes = int(lines[4].split(':')[1].strip())
        self._expressions = int(lines[5].split(':')[1].strip())
        self.variables = lines[6].split(':')[1].strip().split(', ')
        if not lines[7].split(';')[0].strip() == 'X':
            raise ValueError('First row of data must be X')

        self._times = []
        self._data_vals = []
        last_time = None
        for val in lines[7].split(';')[1:]:
            if '@' not in val:
                raise ValueError(f'All data descriptors must be in the form "value@time", got: {val}')
            val, time_val = val.split(' @ ')
            self._data_vals.append(self._data_name(val))
            if last_time != time_val:
                self._times.append(float(time_val.split('=')[1]))
                last_time = time_val

        self.data = {x: [] for x in set(self._data_vals)}
        data = np.genfromtxt(loc, delimiter=delimiter, comments=comments, unpack=True)
        self._x_name, self.x = self._set_distance(data[0])
        for dat, val in zip(data[1:], self._data_vals, strict=True):
            self.data[val].append(dat)

        for key, value in self.data.items():
            self.data[key] = np.array(value)
            if not self.data[key].shape == (len(self._times), self._nodes):
                raise ValueError('Data has wrong shape')

    @staticmethod
    def _set_distance(dist_meter):
        order_of_mag = math.floor(math.log10(max(dist_meter)))
        if order_of_mag >= 0:
            name, val = 'm', 1
        elif order_of_mag >= -2:
            name, val = 'cm', 1e2
        elif order_of_mag >= -3:
            name, val = 'mm', 1e3
        elif order_of_mag >= -6:
            name, val = 'um', 1e6
        else:
            name, val = 'nm', 1e9
        return f'Locations [{name}]', dist_meter*val

    def _normalizer(self, time_scale: str):
        if time_scale == 'log':
            return mpl.colors.LogNorm(vmin=self._times[1], vmax=self._times[-1])
        elif time_scale == 'linear':
            return mpl.colors.Normalize(vmin=self._times[1], vmax=self._times[-1])
        else:
            raise ValueError(f'Unknown time scale: {time_scale}')

    def _color(self, index, order, cmap, normalizer):
        return cmap(normalizer(self._times[::order][index]))

    def var_vs_time(self, variable: str, *, cmap='jet', time_scale='log', order=1, cbarticks=False, show=False):
        normalizer = self._normalizer(time_scale)
        cmap = plt.get_cmap(cmap)
        fig, ax = plt.subplots()
        name, data = self.data_getter(variable, order)
        for index, dat in enumerate(data[1:]):
            plt.plot(self.x, dat, color=self._color(index, order, cmap, normalizer))
        plt.xlabel(self._x_name)
        plt.ylabel(name)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalizer)
        cbar = plt.colorbar(sm, label='Time (s)', ax=ax)
        if cbarticks:
            self._colorbar_ticker(cbar)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            return fig, ax

    def vars_vs_time(self, variable: tuple[str, str], *, cmap='jet', time_scale='log', order=1, plot_kwargs=None,
                     cbarticks=False):
        normalizer = self._normalizer(time_scale)
        cmap = plt.get_cmap(cmap)
        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        name1, data1 = self.data_getter(variable[0], order)
        for index, dat in enumerate(data1[1:]):
            ax.plot(self.x, dat, color=self._color(index, order, cmap, normalizer), linestyle='--')
        ax.set_ylabel(name1)

        name2, data2 = self.data_getter(variable[1], order)
        for index, dat in enumerate(data2[1:]):
            ax2.plot(self.x, dat, color=self._color(index, order, cmap, normalizer), linestyle=':')
        ax2.set_ylabel(name2)

        ax.set_xlabel(self._x_name)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalizer)
        plt.legend([plt.Line2D([0], [0], linestyle='--', color='black'),
                          plt.Line2D([0], [0], linestyle=':', color='black')], [name1, name2])

        cbar = plt.colorbar(sm, label='Time (s)', ax=ax)
        if cbarticks:
            self._colorbar_ticker(cbar)

        self.setting_setter(ax, **(plot_kwargs or {}))
        plt.tight_layout()
        plt.show()

    def diff_var_vs_time(self, variables: tuple[str, str], rel_strength: tuple[float | int, float | int] = (1, 1),
                         *, cmap='jet', time_scale='log', order=1, plot_kwargs: dict = None, cbarticks=False):
        normalizer = self._normalizer(time_scale)
        cmap = plt.get_cmap(cmap)
        fig, ax = plt.subplots()
        name1, data1 = self.data_getter(variables[0], order)*rel_strength[0]
        name2, data2 = self.data_getter(variables[1], order)*rel_strength[1]
        for index, (dat1, dat2) in enumerate(zip(data1[1:], data2[1:])):
            ax.plot(self.x, dat1-dat2, color=self._color(index, order, cmap, normalizer))
        plt.xlabel(self._x_name)
        plt.ylabel(f'{name1} - {name2}')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalizer)
        cbar = plt.colorbar(sm, label='Time (s)', ax=ax)
        if cbarticks:
            self._colorbar_ticker(cbar)
        self.setting_setter(ax, **(plot_kwargs or {}))
        plt.tight_layout()
        plt.show()

    def _colorbar_ticker(self, cbar, linewidth=0.25):
        axis = cbar.ax
        for t in self._times:
            axis.plot([0, 1], [t, t], 'k', linewidth=linewidth)


    @staticmethod
    def _data_name(name: str):
        variable, unit = name.split(' ')
        unit = unit.replace('(', '[').replace(')', ']')

        # replace 1/[..]^n with [...]^-n
        value = re.search(r'1/(.{1,})\^([0-9]{1,})', unit)
        if value is not None:
            replacement = f'{value.group(1)}$^{{-{value.group(2)}}}$'
            unit = re.sub(r'(1/.{1,}\^)([0-9]{1,})', replacement, unit)

        vars = {'plas.ne': 'Electron density', 'plas.Te': 'Electron temperature', 'V': 'Electric potential'}
        if variable in vars.keys():
            variable = vars[variable]

        relative_dens = re.search(r'plas\.w.+\(1\)', name)
        if relative_dens is not None:
            variable = variable.replace('plas.w', 'Relative ')
            unit = 'density'

        variable = variable.replace('plas.n_w', '')

        charge = re.search(r'_([1-9])p', variable)
        if charge is not None:
            variable = re.sub(r'(_[1-9]p)', f'$^{{{charge.group(1)}+}}$', variable)

        return f'{variable} {unit}'

    def data_getter(self, name: str, order: int):
        if order not in (-1, 1):
            raise ValueError('Order must be -1 or 1')

        if name in self.data.keys():
            return name, self.data[name][::order]

        vals = {'ne': 'Electron density', 'te': 'Electron temperature', 'v': 'Electric potential'}
        if name.lower() in vals.keys():
            name = vals[name.lower()]

        for key in self.data.keys():
            if (name in key) or (name.lower() in key.lower()):
                return key, self.data[key][::order]

        raise ValueError(f'No data found with name {name}')


if __name__ == '__main__':
    test = ComsolAnalyzer.read_txt(r'C:\Users\20222772\Downloads\Export.txt', delimiter=';')
    test2 = Comsol1DAnalyzer(r'C:\Users\20222772\Downloads\Export.txt')
    # test = ComsolAnalyzer.read_txt(r'C:\Users\20222772\Downloads\Output1.txt', delimiter='  ')
    print('hi')
