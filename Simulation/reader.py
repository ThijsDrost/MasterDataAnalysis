from typing import Union
import math
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from General.Plotting.Plot import Plot


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

    def _set_distance(self, dist_meter):
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

    def var_vs_time(self, variable: str, *, cmap='jet', time_scale='log', order=1, cbarticks=False):
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
        plt.show()

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

        self._setting_setter(ax, **(plot_kwargs or {}))
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
        self._setting_setter(ax, **(plot_kwargs or {}))
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

        relative_dens = re.search('plas\.w.{1,}\(1\)', name)
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
    test = Comsol1DAnalyzer(r'C:\Users\20222772\Downloads\Export.txt')
