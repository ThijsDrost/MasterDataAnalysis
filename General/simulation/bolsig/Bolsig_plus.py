"""
This subpackage contains three classes: BolsigRun, BolsigRuns, and Bolsig2DRun. BolsigRuns and Bolsig2DRun are used for reading Bolsig+
output, BolsigRun is for saving individual runs inside BolsigRuns. These functions are meant for reading the results from
BolsigMinus runs and 2Druns. The BolsigRuns can also read the output from normal Bolsig+ runs. The BolsigRuns and Bolsig2DRun
classes also contain some functions for plotting the results.
"""

import warnings
from pathlib import Path
import re

import numpy as np
import pandas as pd

from General.plotting import plot, linestyles


def read_parameters(data: list[str], delimiter='    '):
    """
    Read the parameters from the output of a Bolsig+ run

    Parameters
    ----------
    data
    delimiter

    Returns
    -------

    """
    data = [line.removesuffix('\n').removesuffix('\r') for line in data]
    data = [[val.strip() for val in line.split(delimiter) if val.strip() != ''] for line in data]
    name_to_param = {
        'Electric field / N (Td)': 'E_field',
        'Angular field frequency / N (m3/s)': 'angular_field_frequency',
        'Cosine of E-B field angle': 'cosine_EB_field_angle',
        'Gas temperature (K)': 'T_gas',
        'Excitation temperature (K)': 'T_exc',
        'Transition energy (eV)': 'transition_energy',
        'Ionization degree': 'ionization_degree',
        'Plasma density (1/m3)': 'plasma_density',
        'Ion charge parameter': 'ion_charge_parameter',
        'Ion/neutral mass ratio': 'ion_neutral_mass_ratio',
        'Coulomb collision model': 'coulomb_collision_model',
        'Energy sharing': 'energy_sharing',
        'Growth': 'growth',
        'Maxwellian mean energy (eV)': 'maxwellian_mean_energy',
        '# of grid points': 'n_grid_points',
        'Grid type': 'grid_type',
        'Maximum energy (eV)': 'max_energy',
        'Precision': 'precision',
        'Convergence': 'convergence',
        'Maximum # of iterations': 'max_iterations',
    }
    output = {'mole_fractions' : {}}
    for line in data:
        if line[0] in name_to_param:
            output[name_to_param[line[0]]] = float(line[1])
        elif line[0].startswith('Mole fraction'):
            output['mole_fractions'][line[0].removeprefix('Mole fraction ')] = float(line[1])
        else:
            raise ValueError(f'Unknown parameter: {line[0]}')
    return output


class BolsigRun:
    """
    A class to store the results of a single Bolsig+ run
    """
    def __init__(self, energies, eedf, rate_coeffs, inv_rate_coeffs, *, run_num, E_field, T_gas, ionization_degree, plasma_density, mole_fractions,
                 mean_energy, extrapolated: bool, **kwargs):
        self.energies = energies
        self.eedf = eedf
        self.E_field = E_field
        self.ionization_degree = ionization_degree
        self.plasma_density = plasma_density
        self.mole_fractions = mole_fractions
        self.run_num = run_num
        self.T_gas = T_gas
        self.rate_coeffs = rate_coeffs
        self.inv_rate_coeffs = inv_rate_coeffs
        self.mean_energy = mean_energy
        self.extrapolated = extrapolated
        
        for key, value in kwargs.items():
            warnings.warn(f'Unknown keyword argument: {key}, set via setattr()')
            setattr(self, key, value)

    @staticmethod
    def read_bolsig(data: list[str], read_failed=False):
        data = [line.removesuffix('\n').removesuffix('\r') for line in data]
        if not (data[0].startswith('R') and data[1].startswith('-----')):
            if not data[0].startswith('R'):
                raise ValueError(f'Invalid data, first line does not start with "R": {data[0]}')
            if not data[1].startswith('-----'):
                raise ValueError(f'Invalid data, second line does not start with "-----": {data[1]}')
        run_num = int(data[0][1:])

        def readline(lines, index, name, splitter, return_type=float):
            if not lines[index].startswith(name):
                raise ValueError(f'Invalid data, line {index+1} does not start with "{name}": {lines[index]}')
            return return_type(lines[index].split(splitter)[1].strip())

        E_field = readline(data, 2, 'Electric field', '(Td)')
        T_gas = readline(data, 5, 'Gas temperature', '(K)')
        ionization_degree = readline(data, 8, 'Ionization degree', 'degree')
        plasma_density = readline(data, 9, 'Plasma density', '(1/m3)')
        # max_energy = readline(data, 18, 'Maximum energy', 'energy')
        # if math.isclose(max_energy, -1):
        #     if not read_failed:
        #         return None

        mole_index = 0
        mole_fraction = {}
        while True:
            if 'Mole fraction' in data[22 + mole_index]:
                _, _, start, end = data[22 + mole_index].split()
                mole_fraction[start] = float(end.strip())
            mole_index += 1
            if data[22 + mole_index].startswith('-----'):
                break

        start_index = 23 + mole_index

        def read_EEDF(data):
            energies = []
            eedfs = []
            for dat in data:
                if dat.startswith('-----'):
                    raise ValueError(f'Invalid data, line starts with "-----": {dat}')
                if dat.strip() == '':
                    raise ValueError(f'Invalid data, line is empty: {dat}')
                energy, eedf, _ = dat.split()
                energies.append(float(energy.strip()))
                eedfs.append(float(eedf.strip()))
            return np.array(energies), np.array(eedfs)

        def rate_coeffs(data):
            coeffs = {}
            for dat in data:
                if not dat.startswith('C'):
                    raise ValueError(f'Invalid data, line {index+1} does not start with "C": {dat}')
                if 'eV' in dat:
                    _, specie, name, energy, _, rate = dat.split()
                else:
                    _, specie, name, rate = dat.split()
                coeffs[specie, name] = float(rate)
            return coeffs

        def transport_coeffs(data):
            data_names = ('Mean energy', 'Error code')
            coeffs = {}
            for dat in data:
                for name in data_names:
                    if dat.startswith(name):
                        coeffs[name] = float(dat[42:].strip())
                        break
            return coeffs

        index = start_index
        name = data[index]

        energies, eedf = None, None
        rate_coef = None
        inv_rate_coeffs = None
        trans_coeffs = None

        while True:
            if data[index].startswith('-----') or (data[index].strip() == ''):
                if name.startswith('Energy (eV) EEDF (eV-3/2) Anisotropy'):
                    energies, eedf = read_EEDF(data[start_index:index])
                if name.startswith('Rate coefficients'):
                    rate_coef = rate_coeffs(data[start_index:index])
                if name.startswith('Inverse rate coefficients'):
                    inv_rate_coeffs = rate_coeffs(data[start_index:index])
                if name.startswith('Mean energy'):
                    trans_coeffs = transport_coeffs(data[start_index:index])

                if (index == len(data)-1):
                    break
                start_index = index + 2
                name = data[index + 1]
            index += 1

        trans_coeffs = trans_coeffs or {}
        mean_energy = trans_coeffs.get('Mean energy', None)
        error_code = trans_coeffs.get('Error code', None)

        extrapolated = False
        if error_code is not None:
            if float(error_code) == -1:
                extrapolated = True
            elif float(error_code) != 0.0:
                if not read_failed:
                    return None

        # check data
        if rate_coef is not None:
            for key, value in rate_coef.items():
                if value < 0:
                    warnings.warn(f'Negative rate coefficient: {key}')

        return BolsigRun(energies, eedf, rate_coef, inv_rate_coeffs, E_field=E_field, T_gas=T_gas, ionization_degree=ionization_degree, plasma_density=plasma_density,
                         mole_fractions=mole_fraction, run_num=run_num, mean_energy=mean_energy, extrapolated=extrapolated)


class Bolsig2DRun:
    def __init__(self, data: dict[str, pd.DataFrame], row_name, column_name, parameters):
        self.data = data
        self.row_name = row_name
        self.column_name = column_name
        self.parameters = parameters

    @staticmethod
    def read_txt(loc):
        data = Path(loc).read_text().splitlines()

        parameters_indexes = None
        index = 0
        for index, line in enumerate(data):
            if line.startswith('Electric field / N (Td)'):
                parameters_indexes = (index, None)
            if parameters_indexes is not None and line.startswith('---------'):
                parameters_indexes = (parameters_indexes[0], index)
                break
        parameters = read_parameters(data[parameters_indexes[0]:parameters_indexes[1]])

        row_name = None
        for index, line in enumerate(data[index:], index):
            if 'Rows:' in line:
                row_name = line.split('Rows:')[1].strip()
                break

        row_values = []
        for index, line in enumerate(data[index+1:], index+1):
            if line.strip():
                row_values.append(float(line.strip()))
                # row_values.append(float(line.strip()))
            else:
                break

        column_name = None
        for index, line in enumerate(data[index:], index):
            if 'Columns:' in line:
                column_name = line.split('Columns:')[1].strip()
                break

        column_values = [float(val) for val in data[index+1].split()]

        def read_table(data_list):
            def to_float(val):
                try:
                    return float(val)
                except ValueError:
                    val = val.lower()
                    if ('+' in val) and ('e' not in val):
                        return float(val.replace('+', 'e+'))

            table = []
            for index, line in enumerate(data_list):
                if line.strip():
                    table.append([to_float(val) for val in line.split()])
                else:
                    raise ValueError(f'Invalid data, (sub)line {index} is empty')
            return np.array(table)

        index = index + 2
        output = {}
        while True:
            name = data[index+1].strip()
            start_index = index + 2
            index = start_index + 1

            while data[index].strip():
                index += 1
                if index >= len(data):
                    break
            if index >= len(data):
                break
            table = read_table(data[start_index:index])
            if table.shape != (len(column_values), len(column_values)):
                raise ValueError(f'Invalid data, table {name} has shape {table.shape}, expected ({len(column_values)}, {len(column_values)})')
            output[name] = pd.DataFrame(table, index=column_values, columns=column_values)
        return Bolsig2DRun(output, row_name, column_name, parameters)


class BolsigRuns:
    def __init__(self, runs: list[BolsigRun]):
        self.runs = runs
        self.particles = list(runs[0].rate_coeffs.keys())

    def _get_val(self, name):
        if not hasattr(self.runs[0], name):
            raise AttributeError(f'`{name}` is not an attribute of BolsigRun')
        if getattr(self.runs[0], name) is None:
            raise ValueError(f'{name} is not available')
        return [getattr(run, name) for run in self.runs]

    def _get_dict(self, name):
        if not hasattr(self.runs[0], name):
            raise AttributeError(f'`{name}` is not an attribute of BolsigRun')
        if (getattr(self.runs[0], name) is None) or (getattr(self.runs[0], name) == {}):
            raise ValueError(f'{name} is not available')
        return {part: [getattr(run, name)[part] for run in self.runs] for part in self.particles}

    def plot_cross_section_with_energy(self, *, plot_kwargs=None, legend_kwargs=True, show=True, **kwargs):
        cross_sections_dict = self._get_dict('rate_coeffs')
        cross_sections = list(cross_sections_dict.values())
        inv_cross_sections_dict = self._get_dict('inv_rate_coeffs')
        inv_cross_sections = list(inv_cross_sections_dict.values())

        cross_sections = np.array(cross_sections + inv_cross_sections)
        labels = [f'{' '.join(label)}' for label in cross_sections_dict.keys()] + [f'{' '.join(label)} (rev)' for label in inv_cross_sections_dict.keys()]
        mask = [np.any(val > 0) for val in cross_sections]
        cross_sections = cross_sections[mask]
        labels = [label for label, m in zip(labels, mask, strict=True) if m]

        energies = self._get_val('mean_energy')
        if legend_kwargs:
            legend_kwargs = plot.set_defaults(legend_kwargs, labels=labels)

        plot_kwargs = plot.set_defaults(plot_kwargs, xlabel='Energy (eV)', ylabel='Cross section (m^3/s)', yscale='log', xscale='log', ylim=(1e-23, None))

        if 'line_kwargs' not in kwargs:
            line_kwargs_iter = linestyles.linelook_by(labels, linestyles=True, colors=True)
        else:
            line_kwargs_iter = None

        return plot.lines(energies, cross_sections, labels=labels, plot_kwargs=plot_kwargs, legend_kwargs=legend_kwargs, line_kwargs_iter=line_kwargs_iter,
                          show=show, **kwargs)

    @staticmethod
    def read_file(loc: str):
        data = Path(loc).read_text().splitlines()
        runs = []
        index = 0

        while True:
            if re.search(r'(R\d+)', data[index]) is not None:
                start = index
                num = int(data[index][1:])
                break
            if index == len(data):
                raise ValueError('No run found')
            index += 1
        index += 1

        while index < len(data):
            if data[index].startswith(f'R{num+1}'):
                run = BolsigRun.read_bolsig(data[start:(index - 1)])
                runs.append(run)
                start = index
                num += 1
            index += 1

        runs = [run for run in runs if run is not None]
        return BolsigRuns(runs)

