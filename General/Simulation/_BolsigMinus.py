"""
This is a wrapper around the bolsigminus command line executable.

It has the same capability as Bolsig+ in running normal runs, for the meaning of all the setting, see the Bolsig+ documentation.
Next to normal runs it can also run with one extra variable (run_2D), where one of the setting of Bolsig+ can be varied.
The resulting output file are slightly different from those generated by bolsigminus, since before the normal output the
conditions of the simulation are prepended. The data can be analyzed using the code in Bolsig_plus.py
"""


import subprocess
from typing import Sequence, Mapping
from typing import Literal as TypeLiteral
from tempfile import NamedTemporaryFile
import pathlib
import warnings
import os
import time
import threading
from dataclasses import dataclass

from General.Properties import property_literal, property_float, property_positive_int
from General.Data_handling import check_literal, check_positive_int, check_positive_float
from General.Descriptors import Positive, Integer, Float, Literal, Default

z_o = TypeLiteral[0, 1]
z_t = TypeLiteral[0, 1, 2]
o_t = TypeLiteral[1, 2]
o_tr= TypeLiteral[1, 2, 3]
o_f = TypeLiteral[1, 2, 3, 4]
o_s = TypeLiteral[1, 2, 3, 4, 5, 6]


class _VarVal:
    """
    This class is used to put the VAR value easily into the input file for the BolsigMinus simulation.
    """
    def __format__(self, format_spec):
        return 'VAR'


class BolsigMinus:  # class would be better as a dataclass with descriptor-typed fields
    E_field = property_float('_E_field', 'Electric field / N (Td)')
    tgas = property_float('_tgas', 'Gas temperature (K)')
    A_field = property_float('_A_field', 'Angular field frequency / N (m3/s)')
    cos_field = property_float('_cos_field', 'Cosine of E-B field angle')
    texc = property_float('_texc', 'Excitation temperature (K)')
    trans_energy = property_float('_trans_energy', 'Transition energy (eV)')
    ion_degree = property_float('_ion_degree', 'Ionization degree')
    plasma_density = property_float('_plasma_density', 'Plasma density (1/m3)')
    ion_charge_parameter = property_float('_ion_charge_parameter', 'Ion charge parameter')
    ion_neutral_mass_ratio = property_float('_ion_neutral_mass_ratio', 'Ion/neutral mass ratio')
    e_e_momentum_effect = property_float('_e_e_momentum_effect', 'e-e momentum effects: 0=No; 1=Yes')
    energy_sharing = property_float('_energy_sharing', 'Energy sharing: 1=Equal; 2=One takes all')
    growth = property_float('_growth', 'Growth: 1=Temporal; 2=Spatial; 3=Not included; 4=Grad-n expansion')
    maxwellian_mean_energy = property_float('_maxwellian_mean_energy', 'Maxwellian mean energy (eV)')
    n_grid = property_positive_int('_n_grid', '# of grid points')
    manual_grid = property_literal([0, 1, 2], '_manual_grid', 'Manual grid: 0=No; 1=Linear; 2=Parabolic')
    max_energy = property_float('_max_energy', 'Manual maximum energy (eV)')
    precision = property_float('_precision', 'Precision')
    convergence = property_float('_convergence', 'Convergence')
    max_iterations = property_positive_int('_max_iterations', 'Maximum # of iterations')

    def __init__(self, bolsig_loc, species_files: Mapping[str, list[str] | str], gas_fractions: Sequence[float | int],
                 *, E_field=0.0, tgas=300., A_field=0., cos_field=0., texc=0., trans_energy=0., ion_degree=1e-6, plasma_density=1e25, ion_charge_parameter=1,
                 ion_neutral_mass_ratio=1, e_e_momentum_effect: z_o = 1, energy_sharing: o_t = 1, growth: o_f = 1, maxwellian_mean_energy=0.0, n_grid=200,
                 manual_grid: z_t = 0, max_energy=200., precision=1e-10, convergence=1e-4, max_iterations=2000):
        """
        Initialize the BolsigMinus simulation.

        Parameters
        ----------
        bolsig_loc : str
            The location of the bolsigminus executable.
        species_files : Mapping[str, list[str] | str]
            A mapping of the location of the collision file to species to import from that file. If the file contains multiple
            species, the species should be in a list.

        Notes
        -----
        Other parameters are described in the attributes

        """
        self._species = []
        for specie in species_files.values():
            if isinstance(specie, str):
                self._species.append(specie)
            else:
                self._species.extend(specie)

        self.bolsig_loc = bolsig_loc
        self.species_files = species_files
        self.gas_fractions = gas_fractions
        self.E_field = E_field
        self.tgas = tgas
        self.A_field = A_field
        self.cos_field = cos_field
        self.texc = texc
        self.trans_energy = trans_energy
        self.ion_degree = ion_degree
        self.plasma_density = plasma_density
        self.ion_charge_parameter = ion_charge_parameter
        self.ion_neutral_mass_ratio = ion_neutral_mass_ratio
        self.e_e_momentum_effect = e_e_momentum_effect
        self.energy_sharing = energy_sharing
        self.growth = growth
        self.maxwellian_mean_energy = maxwellian_mean_energy
        self.n_grid = n_grid
        self.manual_grid = manual_grid
        self.max_energy = max_energy
        self.precision = precision
        self.convergence = convergence
        self.max_iterations = max_iterations

    @property
    def gas_fractions(self):
        """The gas fractions of the species in the simulation. The sum of the gas fractions must be equal to 1."""
        return self._gas_fractions

    @gas_fractions.setter
    def gas_fractions(self, value):
        if not sum(value) == 1:
            raise ValueError('Sum of `gas_fractions` must be equal to 1')
        if len(value) != len(self._species):
            raise ValueError(f'Number of species ({len(self._species)})  and number of gas_fractions ({len(value)}) must be equal')
        self._gas_fractions = value

    def _write_conditions(self, file_handle):
        file_handle.write('CLEARCOLLISIONS\n')
        file_handle.write('\n')
        for key, value in self.species_files.items():
            file_handle.write('READCOLLISIONS\n')
            file_handle.write(f'{key}\n')
            if isinstance(value, str):
                file_handle.write(f'{value}\n')
            else:
                for val in value:
                    file_handle.write(f'{val} ')
                file_handle.write('\n')
        file_handle.write('\n')
        file_handle.write('CONDITIONS\n')
        file_handle.write(f'{self.E_field:f}\n'
                   f'{self.A_field:f}\n'
                   f'{self.cos_field:f}\n'
                   f'{self.tgas:f}\n'
                   f'{self.texc:f}\n'
                   f'{self.trans_energy:f}\n'
                   f'{self.ion_degree:f}\n'
                   f'{self.plasma_density:e}\n'
                   f'{self.ion_charge_parameter:f}\n'
                   f'{self.ion_neutral_mass_ratio:f}\n'
                   f'{self.e_e_momentum_effect}\n'
                   f'{self.energy_sharing}\n'
                   f'{self.growth}\n'
                   f'{self.maxwellian_mean_energy:f}\n'
                   f'{self.n_grid}\n'
                   f'{self.manual_grid}\n'
                   f'{self.max_energy:f}\n'
                   f'{self.precision:e}\n'
                   f'{self.convergence:e}\n'
                   f'{self.max_iterations}\n'
                   )
        for item in self.gas_fractions[:-1]:
            file_handle.write(f'{item:f} ')
        file_handle.write(f'{self.gas_fractions[-1]}\n')
        file_handle.write('1\n')
        file_handle.write('\n')

    @staticmethod
    def _write_save(file, save_loc: str, save_format: o_s = 1, save_conditions: z_o = 1, save_transport_coefficients: z_o = 1,
                    save_rate_coefficients: z_o = 1, save_rate_coefficients_inv: z_o = 1, save_energy_loss_coeffs: z_o = 0,
                    save_eedf: z_o = 1, skip_failed: z_o = 0):
        check_literal(save_format, [1, 2, 3, 4, 5, 6], 'save_format')
        check_literal(save_conditions, [0, 1], 'save_conditions')
        check_literal(save_transport_coefficients, [0, 1], 'save_transport_coefficients')
        check_literal(save_rate_coefficients, [0, 1], 'save_rate_coefficients')
        check_literal(save_rate_coefficients_inv, [0, 1], 'save_rate_coefficients_inv')
        check_literal(save_energy_loss_coeffs, [0, 1], 'save_energy_loss_coeffs')
        check_literal(save_eedf, [0, 1], 'save_eedf')
        check_literal(skip_failed, [0, 1], 'skip_failed')

        if ' ' in save_loc:
            if save_loc[0] not in ("'", '"'):
                save_loc = f"'{save_loc}'"

        file.write(f'SAVERESULTS\n')
        file.write(f'{save_loc}\n')
        file.write(f'{save_format}\n'
                   f'{int(save_conditions)}\n'
                   f'{int(save_transport_coefficients)}\n'
                   f'{int(save_rate_coefficients)}\n'
                   f'{int(save_rate_coefficients_inv)}\n'
                   f'{int(save_energy_loss_coeffs)}\n'
                   f'{int(save_eedf)}\n'
                   f'{int(skip_failed)}\n')
        file.write('\n')

    @staticmethod
    def print_progress(file_loc, total_num=None, preprint=''):
        def mktime(time_stamp):
            if time_stamp > 3600:
                return f'{int(time_stamp//3600):02}:{int(time_stamp%3600//60):02}:{int(time_stamp%60):02}'
            else:
                return f'{int(time_stamp//60):02}:{int(time_stamp%60):02}'
        max_len = len(str(total_num))
        start_time = time.time()
        done = False
        num = 0
        while not done:
            if not pathlib.Path(file_loc).exists():
                time.sleep(1)
                continue
            contents = pathlib.Path(file_loc).read_text().split('\n')
            for line in contents[::-1]:
                if 'FINISHED' in line:
                    done = True
                    break
                elif line.startswith('R') and line[1].isdigit():
                    new_num = int(line.split()[0][1:])
                    if new_num < num:
                        done = True
                        break
                    num = new_num

                    time_stamp = time.time()-start_time
                    if total_num is not None:
                        print('\r', preprint, f'{num:>{max_len}}/{total_num:>{max_len}} done after {mktime(time_stamp)}', end='', flush=True)
                    else:
                        print('\r', preprint, f'{num:} done after {mktime(time_stamp)}', end='', flush=True)
                    break
            time.sleep(1)
        print('\r', preprint, f'Done in {mktime(time.time()-start_time)}')

    def _run(self, write_between: list[str], save=True, *, print_stdout=False, print_progress=True, pre_print='', run_num=None, save_kwargs=None):
        with NamedTemporaryFile('w+', delete_on_close=False) as file:
            self._write_conditions(file)
            file.writelines(write_between)
            if save:
                save_kwargs = save_kwargs or {}
                self._write_save(file, **save_kwargs)
            file.write('END\n')
            file.close()
            dir_path = os.path.dirname(os.path.realpath(__file__))
            if print_progress:
                bolsiglog_loc = f'{dir_path}/bolsiglog.txt'
                if pathlib.Path(bolsiglog_loc).exists():
                    os.remove(bolsiglog_loc)
                thread = threading.Thread(target=self.print_progress, args=(bolsiglog_loc, run_num, pre_print))
                thread.start()
            values = subprocess.run([self.bolsig_loc, file.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dir_path)
            if print_progress:
                thread.join()
            if 'Error in input file line' in values.stdout.decode():
                print('Input: \n', )
                for index, line in enumerate(pathlib.Path(file.name).read_text().split('\n'), 1):
                    print(index, ' ', line)
                print('Output: \n', values.stdout.decode())
                raise ValueError(f'Error in BolsigMinus')
            if print_stdout:
                print(values.stdout.decode())
        return values

    def run_series(self, save_loc: str, variable_min_max: Sequence[float | int], *, save_format: o_s = 1, save_conditions: z_o = 1, save_rate_coefficients: z_o = 1,
                   save_transport_coefficients: z_o = 1, save_rate_coefficients_inv: z_o = 1, save_energy_loss_coeffs: z_o = 0, save_eedf: z_o = 1, num=200,
                   run_variable: o_tr = 1, growth_type: o_tr = 3, skip_failed: z_o = 0, print_stdout=False, pre_print=''):
        check_literal(run_variable, [1, 2, 3], 'run_variable')
        check_literal(growth_type, [1, 2, 3], 'growth_type')
        check_positive_int(num, 'num')

        write_between = [f'RUNSERIES\n',
                         f'{int(run_variable)}\n',
                         f'{variable_min_max[0]:f} {variable_min_max[1]:f}\n',
                         f'{int(num)}\n',
                         f'{int(growth_type)}\n',
                         '\n']
        save_kwargs = {'save_loc': save_loc, 'save_format': save_format, 'save_conditions': save_conditions, 'save_rate_coefficients': save_rate_coefficients,
                       'save_transport_coefficients': save_transport_coefficients, 'save_rate_coefficients_inv': save_rate_coefficients_inv,
                       'save_energy_loss_coeffs': save_energy_loss_coeffs, 'save_eedf': save_eedf, 'skip_failed': skip_failed}
        return self._run(write_between, run_num=num, save_kwargs=save_kwargs, print_stdout=print_stdout, pre_print=pre_print)

    def run_2D(self, save_loc: str, variable_min_max: Sequence[float | int], variable2: tuple[tuple[float, float], int, int], variable_name: str, *, num=200,
               run_variable: o_tr = 1, growth_type: o_tr = 3, print_stdout=False, pre_print=''):
        if variable2[0][0] > variable2[0][1]:
            warnings.warn('variable2 from big to small is not recommended')

        check_literal(run_variable, [1, 2, 3], 'run_variable')
        check_literal(growth_type, [1, 2, 3], 'growth_type')

        if not hasattr(self, variable_name):
            raise ValueError(f'Variable {variable_name} is not an attribute of BolsigMinus')
        value = getattr(self, variable_name)
        setattr(self, variable_name, _VarVal())

        check_positive_int(num, 'num')
        check_positive_float(variable2[0][0], 'variable2[0][0]')
        check_positive_float(variable2[0][1], 'variable2[0][1]')
        check_positive_int(variable2[1], 'variable2[1]')
        check_positive_int(variable2[2], 'variable2[2]')
        with NamedTemporaryFile('w+', delete_on_close=False) as file:
            write_between = [f'RUN2D\n',
                             f'{int(run_variable)}\n',
                             f'{variable_min_max[0]:e} {variable_min_max[1]:e} {variable2[0][0]:e} {variable2[0][1]:e}\n',
                             f'{int(num)} {int(variable2[1])}\n',
                             f'{int(growth_type)} {int(variable2[2])}\n',
                             f"'{file.name}'\n",
                             '\n']
            file.close()
            result = self._run(write_between, save=False, print_stdout=print_stdout, run_num=num*variable2[1], pre_print=pre_print)
            output = pathlib.Path(file.name).read_text()
        setattr(self, variable_name, value)

        with open(save_loc, 'w+') as file:
            self._write_output_conditions(file)
            file.write('------------------------------------------------------------\n')
            file.write(output)

        return result

    def _write_output_conditions(self, file_handle):
        file_handle.write('Collision input data:\n')
        for key, value in self.species_files.items():
            file_handle.write('--------------------------------------------------\n')
            if isinstance(value, str):
                file_handle.write(f'{value}\n')
            else:
                for val in value:
                    file_handle.write(f'{val} ')
                file_handle.write('\n')
            file_handle.write(f'{key}\n')
        file_handle.write('\n\n\n')

        file_handle.write(f'Electric field / N (Td)    {self.E_field:e}\n')
        file_handle.write(f'Angular field frequency / N (m3/s)    {self.A_field}\n')
        file_handle.write(f'Cosine of E-B field angle    {self.cos_field}\n')
        file_handle.write(f'Gas temperature (K)    {self.tgas}\n')
        file_handle.write(f'Excitation temperature (K)    {self.texc}\n')
        file_handle.write(f'Transition energy (eV)    {self.trans_energy}\n')
        file_handle.write(f'Ionization degree    {self.ion_degree:e}\n')
        file_handle.write(f'Plasma density (1/m3)    {self.plasma_density:e}\n')
        file_handle.write(f'Ion charge parameter    {self.ion_charge_parameter}\n')
        file_handle.write(f'Ion/neutral mass ratio    {self.ion_neutral_mass_ratio}\n')
        file_handle.write(f'Coulomb collision model    {self.e_e_momentum_effect}\n')
        file_handle.write(f'Energy sharing    {self.energy_sharing}\n')
        file_handle.write(f'Growth    {self.growth}\n')
        file_handle.write(f'Maxwellian mean energy (eV)    {self.maxwellian_mean_energy}\n')
        file_handle.write(f'# of grid points    {self.n_grid}\n')
        file_handle.write(f'Grid type    {self.manual_grid}\n')
        file_handle.write(f'Maximum energy (eV)    {self.max_energy}\n')
        file_handle.write(f'Precision    {self.precision:e}\n')
        file_handle.write(f'Convergence    {self.convergence:e}\n')
        file_handle.write(f'Maximum # of iterations    {self.max_iterations}\n')
        for name, fraction in zip(self._species, self.gas_fractions, strict=True):
            file_handle.write(f'Mole fraction {name}    {fraction}\n')
