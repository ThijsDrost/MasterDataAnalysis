"""
This subpackage contains classes for storing and testing chemical reactions.
"""

import re


class Molecule:
    atoms = ['Ar', 'He', 'H', 'N', 'O', 'e', 'M']

    def __init__(self, atoms: dict[str, int], *, excited=False, charge=0):
        self.atoms = atoms
        self.charge = charge
        self.excited = excited

    @staticmethod
    def from_string(molecule: str):
        atoms = {}
        excited = False

        if isinstance(molecule, str):
            for atom in Molecule.atoms:
                found_atom = re.findall(f'({atom})[1-9]?', molecule)
                if found_atom:
                    num = re.findall(f'{atom}([1-9])', molecule)
                    nums = sum([int(n) for n in num]) + len(found_atom) - len(num)
                    atoms[atom] = nums
                else:
                    continue

            if '*' in molecule:
                excited = True
            plusses = re.findall(r'\+', molecule)
            minusses = re.findall(r'−', molecule)
            minusses2 = re.findall(r'-', molecule)
            charge = len(plusses) - len(minusses) - len(minusses2)
        else:
            raise TypeError(f'Can only convert str to Molecule, not {type(molecule)}')

        if atoms == {'e': 1} and charge == 0:
            charge = -1

        return Molecule(atoms, excited=excited, charge=charge)

    def __add__(self, other):
        if isinstance(other, Molecule):
            atoms = self.atoms.copy()
            for atom, num in other.atoms.items():
                if atom in atoms:
                    atoms[atom] += num
                else:
                    atoms[atom] = num
            charge = self.charge + other.charge
            excited = self.excited or other.excited
            return Molecule(atoms, excited=excited, charge=charge)
        else:
            raise TypeError(f'Can only add Molecule to Molecule, not {type(other)}')

    def __mul__(self, other):
        if isinstance(other, int):
            atoms = {atom: num * other for atom, num in self.atoms.items()}
            charge = self.charge * other
            return Molecule(atoms, excited=self.excited, charge=charge)
        else:
            raise TypeError(f'Can only multiply Molecule with int, not {type(other)}')

    def __sub__(self, other):
        if isinstance(other, Molecule):
            atoms = self.atoms.copy()
            for atom, num in other.atoms.items():
                if atom in atoms:
                    atoms[atom] -= num
                    if atoms[atom] == 0:
                        del atoms[atom]
                else:
                    raise ValueError(f'Cannot subtract {other} from {self}: cannot have a negative number of atoms of {atom}')
            charge = self.charge - other.charge
            if other.excited:
                excited = False
            else:
                excited = self.excited
            return Molecule(atoms, excited=excited, charge=charge)
        else:
            raise TypeError(f'Can only subtract Molecule from Molecule, not {type(other)}')

    def __str__(self):
        return ''.join([f'{atom}{num}' for atom, num in self.atoms.items()]) + f', c={self.charge}'

    def __repr__(self):
        return f'Molecule({str(self)})'

    def __eq__(self, other):
        if isinstance(other, Molecule):
            if self.charge != other.charge:
                return False
            if self.atoms != other.atoms:
                diff = set(self.atoms.items()) ^ set(other.atoms.items())
                for d in diff:
                    if d[0] != 'e':
                        return False
            return True
        else:
            raise TypeError(f'Can only compare Molecule with Molecule, not {type(other)}')

    def representation(self):
        atom_strings = []
        for atom, num in self.atoms.items():
            if num == 1:
                atom_strings.append(atom)
            else:
                atom_strings.append(f'{atom}{num}')
        if self.charge == 1:
            atom_strings[-1] += '+'
        elif self.charge == -1:
            if self.atoms != {'e': 1}:
                atom_strings[-1] += '-'
        elif self.charge > 1 or self.charge < -1:
            raise NotImplementedError(f'Cannot handle charge {self.charge}')
        if self.excited:
            atom_strings[-1] += '*'

        return ''.join(atom_strings)

    def latex(self):
        atom_strings = []
        for atom, num in self.atoms.items():
            if num == 1:
                atom_strings.append(atom)
            else:
                atom_strings.append(f'{atom}_{{{num}}}')
        if self.charge == 1:
            atom_strings[-1] += '^+'
        elif self.charge == -1:
            atom_strings[-1] += '^-'
        elif self.charge > 1:
            atom_strings[-1] += f'^{{{self.charge}+}}'
        elif self.charge < -1:
            atom_strings[-1] += f'^{{{-self.charge}-}}'
        if self.excited:
            atom_strings[-1] += '^*'
        return ''.join(atom_strings)


class Reaction:
    def __init__(self, reactants: list[Molecule], products: list[Molecule]):
        self.reactants = reactants
        self.products = products
        left = sum(reactants, Molecule({}))
        right = sum(products, Molecule({}))
        if left != right:
            raise ValueError(f'{reactants} -> {products}:\n'
                             f'Left side of reaction ({left}) does not equal right side ({right})')

    def __str__(self):
        return f'{' + '.join([str(react) for react in self.reactants])} -> {' + '.join([str(prod) for prod in self.products])}'

    def __repr__(self):
        return f'Reaction({str(self)})'

    def representation(self):
        return f'{' + '.join([react.representation() for react in self.reactants])} => ' \
               f'{' + '.join([prod.representation() for prod in self.products])}'