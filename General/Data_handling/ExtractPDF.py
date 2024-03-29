"""
This subpackage contains functions to read cross-section data from pdf tables.
"""

import re
import itertools

from General.Data_handling.Reactions import Molecule, Reaction


def read_reaction(reaction: str):
    try:
        reactants, products = reaction.split('→')
    except Exception as e:
        raise ValueError(f'Invalid reaction: {reaction}') from e
    # changes = {'+2': '2+', 'H 2': 'H2 ', 'HO2': 'H2O', ' ++': '+ +', ' + +': '+ +', 'O 2': 'O2 ', '−': '-'}
    reactants, products = reactants.strip(), products.strip()
    changes = {'−': '-', '+\n2': '2+', '-\n2': '2-', 'HO2+': 'H2O+', 'HO2-': 'H2O-', '-+': '- +'}
    for old, new in changes.items():
        reactants = reactants.replace(old, new)
        products = products.replace(old, new)
    reactants_list = [Molecule.from_string(react) for react in reactants.split(' +')]
    products_list = [Molecule.from_string(prod) for prod in products.split(' +')]
    try:
        reaction = Reaction(reactants_list, products_list)
    except ValueError as e:
        raise ValueError(f'Invalid reaction: {reaction}') from e
    return reaction


def equation(equation: str, electrons: bool = False):
    # change different minus sign to correct one
    equation = equation.replace('−', '-')

    # Extract the coefficient and exponent of the first factor
    _, coefficient, exponent, rest = re.split(r'(\d+.?\d*) × 10(-?\d+)', equation)

    # Extract values with powers
    values = re.split(r'(ε|T[ge]?\)?|\)) ?(-?\d+\.?\d*)', rest, maxsplit=0)
    values = [val.strip() for val in values]

    # Extract factors in exp blocks
    exp_vals = re.findall(r'exp(\([^(]+\)|\[[^\[]+])', values[-1])
    first = coefficient + 'e' + exponent

    # Check if some factor is left before the exp blocks
    preval = re.split(r'exp(\([^(]+\)|\[[^\[]+])', values[-1])[0]
    after = ''
    if len(values) != 1:
        after = values[0] + values[1] + '^' + values[2]
    else:
        if preval:
            if preval[0] not in ('(', '['):
                after = preval.strip()

    after = after.strip()
    if after:
        if after[0] == '/':
            # Change division by exponent to multiplication by negative exponent
            after = r'^-'.join(after[1:].split('^'))
        elif after[0] in ('+', '-', '*'):
            raise NotImplementedError(f'Cannot handle {equation} yet')
        first += '*' + after

    # Put the equation back together
    total = '*'.join([first] + ['exp' + val for val in exp_vals])

    # Replace T values to Te or Tg based on whether electrons are present in the reaction
    if electrons:
        t_val = 'Te^'
    else:
        t_val = 'Tg^'
    total = re.sub(r'(?P<match>T ?\^)', t_val, total)

    # Put multiplication sign between two symbols
    symbols = ['Te', 'Tg', 'ε']
    for vals in itertools.permutations(symbols, 2):
        total = total.replace(f'{vals[0]}{vals[1]}', f'{vals[0]}*{vals[1]}')

    # Replace ε with the correct value
    total = total.replace('ε', '(3*Te*k_B_const/(2*e_const))')
    return total
