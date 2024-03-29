"""
This subpackage contains functions to check the validity of data.
"""


def check_literal(value, literals, val_name):
    if value not in literals:
        raise ValueError(f'`{val_name}` must be one of {literals}')


def check_positive_int(value, val_name):
    if not isinstance(value, int) or value < 0:
        raise ValueError(f'`{val_name}` must be a positive integer')


def check_positive_float(value, val_name):
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f'`{val_name}` must be a positive float')