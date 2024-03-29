"""
This subpackage contains `the drive_letter` function, which is used to find the drive letter of the hard drive. This was needed
since the drive with data had a different letter on different computers.
"""

import os

import numpy as np
import h5py

from .DataSets import SimpleDataSet


def drive_letter(test_drives=('D', 'E')):
    for letter in test_drives:
        if os.path.exists(f'{letter}:'):
            return letter
    raise FileNotFoundError(f'No drives found, tested {', '.join(test_drives[:-1])}, and {test_drives[-1]}')
