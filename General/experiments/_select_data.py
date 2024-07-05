import os
import re
import warnings


def read_title(file):
    file_split = file.removesuffix('.hdf5').split('_')
    results = {}

    while file_split:
        if not re.search(r'\d', file_split[0]):
            results[file_split[0].lower()] = float(file_split[1].replace('slm', ''))
            file_split = file_split[2:]
        else:
            if 'kv' in file_split[0].lower():
                results['voltage (kV)'] = float(file_split[0].replace('kV', ''))
                file_split = file_split[1:]

            elif 'us' in file_split[0].lower():
                results['peak_width (us)'] = float(file_split[0].replace('us', ''))
                file_split = file_split[1:]

            else:
                raise ValueError(f'Unknown value: {file_split[0]} in {file}')

    return results


def select_files(loc, gasses: dict = None, voltage = None, peak_width = None):
    def check_value(value, needed):
        if isinstance(needed, tuple):
            if needed[0] <= value <= needed[1]:
                return True
        else:
            if value == needed:
                return True
        return False

    files = os.listdir(loc)

    files_out = []
    for file in files:
        names = read_title(file)

        found_gas = True
        if gasses:
            for gas, amount in gasses.items():
                if gas not in names or not check_value(names[gas], amount):
                    found_gas = False
                    break
        found_voltage = True
        if voltage:
            if 'voltage (kV)' not in names or not check_value(names['voltage (kV)'], voltage):
                found_voltage = False
        found_peak_width = True
        if peak_width:
            if 'peak_width (us)' not in names or not check_value(names['peak_width (us)'], peak_width):
                found_peak_width = False

        if found_gas and found_voltage and found_peak_width:
            files_out.append(os.path.join(loc, file))
    if not files_out:
        warnings.warn('No files found')

    return files_out


def select_files_gas(loc, argon=None, air=None, voltage=None, peak_width=None):
    if argon is not None:
        if air is not None:
            gasses = {'Ar': argon, 'air': air}
        else:
            gasses = {'Ar': argon}
    elif air is not None:
        gasses = {'air': air}
    else:
        gasses = None
    return select_files(loc, gasses, voltage, peak_width)
