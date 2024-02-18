import os
import warnings
import shutil

from General.Data_handling import drive_letter

spectrometers = ('2112120U1', '2203047U1')
loc = rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_02_16\Stability'
folders = [d for d in os.scandir(loc) if (d.is_dir() and (not any(s in d.name for s in spectrometers)))]

for folder in folders:
    for spectrometer in spectrometers:
        if os.path.exists(f'{folder.path}_{spectrometer}'):
            warnings.warn(f'{folder.path}_{spectrometer} already exists, skipping')
            continue
        os.mkdir(f'{folder.path}_{spectrometer}')
        files = [f for f in os.scandir(folder.path) if (spectrometer in f.name)]
        for file in files:
            shutil.copy2(file.path, f'{folder.path}_{spectrometer}')
