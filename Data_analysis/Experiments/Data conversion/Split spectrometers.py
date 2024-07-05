import os
import warnings
import shutil

spectrometers = ('2203047U1', )
loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_01_25 NO2- Conc'

# %%
folders = [d for d in os.scandir(loc) if (d.is_dir() and (not any(s in d.name for s in spectrometers)))]
for folder in folders:
    for spectrometer in spectrometers:
        if os.path.exists(f'{folder.path}_{spectrometer}'):
            warnings.warn(f'{folder.path}_{spectrometer} already exists, skipping')
            continue
        files = [f for f in os.scandir(folder.path) if (spectrometer in f.name)]
        os.mkdir(rf'C:\Users\20222772\Downloads\Output\{folder.name}_{spectrometer}')
        for file in files:
            shutil.copy2(file.path, rf'C:\Users\20222772\Downloads\Output\{folder.name}_{spectrometer}')

# %%
for spectrometer in spectrometers:
    if os.path.exists(f'{loc}_{spectrometer}'):
        warnings.warn(f'{loc}_{spectrometer} already exists, skipping')
        continue
    os.mkdir(rf'{loc}_{spectrometer}')
    files = [f for f in os.scandir(loc) if (f.is_file() and (spectrometer in f.name))]
    for file in files:
        shutil.copy2(file.path, f'{loc}_{spectrometer}')