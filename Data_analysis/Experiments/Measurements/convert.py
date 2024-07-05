import shutil
import os

import h5py

from General.experiments.hdf5 import makeHDF5
from General.experiments.hdf5.split_files import move_files_filtered, filter_and_move_files

pulse = '0.5us'
voltage = '10kV'
# name = 'Air_3slm_Ar_1slm'
# name = 'Ar_3slm'
# name = 'Air_4slm'
name = 'Air_2slm_Ar_2slm'
# name = 'Air_1slm_Ar_3slm'
# name = 'Ar_1slm_Air_3slm'
# input_loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_05_08\{name}_{pulse}_{voltage}'
# input_loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_05_15\{name}_{voltage}_{pulse}'
input_loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_04_23\{name}_{pulse}'
output_name = f'{name}_{voltage}_{pulse}'
hdf5_loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Results\{output_name}.hdf5'
raw_loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Results\raw\{output_name}'
# input_loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_05_03 --\{name}'
# output_loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Results\Ar_3slm_5kV_5us.hdf5'

# shutil.rmtree(raw_loc)
shutil.copytree(input_loc, raw_loc)

# %%
files_dict = {
    "Absorption": "2203047U1",
    "Emission": "2201415U1",
}
move_files_filtered(input_loc, files_dict, raise_exists=True)

# %%
# waveform_loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_05_17\waveforms\{name}\Autosave'
waveform_loc = rf'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_04_23\Waveforms'
waveform_dest = rf'{input_loc}\Waveforms'

filter_and_move_files(waveform_loc, waveform_dest, relative_path=False, filter_func=lambda x: pulse in x)

# %% Combine multiple conductivity files
hdf5_files = [file for file in os.listdir(input_loc) if (file.endswith('.hdf5') and 'conductivity' in file)]

os.mkdir(rf'{input_loc}\Conductivity')

times = []
temps = []
conds = []

for hdf5_file in hdf5_files:
    shutil.move(rf'{input_loc}\{hdf5_file}', rf'{input_loc}\Conductivity\{hdf5_file}')

    with h5py.File(rf'{input_loc}\Conductivity\{hdf5_file}', 'r') as f:
        group = f['conductivity']
        times.extend(group['time'][:])
        temps.extend(group['temperature'][:])
        conds.extend(group['value'][:])

with h5py.File(rf'{input_loc}\conductivity.hdf5', 'w') as f:
    group = f.create_group('conductivity')
    group.create_dataset('time', data=times)
    group.create_dataset('temperature', data=temps)
    group.create_dataset('value', data=conds)

# %%
absorption_loc = rf'{input_loc}\Absorption'
emission_loc = rf'{input_loc}\Emission'
waveform_loc = rf'{input_loc}\Waveforms'
conductivity_loc = None  # rf'{input_loc}\conductivity.hdf5'
# conductivity_photo_loc = rf'{input_loc}\Conductivity.txt'

if conductivity_loc:
    conductivity_kwargs = {'conductivity_hdf5_loc': conductivity_loc, 'photo': False}
else:
    conductivity_kwargs = None

makeHDF5.make_hdf5(hdf5_loc, absorption_kwargs={'folder_loc': absorption_loc},
                   emission_kwargs={'folder_loc': emission_loc}, waveform_kwargs={'folder_loc': waveform_loc},
                   conductivity_kwargs=conductivity_kwargs)
