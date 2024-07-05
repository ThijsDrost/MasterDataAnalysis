import os
import time

from General.experiments.hdf5 import makeHDF5

input_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results\raw'
output_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results'
folders = [folder for folder in os.listdir(input_loc) if os.path.isdir(os.path.join(input_loc, folder))]

start_time = time.time()

for index, folder in enumerate(folders):
    hdf5_loc = rf'{output_loc}\{folder}.hdf5'
    loc = rf'{input_loc}\{folder}'

    absorption_loc = rf'{loc}\Absorption'
    emission_loc = rf'{loc}\Emission'
    waveform_loc = rf'{loc}\Waveforms'
    conductivity_kwargs = None
    if os.path.exists(rf'{loc}\conductivity.hdf5'):
        conductivity_kwargs = {'conductivity_hdf5_loc': rf'{loc}\conductivity.hdf5', 'photo': False}
    conductivity_photo_kwargs = None
    if os.path.exists(rf'{loc}\conductivity.txt'):
        conductivity_photo_kwargs = {'conductivity_hdf5_loc': rf'{loc}\Conductivity.txt', 'photo': True}

    makeHDF5.make_hdf5(hdf5_loc, absorption_kwargs={'folder_loc': absorption_loc},
                       emission_kwargs={'folder_loc': emission_loc}, waveform_kwargs={'folder_loc': waveform_loc},
                       conductivity_kwargs=conductivity_kwargs, conductivity_photo_kwargs=conductivity_photo_kwargs)

    print(f'Done with {index+1}/{len(folders)} after {(time.time()-start_time)/60:.2f} min')
