from General.experiments.absorption.analyse_directory import *
from General.experiments.absorption.Models import multi_species_model

data_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results\Air_2slm_Ar_2slm'
pulse_lengths = ['0.3us', '0.5us', '1us', '2us', '3us', '5us']  # ['0.3us', '0.5us', '1us', '1.5us', '2us', '3us', '5us']
voltages = ['8kV', '9kV']

# # %%
# model_loc = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Cuvette4.hdf5'
# model = multi_species_model(model_loc, add_zero=True, add_constant=True, add_slope=True)
#
# data_file = f'{data_loc}_{voltages[0]}_{pulse_lengths[2]}.hdf5'
# # analyse_fit(data_file, 10, None, model, corrected=False)
# save_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Results\Absorption\Air_2slm_Ar_2slm'
# analyse_directory_absorption(data_loc, voltages, pulse_lengths, save_loc=save_loc)

# %%
out_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Pulses\\Nitrogen 2 + Argon 2\\'
out_loc2 = f'{out_loc}Current\\'
channels = {1: 'voltage', 2: 'current', 3: 'pulse_generator', 4: 'ground_current'}
analyse_directory_pulse(data_loc, voltages, pulse_lengths, channels, save_loc=out_loc, save_loc2=out_loc2)