from General.experiments.absorption.analyse_directory import *
from General.experiments.absorption.Models import multi_species_model

data_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results\Ar_3slm'
pulse_lengths = ['0.3us', '0.5us', '1us', '2us', '5us']  # ['0.3us', '1us', '5us']  # ['0.3us', '0.5us', '1us', '1.5us', '2us', '3us', '5us']
voltages = ['4kV', '4.5kV', '5kV']

# %%
# model_loc = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Cuvette4.hdf5'
# model = multi_species_model(model_loc, add_zero=True, add_constant=True, add_slope=True)
#
# data_file = f'{data_loc}_{voltages[0]}_{pulse_lengths[2]}.hdf5'
# analyse_fit(data_file, 10, None, model, corrected=False)
# save_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Results\Absorption\back_Ar_3slm'
# model_loc = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Cuvette5.hdf5'
# analyse_directory_absorption(data_loc, voltages, pulse_lengths, save_loc=save_loc, add_slope=True, model_loc=model_loc,
#                              show=True, show2=True)

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Argon\\Argon\\'
# save_loc2 = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Argon\\Argon\\Rel_intensity\\'
# analyse_directory_argon_emission(data_loc, voltages, pulse_lengths, save_loc=save_loc, save_loc2=save_loc2)

# %%
save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Pulses\\Argon\\'
save_loc2 = f'{save_loc}Current\\'
channels = {1: 'voltage', 2: 'current', 3: 'pulse_generator', 4: 'ground_current'}
analyse_directory_pulse(data_loc, voltages, pulse_lengths, channels, save_loc=save_loc, save_loc2=save_loc2)

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Conductivity\\Argon 3\\'
# analyse_directory_conductivity(data_loc, voltages, pulse_lengths, save_loc=save_loc, show=True, align='end')
# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Argon\\H alpha\\'
# save_loc1 = f'{save_loc}Voltages\\'
# save_loc2 = None  # f'{save_loc1}Individual\\'
# analyse_directory_H_alpha(data_loc, voltages, pulse_lengths, save_loc1=save_loc, save_loc2=save_loc1, save_loc3=save_loc2)

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Argon\\total_intensity\\'
# save_loc1 = f'{save_loc}Voltages\\'
# save_loc2 = f'{save_loc1}Individual\\'
# # pulse_lengths = ['2us']
# emission_ranges(data_loc, voltages, pulse_lengths, save_loc1=save_loc, save_loc2=save_loc1, save_loc3=save_loc2,
#                 wavelength_ranges=((270, 330), (654, 659), ((690, 775), (780, 860)), (776, 779)), labels=(r'OH', r'H$_{\alpha}$', 'Ar', 'O'))
#
# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Argon\\total_intensity_p\\'
# save_loc1 = f'{save_loc}Voltages\\'
# save_loc2 = f'{save_loc1}Individual\\'
# peaks = {'Ha': 656.3, 'Ar': 763.5, 'OH': 309, 'O': 777.5}
# emission_peaks(data_loc, voltages, pulse_lengths, save_loc1=save_loc, save_loc2=save_loc1, save_loc3=save_loc2,
#                wavelength_peaks=peaks)

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Argon\\total_intensity_p2\\'
# save_loc1 = f'{save_loc}Voltages\\'
# save_loc2 = f'{save_loc1}Individual\\'
# emission_ranges(data_loc, voltages, pulse_lengths, save_loc1=save_loc, save_loc2=save_loc1, save_loc3=save_loc2,
#                 wavelength_ranges=((270, 400), (654, 659), ((690, 775), (780, 860)), (776, 779)),
#                 labels=(r'OH', r'H$_{\alpha}$', 'Ar', 'O'),
#                 peaks=((309,), (763.5, 811.5, 850.5), (763.5,), (777.5,)))

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Argon\\spectra\\'
# spectra_over_time(data_loc, voltages, pulse_lengths, {'OH': (270, 340)}, save_loc=save_loc)
