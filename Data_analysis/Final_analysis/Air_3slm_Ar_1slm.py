from General.experiments.absorption.analyse_directory import *
from General.experiments.absorption.Models import multi_species_model

data_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results\Air_3slm_Ar_1slm'
pulse_lengths = ['0.3us', '0.5us', '1us', '1.5us', '2us', '3us', '5us']
# voltages = ['8kV', '9kV', '10kV']
voltages = ['8kV', '9kV']

# %%
# model_loc = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Cuvette4.hdf5'
# model = multi_species_model(model_loc, add_zero=True, add_constant=True, add_slope=True)
#
# data_file = f'{data_loc}_{voltages[0]}_{pulse_lengths[2]}.hdf5'
# analyse_fit(data_file, 10, None, model, corrected=False)
# # save_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Results\Absorption\Air_3slm_Ar_1slm'
# # analyse_directory(data_loc, voltages, pulse_lengths, save_loc=save_loc)

# # %%
# out_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Results\Emission\Nitrogen 3 + Argon 1'
# out_loc2 = f'{out_loc}\\Voltages\\'
# out_loc3 = f'{out_loc2}Individual\\'
#
# specair_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\SpecAir'
# interp_N2 = rf'{specair_loc}\N2_fps_rot_500-5000_vib_1000-11000_elec_12000.hdf5'
# interp_OH = rf'{specair_loc}\OH_AX_rot_600-2400_vib_1000-11000_elec_12000.hdf5'
# analyse_directory_nitrogen_oh_emission(data_loc, voltages, pulse_lengths, interp_N2, interp_OH,
#                                        save_loc=out_loc, save_loc2=out_loc2, save_loc3=out_loc3, show2=True)

# %%
out_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Pulses\\Nitrogen 3 + Argon 1\\'
out_loc2 = f'{out_loc}Current\\'
channels = {1: 'voltage', 2: 'current', 3: 'pulse_generator', 4: 'ground_current'}
analyse_directory_pulse(data_loc, voltages, pulse_lengths, channels, save_loc=out_loc, save_loc2=out_loc2)

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Conductivity\\Nitrogen 3 + Argon 1\\'
# analyse_directory_conductivity(data_loc, voltages, pulse_lengths, save_loc=save_loc, show=True, align='end', end_time=1920)

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Nitrogen 3 + Argon 1\\total_intensity\\'
# save_loc1 = f'{save_loc}Voltages\\'
# save_loc2 = f'{save_loc1}Individual\\'
# # pulse_lengths = ['2us']
# emission_ranges(data_loc, voltages, pulse_lengths, save_loc1=save_loc, save_loc2=save_loc1, save_loc3=save_loc2,
#                 wavelength_ranges=((270, 400), (654, 659), ((690, 775), (780, 860)), (776, 779)), labels=(r'N$_{2}$+OH', r'H$_{\alpha}$', 'Ar', 'O'))

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Nitrogen 3 + Argon 1\\total_intensity_p2\\'
# save_loc1 = f'{save_loc}Voltages\\'
# save_loc2 = f'{save_loc1}Individual\\'
# emission_ranges(data_loc, voltages, pulse_lengths, save_loc1=save_loc, save_loc2=save_loc1, save_loc3=save_loc2,
#                 wavelength_ranges=((270, 400), (654, 659), ((690, 775), (780, 860)), (776, 779)),
#                 labels=(r'N$_{2}$', r'H$_{\alpha}$', 'Ar', 'O'),
#                 peaks=((316, 337, 357.5, 380), (763.5, 811.5, 850.5), (763.5,), (777.5,)))

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Nitrogen 3 + Argon 1\\total_intensity_p\\'
# save_loc1 = f'{save_loc}Voltages\\'
# save_loc2 = f'{save_loc1}Individual\\'
# peaks = {'Ha': 656.3, 'Ar': 763.5, 'OH': 309, 'O': 777.5}
# emission_peaks(data_loc, voltages, pulse_lengths, save_loc1=save_loc, save_loc2=save_loc1, save_loc3=save_loc2,
#                wavelength_peaks=peaks)

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Nitrogen 2 + Argon 2\\Argon\\'
# save_loc2 = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Nitrogen 2 + Argon 2\\Argon\\Rel_intensity\\'
# analyse_directory_argon_emission(data_loc, voltages, pulse_lengths, save_loc=save_loc, save_loc2=save_loc2)

# # %%
# save_loc = 'E:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Results\\Emission\\Nitrogen 3 + Argon 1\\spectra\\'
# spectra_over_time(data_loc, voltages, pulse_lengths, {'N$_2$': (270, 420)}, save_loc=save_loc)