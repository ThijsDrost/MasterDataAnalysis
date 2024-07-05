import numpy as np
import matplotlib.pyplot as plt

from General.experiments.absorption.Models import multi_species_model
from General.experiments import SpectroData
from General.plotting import plot

loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_05_31_cali'
names = ['NO2 0.4g_L NO3 0.1g_L.txt', 'NO2 0.1g_L NO3 0.4g_L.txt', 'NO2 0.1g_L NO3 0.1g_L maybe.txt']
model = multi_species_model(r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Cuvette4.hdf5',
                            add_zero=True, add_constant=True)

wav_range = (250, 400)

dark = rf'{loc}\dark.txt'
reference = rf'{loc}\reference4.txt'

dark_spec = SpectroData.read_txt(dark).spectrum.intensities
reference_spec = SpectroData.read_txt(reference).spectrum.intensities

fig, ax = plt.subplots()
for index, name in enumerate(names):
    spectrum = SpectroData.read_txt(rf'{loc}\{name}')
    absorption = -np.log10((spectrum.spectrum.intensities - dark_spec) / (reference_spec - dark_spec))
    mask = (spectrum.spectrum.wavelengths > wav_range[0]) & (spectrum.spectrum.wavelengths < wav_range[1])

    params = model.make_params()
    result = model.fit(absorption[mask], params, x=spectrum.spectrum.wavelengths[mask])

    NO2_conc = result.params['NO2conc_mol_lconc'].value
    NO3_conc = result.params['NO3conc_mol_lconc'].value
    H2O2_conc = result.params['H2O2conc_mol_lconc'].value
    O3_conc = result.params['O3conc_mol_lconc'].value
    NO2_std = result.params['NO2conc_mol_lconc'].stderr
    NO3_std = result.params['NO3conc_mol_lconc'].stderr
    H2O2_std = result.params['H2O2conc_mol_lconc'].stderr
    O3_std = result.params['O3conc_mol_lconc'].stderr
    NO2_conc_name = float(name.split('g_L')[0].split(' ')[1])/46.0055
    NO3_conc_name = float(name.split('g_L')[1].split(' ')[2])/62.0049
    print(f'NO2: {1000*NO2_conc_name:.2f} vs {NO2_conc:.2f} +/- {NO2_std:.2f}')
    print(f'NO3: {1000*NO3_conc_name:.2f} vs {NO3_conc:.2f} +/- {NO3_std:.2f}')
    print(f'H2O2: {H2O2_conc:.2f} +/- {H2O2_std:.2f}')
    print(f'O3: {O3_conc:.2f} +/- {O3_std:.2f}')

    plt.plot(spectrum.spectrum.wavelengths[mask], absorption[mask], label=f'Spectrum {index+1}')
    plt.plot(spectrum.spectrum.wavelengths[mask], result.best_fit, '--', label=f'Fit {index+1}')
ax.set_xlim(250, 400)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Absorption')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Results\Calibrations\Absorption_test.pdf')
fig.show()
