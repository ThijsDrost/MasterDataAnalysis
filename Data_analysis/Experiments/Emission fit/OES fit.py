import numpy as np
import scipy
import matplotlib.pyplot as plt

from General.simulation.specair.specair import Spectrum, SpecAirSimulations
from General.experiments.hdf5.readHDF5 import read_hdf5
from General.experiments.oes import OESData
from General.plotting import plot
from General.experiments.spectrum import TemporalSpectrum
from General.experiments import select_files, read_title

# loc = r"E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\OH_AX_rot_600-2400_vib_1000-11000_elec_12000.hdf5"
loc = r"E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\N2_fps_rot_500-5000_vib_1000-11000_elec_12000.hdf5"
fwhm = 0.5
voltage = 9  # kV
peak_width = None  # us
gasses = {'ar': 2, 'air': 2}  # slm
parameter = 'rot_energy'
# wav_range = (275, 340)
wav_range = (270, 450)


image_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Tex\Images\Results\N2 fit'
data_files = select_files(r'E:\OneDrive - TU Eindhoven\Master thesis\Results', gasses=gasses, voltage=voltage, peak_width=peak_width)
wavs = np.linspace(-6, 6, 250)
peak = scipy.stats.norm.pdf(wavs, 0, fwhm)

print('Preparing done')

name = ''
loop_variable = None
if voltage is None:
    loop_variable = 'voltage'

    name = f'{peak_width}us'
    for gas, amount in gasses.items():
        name += f'_{gas}_{amount}slm'
elif peak_width is None:
    if loop_variable is not None:
        raise ValueError('Only one loop variable is allowed.')
    loop_variable = 'peak_width (us)'

    name = f'{voltage}kV'
    for gas, amount in gasses.items():
        name += f'_{gas}_{amount}slm'
elif gasses is None:
    if loop_variable is not None:
        raise ValueError('Only one loop variable is allowed.')
    loop_variable = 'gas'

    name = f'{voltage}kV_{peak_width}us'
else:
    raise ValueError('No loop variable found.')


var_value = []
for data_loc in data_files:
    result = []
    result_std = []
    data: OESData = read_hdf5(data_loc)['emission']
    var_value.append(read_title(data_loc)[loop_variable])

    spectrum = data.spectrum
    spectrum = spectrum.clean(wavelength_range=wav_range, block_average=5, background_index=-1, background_wavelength=(290, 304))
    oes_data = OESData.new(spectrum.wavelengths, spectrum.intensities, spectrum.times)

    # plot_kwargs = {'xlim': wav_range}
    # save_loc = f'{image_loc}/{name}_{loop_variable}_{var_value[-1]}.pdf'
    # oes_data.intensity_vs_wavelength_with_time(plot_kwargs=plot_kwargs, show=False, save=save_loc)

    meas_interp = SpecAirSimulations.from_hdf5(loc, Spectrum(wavs, peak), spectrum.wavelengths)
    model = meas_interp.model()
    params = model.make_params()
    print('Fitting')

    for index in range(len(spectrum.intensities)):
        print(f'\r{index}')
        res = model.fit(spectrum.intensities[index], params)
        result.append(res.params[parameter].value)
        result_std.append(res.params[parameter].stderr)

    save_loc = f'{image_loc}/{name}_{loop_variable}_fit_res.pdf'
    plot_kwargs = {'xlabel': 'Time (s)', 'ylabel': 'T$_{rot}$ (K)'}
    plot.errorbar(spectrum.times, result, yerr=result_std, show=True, save_loc=save_loc, plot_kwargs=plot_kwargs)
