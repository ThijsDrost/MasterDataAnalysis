from General.Analysis import Analyzer
from General.Data_handling import drive_letter


line_width = 2  # nm
lines = [215, 225, 235, 260]  # nm
ranges = [(lines[i] - line_width, lines[i] + line_width) for i in range(len(lines))]


analyzer_H2O2 = Analyzer.standard(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\H2O2 cuvette\data.hdf5',
                                  'H2O2', 'H2O2 [mM]')
analyzer_H2O2.absorbances_wavelength_ranges_vs_variable(ranges)

analyzer_NO2 = Analyzer.standard(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2 cuvette\data.hdf5',
                                 'NO2-', 'NO2 [mM]')
analyzer_NO2.absorbances_wavelength_ranges_vs_variable(ranges, plot_kwargs={'ylim': (-0.001, 0.02)})
analyzer_NO2.absorbances_wavelength_ranges_vs_variable(ranges)

analyzer_NO3 = Analyzer.standard(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO3 cuvette\data.hdf5',
                                 'NO3-', 'NO3 [mM]')
analyzer_NO3.absorbances_wavelength_ranges_vs_variable(ranges, plot_kwargs={'ylim': (-0.001, 0.02)})
analyzer_NO3.absorbances_wavelength_ranges_vs_variable(ranges)
