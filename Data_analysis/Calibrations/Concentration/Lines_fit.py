from Data_analysis.Calibrations.Calibration_analysis import Analyzer
from General.Data_handling.Import import drive_letter


line_width = 2  # nm
lines = [215, 230, 245, 260]  # nm
ranges = [(lines[i] - line_width, lines[i] + line_width) for i in range(len(lines))]


analyzer_H2O2 = Analyzer.standard(rf'{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\H2O2 cuvette\data.hdf5',
                                  'H2O2', 'H2O2 [mM]')
analyzer_H2O2.range



