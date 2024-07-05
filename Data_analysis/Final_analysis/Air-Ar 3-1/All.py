import matplotlib.pyplot as plt

from General.experiments.absorption import MeasurementsAnalyzer
from General.experiments.hdf5.readHDF5 import read_hdf5, DataSet
from General.experiments.absorption.Models import multi_species_model
from General.plotting import plot
from General.plotting.linestyles import linelooks_by, legend_linelooks


data_loc = (r'E:\OneDrive - TU Eindhoven\Master thesis\Results\Air_3slm_Ar_1slm')
pulse_lengths = ['0.5us', '1us', '2us', '3us']  # ['0.3us', '0.5us', '1us', '1.5us', '2us', '3us', '5us']
voltages = ['8kV', '9kV', '10kV']

model_loc = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Cuvette2.hdf5'
model = multi_species_model(model_loc, add_zero=True, add_constant=True)

voltages_ = []
pulses_ = []
results_ = []

for voltage in voltages:
    for pulse in pulse_lengths:
        loc = f'{data_loc}_{voltage}_{pulse}.hdf5'

        data: DataSet = read_hdf5(loc)['absorbance'].remove_index(-1)
        analyzer = MeasurementsAnalyzer(data)

        result, _ = analyzer.fit(model, wavelength_range=(250, 400))
        results_.append(result)
        pulses_.append(pulse)
        voltages_.append(voltage)


# %%
times_ = [result[1] for result in results_]
h2o2_ = [result[2][0] for result in results_]
h2o2_new = [h2o2 - h2o2[3] for h2o2 in h2o2_]

no2 = [result[2][1] for result in results_]
no3 = [result[2][2] for result in results_]

line_kwargs = linelooks_by(color_values=voltages_, linestyle_values=pulses_)
legend_kwargs = legend_linelooks(line_kwargs, color_labels=voltages_, linestyle_labels=pulses_)
plot_kwargs = {'ylabel': 'Concentration [mM]', 'xlabel': 'Time [min]'}
plot.lines(times_, h2o2_new, line_kwargs_iter=line_kwargs, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs)
plot_kwargs = {'ylabel': 'Concentration', 'xlabel': 'Time [min]'}
plot.lines(times_, no2, line_kwargs_iter=line_kwargs, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs)
plot.lines(times_, no3, line_kwargs_iter=line_kwargs, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs)

