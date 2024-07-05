from General.experiments.hdf5.readHDF5 import read_hdf5
from General.plotting import plot

loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results'
time_us = [1, 2, 3]
time_vals = []
conductivities = []
temperatures = []

for t in time_us:
    print(t)
    file = rf'{loc}\Air_2slm_Ar_2slm_{t}us_9kV.hdf5'
    data = read_hdf5(file)
    times, conductivity, temp = data['conductivity']
    times = [(t - times[0])/60 for t in times]
    time_vals.append(times)
    conductivities.append(conductivity)
    temperatures.append(temp)

legend_kwargs = {'title': 'Pulse width [us]'}
plot_kwargs = {'xlabel': 'Time [min]', 'ylabel': 'Conductivity [S/cm]'}
plot.lines(time_vals, conductivities, labels=time_us, legend_kwargs=legend_kwargs, plot_kwargs=plot_kwargs)
