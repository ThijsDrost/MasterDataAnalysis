from General.experiments.hdf5.readHDF5 import read_hdf5
from General.plotting import plot

data_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results'
times = []
conductivities = []
t_vals = (0.3, 0.5, 1, 2, 3)
for t in t_vals:
    file = rf'Air_3slm_Ar_1slm_9kV_{t}us.hdf5'
    loc = f'{data_loc}\\{file}'
    data = read_hdf5(loc)['conductivity']
    times.append(data[0] - data[0][0])
    conductivities.append(data[1])

# %%
plot.lines(times, conductivities, labels=t_vals)

# %%
times = []
conductivities = []
t_vals = (0.5, 1, 2, 3, 5)
for t in t_vals:
    file = rf'Air_3slm_Ar_1slm_8kV_{t}us.hdf5'
    loc = f'{data_loc}\\{file}'
    data = read_hdf5(loc)['conductivity']
    times.append(data[0] - data[0][0])
    conductivities.append(data[1])

# %%
plot.lines(times, conductivities, labels=t_vals)
