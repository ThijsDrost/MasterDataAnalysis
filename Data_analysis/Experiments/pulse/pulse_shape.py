import lmfit
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn

from General.experiments.hdf5.readHDF5 import read_hdf5
from General.experiments.waveforms import Waveforms, MeasuredWaveforms
import General.numpy_funcs as npf
from General.plotting import plot

# %%
data_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results\Air_3slm_Ar_1slm'
data = read_hdf5(data_loc + '_9kV_3us.hdf5')['waveforms']
channels = {1: 'voltage', 2: 'current', 3: 'pulse_generator', 4: 'ground_current'}
wavs = MeasuredWaveforms.from_waveforms(data, channels)

# %%
wavs.plot()

# %%
def model(x, amplitude, decay, phase, length, offset):
    return amplitude * np.exp(-(x-x[0]) / decay) * np.sin((2 * np.pi * x) / length + phase) + offset


current = wavs.currents
lmfit_model = lmfit.Model(model)
lmfit_model.set_param_hint('amplitude', value=0.25, min=0)
lmfit_model.set_param_hint('decay', value=2e-6)
lmfit_model.set_param_hint('phase', value=-1.5)
lmfit_model.set_param_hint('length', value=1.9e-7)
lmfit_model.set_param_hint('offset', value=0.8)

model2 = lmfit.models.SineModel() + lmfit.models.ConstantModel()
model2.set_param_hint('amplitude', value=0.25, min=0.1)
model2.set_param_hint('frequency', value=3e7, min=1e5)
model2.set_param_hint('c', value=0.8)

# %%
idxs = (340, 360)
times = np.average(wavs.time[idxs[0]:idxs[1]], axis=0)
currents = np.average(current[idxs[0]:idxs[1]], axis=0)

mask = (times > (times[np.argmin(currents)]-1.75e-6)) & (times < (times[np.argmin(currents)]-0.75e-7))
# mask = (times > (times[np.argmax(currents)]+1.25e-7)) & (times < (times[np.argmin(currents)]-0.75e-7))
lmfit_model.set_param_hint('offset', value=np.average(currents[mask]))
params = lmfit_model.make_params()
result = lmfit_model.fit(currents[mask], x=times[mask], params=params)
result.plot()
plt.plot(times[mask], lmfit_model.eval(lmfit_model.make_params(), x=times[mask]))
plt.show()
print(result.fit_report())

# %%
result2 = model2.fit(currents[mask], x=times[mask])
result2.plot()
plt.show()
print(result2.fit_report())


