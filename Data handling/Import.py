import numpy as np

loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\Calibration\NO2 pH\data.hdf5'
image_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Plots\Calibration\NO2 pH'
dependent = 'pH'
variable_name = 'pH'
variable_factor = 1
wavelength_range = [180, 400]
r2_values = [0.99, 1]
wavelength_plot_every = 5
plot_number = 2
baseline_correction = [370, 390]
#%%
if not os.path.exists(image_loc):
    os.makedirs(image_loc)

def save_loc(loc):
    return os.path.join(image_loc, loc)
#%%
with h5py.File(loc, 'r') as file:
    # dark = file.attrs['dark'][:]
    # reference = file.attrs['reference'][:]
    wavelength = file.attrs['wavelength'][:]
    absorbance = []
    variable = []
    number = []
    for key in file.keys():
        absorbance.append(file[key][:])
        variable.append(file[key].attrs[dependent])
        number.append(int(key.split('_')[1].split('.')[0]))

absorbance = np.array(absorbance)
# dark = np.array(dark)
# reference = np.array(reference)
wavelength = np.array(wavelength)
variable = np.array(variable)
number = np.array(number)
# absorbance = -np.log10((measurements-dark) / (reference-dark))
mask = (wavelength_range[0] < wavelength) & (wavelength < wavelength_range[1])

absorbance_m_uncorr = absorbance[:, mask]
wavelength_m = wavelength[mask]
# dark_m = dark[mask]
# reference_m = reference[mask]

if baseline_correction is not None:
    correction_mask = (baseline_correction[0] < wavelength) & (wavelength < baseline_correction[1])
    absorbance_m = absorbance_m_uncorr - np.mean(absorbance[:, correction_mask], axis=1)[:, np.newaxis]
    absorbance_m_num = absorbance_m[number == plot_number]
    absorbance_m_uncorr_num = absorbance_m_uncorr[number == plot_number]
else:
    absorbance_m = absorbance_m_uncorr
    absorbance_m_num = absorbance_m_uncorr[number == plot_number]
    absorbance_m_uncorr_num = absorbance_m_uncorr[number == plot_number]

variable_num = variable[number==plot_number]

absorbance_best_num = np.zeros((len(np.unique(variable)), len(wavelength_m)))
variable_best_num = np.zeros(len(np.unique(variable)))
for i, v in enumerate(np.unique(variable)):
    v_absorbances = absorbance_m[variable == v]
    value = []
    pair_values = []
    for pair in itertools.combinations(range(len(v_absorbances)), 2):
        # value.append(np.sum((v_absorbances[pair[0]]-v_absorbances[pair[1]])**2))
        mask = v_absorbances[0] > 0.5*np.max(v_absorbances[0])
        value.append(np.sum((v_absorbances[pair[0]][mask]-v_absorbances[pair[1]][mask])**2))
        pair_values.append(pair)
        # pair_values.append([1, 1])
    min_num = pair_values[np.argmin(value)]
    print(f'{v}, {min_num}')
    absorbance_best_num[i] = (absorbance_m[variable == v][min_num[0]]+absorbance_m[variable == v][min_num[1]])/2
    variable_best_num[i] = v

class data:
    def __init__(self, ):

    @staticmethod
    def from_hdf5(loc):
