import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from General.experiments import WavelengthCalibration
from General.import_funcs import drive_letter

plt.rcParams.update({'font.size': 14})

center_num = 250
sigma_num = 100
center_range = (0, 0.5)
sigma_range = (0.1, 4)


def gaussian(x, center, width):
    return np.exp(-((x - center) / width) ** 2)


pixels = np.arange(-1, 2, 1)
result = np.zeros((center_num, sigma_num))
result2 = np.zeros((center_num, sigma_num))
for index_center, center in enumerate(np.linspace(*center_range, center_num)):
    for index_sigma, sigma in enumerate(np.linspace(*sigma_range, sigma_num)):
        y = gaussian(pixels, center, sigma)
        found_peak = WavelengthCalibration.quadratic_peak(pixels, y)
        result[index_center, index_sigma] = center - found_peak
        a, b, c = WavelengthCalibration.quadratic(pixels, y)
        sig = (((1-1/np.sqrt(np.e))*((b**2) - 4*a*c))**0.5) / abs(a)
        result2[index_center, index_sigma] = sig/2 - sigma

assert np.all(result >= 0)

fig, ax = plt.subplots()
image = ax.imshow(np.abs(result), extent=(*sigma_range, *center_range), origin='lower', norm=LogNorm(vmin=1e-5, vmax=1), aspect='auto', cmap='tab20c_r')
cbar = fig.colorbar(image, ax=ax, label='Error')
plt.grid()
plt.xlabel('Sigma')
plt.ylabel('Center')
plt.tight_layout()
plt.savefig(f'{drive_letter()}:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Appendices\\Qaudratic_peak_error.pdf')
plt.show()

fig, ax = plt.subplots()
image = ax.imshow(np.abs(result), extent=(*(np.sqrt(2*np.log(2))*np.array(sigma_range)), *center_range), origin='lower', norm=LogNorm(vmin=1e-5, vmax=1), aspect='auto', cmap='tab20c_r')
cbar = fig.colorbar(image, ax=ax)
plt.grid()
plt.xlabel('HWHM')
plt.ylabel('Center')
plt.show()

avg_error = np.average(result, axis=0)
max_error = np.max(result, axis=0)
plt.figure()
plt.semilogy(np.linspace(*sigma_range, sigma_num), avg_error, label='Average')
plt.semilogy(np.linspace(*sigma_range, sigma_num), max_error, label='Maximum')
plt.xlabel('Sigma')
plt.ylabel('Error')
plt.xlim(0, sigma_range[1])
plt.legend()
plt.grid(which='both')
plt.tight_layout()
plt.savefig(f'{drive_letter()}:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Appendices\\Qaudratic_peak_error_avg.pdf')
plt.show()

fig, ax = plt.subplots()
image = ax.imshow(np.abs(result2), extent=(*sigma_range, *center_range), origin='lower',
                  norm=LogNorm(vmin=10**(-2.5), vmax=1), aspect='auto', cmap='tab20c_r')
cbar = fig.colorbar(image, ax=ax, label='Error')
plt.grid()
plt.xlabel('Sigma')
plt.ylabel('Center')
plt.tight_layout()
plt.savefig(f'{drive_letter()}:\\OneDrive - TU Eindhoven\\Master thesis\\Tex\\Images\\Appendices\\Qaudratic_HWHM_error.pdf')
plt.show()

fig, ax = plt.subplots()
image = ax.imshow(result2, extent=(*(np.sqrt(2*np.log(2))*np.array(sigma_range)), *center_range), origin='lower',
                  norm=SymLogNorm(vmin=-1, linthresh=10**(-2), vmax=1, linscale=0.5), aspect='auto', cmap='tab20c_r')
cbar = fig.colorbar(image, ax=ax)
plt.grid()
plt.xlabel('HWHM')
plt.ylabel('Center')
plt.show()

