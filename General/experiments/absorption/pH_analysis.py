import numpy as np

from General.import_funcs import drive_letter
from General.experiments.absorption import pH


def pH_concentration(pKa, pH):
    return 1/(10**(pKa-pH) + 1)


def theoretical_ratio(pH_val: float, offset=0):
    no2_val = pH_concentration(3.3, pH_val)
    hno2_val = 1 - no2_val

    # TODO: Move files to better location
    hno2 = np.loadtxt(rf"{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Spectral_data\HNO2.txt", unpack=True)
    no2 = np.loadtxt(rf"{drive_letter()}:\OneDrive - TU Eindhoven\Master thesis\Spectral_data\NO2-.txt", unpack=True)

    ranges1 = pH.NO2_ranges[0]
    ranges2 = pH.NO2_ranges[1]

    mask1 = (ranges1[0][0] < no2[0]) & (no2[0] < ranges1[0][1])
    for r in ranges1[1:]:
        mask1 = mask1 | ((r[0] < no2[0]) & (no2[0] < r[1]))
    mask2 = (ranges2[0][0] < no2[0]) & (no2[0] < ranges2[0][1])
    for r in ranges2[1:]:
        mask2 = mask2 | ((r[0] < no2[0]) & (no2[0] < r[1]))

    spectrum = no2_val*no2[1] + hno2_val*hno2[1]
    ratio = (np.average(spectrum[mask1])+offset)/(np.average(spectrum[mask2])+offset)
    return ratio
