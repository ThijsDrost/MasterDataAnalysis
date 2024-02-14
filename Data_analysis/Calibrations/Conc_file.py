import numpy as np

loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_01_26 H2O2 Conc\H2O2.txt'
stock_molarity = 9.8  # M
volumes_added = np.array([0.015, 0.05, 0.15, 0.5, 1.5, 5, 15, 50, 150, 500])*1e-6  # L
total_volume = 100e-3  # L
moles_added = stock_molarity * volumes_added
moles_total = moles_added / total_volume
nums = np.arange(0, len(moles_total))
with open(loc, 'w') as f:
    for i in range(len(moles_total)):
        f.write(f'{nums[i]}\t{moles_total[i]:.3e}\n')

