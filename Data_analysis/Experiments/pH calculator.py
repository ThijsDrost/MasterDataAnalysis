import numpy as np

HCl_concentration = 4.0  # mol/L
NaOH_concentration = 1.0  # mol/L
NO2_moles = 0  # mol
pka = 3
start_pH = 5.6

pH_values = np.linspace(2.5, 6.5, 9)
volume = 100  # mL

HCl_volume = []
NaOH_volume = []

for pH in pH_values:
    ratio = 10**(pka - pH)/(1 + 10**(pka - pH))
    HCl_moles = 10**(-pH)*volume/1000 + NO2_moles*ratio - 10**(-start_pH)*volume/1000
    HCl_volume.append(1_000_000*HCl_moles/HCl_concentration)
    NaOH_volume.append(-1_000_000*HCl_moles/NaOH_concentration)

print(', '.join([f'{vol:.2f} uL' for vol in HCl_volume]))
print(', '.join([f'{vol:.2f} uL' for vol in NaOH_volume]))
