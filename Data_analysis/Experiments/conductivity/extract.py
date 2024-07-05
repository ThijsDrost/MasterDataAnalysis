import numpy as np

from GUIs.conductivity_photo import convert_image

loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_05_13\Conductivity'
name = '5us'
real_loc = rf'{loc}\{name}.jpg'

values = convert_image(real_loc, (100, 900), (0, 7), True)
np.savetxt(rf'{loc}\{name}.txt', np.array(values).T, fmt='%.2f', delimiter='\t')
