import numpy as np

from General.simulation.bolsig.Bolsig_plus import Bolsig2DRun

base_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Bolsig+\data\Argon\var_En_nIon_with_Ars'
loc = rf'{base_loc}\Ars_exp(-2).txt'
runs = Bolsig2DRun.read_txt(loc)

diffusion = runs.data['Energy diffusion coef. *N (1/m/s)'].to_numpy()/1e26
collision_freq = runs.data['Total collision freq. /N (m3/s)'].to_numpy()*1e26

values = 2*diffusion/(np.sqrt(collision_freq))
print(values)



