from General.simulation.specair import to_hdf5

loc_in = r'E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\second'
loc_out = r'E:\OneDrive - TU Eindhoven\Master thesis\SpecAir\OH_A-X_rot_500-5000_vib_500-8500_elec_12000.hdf5'
to_hdf5(loc_in, loc_out, resolution=0.025, rel_cutoff=None)
