from GUIs.SpectrometerGUI.Spectrometer.Measure_GUI import MeasureGUI, create_background, create_background_reference

database_loc = r'C:\Users\20222772\Downloads\test.hdf5'
measurement = r'2022-10-24_test'
data_loc = r''
background_loc = r''
reference_loc = r''
mode = 'scope'  # absorption

if mode == 'scope':
    create_background(database_loc, measurement, background_loc)
elif mode == 'absorption':
    create_background_reference(database_loc, measurement, background_loc, reference_loc)
else:
    raise ValueError('`mode` value not recognized')

gui = MeasureGUI(data_loc, database_loc, measurement, mode,
                 kwargs_plot1={'xlabel': 'Wavelength [nm]', 'ylabel': 'Intensity [AU]'})

