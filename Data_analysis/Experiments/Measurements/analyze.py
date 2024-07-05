from General.experiments.hdf5.readHDF5 import read_hdf5
from General.experiments.oes import OESData
from General.experiments.absorption import MeasurementsAnalyzer

loc = r"E:\OneDrive - TU Eindhoven\Master thesis\Measurements\24_04_10\Ar 1slm 1us 1kHz 7.5kV\data.hdf5"
data = read_hdf5(loc)

# %% OES data
oes: OESData = data['emission']
for i in range(10):
    plot_kwargs = {'xlim': (200+100*i, 300+100*i)}
    blocks = ((305, 320), (654, 659), (695, 699), (704, 709), (724, 729), (735, 740), (745, 755), (761, 766), (770, 775), (775, 780),
              (792, 797), (798, 803), (808, 813), (823, 828), (838, 845), (849, 854), (908, 913), (919, 924))
    fig, ax = oes.intensity_vs_wavelength_with_time(block_average=100, plot_kwargs=plot_kwargs, show=False, background_index=-1)
    for block in blocks:
        ax.axvline(block[0], color='black')
        ax.axvline(block[1], color='black')
    fig.show()

plot_kwargs = {'xlim': (300, 920)}
oes.intensity_vs_wavelength_with_time(block_average=100, plot_kwargs=plot_kwargs, show=True, background_index=-1)

oes.ranged_intensity_vs_wavelength_with_time(blocks, block_average=10, show=True)

# %% Absoprtion data
absorbance = MeasurementsAnalyzer(data['absorbance'])
absorbance.total_absorbance_over_time(10, wav_range=(220, 300))
absorbance.total_absorbance_over_time(10, wav_range=(220, 300), corrected=False)
