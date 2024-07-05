from General.experiments.absorption import MeasurementsAnalyzer
from General.experiments.hdf5.readHDF5 import read_hdf5, DataSet
from General.experiments.absorption.Models import multi_species_model


data_loc = r'E:\OneDrive - TU Eindhoven\Master thesis\Results\Ar_3slm_4.5kV_0.3us.hdf5'
data: DataSet = read_hdf5(data_loc)['absorbance'].remove_index(-1)
analyzer = MeasurementsAnalyzer(data)

model_loc = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Cuvette2.hdf5'
model = multi_species_model(model_loc, add_zero=True, add_constant=True)

result = analyzer.fit(model, wavelength_range=(250, 400))
