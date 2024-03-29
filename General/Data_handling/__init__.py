"""
This module contains all the classes and functions for importing, storing, cleaning and handling data.
"""

from General.Data_handling.Import import drive_letter
from General.Data_handling.DataSets import SimpleDataSet, DataSet, InterpolationDataSet, import_hdf5
from General.Data_handling.Conversion import make_hdf5, interpolate_weights
from General.Data_handling._SpectroData import SpectroData
from General.Data_handling.CrossSections import CrossSectionData, plot_CrossSections, plot_spread, plot_averageCrossSections
from General.Data_handling.DataChecker import check_positive_float, check_positive_int, check_literal

__all_exports = [SpectroData, ]

for _e in __all_exports:
    _e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]
