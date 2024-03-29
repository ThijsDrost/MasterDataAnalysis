from ._DataSet import DataSet, import_hdf5
from ._SimpleDataSet import SimpleDataSet
from ._InterpolationDataSet import InterpolationDataSet

__all_exports = [DataSet, SimpleDataSet, InterpolationDataSet, import_hdf5]

for _e in __all_exports:
    _e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]
