from General.Analysis import CalibrationAnalyzer, Models
from General.Data_handling import drive_letter, InterpolationDataSet, import_hdf5

class LinesModel:
    def __init__(self, ranges, corrected=True, num=1, **kwargs):
        self.ranges = ranges
        self.corrected = corrected
        self.num = num
        self.kwargs = kwargs