
class LinesModel:
    def __init__(self, ranges, corrected=True, num=1, **kwargs):
        self.ranges = ranges
        self.corrected = corrected
        self.num = num
        self.kwargs = kwargs