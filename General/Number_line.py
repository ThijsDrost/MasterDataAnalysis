from typing import assert_never


class Bound:
    def __init__(self, value, inclusive):
        self.value = value
        self.inclusive = inclusive

    def smaller(self, other):
        """Check if the value is smaller than the other value"""
        def less(first, second): return first < second

        return self._compare(other, less)

    def bigger(self, other):
        """Check if the value is bigger than the other value"""
        def more(first, second): return first > second

        return self._compare(other, more)

    def _compare(self, other, operator):
        if not isinstance(other, (Bound, int, float)):
            raise TypeError(f'`Bound` can only be compared to `Bound`, `int`, or `float`, not {type(other).__name__}')
        if isinstance(other, Bound):
            result = operator(self.value, other.value)
            if self.inclusive and other.inclusive:
                return result or (self.value == other.value)
            else:
                return result
        else:
            result = operator(self.value, other)
            if self.inclusive:
                return result or (self.value == other)
            else:
                return result

    def __eq__(self, other):
        if not isinstance(other, Bound):
            raise TypeError(f'`Bound` can only be compared to `Bound`, not {type(other).__name__}')
        return self.value == other.value and self.inclusive == other.inclusive

    def __repr__(self):
        return f'Bound({self.value}, {self.inclusive})'


MinusInfinity = Bound(float('-inf'), True)
Infinity = Bound(float('inf'), True)


class Range:
    def __init__(self, lower: Bound, upper: Bound, *, _check=True):
        self.lower = lower
        self.upper = upper
        if _check and not self.lower.smaller(self.upper):
            raise ValueError(f'Lower bound ({self.lower.value}) cannot be bigger than upper bound ({self.upper.value})')

    def __contains__(self, item):
        return self.lower.smaller(item) and self.upper.bigger(item)

    def __bool__(self):
        return self.lower.smaller(self.upper)

    def __add__(self, other):
        if isinstance(other, Range):
            if ((self.lower.value > other.upper.value) or (self.upper.value < other.lower.value)
                    or (self.lower.value == other.upper.value
                        and (not self.lower.inclusive) and (not other.upper.inclusive))):
                return self, other

            if self.lower.value < other.lower.value:
                lower_bound = self.lower
            elif self.lower.value > other.lower.value:
                lower_bound = other.lower
            else:
                lower_bound = Bound(self.lower.value, self.lower.inclusive or other.lower.inclusive)

            if self.upper.value > other.upper.value:
                upper_bound = self.upper
            elif self.upper.value < other.upper.value:
                upper_bound = other.upper
            else:
                upper_bound = Bound(self.upper.value, self.upper.inclusive or other.upper.inclusive)

            return Range(lower_bound, upper_bound)
        else:
            raise TypeError(f'`Range` can only be added to `Range`, not {type(other).__name__}')

    def __sub__(self, other):
        if isinstance(other, Range):
            lower_bound = Bound(other.lower.value, not other.lower.inclusive)
            upper_bound = Bound(other.upper.value, not other.upper.inclusive)
            if self.lower.bigger(upper_bound) or self.upper.smaller(lower_bound):
                return self
            elif self.lower.smaller(lower_bound) and self.upper.bigger(upper_bound):
                return Range(self.lower, lower_bound), Range(upper_bound, self.upper)
            elif self.lower.smaller(lower_bound) and self.upper.smaller(other.upper):
                return Range(self.lower, lower_bound)
            elif self.lower.bigger(other.lower) and self.upper.bigger(upper_bound):
                return Range(upper_bound, self.upper)
            elif self.lower.bigger(other.lower) and self.upper.smaller(other.upper):
                return EmptyRange
            else:
                assert_never("This should never happen")
        else:
            raise TypeError(f'`Range` can only be subtracted from `Range`, not {type(other).__name__}')

    def __eq__(self, other):
        if not isinstance(other, Range):
            raise TypeError(f'`Range` can only be compared to `Range`, not {type(other).__name__}')
        return self.lower == other.lower and self.upper == other.upper

    def __repr__(self):
        return f'Range({self.lower}, {self.upper})'

    def __str__(self):
        lower = '('
        if self.lower.inclusive:
            lower = '['
        upper = ')'
        if self.upper.inclusive:
            upper = ']'
        return f'{lower}{self.lower.value}, {self.upper.value}{upper}'


EmptyRange = Range(Infinity, MinusInfinity, _check=False)
FullRange = Range(MinusInfinity, Infinity)


class NumberLine:
    def __init__(self, ranges: list[Range] | Range = FullRange, simplify=True):
        if isinstance(ranges, Range):
            self.ranges = [ranges]
        elif isinstance(ranges, (list, tuple)):
            self.ranges = ranges
        else:
            raise TypeError(f'`NumberLine` can only be created with `Range` or `tuple` of `Range`, not {type(ranges).__name__}')
        if simplify:
            self.simplify()

    @staticmethod
    def include_from_floats(start: float = float('-inf'), end: float = float('inf'), start_inclusive=True,
                            end_inclusive=True):
        return NumberLine.include(Bound(start, start_inclusive), Bound(end, end_inclusive))

    @staticmethod
    def empty():
        return NumberLine()

    @staticmethod
    def include(start: Bound = MinusInfinity, end: Bound = Infinity):
        if start.bigger(end):
            raise ValueError(f'Start value ({start.value}) cannot be bigger than end value ({end.value})')
        return NumberLine(Range(start, end))

    @staticmethod
    def bigger_than(value: Bound):
        return NumberLine.include(value, Infinity)

    @staticmethod
    def bigger_than_float(value: float, inclusive=True):
        return NumberLine.include_from_floats(start=value, start_inclusive=inclusive)

    @staticmethod
    def smaller_than(value: Bound):
        return NumberLine.include(MinusInfinity, value)

    @staticmethod
    def smaller_than_float(value: float, inclusive=True):
        return NumberLine.include_from_floats(end=value, end_inclusive=inclusive)

    @staticmethod
    def exclude(start: Bound, end: Bound):
        if start.bigger(end):
            raise ValueError(f'Start value ({start.value}) cannot be bigger than end value ({end.value})')
        if start == MinusInfinity and end == Infinity:
            return NumberLine()
        return NumberLine([Range(MinusInfinity, start), Range(end, Infinity)])

    @staticmethod
    def exclude_from_floats(start: float = float('-inf'), end: float = float('inf'), start_inclusive=True,
                            end_inclusive=True):
        return NumberLine.exclude(Bound(start, start_inclusive), Bound(end, end_inclusive))

    def simplify(self):
        busy = True
        if len(self.ranges) <= 1:
            return

        while busy:
            for i, range1 in enumerate(self.ranges[:-1]):
                if range1 == EmptyRange:
                    self.ranges.pop(i)
                    break

                for j, range2 in enumerate(self.ranges[i + 1:], i + 1):
                    new_range = range1 + range2
                    if isinstance(new_range, Range):
                        self.ranges[i] = new_range
                        self.ranges.pop(j)
                        break
                else:
                    continue
                break
            else:
                busy = False
        self.ranges.sort(key=lambda x: x.lower.value)

    def value_in_range(self, value):
        return self.__contains__(value)

    def __add__(self, other):
        if isinstance(other, NumberLine):
            return NumberLine(self.ranges + other.ranges)
        elif isinstance(other, Range):
            return NumberLine(self.ranges + [other])
        elif isinstance(other, (int, float)):
            return NumberLine(self.ranges + [Range(Bound(other, True), Bound(other, True))])
        else:
            raise TypeError(
                f'Only `NumberLine`, `Range`, `int` and `float` can be added to `NumberLine`, not {type(other).__name__}')

    def __sub__(self, other):
        if isinstance(other, NumberLine):
            new_ranges = self.ranges
            for range_ in other.ranges:
                new_ranges = self._subtract_range(new_ranges, range_)
            return NumberLine(new_ranges, simplify=False)
        elif isinstance(other, Range):
            return NumberLine(self._subtract_range(self.ranges, other), simplify=False)
        elif isinstance(other, (int, float)):
            return NumberLine(self._subtract_range(self.ranges, Range(Bound(other, True), Bound(other, True))),
                              simplify=False)
        else:
            raise TypeError(
                f'Only `NumberLine`, `Range`, `int` and `float` can be subtracted from `NumberLine`, not {type(other).__name__}')

    def __contains__(self, value):
        if isinstance(value, (float, int)):
            return any(value in _range for _range in self.ranges)
        else:
            raise TypeError(f'`NumberLine` can check if `float` or `int` is in range, not {type(value).__name__}')

    def __bool__(self):
        self.simplify()
        return bool(self.ranges)

    @staticmethod
    def _subtract_range(_ranges, range_):
        new_ranges = []
        for _range in _ranges:
            new_range = _range - range_
            if isinstance(new_range, Range):
                new_ranges.append(new_range)
            else:
                new_ranges.extend(new_range)
        return new_ranges

    def __repr__(self):
        return f'NumberLine({self.ranges})'

    def __str__(self):
        return f'NumberLine({', '.join(str(range_) for range_ in self.ranges)})'


def tests():
    range1 = Range(Bound(0, True), Bound(10, True))
    range2 = Range(Bound(5, True), Bound(15, True))
    range3 = Range(Bound(5, False), Bound(10, True))
    range4 = Range(Bound(10, True), Bound(15, True))
    range5 = Range(Bound(0, True), Bound(5, True))
    range6 = Range(Bound(0, False), Bound(10, False))
    range7 = Range(Bound(0, True), Bound(10, True))
    range8 = Range(Bound(0, True), Bound(0, True))
    range9 = Range(Bound(10, True), Bound(10, True))
    range10 = Range(Bound(0, False), Bound(10, True))
    range11 = Range(Bound(4, True), Bound(4, True))
    range12 = Range(Bound(0, True), Bound(10, False))

    assert range1 + range2 == Range(Bound(0, True), Bound(15, True)), range1 + range2
    assert range1 - range2 == Range(Bound(0, True), Bound(5, False)), range1 - range2
    assert range2 - range1 == Range(Bound(10, False), Bound(15, True)), range2 - range1
    assert range1 - range4 == Range(Bound(0, True), Bound(10, False)), range1 - range4
    assert range1 - range3 == Range(Bound(0, True), Bound(5, True)), range1 - range3
    assert range1 - range5 == Range(Bound(5, False), Bound(10, True)), range1 - range5
    assert range1 - range6 == (Range(Bound(0, True), Bound(0, True)),
                               Range(Bound(10, True), Bound(10, True))), range1 - range6
    assert range1 - range7 == EmptyRange, range1 - range7
    assert range1 - range8 == Range(Bound(0, False), Bound(10, True)), range1 - range8
    assert range1 - range9 == Range(Bound(0, True), Bound(10, False)), range1 - range9
    assert range1 - range10 == Range(Bound(0, True), Bound(0, True)), range1 - range10
    assert range1 - range11 == (Range(Bound(0, True), Bound(4, False)),
                                Range(Bound(4, False), Bound(10, True))), range1 - range11
    assert range1 - range12 == Range(Bound(10, True), Bound(10, True)), range1 - range12

    range1 = Range(Bound(0, False), Bound(10, False))
    range2 = Range(Bound(0, True), Bound(10, True))
    range3 = Range(Bound(0, True), Bound(10, False))
    range4 = Range(Bound(0, False), Bound(10, True))
    range5 = Range(Bound(10, True), Bound(20, True))
    range6 = Range(Bound(5, False), Bound(15, False))
    range7 = Range(Bound(5, True), Bound(15, True))

    assert range1 + range2 == Range(Bound(0, True), Bound(10, True)), range1 + range2
    assert range2 + range1 == Range(Bound(0, True), Bound(10, True)), range2 + range1
    assert range1 + range3 == Range(Bound(0, True), Bound(10, False)), range1 + range3
    assert range3 + range1 == Range(Bound(0, True), Bound(10, False)), range3 + range1
    assert range1 + range4 == Range(Bound(0, False), Bound(10, True)), range1 + range4
    assert range4 + range1 == Range(Bound(0, False), Bound(10, True)), range4 + range1
    assert range1 + range5 == Range(Bound(0, False), Bound(20, True)), range1 + range5
    assert range5 + range1 == Range(Bound(0, False), Bound(20, True)), range5 + range1
    assert range1 + range6 == Range(Bound(0, False), Bound(15, False)), range1 + range6
    assert range6 + range1 == Range(Bound(0, False), Bound(15, False)), range6 + range1
    assert range1 + range7 == Range(Bound(0, False), Bound(15, True)), range1 + range7
    assert range7 + range1 == Range(Bound(0, False), Bound(15, True)), range7 + range1


if __name__ == '__main__':
    tests()
    print('All tests passed')
