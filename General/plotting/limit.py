from collections.abc import Sequence


class Limit:
    def __init__(self, outer_limit, limit=None, inverted=False):
        self.inverted = inverted
        self._outer_limit = tuple(outer_limit)
        if inverted:
            self._outer_limit = (outer_limit[1], outer_limit[0])

        if limit is None:
            self._limit = self._outer_limit
        else:
            self._limit = self.parse_limit(limit)

        self.check_order(self._outer_limit, 'outer_limit')
        self.check_order(self._limit, 'limit')

    def parse_limit(self, limit) -> tuple[float, float]:
        if isinstance(limit, (float, int)):
            return limit, self._outer_limit[1]
        elif isinstance(limit, Sequence):
            if len(limit) == 0:
                raise ValueError("`limit` cannot be an empty Sequence")
            if len(limit) == 1:
                return self.parse_limit(limit[0])

            if self.inverted:
                limit = (limit[1], limit[0])
            else:
                limit = (limit[0], limit[1])

            if limit[0] is None:
                limit = (self._outer_limit[0], limit[1])
            if limit[1] is None:
                limit = (limit[0], self._outer_limit[1])

            return limit
        else:
            raise TypeError("`limit` should be float, integer or Sequence")

    def _new_limit(self, limit, inverted=None) -> tuple[float, float]:
        inverted = inverted or self.inverted
        if inverted:
            limit = (limit[1], limit[0])

        limit = self.parse_limit(limit)
        if limit[0] < self._outer_limit[0]:
            limit = (self._outer_limit[0], limit[1] + self._outer_limit[0] - limit[0])
            if limit[1] > self._outer_limit[1]:
                return self._outer_limit
            return limit
        if self._outer_limit[1] < limit[1]:
            limit = (limit[0] - (limit[1] - self._outer_limit[1]), self._outer_limit[1])
            if limit[0] < self._outer_limit[0]:
                return self._outer_limit
            return limit
        return limit

    def set_limit(self, limit, inverted=None):
        limit = self._new_limit(limit, inverted)
        self.check_order(limit, 'limit')
        self._limit = limit

    def width(self):
        return self._limit[1] - self._limit[0]

    def zoom(self, middle, factor):
        new_width = factor*self.width()
        new_limits = (middle-0.5*new_width, middle+0.5*new_width)
        self.set_limit(new_limits, False)

    @property
    def limit(self):
        if self.inverted:
            return self._limit[::-1]
        else:
            return self._limit

    @property
    def outer_limit(self):
        if self.inverted:
            return self._outer_limit[::-1]
        else:
            return self._outer_limit

    @staticmethod
    def check_order(value, name):
        if value[0] == value[1]:
            raise ValueError(f"`{name}` bounds must be different")
        if value[1] < value[0]:
            raise ValueError(f"`{name}` bounds are in wrong order: {value}")

    def __repr__(self):
        return f"Limit({self.limit=}, {self.outer_limit=})"
