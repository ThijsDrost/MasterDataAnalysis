from collections.abc import Collection
from typing import Callable


def argmax[T](values: Collection[T], key: Callable[[T], any] = None) -> int:
    """Returns the first index of the maximum value in the sequence."""
    return _argfunc(max, values, key=key)


def argmin[T](values: Collection[T], key: Callable[[T], any] = None) -> int:
    """Returns the first index of the minimum value in the sequence."""
    return _argfunc(min, values, key=key)


def _argfunc[T](func: callable, values: Collection[T], key: Callable[[T], any] = None) -> int:
    if key is None:
        def key(x):
            return x
    if not isinstance(values, Collection):
        raise TypeError('`values` should be a collection')
    return func(range(len(values)), key=lambda i: key(values[i]))
