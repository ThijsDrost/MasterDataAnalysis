from collections.abc import Iterable


def flatten_2D(lst):
    return [item for sublist in lst for item in sublist]


def flatten_iter(lst):
    for item in lst:
        if isinstance(item, Iterable):
            yield from flatten_iter(item)
        else:
            yield item


def flatten(lst):
    return list(flatten_iter(lst))
