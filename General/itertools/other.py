def transpose(iterable):
    """
    Transpose an iterable of iterables.
    """
    return list(map(list, zip(*iterable)))
