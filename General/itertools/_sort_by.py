def sort_by(sorter, *to_sort, key=None, reverse=False) -> tuple[tuple, ...]:
    values = sort_together(sorter, *to_sort, key=key, reverse=reverse)
    return values[1:]


def sort_together(sorter, *others, key=None, reverse=False) -> tuple[tuple, ...]:
    if not all(len(sorter) == len(other) for other in others):
        raise ValueError('The lengths of the lists must be the same.')
    if key is not None:
        key_sorter = lambda x: key(x[0])
    else:
        key_sorter = lambda x: x[0]
    sorted_values = sorted(zip(sorter, *others), key=key_sorter, reverse=reverse)
    return tuple(zip(*sorted_values))
