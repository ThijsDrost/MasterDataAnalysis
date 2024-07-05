"""
This submodule contains the `moving_average` and `block_averages` functions for averaging numpy arrays.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import bottleneck as bn


def moving_average(a, n):
    """
    Calculates the moving average of a (over axis zero) with a window of n. The length of the returned array is len(a) - n + 1.

    Parameters
    ----------
    a: np.ndarray
    n: int

    Returns
    -------
    np.ndarray
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def block_average(a, n):
    """
    Calculates the average of every n values in a (over axis zero). If len(a) is not divisible by n, the last value is the average of
    the remaining values.

    Parameters
    ----------
    a: np.ndarray
    n: int

    Returns
    -------
    np.ndarray
    """
    num = len(a) // n
    extra = 1 if len(a) % n != 0 else 0
    shape = [num, n] + [-1] * (a.ndim - 1)
    values = np.zeros((len(a) // n + extra, *a.shape[1:]))
    values[:num] = np.average(a[:n * num].reshape(*shape), axis=1)
    if a.shape[0] % n != 0:
        values[-1] = np.average(a[n * num:], axis=0)
    return values


def averaging(values, block_average_num=None, moving_average_num=None):
    """
    Returns the appropriate averaging function.

    Parameters
    ----------
    block_average_num: int
    moving_average_num: int

    Returns
    -------
    Callable
    """
    if block_average_num is not None and moving_average_num is not None:
        raise ValueError('Only one of block_average and moving_average can be set.')
    if block_average_num is not None:
        return block_average(values, block_average_num)
    if moving_average_num is not None:
        return moving_average(values, moving_average_num)
    return values


def outlier_filter(values, rel_diff=0.2):
    diff = np.diff(values)
    backward_rel_diff = np.abs(diff / values[:-1])[1:]
    forward_rel_diff = np.abs(diff / values[1:])[:-1]
    mask = (backward_rel_diff > rel_diff) & (forward_rel_diff > rel_diff)
    indexes = 1 + np.where(mask)[0]
    result = values.copy()
    result[1:-1][mask] = np.interp(indexes, np.arange(len(values)), values)
    return result


def median_interp_filter(values, rel_diff=0.2, window_size=9):
    """
    Interpolates the values that are more than rel_diff away from the median of the window_size values around them.

    Parameters
    ----------
    values: np.ndarray
        The values to filter
    rel_diff: float
        The relative difference from the median to interpolate.
    window_size: int
        The size of the window to calculate the median, must be odd.

    Returns
    -------
    np.ndarray

    Notes
    -----
    The values closer than window_size//2 to the edges are not interpolated and are thus always returned as is.
    """
    if window_size % 2 == 0:
        raise ValueError('Window size should be odd.')

    offset = window_size // 2
    medians = bn.move_median(values, window=window_size)[offset:-offset]
    mask = np.abs(1 - values[offset:-offset]/medians) > rel_diff
    result = values.copy()

    indexes = np.where(mask)[0] + offset
    result[offset:-offset][mask] = np.interp(indexes, np.arange(len(values))[offset:-offset][~mask], values[offset:-offset][~mask])
    return result
