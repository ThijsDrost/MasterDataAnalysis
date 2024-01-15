import numpy as np


def moving_average(a, n):
    """
    Calculates the moving average of a with a window of n. The length of the returned array is len(a) - n + 1.

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


def block_averages(a, n):
    """
    Calculates the average of every n values in a. If len(a) is not divisible by n, the last value is the average of
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
