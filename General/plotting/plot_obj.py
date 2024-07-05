from __future__ import annotations

from abc import ABC
from collections import Sequence, Iterable

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, plot_objs: PlotObj | Iterable[PlotObj], plot_settings: dict = None):
        if not isinstance(plot_objs, Sequence):
            plot_objs = [plot_objs]
        self.plot_objs = list(plot_objs)
        self._plot_settings = SkipDict()
        self.plot_settings(**(plot_settings or {}))

    def plot(self, fig_ax=None):
        if fig_ax is None:
            fig_ax = plt.subplots()
        fig, ax = fig_ax
        for plot_obj in self.plot_objs:
            plot_obj.plot(ax)

    def _set_settings(self, ax: plt.Axes):
        for key, value in self._plot_settings:
            if key == 'xlabel':
                ax.set_xlabel(value)
            elif key == 'ylabel':
                ax.set_ylabel(value)
            elif key == 'title':
                ax.set_title(value)
            elif key == 'grid':
                ax.grid(value)
            elif key == 'xlim':
                ax.set_xlim(value)
            elif key == 'ylim':
                ax.set_ylim(value)
            elif key == 'xticks':
                ax.set_xticks(value)
            elif key == 'yticks':
                ax.set_yticks(value)
            elif key == 'xscale':
                ax.set_xscale(value)
            elif key == 'yscale':
                ax.set_yscale(value)
            elif key == 'xticklabels':
                ax.set_xticklabels(value)
            elif key == 'yticklabels':
                ax.set_yticklabels(value)



    def plot_settings(self, xlabel='', ylabel='', title='', grid=True, xlim=None, ylim=None, xticks=None,
                      yticks=None, xscale=None, yscale=None, xticklabels=None, yticklabels=None):
        self._plot_settings.update(xlabel=xlabel, ylabel=ylabel, title=title, grid=grid, xlim=xlim, ylim=ylim,
                                   xticks=xticks, yticks=yticks, xscale=xscale, yscale=yscale, xticklabels=xticklabels,
                                   yticklabels=yticklabels)


class PlotObj(ABC):
    def plot(self, ax):
        pass


class Lines(PlotObj):
    def __init__(self, x, y=None, /, *, line_kwargs=None, line_kwargs_iter=None):
        if not isinstance(x, (np.ndarray, Sequence)):
            raise ValueError(f'`x` should be a numpy array or sequence, not {type(x)}')
        if not (isinstance(y, (np.ndarray, Sequence)) or y is None):
            raise ValueError(f'`y` should be a numpy array or sequence, not {type(y)}')
        if y is None:
            y = x
            if isinstance(x[0], (np.ndarray, Sequence)):
                x = [np.arange(len(x[0]))] * len(x)
            else:
                x = np.arange(len(x))

        if not isinstance(x[0], (np.ndarray, Sequence)):
            if not isinstance(y[0], (np.ndarray, Sequence)):
                if len(x) != len(y):
                    raise ValueError(f'`x` and `y` must have the same length, not {len(x)} and {len(y)}')
                else:
                    x = [x]
                    y = [y]
            elif len(x) == len(y[0]):
                x = [x] * len(y)
            else:
                raise ValueError('`x` and `y` must have the same shape or `x` must have the same length lists in `y`')
        else:
            if len(x) != len(y):
                raise ValueError(f'`x` and `y` must have the same length, not {len(x)} and {len(y)}')

        for index, (x_v, y_v) in enumerate(zip(x, y)):
            if len(x_v) != len(y_v):
                raise ValueError(
                    f'`x` and `y` must have the same length, for line {index}, lengths are `x`:{len(x_v)} and `y`:{len(y_v)}')

        if len(line_kwargs_iter) != len(x):
            raise ValueError(f'`line_kwargs_iter` must have the same length as `x` and `y`, not {len(line_kwargs_iter)} and {len(x)}')

        self._x = x
        self._y = y
        self._line_kwargs: dict = line_kwargs or {}
        self._line_kwargs_iter: list[dict] = line_kwargs_iter or [{} for _ in range(len(x))]

    def plot(self, ax):
        for x, y, line_kwargs in zip(self._x, self._y, self._line_kwargs_iter):
            this_line_kwargs = self._line_kwargs.copy()
            this_line_kwargs.update(line_kwargs)
            ax.plot(x, y, **this_line_kwargs)


class SkipDict(dict):
    def __init__(self, skip_values=None, *args, **kwargs):
        self.skip_values = skip_values or [None]
        super().__init__(*args, **kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value in self.skip_values:
                self[key] = value


def add_true_mask(mask: np.ndarray | Sequence) -> np.ndarray:
    """
    Add an extra True before the first and after the last True value in the mask. If the first value is True, no extra
    True value is added, same for the last value.

    Examples
    --------
    >>> add_true_mask([False, True, False, False])
    [True, True, True, False]
    >>> add_true_mask([False, False, True, True, True, False, False, False])
    [False, True, True, True, True, True, False, False]
    >>> add_true_mask([False, False, True, False, True, True, False, False])
    [False, True, True, False, True, True, True, False]
    """
    if not mask[0]:
        start = np.argmax(mask)
        mask[start - 1] = True
    if not mask[-1]:
        end = len(mask) - np.argmax(mask[::-1])
        mask[end] = True
    return mask


def make_mask(values: np.ndarray, limits: float | int | tuple[float | int] | tuple[float | int | None, float | int | None]):
    """
    Make a mask based on the limits.

    Parameters
    ----------
    values:
        The values to mask
    limits:
        If the limits is a float or int, the mask is True for values greater than the limit. A tuple with one value is
        interpreted as a lower limit, a tuple with two values is interpreted as a lower and upper limit. If a limit is
        None, that limit is not used.

    Returns
    -------
    np.ndarray
    """
    mask = np.ones(len(values), dtype=bool)
    if isinstance(limits, (float, int)):
        mask &= limits < values
    elif len(limits) == 1:
        mask &= limits[0] < values
    else:
        if limits[0] is not None:
            mask &= limits[0] < values
        if limits[1] is not None:
            mask &= values < limits[1]
    return add_true_mask(mask)