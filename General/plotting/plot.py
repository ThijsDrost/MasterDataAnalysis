"""
Contains the Plot class, which is a wrapper around matplotlib, to make it easier to make plots. The main plotting functions
are `lines`, `errorbar` and `errorrange`. The `lines` function is used to plot lines, the `errorbar` function is used to plot lines
with errorbars and the `errorrange` function is used to plot lines with a shaded error area around them.

Next to these plotting functions, it contains some convenience functions to make it easier to make plots. The `set_defaults`
function is used to set default values for a dictionary, the `marker_cycler`, `color_cycler` and `linestyle_cycler` functions
are used to get a marker, color or linestyle based on an index. The `linelook_by` function is used to get a list of dictionaries
with markers, colors and linestyles based on a list of values, to give each line a unique look.

This class is used for (almost) all plots made in this python project.
"""

import warnings
from collections.abc import Sequence
import operator

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from General.plotting.geometry import two_ellipse_tangent, ellipse_inside, Ellipse, Point, LineSegment, Line, Rectangle
from General.checking import Validator


def setting_setter(ax, *, xlabel='', ylabel='', title='', grid=True, xlim=None, ylim=None, xticks=None,
                   yticks=None, xscale=None, yscale=None, xticklabels=None, yticklabels=None, before=None,
                   fontsize=14, xticks_minor=False, yticks_minor=False, grid_which='major', grid_axis='both'):
    """
    Set the settings for a matplotlib plot. The kwargs correspond one to one with the matplotlib settings. If `before`
    is True, all the settings except the limits are set, if `before` is False, only the limits are set, if `before` is None,
    all settings are set.

    Notes
    -----
    The reason the split the settings into before in after, is since when using partial limits (only upper or lower limit),
    the other limit is based on the dat in the plot, so the limits should be set after the data is plotted. The grid is set
    before, so that the grid is behind the data. For the other settings it does not matter if they are set before or after.
    """

    def set_lim(func, value):
        if isinstance(value, (int, float)):
            func(value)
        else:
            func(*value)

    if before or before is None:
        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)
        if xticks is not None:
            ax.set_xticks(xticks, xticklabels)
        if xticks_minor is True:
            ax.minorticks_on()
        if yticks is not None:
            ax.set_yticks(yticks, yticklabels, yticks_minor)
        if yticks_minor is True:
            ax.minorticks_on()
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if grid is not None:
            ax.grid(grid, grid_which, grid_axis)

    if (not before) or before is None:
        if xlim is not None:
            set_lim(ax.set_xlim, xlim)
        if ylim is not None:
            set_lim(ax.set_ylim, ylim)
        if fontsize is not None:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)


def _check_sequence_or_ndarray(value):
    return isinstance(value, (np.ndarray, Sequence)) and (value is not np.ma.masked)


def _check_data(xs, ys):
    if not isinstance(xs, (np.ndarray, Sequence)):
        raise ValueError(f'`xs` should be a numpy array or sequence, not {type(xs)}')
    if not (isinstance(ys, (np.ndarray, Sequence)) or ys is None):
        raise ValueError(f'`ys` should be a numpy array or sequence, not {type(ys)}')
    if ys is None:
        ys = xs
        if len(xs) == 0:
            return [[]], [[]]
        if _check_sequence_or_ndarray(xs[0]):
            xs = [np.arange(len(xs[0])) for _ in xs]
        else:
            xs = np.arange(len(xs))

    if len(xs) == 0:
        if len(ys) == 0:
            return [[]], [[]]
        else:
            raise ValueError(f'`xs` should have the same length as `ys`, not {len(xs)} and {len(ys)}')

    if (not _check_sequence_or_ndarray(xs[0])) or isinstance(xs[0], str):
        if not _check_sequence_or_ndarray(ys[0]):
            if len(xs) != len(ys):
                raise ValueError(f'`xs` and `ys` must have the same length, not {len(xs)} and {len(ys)}')
            else:
                xs = [xs]
                ys = [ys]
        elif len(xs) == len(ys[0]):
            xs = [xs] * len(ys)
        else:
            raise ValueError('`xs` and `ys` must have the same shape or `xs` must have the same length lists in `ys`')
    else:
        if len(xs) != len(ys):
            raise ValueError(f'`xs` and `ys` must have the same length, not {len(xs)} and {len(ys)}')

    for index, (x, y) in enumerate(zip(xs, ys)):
        if len(x) != len(y):
            raise ValueError(f'`xs` and `ys` must have the same length, for line {index}, lengths are `xs`:{len(x)} and `ys`:{len(y)}')

    return xs, ys



def plot_lines(plot_func, xs: np.ndarray | Sequence, ys: np.ndarray | Sequence = None, /, *, colors=None, labels=None,
               legend_kwargs: dict = None, save_loc: str = None, show: bool = False, plot_kwargs: dict = None,
               cbar_kwargs: dict = None, line_kwargs: dict = None, save_kwargs: dict = None, close=True, fig_ax=None,
               line_kwargs_iter: list[dict] = None, tight_layout=True):
    """
    Plot lines on a matplotlib plot. The `plot_func` is used to plot the lines, this can be `plt.plot` or `plt.errorbar`.

    Parameters
    ----------
    plot_func: str
    xs
        The x-values for the lines. Should have the same shape as `ys`, or the same length of the first dimension of `ys`.
    ys
        The y-values for the lines. If `ys` is 2D, multiple lines are plotted.
    colors
        The colors for the lines. If None, the default color cycle is used. If a single color is given, all lines are plotted
        in that color. If a list of colors is given, the colors are used as a colorwheel. If the colorwheel is exhausted, the
        colors are repeated.
    labels
        The labels for the lines. If None, no labels are used.
    show
        Whether to show the plot. If False, the plot is not shown.
    close
        Whether to close the plot after showing or saving. If False, the plot is not closed.
    legend_kwargs
        The kwargs for :py:func:`plt.legend`. If None, no legend is used.
    save_loc
        The location to save the plot. If None, the plot is not saved.
    save_kwargs
        The kwargs for saving the plot (:py:func:`plt.savefig`). If None, the default values are used.
    plot_kwargs
        The kwargs to use in the :py:func:`setting_setter` function. If None, no settings are set.
    cbar_kwargs
        The kwargs for the colorbar (:py:func:`plt.colorbar`). If None, no colorbar is used.
    line_kwargs
        The kwargs for the plotting functions, the same kwargs are used for each line. If there is overlap with the values in
        `line_kwargs_iter`, the values from `line_kwargs_iter` are used.
    line_kwargs_iter
        An iterable of dictionaries with the kwargs for the plotting functions. This should contain one dictionary for each line
        to be plotted. These values take president over the values in `line_kwargs`. If None, no extra kwargs are used.
    fig_ax
        The figure and axis to use for the plot. If None, a new figure is created.
    tight_layout
        Whether to use :py:func:plt.tight_layout for the plot.

    Returns
    -------
    fig, ax
        The figure and axis of the plot if `show` and `close` are False, else None.

    Notes
    -----
    The plot_func is passes a default `zorder` value of 2
    """
    xs, ys = _check_data(xs, ys)

    for index, (x, y) in enumerate(zip(xs, ys)):
        if len(x) != len(y):
            raise ValueError(f'`xs` and `ys` must have the same length, for line {index}, lengths are `xs`:{len(x)} and `ys`:{len(y)}')

    if line_kwargs_iter is None:
        line_kwargs_iter = [{} for _ in xs]
    else:
        if len(line_kwargs_iter) != len(xs):
            raise ValueError(f'`line_kwargs_iter` should have the same length as xs, not {len(line_kwargs_iter)} and {len(xs)}')
        for i, kwargs in enumerate(line_kwargs_iter):
            if not isinstance(kwargs, dict):
                raise TypeError(f'`line_kwargs_iter[{i}]` should be a dict, not {type(kwargs)}')

    if (labels is not None) and (len(labels) != len(xs)):
        raise ValueError(f'`labels` should have the same length as `xs`, not {len(labels)} and {len(xs)}')

    if 'c' in line_kwargs_iter[0]:
        raise ValueError('Use `color` instead of `c` in `line_kwargs_iter`')

    if (colors is None) and ('color' not in line_kwargs_iter[0]):
        offset = 0
        if fig_ax is not None:
            offset = len(fig_ax[1].lines)
        color_wheel = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [color_wheel[(i + offset) % len(color_wheel)] for i in range(len(xs))]
        if len(xs) > len(color_wheel) and ('color' not in line_kwargs_iter[0]):
            warnings.warn(f'Only {len(color_wheel)} colors are available, but {len(xs)} lines are plotted, so colors will be '
                          f'repeated.')

    if colors is not None:
        for index in range(len(line_kwargs_iter)):
            if 'color' not in line_kwargs_iter[index]:
                line_kwargs_iter[index]['color'] = colors[index]

    if labels is not None:
        for index in range(len(line_kwargs_iter)):
            line_kwargs_iter[index]['label'] = labels[index]

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    plot_func = getattr(ax, plot_func.__name__)

    def add_true_mask(mask: np.ndarray) -> np.ndarray:
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
        [False, True, True, True, True, True, True, False]
        >>> add_true_mask([False, False, True, False, False, True, True, True])
        [False, True, True, True, False, True, True, True]
        >>> add_true_mask([True, True, True, False, False, True, False, False])
        [True, True, True, True, True, True, True, False]
        """
        if (not np.any(mask)) or np.all(mask):
            return mask

        mask_mask = np.zeros(len(mask), dtype=bool)
        diff = np.diff(mask.astype(int))
        mask_mask[:-1] |= diff == 1
        mask_mask[1:] |= diff == -1
        mask[mask_mask] = True
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
            mask &= limits <= values
        elif len(limits) == 1:
            mask &= limits[0] <= values
        else:
            if limits[0] is not None:
                mask &= limits[0] <= values
            if limits[1] is not None:
                mask &= values <= limits[1]
        return add_true_mask(mask)

    def mk_lims(xs):
        xlims = (min((min(x) for x in xs)), max((max(x) for x in xs)))
        return xlims[0] if not np.isnan(xlims[0]) else None, xlims[1] if not np.isnan(xlims[1]) else None

    if (fig_ax is None) and (len(xs[0]) > 0) and (not isinstance(xs[0][0], str)):
        xlims = mk_lims(xs)
        plot_kwargs = set_defaults(plot_kwargs, xlim=xlims)
    elif (len(xs[0]) == 0) or isinstance(xs[0][0], str):
        plot_kwargs = plot_kwargs or {}
    else:
        _, ax = fig_ax
        xlim = ax.get_xlim()

        # xlims of (-0.05500000000000001, 0.05500000000000001) are the default, so ignore these.
        xlim = (xlim[0] if xlim[0] != -0.05500000000000001 else None,
                xlim[1] if xlim[1] != 0.05500000000000001 else None)

        xlims = mk_lims(xs)

        def get_val(val1, val2, operator):
            try:
                return operator(val1, val2)
            except:
                return val1

        new_xlim = (get_val(xlim[0], xlims[0], min) if xlim[0] is not None else xlims[0],
                    get_val(xlim[1], xlims[1], max) if xlim[1] is not None else xlims[1])  # TODO: find better solution

        plot_kwargs = set_defaults(plot_kwargs, xlim=new_xlim)
    setting_setter(ax, before=True, **plot_kwargs)

    line_kwargs = line_kwargs or {}
    for x, y, kwargs in zip(xs, ys, line_kwargs_iter, strict=True):
        x, y = np.array(x), np.array(y)

        if 'xlim' in plot_kwargs:
            mask = make_mask(x, plot_kwargs['xlim'])
            x = np.ma.masked_where(~mask, x)
            y = np.ma.masked_where(~mask, y)

        if 'ylim' in plot_kwargs:
            mask = make_mask(y, plot_kwargs['ylim'])
            x = np.ma.masked_where(~mask, x)
            y = np.ma.masked_where(~mask, y)

        if len(x) == 0:
            continue

        kwargs = set_defaults(kwargs, **line_kwargs, zorder=3)
        plot_func(x, y, **kwargs)

    setting_setter(ax, before=False, **plot_kwargs)

    if (legend_kwargs is not None) or ((labels is not None) and any(labels)):
        legend_kwargs = set_defaults(legend_kwargs, fontsize=plot_kwargs.get('fontsize', 14))
        legend = ax.legend(**legend_kwargs)

        # If `visible=False` for an entry, assume it's a title.
        # From https://stackoverflow.com/questions/24787041/multiple-titles-in-legend-in-matplotlib
        for col in legend._legend_handle_box.get_children():
            row = col.get_children()
            new_children: list[plt.Artist] = []
            for hpacker in row:
                if not isinstance(hpacker, mpl.offsetbox.HPacker):
                    new_children.append(hpacker)
                    continue
                drawing_area, text_area = hpacker.get_children()
                handle_artists = drawing_area.get_children()
                if not all(a.get_visible() for a in handle_artists):
                    new_children.append(text_area)
                else:
                    new_children.append(hpacker)
            col._children = new_children

    if cbar_kwargs is not None:
        plt.colorbar(**cbar_kwargs, ax=ax)

    if tight_layout:
        fig.tight_layout()
    if save_loc is not None:
        save_kwargs = save_kwargs or {}
        fig.savefig(save_loc, **save_kwargs)
    if show:
        fig.show()
    elif close:
        plt.close(fig)
    else:
        return fig, ax


def lines(xs: np.ndarray | list, ys: np.ndarray | list = None, /, *, colors=None, labels=None, legend_kwargs: dict = None,
          save_loc: str = None, show: bool = True, plot_kwargs: dict = None, cbar_kwargs: dict = None, fig_ax=None,
          line_kwargs: dict = None, save_kwargs: dict = None, close=False, line_kwargs_iter: list[dict] = None):
    """
    Wrapper around the :py:func:`plot_lines` function, with the `plot_func` set to `plt.plot`.
    """
    return plot_lines(plt.plot, xs, ys, colors=colors, labels=labels, legend_kwargs=legend_kwargs, save_loc=save_loc,
                      show=show, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs,
                      save_kwargs=save_kwargs, close=close, fig_ax=fig_ax, line_kwargs_iter=line_kwargs_iter)

def semilogy(xs: np.ndarray | list, ys: np.ndarray | list=None, /, *, colors=None, labels=None, legend_kwargs: dict = None,
             save_loc: str = None, show: bool = True, plot_kwargs: dict = None, cbar_kwargs: dict = None, fig_ax=None,
             line_kwargs: dict = None, save_kwargs: dict = None, close=False, line_kwargs_iter: list[dict] = None):
    """
    Wrapper around the :py:func:`plot_lines` function, with the `plot_func` set to `plt.semilogy`.
    """
    return plot_lines(plt.semilogy, xs, ys, colors=colors, labels=labels, legend_kwargs=legend_kwargs, save_loc=save_loc,
                      show=show, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs,
                      save_kwargs=save_kwargs, close=close, fig_ax=fig_ax, line_kwargs_iter=line_kwargs_iter)


def _make_err(err, ref_data, name):
    if err is None:
        return err
    if isinstance(err, (int, float)):
        err = [[err for _ in x] for x in ref_data]
    if len(err) != len(ref_data):
        if len(err) == len(ref_data[0]):
            err = [err for _ in ref_data]
        else:
            raise ValueError(f'`{name}` should be a single value or a list with the same length as the number of lines or the '
                             f'number of values')
    else:
        if isinstance(err[0], (int, float)):
            err = [[err[i] for _ in ref_data[i]] for i in range(len(ref_data))]
        elif len(err[0]) != len(ref_data[0]):
            raise ValueError(f'`{name}` should be a single value or a list with the same length as the number of lines or the '
                             f'number of values')
    return err


def errorbar(xs: np.ndarray | list, ys: np.ndarray | list, *, xerr=None, yerr=None, colors=None, labels=None, legend_kwargs: dict = None,
             save_loc: str = None, show: bool = True, plot_kwargs: dict = None, cbar_kwargs: dict = None, fig_ax=None,
             line_kwargs: dict = None, save_kwargs: dict = None, close=False, line_kwargs_iter: list[dict] = None):
    """
    Wrapper around the :py:func:`plot_lines` function, with the `plot_func` set to `plt.errorbar`.

    Notes
    -----
    Default value for the capsize is set to 2.
    """
    if (xerr is None) and (yerr is None):
        warnings.warn('No error given, so no error is plotted')

    xs, ys = _check_data(xs, ys)
    xerr = _make_err(xerr, xs, 'xerr')
    yerr = _make_err(yerr, ys, 'yerr')

    if line_kwargs_iter is None:
        if xerr is not None:
            line_kwargs_iter = [{} for _ in xerr]
        if yerr is not None:
            line_kwargs_iter = [{} for _ in yerr]
    for i, kwargs in enumerate(line_kwargs_iter):
        line_kwargs_iter[i]['xerr'] = xerr[i] if (xerr is not None) else None
        line_kwargs_iter[i]['yerr'] = yerr[i] if (yerr is not None) else None
    line_kwargs = set_defaults(line_kwargs, capsize=2)
    return plot_lines(plt.errorbar, xs, ys, colors=colors, labels=labels, legend_kwargs=legend_kwargs, save_loc=save_loc,
                      show=show, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs,
                      save_kwargs=save_kwargs, close=close, fig_ax=fig_ax, line_kwargs_iter=line_kwargs_iter)


def errorrange(xs: np.ndarray | list, ys: np.ndarray | list, *, xerr=None, yerr=None, error_shape='ellipse', save_loc: str = None,
               show: bool = False, close=True, save_kwargs=None, colors=None, line_kwargs_iter=None, fig_ax=None, **kwargs):
    """
    Plot lines with the error as a shaded error area around them. All kwargs (except `xerr`, `yerr`, and `error_shape`) are the
    same as in :py:func:`plot_lines`.

    Parameters
    ----------
    xerr
        The error in the x-values. If a single value is given, the same error is used for all values in all lines. If the length
        of the error is the same as the number of lines, each line gets its own error (with the same value for each point in the
        line). If the length of the error is the same as the number of values in the lines, every line gets the same error, but
        the points within each line get different values. If the error has the same shape as the lines, every point gets its own
        error. Only symmetric errors are supported.
    yerr
        The error in the y-values. The same rules apply as for `xerr`.
    error_shape: str
        The shape of the error, if both `xerr` and `yerr` are given. If 'ellipse', the error is plotted as an ellipse. If
        'rectangle', the error is plotted as a circle.

    Notes
    -----
    `xs` should be sorted

    The range is plotted with `zorder` 2.
    """
    if (xerr is None) and (yerr is None):
        warnings.warn('No error given, so no error is plotted')

    if error_shape not in ['ellipse', 'rectangle', 'rect']:
        raise ValueError(f'`error_shape` should be "ellipse" or "rectangle", not {error_shape}')

    fig, ax = plot_lines(plt.plot, xs, ys, save_loc=None, show=False, close=False, colors=colors, line_kwargs_iter=line_kwargs_iter,
                         fig_ax=fig_ax, **kwargs)
    xs, ys = _check_data(xs, ys)

    # is_sorted = Validator.sorted()
    # for line in ax.lines:
    #     is_sorted(line.get_xdata(), 'xs')

    def set_err(err, values, name):
        if err is None:
            return err

        if isinstance(err, (int, float)):
            return [err] * len(values)
        if len(err) != len(values):
            if len(err) == len(values[0].get_xdata()):
                return [err] * len(values)
        else:
            return err
        raise ValueError(f'`{name}` should be a single value or a list/array with the same length as the number of lines or the '
                         f'number of values. `yerr` has length {len(yerr)}, `xs` has length {len(values[0].get_xdata())}')


    xerr = set_err(xerr, ax.lines[-len(ys):], 'xerr')
    yerr = set_err(yerr, ax.lines[-len(ys):], 'yerr')
    if xerr is None:
        for i, line in enumerate(ax.lines[-len(ys):]):
            ax.fill_between(line.get_xdata(), line.get_ydata() - yerr[i], line.get_ydata() + yerr[i], color=line.get_color(),
                            alpha=0.5, zorder=2)
    elif yerr is None:
        for i, line in enumerate(ax.lines[-len(ys):]):
            ax.fill_betweenx(line.get_ydata(), line.get_xdata() - xerr[i], line.get_xdata() + xerr[i], color=line.get_color(),
                             alpha=0.5, zorder=2)
    else:
        xdata = [line.get_xdata() for line in ax.lines]
        ydata = [line.get_ydata() for line in ax.lines]

        if error_shape == 'ellipse':
            def remove_inside(ellipses: Sequence[Ellipse]):
                """
                Remove (recursively) ellipses that are inside other ellipses.
                """
                out_ellipses = []
                change = False

                if ellipse_inside(ellipses[0], ellipses[1]):
                    return remove_inside(ellipses[1:])

                for index in range(len(ellipses) - 1):
                    if ellipse_inside(ellipses[index], ellipses[index + 1]):
                        change = True
                    else:
                        out_ellipses.append(ellipses[index + 1])

                if change:
                    return remove_inside(out_ellipses)
                else:
                    return out_ellipses

            for line_index in range(len(xdata)):
                ellipses = []
                for point_index in range(len(xdata[line_index])):
                    point1 = Point(xdata[line_index][point_index], ydata[line_index][point_index])
                    ellipse = Ellipse(point1, Point(xerr[line_index], yerr[line_index]))
                    ellipses.append(ellipse)
                ellipses = remove_inside(ellipses)

                upper_segments = [LineSegment.zeros()] * (len(ellipses) - 1)
                lower_segments = [LineSegment.zeros()] * (len(ellipses) - 1)
                for index in range(len(ellipses) - 1):
                    upper, lower = two_ellipse_tangent(ellipses[index], ellipses[index + 1])
                    upper_segments[index] = LineSegment(upper[0], upper[1])
                    lower_segments[index] = LineSegment(lower[0], lower[1])

                def add_ellipse_top(segment1, segment2, ellipse, sign=1):
                    if (segment1.intersection(segment2) is None) \
                            and segment1[1].x < ellipse.center.x < segment2[0].x:
                        return (LineSegment(segment1[1], ellipse.center + Point(0, ellipse.radii.y)),
                                LineSegment(ellipse.center + sign * Point(0, ellipse.radii.y), segment2[0]))
                    return ()

                result_upper_segments = []
                result_lower_segments = []
                if ellipses[0].center.x < upper_segments[0][0].x:
                    result_upper_segments.append(
                        LineSegment(Point(ellipses[0].center.x, ellipses[0].center.y + ellipses[0].radii.y),
                                    upper_segments[0][0]))
                if ellipses[0].center.x < lower_segments[0][0].x:
                    result_lower_segments.append(
                        LineSegment(Point(ellipses[0].center.x, ellipses[0].center.y - ellipses[0].radii.y),
                                    lower_segments[0][0]))

                for circle_index in range(1, len(ellipses) - 1):
                    upper_segments = (upper_segments[circle_index - 1], upper_segments[circle_index])
                    result_upper_segments.append(upper_segments[0])
                    for val in add_ellipse_top(upper_segments[0], upper_segments[1], ellipses[circle_index]):
                        result_upper_segments.append(val)
                    result_upper_segments.append(upper_segments[1])

                    lower_segments = (lower_segments[circle_index - 1], lower_segments[circle_index])
                    result_lower_segments.append(lower_segments[0])
                    for val in add_ellipse_top(lower_segments[0], lower_segments[1], ellipses[circle_index], -1):
                        result_lower_segments.append(val)
                    result_lower_segments.append(lower_segments[1])

                if upper_segments[-1][1].x < ellipses[-1].center.x:
                    result_upper_segments.append(LineSegment(upper_segments[-1][1],
                                                             Point(ellipses[-1].center.x,
                                                                   ellipses[-1].center.y + ellipses[-1].radii.y)))
                if lower_segments[-1][1].x < ellipses[-1].center.x:
                    result_lower_segments.append(LineSegment(lower_segments[-1][1],
                                                             Point(ellipses[-1].center.x,
                                                                   ellipses[-1].center.y - ellipses[-1].radii.y)))

                upper_line = Line(result_upper_segments)
                lower_line = Line(result_lower_segments)
                ax.fill_between(upper_line.x_values, upper_line.y_values, lower_line.y_values,
                                color=ax.lines[line_index].get_color(),
                                alpha=0.5)
        else:
            for line_index in range(len(xdata)):
                upper_points = []
                lower_points = []
                for point_index in range(len(xdata[line_index])):
                    point1 = Point(xdata[line_index][point_index], ydata[line_index][point_index])
                    rectangle = Rectangle(point1, 2 * Point(xerr[line_index], yerr[line_index]))
                    points = rectangle.corners()
                    upper_points.extend((points[0], (points[0] + points[1]) / 2, points[1]))
                    lower_points.extend((points[2], (points[2] + points[3]) / 2, points[3]))

                def points(p: Sequence[Point], y_comp: callable) -> Sequence[Point]:
                    point = p[0]
                    results = []
                    change = False
                    for new_point in p[1:]:
                        if point.x < new_point.x:
                            results.append(point)
                            continue
                        elif y_comp(point.y, new_point.y):
                            point = new_point
                            change = True
                    results.append(point)

                    if change:
                        return points(results)
                    else:
                        return results

                lower_points = points(lower_points, operator.gt)
                upper_points = points(upper_points, operator.lt)
                lower_line = Line(lower_points)
                upper_line = Line(upper_points)
                x_values = np.union1d(lower_line.x_values, upper_line.x_values)
                ax.fill_between(x_values, lower_line.interpolate(x_values, False),
                                upper_line.interpolate(x_values, False), color=ax.lines[line_index].get_color(),
                                alpha=0.5)

    if save_loc is not None:
        save_kwargs = save_kwargs or {}
        plt.savefig(save_loc, **save_kwargs)
    if show:
        plt.show()
    elif close:
        plt.close()
    else:
        return fig, ax


def set_defaults(kwargs_dict: dict | None | bool, **kwargs) -> dict:
    """
    Set values for a dict. If the dict already has a value for a key, the value is not changed.

    Parameters
    ----------
    kwargs_dict: dict | None
        If None, a new dict is created. The dict is updated with the kwargs, with the values in the dict taking
        precedence over the kwargs.
    kwargs:
        The kwargs to add to the dict

    Returns
    -------
    dict
        The dict with the values from the kwargs added to it
    """

    if (kwargs_dict is None) or (kwargs_dict is True) or (kwargs_dict is False):
        return kwargs
    kwargs.update(kwargs_dict)
    return kwargs


def make_legend(line_kwargs_iter, labels) -> dict[str, list]:
    """
    Make a legend for a plot. The legend is made based on the labels, with the line_kwargs_iter used to get the correct
    lines.
    """

    been = set()
    handles, labels_out = [], []
    for line_kwargs, label in zip(line_kwargs_iter, labels):
        if label in been:
            continue
        labels_out.append(label)
        handles.append(plt.Line2D([0], [0], **line_kwargs))
        been.add(label)
    return {'handles': handles, 'labels': labels_out}
