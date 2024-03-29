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

import matplotlib.pyplot as plt
import numpy as np


class Plot:
    @staticmethod
    def setting_setter(ax, before=None, *, xlabel='', ylabel='', title='', grid=True, xlim=None, ylim=None, xticks=None,
                       yticks=None, xscale=None, yscale=None, xticklabels=None, yticklabels=None):
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
            if yticks is not None:
                ax.set_yticks(yticks, yticklabels)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if title is not None:
                ax.set_title(title)
            if grid is not None:
                ax.grid(grid)

        if (not before) or before is None:
            if xlim is not None:
                set_lim(ax.set_xlim, xlim)
            if ylim is not None:
                set_lim(ax.set_ylim, ylim)

    @staticmethod
    def _lines(plot_func, /, xs: np.ndarray, ys: np.ndarray, *, colors=None, labels=None, legend_kwargs: dict = None, save_loc: str = None,
               show: bool = False, plot_kwargs: dict = None, cbar_kwargs: dict = None, line_kwargs: dict = None,
               save_kwargs: dict = None, close=True, fig_ax=None, line_kwargs_iter: list[dict] = None, font_size=14):
        if not isinstance(xs, (np.ndarray, list)):
            raise ValueError(f'xs should be a numpy array, not {type(xs)}')
        if not isinstance(ys, (np.ndarray, list)):
            raise ValueError(f'ys should be a numpy array, not {type(ys)}')

        if not isinstance(xs[0], (list, np.ndarray)):
            if not isinstance(ys[0], (list, np.ndarray)):
                if len(xs) != len(ys):
                    raise ValueError(f'xs and ys must have the same length, not {len(xs)} and {len(ys)}')
                else:
                    xs = [xs]
                    ys = [ys]
            elif len(xs) == len(ys[0]):
                xs = [xs] * len(ys)
            else:
                raise ValueError('xs and ys must have the same shape or xs must have the same length lists in ys')
        else:
            if len(xs) != len(ys):
                raise ValueError(f'xs and ys must have the same length, not {len(xs)} and {len(ys)}')

        if line_kwargs_iter is None:
            line_kwargs_iter = [{}]*len(xs)
        else:
            if len(line_kwargs_iter) != len(xs):
                raise ValueError(f'line_kwargs_iter should have the same length as xs, not {len(line_kwargs_iter)} and {len(xs)}')
            for i, kwargs in enumerate(line_kwargs_iter):
                if not isinstance(kwargs, dict):
                    raise TypeError(f'line_kwargs_iter[{i}] should be a dict, not {type(kwargs)}')
                if line_kwargs_iter[0].keys() != kwargs.keys():
                    raise ValueError(f'line_kwargs_iter[{i}] should have the same keys as others, not {list(kwargs.keys())} and {list(line_kwargs_iter[0].keys())}')

        if 'c' in line_kwargs_iter[0]:
            raise ValueError('Use `color` instead of `c` in line_kwargs_iter')

        if colors is None:
            color_wheel = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors = [color_wheel[i % len(color_wheel)] for i in range(len(xs))]
            if len(xs) > len(color_wheel) and ('color' not in line_kwargs_iter[0]):
                warnings.warn(f'Only {len(color_wheel)} colors are available, not {len(xs)}, so colors will be repeated.')

        if labels is None:
            labels = [None] * len(xs)

        plt.rc('font', size=font_size)
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        def add_true_mask(mask):
            if not mask[0]:
                start = np.argmax(mask)
                mask[start-1] = True
            if not mask[-1]:
                end = len(mask) - np.argmax(mask[::-1])
                mask[end] = True
            return mask

        def make_mask(values: np.ndarray, limits):
            mask = np.ones(len(values), dtype=bool)
            if isinstance(limits, (float, int)) or len(limits) == 1:
                mask &= limits < values
            elif limits[0] is not None:
                mask &= limits[0] < values
            elif limits[1] is not None:
                mask &= values < limits[1]
            return add_true_mask(mask)

        plot_kwargs = plot_kwargs or {}
        Plot.setting_setter(ax, True, **plot_kwargs)

        line_kwargs = line_kwargs or {}
        for x, y, color, lab, kwargs in zip(xs, ys, colors, labels, line_kwargs_iter, strict=True):
            x, y = np.array(x), np.array(y)
            if 'xlim' in plot_kwargs:
                mask = make_mask(x, plot_kwargs['xlim'])
                x, y = x[mask], y[mask]

            if 'ylim' in plot_kwargs:
                mask = make_mask(y, plot_kwargs['ylim'])
                x, y = x[mask], y[mask]

            new_line_kwargs = Plot.set_defaults(line_kwargs, color=color, label=lab)
            kwargs = Plot.set_defaults(kwargs, **new_line_kwargs)
            plot_func(x, y, **kwargs)

        Plot.setting_setter(ax, False, **plot_kwargs)

        if legend_kwargs is not None:
            plt.legend(**legend_kwargs)
        if cbar_kwargs is not None:
            cbar = plt.colorbar(**cbar_kwargs, ax=plt.gca())
            cbar.ax.minorticks_off()

        plt.tight_layout()
        if save_loc is not None:
            save_kwargs = save_kwargs or {}
            plt.savefig(save_loc, **save_kwargs)
        if show:
            plt.show()
        elif close:
            plt.close()
        else:
            return fig, ax

    @staticmethod
    def lines(xs: np.ndarray | list, ys: np.ndarray | list, *, colors=None, labels=None, legend_kwargs: dict = None, save_loc: str = None,
              show: bool = False, plot_kwargs: dict = None, cbar_kwargs: dict = None, line_kwargs: dict = None,
              save_kwargs: dict = None, close=True, fig_ax=None, line_kwargs_iter: list[dict] = None):
        return Plot._lines(plt.plot, xs, ys, colors=colors, labels=labels, legend_kwargs=legend_kwargs, save_loc=save_loc,
                           show=show, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs,
                           save_kwargs=save_kwargs, close=close, fig_ax=fig_ax, line_kwargs_iter=line_kwargs_iter)

    @staticmethod
    def errorbar(xs: np.ndarray, ys: np.ndarray, *, xerr=None, yerr=None, colors=None, labels=None, legend_kwargs: dict = None, save_loc: str = None,
                 show: bool = False, plot_kwargs: dict = None, cbar_kwargs: dict = None, line_kwargs: dict = None,
                 save_kwargs: dict = None, close=True, fig_ax=None, line_kwargs_iter: list[dict] = None):
        if line_kwargs_iter is None:
            if xerr is not None:
                line_kwargs_iter = [{}]*len(xerr)
            if yerr is not None:
                line_kwargs_iter = [{}]*len(yerr)
        if line_kwargs_iter is not None:
            for i, kwargs in enumerate(line_kwargs_iter):
                line_kwargs_iter[i]['xerr'] = xerr[i] if xerr is not None else None
                line_kwargs_iter[i]['yerr'] = yerr[i] if yerr is not None else None
        line_kwargs = Plot.set_defaults(line_kwargs, capsize=2)
        Plot._lines(plt.errorbar, xs, ys, colors=colors, labels=labels, legend_kwargs=legend_kwargs, save_loc=save_loc,
                    show=show, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs,
                    save_kwargs=save_kwargs, close=close, fig_ax=fig_ax, line_kwargs_iter=line_kwargs_iter)

    @staticmethod
    def errorrange(xs: np.ndarray, ys: np.ndarray, *, xerr=None, yerr=None, colors=None, labels=None, legend_kwargs: dict = None, save_loc: str = None,
                 show: bool = False, plot_kwargs: dict = None, cbar_kwargs: dict = None, line_kwargs: dict = None,
                 save_kwargs: dict = None, close=True, fig_ax=None, line_kwargs_iter: list[dict] = None):
        if (xerr is not None) and (yerr is not None):
            raise ValueError('Only one of xerr and yerr can be used')

        fig, ax = Plot._lines(plt.plot, xs, ys, colors=colors, labels=labels, legend_kwargs=legend_kwargs, save_loc=save_loc,
                              show=show, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs, line_kwargs=line_kwargs,
                              save_kwargs=save_kwargs, close=close, fig_ax=fig_ax, line_kwargs_iter=line_kwargs_iter)

        if yerr is not None:
            if isinstance(yerr, (int, float)):
                yerr = [yerr]*len(ax.lines)
            for i, line in enumerate(ax.lines):
                ax.fill_between(line.get_xdata(), line.get_ydata() - yerr[i], line.get_ydata() + yerr[i], color=line.get_color(), alpha=0.5)
        if xerr is not None:
            if isinstance(xerr, (int, float)):
                xerr = [xerr]*len(ax.lines)
            for i, line in enumerate(ax.lines):
                ax.fill_between(line.get_ydata(), line.get_xdata() - xerr[i], line.get_xdata() + xerr[i], color=line.get_color(), alpha=0.5)
        return fig, ax

    @staticmethod
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

    _marker_list = ['o', 's', 'v', 'D', '^', '<', '>', 'p', '*', 'H', 'd', 'P', 'X']
    @staticmethod
    def marker_cycler(index):
        return Plot._marker_list[index % len(Plot._marker_list)]

    @staticmethod
    def color_cycler(index):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return colors[index % len(colors)]

    @staticmethod
    def _linestyles_maker():
        dash = (6.5, 1.5)
        dot = (1, 1.5)
        return [
            (),
            (1, 2),
            (4, 2),
            (*dash, *dot),
            (*dash, *dot, *dot),
            (*dash, *dash, *dot),
            (*dash, *dash, *dot, *dot),
        ]
    _linestyles = _linestyles_maker()

    @staticmethod
    def linestyle_cycler(index):
        return Plot._linestyles[index % len(Plot._linestyles)]

    @staticmethod
    def linelook_by(values, *, markers=None, linestyles=None, colors=None):
        if markers and isinstance(markers, bool):
            markers = Plot._marker_list
        elif markers is None or (isinstance(markers, bool) and (not markers)):
            markers = [None]
        elif isinstance(markers, list):
            pass
        elif isinstance(markers, str):
            markers = [markers]
        else:
            raise TypeError(f'Unknown type for markers: {type(markers)}. Allowed types are bool, list and str')

        if linestyles and isinstance(linestyles, bool):
            linestyles = Plot._linestyles
        elif linestyles is None or (isinstance(linestyles, bool) and (not linestyles)):
            linestyles = [None]
        elif isinstance(linestyles, list):
            pass
        elif isinstance(linestyles, (str, tuple)):
            linestyles = [linestyles]
        else:
            raise TypeError(f'Unknown type for linestyles: {type(linestyles)}. Allowed types are bool, list, tuple and str')

        if colors and isinstance(colors, bool):
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif colors is None or (isinstance(colors, bool) and (not colors)):
            colors = ['C0']
        elif isinstance(colors, list):
            pass
        elif isinstance(colors, (str, tuple)):
            colors = [colors]
        else:
            raise TypeError(f'Unknown type for colors: {type(colors)}. Allowed types are bool, list, tuple and str')

        given_values = {}
        for i, value in enumerate(values):
            if value not in given_values:
                given_values[value] = {'marker': markers[i % len(markers)], 'linestyle': (0, linestyles[i % len(linestyles)]), 'color': colors[i % len(colors)]}
        return [given_values[value] for value in values]


