import matplotlib.pyplot as plt
import numpy as np


class Plot:
    @staticmethod
    def _setting_setter(ax, *, xlabel='', ylabel='', title='', grid=True, xlim=None, ylim=None, xticks=None,
                        yticks=None, xscale=None, yscale=None, font_size=12, xticklabels=None, yticklabels=None):
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if grid is not None:
            ax.grid(grid)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if xticks is not None:
            ax.set_xticks(xticks, xticklabels)
        if yticks is not None:
            ax.set_yticks(yticks, yticklabels)
        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)
        plt.rc('font', size=font_size)

    @staticmethod
    def _1d_lines(xs, ys, *, colors=None, labels=None, legend_kwargs: dict = None, save_loc: str = None,
                  show: bool = False, plot_kwargs: dict = None, cbar_kwargs: dict = None, line_kwargs: dict = None,
                  save_kwargs: dict = None):
        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if labels is None:
            labels = [None] * len(xs)

        if not isinstance(xs[0], (list, np.ndarray)):
            if len(xs) == len(ys[0]):
                xs = [xs] * len(ys)
            else:
                raise ValueError('xs and ys must have the same shape or xs must have the same length lists in ys')

        fig, ax = plt.subplots()
        line_kwargs = line_kwargs or {}
        for x, y, color, lab in zip(xs, ys, colors, labels):
            plt.plot(x, y, color, label=lab, **line_kwargs)

        if legend_kwargs is not None:
            plt.legend(**legend_kwargs)
        if cbar_kwargs is not None:
            plt.colorbar(**cbar_kwargs, ax=plt.gca())

        plot_kwargs = plot_kwargs or {}
        Plot._setting_setter(ax, **plot_kwargs)

        plt.tight_layout()
        if save_loc is not None:
            save_kwargs = save_kwargs or {}
            plt.savefig(save_loc, **save_kwargs)
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def set_defaults(kwargs_dict: dict | None, **kwargs) -> dict:
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
        kwargs_dict = kwargs_dict or {}
        kwargs.update(kwargs_dict)
        return kwargs
