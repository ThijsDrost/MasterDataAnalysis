from collections.abc import Sequence

import matplotlib.figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class LinePlot:
    def __init__(self, root, figsize, loc, *, xlim=None, ylim=None, xlabel='', ylabel='',
                 add_to_draw: callable | tuple[callable, ...] = None):
        self.root = root
        self._xlims = xlim
        self._ylims = ylim
        self._add_to_draw = add_to_draw if add_to_draw is not None else ()

        self._figure = matplotlib.figure.Figure(figsize=figsize, dpi=100, tight_layout=True)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

        if self._ylims is None:
            self._ax.set_ylim(0, 1)
        else:
            self._ax.set_ylim(*self._ylims)

        if self._xlims is None:
            self._ax.set_xlim(0, 1)
        else:
            self._ax.set_ylim(*self._xlims)

        self._lines = []

        self.canvas = FigureCanvasTkAgg(self._figure, master=self.root)
        self.canvas.get_tk_widget().place(x=loc[0], y=loc[1])

    def draw(self):
        for func in self._add_to_draw:
            func()
        self.update_limits()
        self.canvas.draw()

    def remove_line(self, index):
        self._lines.pop(index).remove()

    def clear(self):
        for line in self._lines:
            line.remove()
        self._lines = []

    def plot_line(self, x, y):
        self._plot_line(x, y, self._lines)

    def _plot_line(self, x, y, lines: list, **kwargs):
        lines.append(self._ax.plot(x, y, **kwargs)[0])

    def plot_lines(self, xs, ys):
        self._plot_lines(xs, ys, self._lines)

    def _plot_lines(self, xs, ys, lines: list):
        if len(xs) != len(ys):
            raise ValueError('`xs` and `ys` have a different length')
        for x, y in zip(xs, ys):
            self._plot_line(x, y, lines)

    def update_limits(self):
        if (self._xlims is not None) and (self._ylims is not None):
            return

        if len(self._lines) == 0:
            x_min, x_max = 0, 1
            y_min, y_max = 0, 1
        else:
            x_data = [line.get_xdata() for line in self._lines]
            x_min = min(map(min, x_data))
            x_max = max(map(max, x_data))

            y_data = [line.get_ydata() for line in self._lines]
            y_min = min(map(min, y_data))
            y_max = max(map(max, y_data))

        if self._ylims is None:
            dy = y_max - y_min
            if dy == 0:
                y_max = 10
            self._ax.set_ylim(y_min-0.05*dy, y_max+0.05*dy)

        if self._xlims is None:
            self._ax.set_xlim(x_min, x_max)

    def change_line(self, index, x, y):
        """Change the data of the line at index `index`"""
        self._lines[index].set_xdata(x)
        self._lines[index].set_ydata(y)
        self.draw()

    def add_to_line(self, index, x, y):
        if isinstance(x, Sequence) != isinstance(y, Sequence):
            raise ValueError('Either `x` and `y` should be Sequences or both should be single values')
        if isinstance(x, Sequence):
            if len(x) != len(y):
                raise ValueError('`x` and `y` should have the same length')

        self._lines[index].set_xdata(np.append(self._lines[index].get_xdata(), x))
        self._lines[index].set_ydata(np.append(self._lines[index].get_ydata(), y))
        self.draw()
