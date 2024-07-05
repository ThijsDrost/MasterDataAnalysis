import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class SpectraPlot:
    def __init__(self, root, figsize, loc, *, colors=None, graph_number=10, xlim=None, ylim=None, xlabel='', ylabel=''):
        self.root = root
        self._xlims = xlim
        self._ylims = ylim

        if colors is not None:
            if len(colors) != graph_number:
                raise ValueError(f'`colors` should have length `show_graphs` ({graph_number}), not {len(colors)}')
            self._colors = colors
        else:
            self._colors = plt.get_cmap('jet')(np.linspace(0, 1, graph_number))

        self._graph_number = graph_number

        self._figure = matplotlib.figure.Figure(figsize=figsize, dpi=100, tight_layout=True)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

        if self._ylims is None:
            self._ax.set_ylim(0, 70_000)
        else:
            self._ax.set_ylim(*self._ylims)

        if self._xlims is None:
            self._ax.set_xlim(200, 700)
        else:
            self._ax.set_ylim(*self._xlims)

        self._lines = []
        self._lines.append(self._ax.plot([], [], c=self._colors[0])[0])

        self.canvas = FigureCanvasTkAgg(self._figure, master=self.root)
        self.canvas.get_tk_widget().place(x=loc[0], y=loc[1])

    def draw(self):
        self.update_colors()
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
        if len(lines) == self._graph_number:
            lines.pop(0).remove()
        lines.append(self._ax.plot(x, y, **kwargs)[0])

    def plot_lines(self, xs, ys):
        self._plot_lines(xs, ys, self._lines)

    def _plot_lines(self, xs, ys, lines: list):
        if len(xs) != len(ys):
            raise ValueError('`xs` and `ys` have a different length')
        for x, y in zip(xs, ys):
            self._plot_line(x, y, lines)

    def update_colors(self):
        for index, line in enumerate(self._lines):
            line.set(color=self._colors[index])

    def update_limits(self):
        # self._ax.relim()
        # self._ax.autoscale_view()
        #
        # if self._xlims is not None:
        #     self._ax.set_xlim(*self._xlims)
        # if self._ylims is not None:
        #     self._ax.set_ylim(*self._ylims)

        if (self._xlims is not None) and (self._ylims is not None):
            return

        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
        for line in self._ax.lines:
            x_data, y_data = line.get_data()
            if self._xlims is None:
                if len(x_data) == 0:
                    pass
                else:
                    x_min = min(min(x_data), x_min)
                    x_max = max(max(x_data), x_max)
            if self._ylims is None:
                if len(y_data) == 0:
                    pass
                else:
                    y_min = min(min(y_data), y_min)
                    y_max = max(max(y_data), y_max)

        x_min = x_min if x_min != np.inf else 0
        x_max = x_max if x_max != -np.inf else 1
        y_min = y_min if y_min != np.inf else 0
        y_max = y_max if y_max != -np.inf else 1

        if self._ylims is None:
            dy = y_max - y_min
            if dy == 0:
                y_max = 10
            self._ax.set_ylim(y_min-0.05*dy, y_max+0.05*dy)

        if self._xlims is None:
            self._ax.set_xlim(x_min, x_max)