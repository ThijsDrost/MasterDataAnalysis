from typing import Iterable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib


class LineErrorPlot:
    def __init__(self, root, figsize, loc, n_lines, ylim=None, labels: Iterable[str] = None):
        self.root = root
        self._ylims = ylim

        self._figure = matplotlib.figure.Figure(figsize=figsize, dpi=100, tight_layout=True)
        self._ax = self._figure.add_subplot(111)
        self._lines = [self._ax.errorbar([], [], yerr=[]) for _ in range(n_lines)]
        if labels is not None:
            self._ax.legend(self._lines, labels, loc='upper right')
        self._ax.set_ylabel('Concentration [mmol]')

        self.canvas = FigureCanvasTkAgg(self._figure, master=self.root)
        self.canvas.get_tk_widget().place(x=loc[0], y=loc[1])

        self._xdatas = [[] for _ in range(n_lines)]
        self._ydatas = [[] for _ in range(n_lines)]
        self._y_errors = [[] for _ in range(n_lines)]

    def plot(self, xs, ys, y_errs):
        for index, (x, y, y_err) in enumerate(zip(xs, ys, y_errs, strict=True)):
            self._xdatas[index].append(x)
            self._ydatas[index].append(y)
            self._y_errors[index].append(y_err)
            self._lines[index].remove()
            self._lines[index] = self._ax.errorbar(self._xdatas[index], self._ydatas[index], yerr=self._y_errors[index],
                                                   capsize=2, fmt=f'C{index}')

    def draw(self):
        self.update_limits()
        self.canvas.draw()

    def update_limits(self):
        if 0 in [len(x) for x in self._xdatas]:
            return
        x_min, x_max = min(map(min, self._xdatas)), max(map(max, self._xdatas))
        dx = x_max - x_min if (x_max - x_min) > 0 else 1
        self._ax.set_xlim(x_min-0.05*dx, x_max+0.05*dx)

        if (len(self._lines[0][0].get_ydata()) > 0) and (self._ylims is None):
            y_min_error = min((line[1][0].get_bbox().y0 for line in self._lines))
            y_max_error = max((line[1][1].get_bbox().y1 for line in self._lines))
            dy_error = y_max_error - y_min_error

            y_min, y_max = min(map(min, self._ydatas)), max(map(max, self._ydatas))
            dy = y_max - y_min

            if dy_error < 2*dy:
                y_min, y_max, dy = y_min_error, y_max_error, dy_error
            dy = dy if dy > 0 else 1

            self._ax.set_ylim(y_min - 0.05*dy, y_max + 0.05*dy)
