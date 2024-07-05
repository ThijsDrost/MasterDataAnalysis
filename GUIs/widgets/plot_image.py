from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

from General.plotting.plot import set_defaults
from General.plotting.limit import Limit


class PlotImage:
    def __init__(self, parent, figsize, loc, image, *, fig_kwargs, zoomable=False, zoom_factor=2,
                 mpl_connect: dict[str, callable] = None):
        self.root = parent

        fig_kwargs = set_defaults(fig_kwargs, figsize=figsize, dpi=100)

        self._figure = matplotlib.figure.Figure(**fig_kwargs)
        # self._ax = self._figure.add_subplot(111)
        self._ax = plt.Axes(self._figure, [0., 0., 1., 1.])
        self._ax.set_axis_off()
        self._figure.add_axes(self._ax)

        self._ax.imshow(image)

        self._xlim = Limit(self._ax.get_xlim())
        self._ylim = Limit(self._ax.get_ylim(), inverted=True)

        self.canvas = FigureCanvasTkAgg(self._figure, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.place(x=loc[0], y=loc[1])

        if mpl_connect is not None:
            for event, func in mpl_connect.items():
                self.canvas.mpl_connect(event, func)

        if zoomable:
            self.canvas.mpl_connect("scroll_event", self._mouse_wheel)
            self.zoom_factor = zoom_factor
            self.zoom = 1

    def plot_on(self, collection):
        self._ax.add_collection(collection)
        self.draw()

    def plot_line(self, collection: plt.Line2D):
        self._ax.add_line(collection)
        self.draw()

    def draw(self):
        self.update()
        self.canvas.draw()

    def update(self):
        self._ax.set_xlim(*self._xlim.limit)
        self._ax.set_ylim(*self._ylim.limit)

    def _set_limits(self, xlim, ylim):
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)

    def _mouse_wheel(self, event: MouseEvent):
        zoom_factor = self.zoom_factor if event.button == 'down' else 1/self.zoom_factor
        self._xlim.zoom(event.xdata, zoom_factor)
        self._ylim.zoom(event.ydata, zoom_factor)
        self.draw()
