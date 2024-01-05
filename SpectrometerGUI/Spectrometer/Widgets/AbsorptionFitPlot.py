import lmfit
import numpy as np

from SpectrometerGUI.Spectrometer.Widgets.SpectraPlot import SpectraPlot
from SpectrometerGUI.Spectrometer.Widgets.LineErrorPlot import LineErrorPlot


class AbsorptionFitPlot(SpectraPlot):
    def __init__(self, root, figsizes, locs, model: lmfit.CompositeModel, reference_intensity, **kwargs):
        super().__init__(root, figsizes[0], locs[0], **kwargs)
        self.n_lines = len(model.components)
        labels = [m.prefix for m in model.components]
        self._error_plot = LineErrorPlot(root, figsizes[1], locs[1], self.n_lines, labels=labels)
        self._fit_lines = []
        self._model = model
        self._params = model.make_params()
        self._results = []
        self._ref_inten = reference_intensity
        self._measurement_num = 0

    def plot_line(self, x, y):
        absorbance = -np.log(y / self._ref_inten)
        super().plot_line(x, absorbance)
        fit_result = self._model.fit(absorbance, params=self._params, wav=x)
        # print(fit_result.fit_report())

        no2 = fit_result.params[('NO2conc')]
        no3 = fit_result.params['NO3conc']
        h2o2 = fit_result.params['H2O2conc']
        print(f'NO2: {no2.value:.2f} pm {no2.stderr:.1e}, NO3: {no3.value:.2f} pm {no3.stderr:.1e},'
              f' H2O2: {h2o2.value:.2f} pm {h2o2.stderr:.1e}')

        super()._plot_line(x, fit_result.best_fit, self._fit_lines, linestyle='--')
        values = []
        errors = []
        for prefix in (m.prefix for m in self._model.components):
            values.append(fit_result.params[f'{prefix}conc'].value)
            errors.append(fit_result.params[f'{prefix}conc'].stderr)

        # errors = [(e if e is not None else 0) for e in errors]
        self._error_plot.plot([self._measurement_num]*self.n_lines, values, errors)
        self._measurement_num += 1

    def plot_lines(self, xs, ys):
        for x, y in zip(xs, ys, strict=True):
            self.plot_line(x, y)

    def update_colors(self):
        super().update_colors()
        for index, line in enumerate(self._fit_lines):
            line.set(color=self._colors[index])

    def draw(self):
        super().draw()
        self._error_plot.draw()

