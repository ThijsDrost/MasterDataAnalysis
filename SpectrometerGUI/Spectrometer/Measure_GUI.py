import os
import tkinter as tk
import tkinter.messagebox as tk_messagebox

import numpy as np
import h5py

from General.Data_handling.Data_import import SpectroData
from SpectrometerGUI.Spectrometer.Widgets.SpectraPlot import SpectraPlot
from SpectrometerGUI.Spectrometer.Widgets.AbsorptionFitPlot import AbsorptionFitPlot
from General.Analysis.Models import multi_species_model


class MeasureGUI:
    def __init__(self, data_loc, hdf5_loc, dataset_loc, mode, *, update_interval_ms=100, size=(1200, 800), plot_every=1,
                 kwargs_plot1: dict = None, kwargs_plot2: dict = None, calibration_loc: tuple[str, str] = None):
        if kwargs_plot2 is None:
            kwargs_plot2 = {}
        if kwargs_plot1 is None:
            kwargs_plot1 = {}
        self._update_interval_ms = update_interval_ms
        self._spectra = []
        self.data_loc = data_loc
        self._hdf5_loc = hdf5_loc
        self._dataset_loc = dataset_loc
        self._dataset_counter = 0
        self._plot_every = plot_every
        self._plot_counter = 0

        self._mode = mode
        allowed_modes = ['scope', 'absorption']
        if mode not in allowed_modes:
            raise ValueError(f'`mode` should be one of: {allowed_modes}')

        with h5py.File(hdf5_loc, 'r') as file:
            data_set = file[dataset_loc]['uv-vis']
            self._background = data_set.attrs.get('background', None)
            if self._background is None:
                raise ValueError('`background` was not found as attribute for uv-vis in the database')
            integration_ms = data_set.attrs.get('integration_ms', None)
            if integration_ms is None:
                raise ValueError('`integration_ms` was not found as attribute for uv-vis in the database')
            serial_number = data_set.attrs.get('serial_number', None)
            if serial_number is None:
                raise ValueError('`serial_number` was not found as attribute for uv-vis in the database')
            n_smoothing = data_set.attrs.get('n_smoothing', None)
            if n_smoothing is None:
                raise ValueError('`n_smoothing` was not found as attribute for uv-vis in the database')
            wavelengths = data_set.attrs.get('wavelength', None)
            if wavelengths is None:
                raise ValueError('`wavelength` was not found as attribute for uv-vis in the database')

            if self._mode == 'absorption':
                self._reference = data_set.attrs.get('reference', None)
                if self._reference is None:
                    raise ValueError('`reference` was not found as attribute for uv-vis in the database')
            else:
                self._reference = None

            self._equality_test_spectrodata = SpectroData(wavelengths, np.zeros(1), serial_number, 2048, integration_ms, 1,
                                                          n_smoothing, 0)

        self.root = tk.Tk()
        self.root.title('Spectra analysis')
        self.root.geometry(f'{size[0]}x{size[1]}')

        self._spectra_plot = SpectraPlot(self.root, (size[0] / (2.22 * 100), size[1] / (2.22 * 100)),
                                         (0.033*size[0], 0.033*size[1]), **kwargs_plot1)

        if self._mode == 'absorption':
            if calibration_loc is None:
                raise ValueError('For absorption mode, `calibration_loc` should be given')
            lmfit_model = multi_species_model(*calibration_loc)

            sizes = ((size[0] / (2.22 * 100), size[1] / (2.22 * 100)), (size[0] / (2.22 * 100), size[1] / (2.22 * 100)))
            locs = ((0.033*size[0], 0.566*size[1]), (0.566*size[0], 0.566*size[1]))
            self._absorption_plots = AbsorptionFitPlot(self.root, sizes, locs, lmfit_model, self._reference,
                                                       **kwargs_plot2)

        self.root.after(self._update_interval_ms, self.update)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self._spectra_plot.draw()
        self.root.mainloop()

    def close(self):
        if tk_messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()

    def save_spectrum(self, spectrum):
        with h5py.File(self._hdf5_loc, 'r+') as file:
            uv_vis = file[self._dataset_loc]['uv-vis']
            dataset = uv_vis.create_dataset(str(self._dataset_counter), data=spectrum.intensity)
            dataset.attrs.create('time_ms', spectrum.time_ms)
            self._dataset_counter += 1

    def save_spectra(self, spectra):
        with h5py.File(self._hdf5_loc, 'r+') as file:
            for spectrum in sorted(spectra, key=lambda x: x.time_ms):
                uv_vis = file[self._dataset_loc]['uv-vis']
                dataset = uv_vis.create_dataset(str(self._dataset_counter), data=spectrum.intensity)
                dataset.attrs.create('time_ms', spectrum.time_ms)
                self._dataset_counter += 1

    # @staticmethod
    # def read_spectrum(loc):
    #     if loc.lower().endswith('raw8'):
    #         return SpectroData.read_raw8(loc)
    #     elif loc.endswith('txt'):
    #         return read_txt(loc)

    def import_spectra(self):
        spectra = []
        with os.scandir(self.data_loc) as it:
            for i in it:
                if (not i.path.endswith('txt')) and (not i.path.lower().endswith('raw8')):
                    continue
                try:
                    # spectrum = self.read_spectrum(i.path)
                    spectrum = SpectroData.read_data(i.path)
                except PermissionError:
                    continue

                if spectrum is None:
                    continue
                self._equality_test_spectrodata.same_measurement(spectrum)
                self.save_spectrum(spectrum)

                spectra.append(spectrum)
                os.remove(i.path)

        y_values = sorted(spectra, key=lambda x: x.time_ms)
        x_values = [list(self._equality_test_spectrodata.wavelength)] * len(y_values)
        y_values = [y.intensity-self._background for y in y_values]
        for x, y in zip(x_values, y_values):
            if self._plot_counter % self._plot_every == 0:
                self._spectra_plot.plot_line(x, y)
            self._plot_counter += 1
        self._absorption_plots.plot_lines(x_values, y_values)

    def update(self):
        self.import_spectra()
        self._spectra_plot.draw()
        if self._mode == 'absorption':
            self._absorption_plots.draw()

        self.root.after(self._update_interval_ms, self.update)


def create_background(database_loc, measurement, background_loc):
    with h5py.File(database_loc, 'r+') as database:
        measurement = database.create_group(measurement)
        uv_vis = measurement.create_group('uv-vis')

        background = SpectroData.read_data(background_loc)
        uv_vis.attrs.create('background', background.intensity)
        uv_vis.attrs.create('n_background', background.n_averages)
        uv_vis.attrs.create('wavelength', background.wavelength)
        uv_vis.attrs.create('serial_number', background.serial_number)
        uv_vis.attrs.create('n_smoothing', background.n_smoothing)
        uv_vis.attrs.create('integration_ms', background.integration_time_ms)


def create_background_reference(database_loc, measurement, background_loc, reference_loc):
    background = SpectroData.read_data(background_loc)
    create_background(database_loc, measurement, background_loc)
    with h5py.File(database_loc, 'r+') as database:
        uv_vis = database[measurement]['uv-vis']
        reference = SpectroData.read_data(reference_loc)
        if not reference.same_measurement(background):
            raise ValueError('Reference and background spectrum do not interoperate')
        uv_vis.attrs.create('reference', reference.intensity)
        uv_vis.attrs.create('n_reference', reference.n_averages)
