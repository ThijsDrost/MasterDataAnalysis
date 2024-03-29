import tkinter as tk
from tkinter import filedialog

import numpy as np
import h5py

from General.Data_handling import SpectroData
from GUIs.Widgets.SpectraPlot import SpectraPlot
from GUIs.Widgets.AbsorptionFitPlot import AbsorptionFitPlot

root = tk.Tk()
root.title('Spectrometer')
root.geometry('1000x900')

# Create a frame with a button to import reference and dark spectra
reference_button = tk.Button(root, text='Import reference', command=lambda: import_data('reference'),
                             width=15, height=1)
reference_button.place(x=10, y=10)

dark_button = tk.Button(root, text='Import dark', command=lambda: import_data('dark'), width=15, height=1)
dark_button.place(x=10, y=50)

data_button = tk.Button(root, text='Import data', command=lambda: import_data('data'), width=15, height=1)
data_button.place(x=10, y=90)

add_button = tk.Button(root, text='Add data', command=lambda: add_data('add'), width=15, height=1, state='disabled')
add_button.place(x=550, y=450, anchor='center')

conversion_selector_values = ['Mean', 'Median', 'Max', 'Min']
conversion_selector_value = tk.StringVar(value=conversion_selector_values[0])
conversion_selector = tk.OptionMenu(root, conversion_selector_value, *conversion_selector_values)
conversion_selector.config(width=10, height=1)
conversion_selector.place(x=450, y=450, anchor='center')

# Create a frame with two spectra plots, equally spaced
plot1 = SpectraPlot(root, figsize=(6, 4), loc=(200, 0), graph_number=-1)
plot2 = SpectraPlot(root, figsize=(6, 4), loc=(200, 500), graph_number=-1)

references = []
darks = []
measurements = []


def import_data(data_type):
    global plot1

    file_paths = filedialog.askopenfilenames()
    if len(file_paths) == 0:
        return

    plot1.clear()
    for file_path in file_paths:
        data = SpectroData.read_data(file_path)
        plot1.plot_line(data.wavelength, data.intensity)


def absorbance(data: SpectroData, references: list[SpectroData] = None, dark: SpectroData = None):
    if darks is not None:
        dark = dark.intensity
    else:
        raise ValueError('Dark spectrum not taken')
    if references is not None:
        reference_weights = weights(data, references)
        reference = np.average([reference.intensity for reference in references], axis=0, weights=reference_weights)
    else:
        raise ValueError('Reference spectrum not taken')
    return -np.log10((data.intensity - dark)/(reference - dark))


def weights(data: SpectroData, values: list[SpectroData], method='closest'):
    if method not in ('closest', 'interpolate', 'before', 'after'):
        raise ValueError(f'`method` should be one of: closest, interpolate, before, after')
    measure_time = data.time_ms
    reference_times = [ref.time_ms for ref in values]
    weights = [0 for _ in values]
    if method == 'closest':
        index = np.argmin(np.abs(np.array(reference_times) - measure_time))
        weights[index] = 1
    elif method == 'interpolate':
        index = np.argmax(np.array(reference_times) < measure_time)
        weights[index] = (reference_times[index+1] - measure_time) / (reference_times[index+1] - reference_times[index])
        weights[index+1] = 1 - weights[index]
    elif method == 'before':
        index = np.argmax(np.array(reference_times) < measure_time)
        weights[index] = 1
    elif method == 'after':
        index = np.argmax(np.array(reference_times) > measure_time)
        weights[index] = 1
    return weights


root.mainloop()
