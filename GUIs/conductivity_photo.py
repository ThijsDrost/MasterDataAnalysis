import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.signal import find_peaks

from GUIs.widgets import PlotImage
from General.itertools import argmax, argmin
import General.numpy_funcs as npf
from General.checking import Validator


def _clicked(event):
    global image
    global click_locks
    global root

    line = plt.Line2D([event.xdata], [event.ydata], marker='o', color='r')
    image.plot_line(line)
    click_locks.append((event.xdata, event.ydata))
    if len(click_locks) == 4:
        root.destroy()

def _clicked2(event):
    global axis_locs

    line = plt.Line2D([event.xdata], [event.ydata], marker='o', color='r')
    image.plot_line(line)
    axis_locs.append((event.xdata, event.ydata))
    if len(axis_locs) == 4:
        root.destroy()


def _trans_matrix(points):
    x_sorted = sorted(points, key=lambda x: x[0])
    y_sorted = sorted(points, key=lambda x: x[1])

    left = x_sorted[:2]
    right = x_sorted[2:]
    top = y_sorted[:2]
    bottom = y_sorted[2:]

    top_left = [x for x in left if x in top][0]
    top_right = [x for x in right if x in top][0]
    bottom_left = [x for x in left if x in bottom][0]
    bottom_right = [x for x in right if x in bottom][0]

    points = [top_left, top_right, bottom_right, bottom_left]

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 500], [1000, 500], [1000, 0], [0, 0]])
    return cv.getPerspectiveTransform(pts1, pts2)


def _scale(points):
    x_sorted = sorted(points, key=lambda x: x[0])

    left = x_sorted[:2]
    right = x_sorted[2:]

    top_axis1 = max(left, key=lambda x: x[1])
    bottom_axis1 = min(left, key=lambda x: x[1])
    left_axis2 = min(right, key=lambda x: x[0])
    right_axis2 = max(right, key=lambda x: x[0])

    return [top_axis1, bottom_axis1, left_axis2, right_axis2]


def convert_image(loc, times, conductivities, rotate=False):
    global click_locks
    global root
    global image
    global axis_locs

    validator = Validator.tuple_of_number() + Validator.length(2)
    validator(times, 'times')
    validator(conductivities, 'conductivities')

    root = tk.Tk()
    root.title('Image plot')
    root.geometry('800x600')

    input_image = cv.imread(loc)
    if rotate is True:
        input_image = cv.rotate(input_image, cv.ROTATE_90_COUNTERCLOCKWISE)

    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)

    click_locks = []
    mpl_collect = {'button_press_event': _clicked}

    image = PlotImage(root, figsize=(8, 6), loc=(0, 0), image=input_image, fig_kwargs={}, zoomable=True, mpl_connect=mpl_collect)
    image.draw()
    root.mainloop()

    matrix = _trans_matrix(click_locks)
    new_image = cv.warpPerspective(input_image, matrix, (1000, 500))

    root = tk.Tk()
    root.title('Image plot')
    root.geometry('800x600')
    axis_locs = []
    mpl_collect = {'button_press_event': _clicked2}
    image = PlotImage(root, figsize=(8, 6), loc=(0, 0), image=input_image, fig_kwargs={}, zoomable=True, mpl_connect=mpl_collect)
    image.draw()
    root.mainloop()

    scale_points = _scale(axis_locs)
    scale_points = np.array([scale_points], dtype='float32')
    pers_scale_points = cv.perspectiveTransform(scale_points, matrix)

    column_fit = np.poly1d(np.polyfit([pers_scale_points[0][2][0], pers_scale_points[0][3][0]], sorted(times), 1))
    row_fit = np.poly1d(np.polyfit([pers_scale_points[0][0][1], pers_scale_points[0][1][1]], sorted(conductivities), 1))

    # column_intensity = np.average(new_image, axis=(0, 2))
    # column_intensity = 1 - column_intensity / column_intensity.max()
    # peaks_column = find_peaks(column_intensity, prominence=0.05)[0][:-1]
    #
    # cond_values = [max(conductivities), min(conductivities)]
    # cond_vals = [np.interp(i, [0, len(peaks_column)], cond_values) for i in range(len(peaks_column))]
    # column_fit = np.polynomial.polynomial.Polynomial.fit(peaks_column, cond_vals, 1)
    #
    # row_intensity = np.average(new_image, axis=(1, 2))
    # row_intensity = 1 - row_intensity / row_intensity.max()
    # peaks_row = find_peaks(row_intensity, prominence=0.05)[0][:-1]
    #
    # time_values = [min(times), max(times)]
    # time_vals = [np.interp(i, [0, len(peaks_row)], time_values) for i in range(len(peaks_row))]
    # row_fit = np.polynomial.polynomial.Polynomial.fit(peaks_row, time_vals, 1)

    avg = np.average(new_image, axis=2)[5:-5, 5:-5]
    avg = 1 - avg / avg.max(axis=0)

    max_val = 5 + np.argmax(avg, axis=0)
    max_val_filt = npf.median_interp_filter(max_val, 0.1)

    indexes = np.arange(len(max_val))

    extent = [column_fit(0), column_fit(1000), row_fit(0), row_fit(500)]

    plt.figure()
    plt.imshow(new_image[::-1], extent=extent, aspect='auto')
    plt.plot(column_fit(indexes)[5:-5], row_fit(max_val_filt)[5:-5], label='Max value')
    plt.ylabel('Conductivity')
    plt.xlabel('Time')
    plt.show()

    return column_fit(indexes)[5:-5], row_fit(max_val_filt)[5:-5]
