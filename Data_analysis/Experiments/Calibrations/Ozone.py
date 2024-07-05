
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import lmfit

image = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Ozone.jpg'
img = cv.imread(image)

# %%
plt.imshow(img[:, :, ::-1])
plt.show()

# %%
plt.imshow(img[635:645, 165:190, ::-1])
plt.show()

x_axis = (640, 641)
y_axis = (179, 180)

x_axis2 = 59
y_axis2 = 1164

axis = img.copy()
axis[x_axis[0]:x_axis[1]+1, :, :] = [255, 0, 0]
axis[:, y_axis[0]:y_axis[1]+1, :] = [255, 0, 0]
axis[x_axis2, :, :] = [0, 255, 0]
axis[:, y_axis2, :] = [0, 255, 0]

plt.imshow(axis[:, :, ::-1])
plt.show()

x_axis = ((0, 1), (640.5, 59))
y_axis = ((220, 320), (179.5, 1164))

# %%
temp_image = img.copy()
temp_image[120:135, 905:920] = [0, 0, 0]

plt.imshow(img[121:135, 905:919, ::-1])
plt.show()

plt.imshow(img[250:263, 370:382, ::-1])
plt.show()

plt.imshow(img[415:428, 814:826, ::-1])
plt.show()

result = cv.matchTemplate(temp_image, img[250:263, 370:382], cv.TM_SQDIFF_NORMED)
result2 = cv.matchTemplate(temp_image, img[415:428, 814:826], cv.TM_SQDIFF_NORMED)
# %%
plt.figure()
plt.imshow(result < 0.1)
plt.show()

plt.figure()
plt.imshow(result2 < 0.1)
plt.show()

plt.figure()
plt.imshow((result+result2) < 0.25)
plt.show()

# %%
model = lmfit.models.PolynomialModel(7)
location = result2.copy() + result.copy()
location[location >= 0.25] = 0
x = np.arange(location.shape[0])
avg_index = np.array([np.sum(location[:, i]*x)/np.sum(location[:, i]) if np.sum(location[:, i]) != 0 else 0 for i in range(location.shape[1])])

mask = avg_index != 0
pixel, avg_index = np.arange(len(avg_index))[mask], avg_index[mask]

a_wav = (y_axis[0][1] - y_axis[0][0])/(y_axis[1][1] - y_axis[1][0])
b_wav = y_axis[0][0] - a_wav*y_axis[1][0]
wavelength = a_wav*pixel + b_wav

a_inten = (x_axis[0][1] - x_axis[0][0])/(x_axis[1][1] - x_axis[1][0])
b_inten = x_axis[0][0] - a_inten*x_axis[1][0]
intensity = a_inten*avg_index + b_inten

plt.figure()
plt.scatter(wavelength, intensity)
plt.show()

params = model.guess(intensity, x=wavelength)
fit = model.fit(intensity, x=wavelength, params=params)
fit.plot()
plt.show()

mask = np.abs(intensity - fit.eval(x=wavelength)) < 0.05
wavelength, intensity = wavelength[mask], intensity[mask]

plt.figure()
plt.scatter(wavelength, intensity)
plt.show()

model = lmfit.models.SkewedVoigtModel()
params = model.guess(intensity, x=wavelength)
fit = model.fit(intensity, x=wavelength, params=params)
fit.plot()
plt.show()

# %%
fit_wavs = np.linspace(220, 320, 250)
fit_intensity = fit.eval(x=fit_wavs)

a_wav = (y_axis[1][1] - y_axis[1][0])/(y_axis[0][1] - y_axis[0][0])
b_wav = y_axis[1][0] - a_wav*y_axis[0][0]
pixel_loc = a_wav*wavelength + b_wav

a_wav = (y_axis[1][1] - y_axis[1][0])/(y_axis[0][1] - y_axis[0][0])
b_wav = y_axis[1][0] - a_wav*y_axis[0][0]
pixel_fit_loc = a_wav*fit_wavs + b_wav

a_inten = (x_axis[1][1] - x_axis[1][0])/(x_axis[0][1] - x_axis[0][0])
b_inten = x_axis[1][0] - a_inten*x_axis[0][0]
pixel_inten = a_inten*intensity + b_inten

a_inten = (x_axis[1][1] - x_axis[1][0])/(x_axis[0][1] - x_axis[0][0])
b_inten = x_axis[1][0] - a_inten*x_axis[0][0]
pixel_fit_inten = a_inten*fit_intensity + b_inten

Hart_wavelength = np.array([190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300])
Hart_intensity = np.array([931, 685, 689, 842, 1292, 2067, 2961, 3300, 2738, 1695, 764, 287])

Hart_wpixels = a_wav*Hart_wavelength + b_wav
Hart_ipixels = a_inten*(Hart_intensity/Hart_intensity.max()) + b_inten

plt.figure()
plt.imshow(axis[:, :, ::-1])
plt.scatter(pixel_loc, pixel_inten, c='r', s=1)
plt.plot(pixel_fit_loc, pixel_fit_inten)
plt.plot(Hart_wpixels, Hart_ipixels, 'o')
plt.show()

# %%
wavelenghts = np.linspace(180, 500, 1000)
intensities = fit.eval(x=wavelenghts)
values = np.array([wavelenghts, intensities]).T
np.savetxt(r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis\Experiments\Calibrations\Models\Ozone.txt', values)
