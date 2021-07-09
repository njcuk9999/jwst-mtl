import sys

import numpy as np
import scipy.constants as sc_cst
import SOSS.trace.tracepol as tp
from scipy.interpolate import interp1d

# Constants
h = sc_cst.Planck
c = sc_cst.speed_of_light
gain = 1.6
area = 25.

def photon_energy(wl):
    """
    wl: Wavelength in microns
    return: Photon energy in J
    """
    return h * c / (wl * 1e-6)

def dispersion(wl):
    """
    wl: Wavelengths array [microns]
    return: Dispersion [microns]
    """
    dw = np.zeros_like(wl, dtype=float)
    for i in range(len(wl)):
        if i == 0:
            dw[i] = wl[0] - wl[1]   #The wl array has to be reversed
        else:
            dw[i] = wl[i - 1] - wl[i]
    return dw

def f_lambda(pixels, im_test, wl, y_trace, radius_pixel=30, area=area, gain=gain):
    """
    pixels: Array of pixels
    im_test: Trace's image [adu/s]
    wl: Array of wavelengths (same size as pixels)  [microns]
    y_trace: Array for the positions of the center of the trace for each column
    radius_pixel: Radius of extraction box [pixels]
    area: Area of photons collection surface [m²]
    gain: Gain [e⁻/adu]
    return: Extracted flux [J/s/m²/micron]
    """
    flux = np.zeros_like(pixels, dtype=float)  # Array for extracted spectrum
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        flux[x_i] = im_test[int(first), x_i] * (1 - first % int(first)) + np.sum(
            im_test[int(first) + 1:int(last) + 1, x_i]) + im_test[int(last) + 1, x_i] * (last % int(last))

    # Calculate the flux in J/s/m²/um
    phot_ener = photon_energy(wl)  # Energy of each photon [J/photon]
    dw = dispersion(wl)   #Dispersion [microns]

    return flux * gain * phot_ener / area / dw

def flambda_adu(pixels, im_test, y_trace, radius_pixel=30):
    """
    pixels: Array of pixels
    im_test: Trace's image [adu/s]
    y_trace: Array for the positions of the center of the trace for each column
    radius_pixel: Radius of extraction box. Default is 30. [pixels]
    return: Extracted flux [adu/s/colonne]
    """
    flux = np.zeros_like(pixels, dtype=float)  # Array for extracted spectrum
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        flux[x_i] = im_test[int(first), x_i] * (1 - first % int(first)) + np.sum(
            im_test[int(first) + 1:int(last) + 1, x_i]) + im_test[int(last) + 1, x_i] * (last % int(last))

    return flux

def flambda_elec(pixels, im_test, y_trace, radius_pixel=30, gain=gain, ng=3, t_read=5.49):
    """
    pixels: Array of pixels
    im_test: Trace's image [adu/s]
    y_trace: Array for the positions of the center of the trace for each column
    radius_pixel: Radius of extraction box. Default is 30. [pixels]
    gain: Gain [e⁻/adu]
    ng: NGROUP
    t_read: Reading time (s)
    return: Extracted flux [e⁻/s/colonne]
    """
    tint = (ng - 1) * t_read  # Integration time [s]

    return flambda_adu(pixels, im_test, y_trace, radius_pixel=radius_pixel) * gain * tint

def flambda_inf_radi_adu(im_test):
    """
    im_test: Trace's image [adu/s]
    return: Extracted flux for infinite radius [J/s/m²/um]
    """
    flux = np.sum(im_test,axis=0)
    return flux

def flambda_inf_radi_ener(im_test, wl, area=area, gain=gain):
    """
    im_test: Trace's image [adu/s]
    wl: Array of wavelengths (same size as pixels)  [um]
    area: Area of photons collection surface [m²]
    gain: Gain [e⁻/adu]
    return: Extracted flux for infinite radius [J/s/m²/um]
    """
    flux = flambda_inf_radi_adu(im_test)

    # Calculate the flux in J/s/m²/um
    phot_ener = photon_energy(wl)  # Energy of each photon [J/photon]
    dw = dispersion(wl)   #Dispersion [um]

    return flux * gain * phot_ener / area / dw

def sigma_flambda(pixels, error, wl, y_trace, radius_pixel=30, area=area, gain=gain):
    """
    pixels: Array of pixels
    variance: Variance of pixels [adu/s]
    wl: Array of wavelengths (same size as pixels)  [microns]
    y_trace: Array for the positions of the center of the trace for each column
    radius_pixel: Radius of extraction box [pixels]
    area: Area of photons collection surface [m²]
    gain: Gain [e⁻/adu]
    return: Sigma of extracted flux [J/s/m²/micron]
    """
    variance = error ** 2  # Variance of each pixel [adu²/s²]

    vari = np.zeros_like(pixels, dtype=float)  # Array for variances
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        vari[x_i] = variance[int(first), x_i] * (1 - first % int(first)) + np.sum(
            variance[int(first) + 1:int(last) + 1, x_i]) + variance[int(last) + 1, x_i] * (last % int(last))

    # Calculate the total error in J/s/m²/um
    delta_flambda = np.sqrt(vari)   #Sigma of column [adu/s]

    phot_ener = photon_energy(wl)  # Energy of each photon [J/photon]
    dw = dispersion(wl)  # Dispersion [microns]

    return delta_flambda * gain * phot_ener / area / dw

def readtrace(os):  # From Loic
    trace_filename = '/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv'
    pars = tp.get_tracepars(trace_filename, disable_rotation=False)
    w = np.linspace(0.7, 3.0, 10000)
    x, y, mask = tp.wavelength_to_pix(w, pars, m=1, oversample=os, subarray='SUBSTRIP256')
    x_index = np.arange(2048 * os)
    # np.interp needs ordered x
    ind = np.argsort(x)
    x, w = x[ind], w[ind]
    wavelength = np.interp(x_index, x, w)
    y_index = np.interp(x_index, x, y)

    return x_index, y_index[::-1], wavelength

def wl_filter(wl, pixels, length=85):
    """
    wl: Wavelengths array
    pixels: Pixels array (same size as wl)
    length: Length of window for which to compute median or mean
    return: New wavelengths array matching filtered flux
    """
    if length%2==1:
        new_w = wl[length//2:-length//2]
    elif length%2==0:
        pixels_new = pixels - 0.5
        pixels_new[0] = 0
        f = interp1d(pixels,wl)
        w_me = f(pixels_new)
        new_w = w_me[length//2:-length//2]
    return new_w

def median_window(flux, start, length):
    """
    flux: Extracted flux
    start: Start point of window for which to compute median
    length: Length of window for which to compute median
    return: Median
    """
    list = flux[start : start + length]
    return np.median(list)

def median_filter(flux, length=85):
    """
    flux: Extracted flux
    length: Length of window for which to compute median
    return: Extracted flux with median filter applied
    """
    m = []
    start = 0
    while start + length < len(flux):
        m.append(median_window(flux, start, length))
        start += 1
    return m

def mean_window(flux, start, length):
    """
    flux: Extracted flux
    start: Start point of window for which to compute mean
    length: Length of window for which to compute mean
    return: Mean
    """
    list = flux[start : start + length]
    return np.mean(list)

def mean_filter(flux, length=50):
    """
    flux: Extracted flux
    length: Length of window for which to compute mean
    return: Extracted flux with mean filter applied
    """
    m = []
    start = 0
    while start + length < len(flux):
        m.append(mean_window(flux, start, length))
        start += 1
    return m

def relative_difference(data, ref_data):
    """
    data: Data to be compared
    ref_data: Reference data with which to compare
    !data and ref_data must be the same size!
    return: Relative difference
    """
    return (data - ref_data) / ref_data