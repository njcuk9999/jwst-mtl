import sys

import numpy as np
import scipy.constants as sc_cst
import SOSS.trace.tracepol as tp
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

def photon_energy(wl):
    """
    wl: Wavelength in microns
    return: Photon energy in J
    """
    h = sc_cst.Planck
    c = sc_cst.speed_of_light
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

def f_lambda(pixels, trace_im, wl, y_trace, radius_pixel=30, area=25., gain=1.6):
    """
    pixels: Array of pixels
    trace_im: Trace's image [adu/s]
    wl: Array of wavelengths (same size as pixels)  [microns]
    y_trace: Array for the positions of the center of the trace for each column
    radius_pixel: Radius of extraction box [pixels]
    area: Area of photons collection surface [m²]. Default is 25.
    gain: Gain [e⁻/adu]. Default is 1.6.
    return: Extracted flux [J/s/m²/micron]
    """
    flux = np.zeros_like(pixels, dtype=float)  # Array for extracted spectrum
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        flux[x_i] = trace_im[int(first), x_i] * (1 - first % int(first)) + np.sum(
            trace_im[int(first) + 1:int(last) + 1, x_i]) + trace_im[int(last) + 1, x_i] * (last % int(last))

    # Calculate the flux in J/s/m²/um
    phot_ener = photon_energy(wl)  # Energy of each photon [J/photon]
    dw = dispersion(wl)   #Dispersion [microns]

    return flux * gain * phot_ener / area / dw

def flambda_adu(pixels, trace_im, y_trace, radius_pixel=30):
    """
    pixels: Array of pixels
    trace_im: Trace's image [adu/s]
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
        flux[x_i] = trace_im[int(first), x_i] * (1 - first % int(first)) + np.sum(
            trace_im[int(first) + 1:int(last) + 1, x_i]) + trace_im[int(last) + 1, x_i] * (last % int(last))

    return flux

def flambda_elec(pixels, trace_im, y_trace, radius_pixel=30, gain=1.6, ng=3, t_read=5.49):
    """
    pixels: Array of pixels
    trace_im: Trace's image [adu/s]
    y_trace: Array for the positions of the center of the trace for each column
    radius_pixel: Radius of extraction box. Default is 30. [pixels]
    gain: Gain [e⁻/adu]. Default is 1.6.
    ng: NGROUP
    t_read: Reading time (s)
    return: Extracted flux [e⁻/s/colonne]
    """
    tint = (ng - 1) * t_read  # Integration time [s]

    return flambda_adu(pixels, trace_im, y_trace, radius_pixel=radius_pixel) * gain * tint

def flambda_inf_radi_adu(trace_im):
    """
    trace_im: Trace's image [adu/s]
    return: Extracted flux for infinite radius [J/s/m²/um]
    """
    flux = np.sum(trace_im,axis=0)
    return flux

def flambda_inf_radi_ener(trace_im, wl, area=25., gain=1.6):
    """
    trace_im: Trace's image [adu/s]
    wl: Array of wavelengths (same size as pixels)  [um]
    area: Area of photons collection surface [m²]. Default is 25.
    gain: Gain [e⁻/adu]. Default is 1.6.
    return: Extracted flux for infinite radius [J/s/m²/um]
    """
    flux = flambda_inf_radi_adu(trace_im)

    # Calculate the flux in J/s/m²/um
    phot_ener = photon_energy(wl)  # Energy of each photon [J/photon]
    dw = dispersion(wl)   #Dispersion [um]

    return flux * gain * phot_ener / area / dw

def sigma_flambda(pixels, error, wl, y_trace, radius_pixel=30, area=25., gain=1.6):
    """
    pixels: Array of pixels
    variance: Variance of pixels [adu/s]
    wl: Array of wavelengths (same size as pixels)  [microns]
    y_trace: Array for the positions of the center of the trace for each column
    radius_pixel: Radius of extraction box [pixels]
    area: Area of photons collection surface [m²]. Default is 25.
    gain: Gain [e⁻/adu]. Default is 1.6.
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
    x, y, w = x[ind], y[ind], w[ind]
    wavelength = np.interp(x_index, x, w)
    y_index = np.interp(x_index, x, y)

    return x_index, y_index, wavelength

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

def make_comb(wave, peak_spacing, peak_width):
    """
    wave: Wavelength array
    peak_spacing: The wavelength spacing between peaks
    peak_width: The spectral width of each peak
    """
    wave_range = wave[-1] - wave[0]
    n_peaks = wave_range / peak_spacing
    peaks = wave[0] + peak_spacing/4 + np.arange(n_peaks) * peak_spacing
    comb = np.zeros_like(wave, dtype=float)
    sigma = peak_width / 2.35
    for p in peaks:
        comb += np.exp(-(wave-p)**2 / 2 / sigma**2)
    return comb, peaks

def create_wave(R, w_min, w_max):
    """
    R: resolving power
    w_min: Minimum wavelength value [um]
    w_max: Maximum wavelength value [um]
    return: Builds a wavelength array with constant resolving power R [um]
    """
    wave = [w_min]
    while wave[-1] < w_max:
        wave.append(wave[-1] + wave[-1]/R)
    return np.array(wave)

def robust_polyfit(fit_resFunc, x, y, p0):
    res = least_squares(fit_resFunc, p0, loss='linear', f_scale=0.1, args=(x, y))   #
    return res.x

def normalize_map(map):
    sum_col = np.sum(map, axis=0)
    norm_map = map / sum_col
    return norm_map