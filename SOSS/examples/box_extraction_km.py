# Example script to perform a box extraction
# on a timeseries
# By K. Morel

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
import scipy.constants as sc_cst
import trace.tracepol as tp   #SOSS.
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


##########################################################
###################### TO UPDATE #########################
##########################################################
# Input and output directory
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/PHY3030/WASP_52/'
# Simulations files names
simu_noiseless_filename = 'wasp_52_20211103/IDTSOSS_clear_noNoise_rateints.fits'      # Noiseless
simu_noisy_filename = 'wasp_52_20211118/IDTSOSS_clear_noisy_gainscalestep_loic.fits'  # Noisy

# Radius for box extraction
radius_pixel = 30   # [pixels]

# If plots are shown
doShow_plots = True

# If figures are saved
doSave_plots = False

# If data is saved
doSave_data = False

##########################################################
# Matplotlib defaults
plt.rc('figure', figsize=(10,6))
plt.rc('font', size=16)
plt.rc('image', cmap='inferno')
plt.rc('lines', lw=1)

##########################################################
###################### FUNCTIONS #########################
##########################################################
def rateints_dms_simulation(file_name):
    with fits.open(file_name) as hdulist:
        data_noisy_rateints = hdulist[1].data  # Images of flux [adu/s]
        # delta_noisy = hdulist[2].data            # Errors [adu/s]
        dq = hdulist[3].data  # Data quality
        i = np.where(dq % 2 != 0)  # Odd values of dq = DO NOT USE these pixels
        data_noisy_rateints[i[0], i[1], i[2]] = 0.
        # delta_noisy[i[0], i[1], i[2]] = 0.

        # Convert data from fits files to float (fits precision is 1e-8)
        data_noisy_rateints = data_noisy_rateints.astype('float64', copy=False)

        simu = hdulist
        data = data_noisy_rateints
    return simu, data

def readtrace(os):  # From Loic
    """
    Returns x and y order 1 trace position with corresponding wavelengths.
    os: Oversampling
    """
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

def normalization(f_lambda, t1, t4):
    """
    Normalize transit light curve by out of transit mean for all wavelengths.
    First dimension is time, second dimension is wavelengths.
    """
    out_transit = np.concatenate((f_lambda[:t1+1], f_lambda[t4:]))
    out_transit_mean = np.mean(out_transit, axis=0)
    return (f_lambda / out_transit_mean)

def relative_difference(data, ref_data):
    """
    data: Data to be compared
    ref_data: Reference data with which to compare
    !data and ref_data must be the same size!
    return: Relative difference
    """
    return ((data - ref_data) / ref_data)

def transit_depth(f_lambda, t1, t2, t3, t4):
    """
    f_lambda: Flux array.
    t1, t2, t3, t4: Characteristic transit times
    :return: Transit light curve.
    """
    # Mean flux value during transit for all wavelengths
    in_transit_mean = np.mean(f_lambda[t2: t3 + 1], axis=0)
    # Mean flux value during out of transit for all wavelengths
    out_transit = np.concatenate((f_lambda[: t1 + 1], f_lambda[t4:]))
    out_transit_mean = np.mean(out_transit, axis=0)
    return out_transit_mean - in_transit_mean  # in_transit_mean / out_transit_mean


##########################################################
################# EXTRACTION, NOISELESS ##################
##########################################################
# Load simulation
simu_noiseless, data_noiseless = rateints_dms_simulation(WORKING_DIR + simu_noiseless_filename)
simulation_noiseless = 'noiseless'

ng = simu_noiseless[0].header['NGROUPS']      # Number of groups
t_read = simu_noiseless[0].header['TGROUP']   # Reading time [s]
tint = (ng - 1) * t_read                      # Integration time [s]

# Characteristic times of transit
# HAS TO BE MODIFIED FOR EACH MODEL TESTED
t1, t2, t3, t4 = 53, 74, 110, 128   # [image]

# Position of trace
x, y, w = readtrace(os=1)

# Wavelengths array
lam_array = w   # [um]

# BOX EXTRACTION
print('Extraction: noiseless')
# To save it:
fbox_noiseless = np.zeros(shape=(np.shape(data_noiseless)[0], np.shape(data_noiseless)[2]), dtype=float)
for t in range(np.shape(data_noiseless)[0]):   # For each image of the timeseries
    fbox_noiseless[t] = flambda_adu(x, data_noiseless[t], y, radius_pixel=radius_pixel)   # [adu/s]
f_array_noiseless = np.nan_to_num(fbox_noiseless)
# Normalization of flux
f_array_noiseless_norm = normalization(f_array_noiseless, t1, t4)

# Time array
time = np.arange(f_array_noiseless.shape[0])
time_min = time * tint / 60.  # [min]

# WHITE LIGHT CURVE
f_white_noiseless = np.sum(f_array_noiseless, axis=1)
# Normalize white light curve
f_white_noiseless_norm = normalization(f_white_noiseless, t1, t4)


##########################################################
################# EXTRACTION, NOISY ######################
##########################################################
# Load simulation
simu_noisy, data_noisy = rateints_dms_simulation(WORKING_DIR + simu_noisy_filename)
simulation_noisy = 'noisy'

ng = simu_noisy[0].header['NGROUPS']      # Number of groups
t_read = simu_noisy[0].header['TGROUP']   # Reading time [s]
tint = (ng - 1) * t_read                  # Integration time [s]

if False:
    # Show trace image
    plt.figure()
    plt.imshow(data_noisy[10], vmin=0, origin="lower", cmap='gray')
    plt.plot(x, y, color="r")
    plt.show()

# BOX EXTRACTION
print('Extraction: noisy')
# To save it:
fbox_noisy = np.zeros(shape=(np.shape(data_noisy)[0], np.shape(data_noisy)[2]), dtype=float)
for t in range(np.shape(data_noisy)[0]):    # For each image of the timeseries
    fbox_noisy[t] = flambda_adu(x, data_noisy[t], y, radius_pixel=radius_pixel)   # [adu/s]
f_array_noisy = np.nan_to_num(fbox_noisy)
# Normalization of flux
f_array_noisy_norm = normalization(f_array_noisy, t1, t4)

# Time array
time = np.arange(f_array_noisy.shape[0])
time_min = time * tint / 60.  # [min]

# WHITE LIGHT CURVE
f_white_noisy = np.sum(f_array_noisy, axis=1)
# Normalize white light curve
f_white_noisy_norm = normalization(f_white_noisy, t1, t4)


##########################################################
####################### ANALYSIS #########################
##########################################################
# Graphic of white light curves
plt.figure()
plt.plot(time_min, f_white_noisy_norm, '.', markersize=4, color='r', label=simulation_noisy)
plt.plot(time_min, f_white_noiseless_norm, '.', markersize=4, color='b', label=simulation_noiseless)
plt.xlabel('Time [min]')
plt.ylabel('Flux')
plt.title('White light')
plt.legend()
if doSave_plots:
    plt.savefig(WORKING_DIR + 'white_light_' + simulation_noisy + 'wasp52')

if False:
    # Plot only one wavelength
    l = 100   # Indice of wavelength to plot
    plt.figure()
    plt.plot(time_min, f_array_noisy_norm[:, l], '.', color='r', label=r'$\lambda$={:.3f}'.format(w[l]))
    plt.xlabel('Time [min]')
    plt.ylabel('Relative flux')
    plt.title(simulation_noisy)
    plt.legend()


# Index where bad wavelengths start
bad_wl = -5


# DISPERSION/PHOTON NOISE RATIO
# Only the uncontaminated portion of the spectrum is used here
i_uncont = 1100   # Start of the uncontaminated portion
# New arrays
new_w = w[i_uncont:bad_wl]
new_f_array_noiseless = f_array_noiseless[:, i_uncont:bad_wl]
new_f_array_noisy = f_array_noisy[:, i_uncont:bad_wl]
new_f_array_noisy_norm = f_array_noisy_norm[:, i_uncont:bad_wl]
if doSave_data:
    # Saves data for transitfit
    np.save(WORKING_DIR + 'wavelengths', new_w)
    np.save(WORKING_DIR + 'extracted_flux_norm', new_f_array_noisy_norm)
    np.save(WORKING_DIR + 'time_array', time_min)

gain = 1.61   # Gain [e⁻]
# To store data:
photon_noise = np.zeros(new_f_array_noiseless.shape[1], dtype='float')
dispersion = np.zeros(new_f_array_noisy.shape[1], dtype='float')
for n in range(new_f_array_noisy.shape[1]):   # for each wavelength
    # Noiseless out of transit spectrum
    out_transit_noiseless = np.concatenate((new_f_array_noiseless[:t1, n], new_f_array_noiseless[t4:, n]))
    # Noisy out of transit spectrum
    out_transit_noisy = np.concatenate((new_f_array_noisy_norm[:t1, n], new_f_array_noisy_norm[t4:, n]))
    # Conversion in electrons (e⁻) for noiseless
    out_transit_noiseless_elec = out_transit_noiseless * tint * gain
    # Photon noise (Poisson) (e⁻)
    photon_noise_elec = np.sqrt(np.mean(out_transit_noiseless_elec))
    # Photon noise (adu/s)
    photon_noise[n] = photon_noise_elec / gain / tint
    # Dispersion in noisy data
    dispersion[n] = np.std(out_transit_noisy)
ratio = dispersion / photon_noise

# Plot ratio between dispersion and photon noise vs wavelength
plt.figure()
plt.plot(new_w, ratio, '.', color='b', label=simulation_noisy)
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel('Dispersion/Photon noise')
plt.legend()
if doSave_plots:
    plt.savefig(WORKING_DIR + 'disp_over_photnoise_' + simulation_noisy + 'wasp52')

print("Mean dispersion over photon noise ratio =", np.mean(ratio))


# For transitfit flux error
# Conversion in electrons for all noiseless spectra (e⁻)
new_f_array_noiseless_elec = new_f_array_noiseless * tint * gain
# Photon noise (Poisson)
phot_noise_elec = np.sqrt(new_f_array_noiseless_elec)   # [e⁻]
phot_noise = phot_noise_elec / gain / tint   # [adu/s]
# Noisy out of transit spectra
out_transit_noisy = np.concatenate((f_array_noisy[: t1 + 1, i_uncont:bad_wl], f_array_noisy[t4:, i_uncont:bad_wl]))
# Noisy out of transit spectra means
out_transit_noisy_mean = np.mean(out_transit_noisy, axis=0)
# Normalized photon noise (adu/s)
phot_noise_norm = phot_noise / out_transit_noisy_mean
if doSave_data:
    # Saves data for transitfit
    np.save(WORKING_DIR + 'phot_noise', phot_noise_norm)


### RELATIVE DIFFERENCE
# Between white light curves
relatDiff_white = relative_difference(f_white_noisy, f_white_noiseless)

plt.figure()
plt.plot(time_min, relatDiff_white * 1.0e6, color='b')
plt.xlabel('Time [min]')
plt.ylabel('Relative difference [ppm]')
plt.title('Relative difference between {} and \n{}'.format(simulation_noisy, simulation_noiseless))
if doSave_plots:
    plt.savefig(WORKING_DIR + 'relatDiff_white_' + simulation_noisy + 'wasp52')


### TRANSIT LIGHT CURVE
transit_curve_noisy = transit_depth(new_f_array_noisy_norm, t1, t2, t3, t4)  # (Rp/Rs)²
transit_curve_noisy = np.sqrt(transit_curve_noisy)  # Rp/Rs

plt.figure()
plt.plot(new_w, transit_curve_noisy * 1e6, color='r')
plt.xlabel(r"Wavelength [$\mu m$]")
plt.ylabel(r'$R_p/R_s$ [ppm]')
plt.title('Transit spectrum')
if doSave_plots:
    plt.savefig(WORKING_DIR + 'transit_spectrum_' + simulation_noisy)


if doShow_plots is True: plt.show()
