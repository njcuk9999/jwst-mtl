import SOSS.trace.tracepol as tp
import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
from sys import path

# Imports from the extraction
#from extract.overlap import TrpzOverlap
#from extract.throughput import ThroughputSOSS
#from extract.convolution import WebbKer

# Imports for plots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For displaying of FITS images.

import box_kim

# Matplotlib defaults
plt.rc('figure', figsize=(13,7))
plt.rc('font', size=14)
plt.rc('image', cmap='inferno')
plt.rc('lines', lw=1)

####################################################################################
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/PHY3030/'

sys.path.insert(0,"/genesis/jwst/jwst-ref-soss/fortran_lib/")

# Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_kim.txt')
pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              # Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars)    # Read in parameter file

####################################################################################
gain = 1.6   # é per adu

####################################################################################
# LOAD SIMULATIONS
# NOISY
#noisy_rateints = fits.open('/genesis/jwst/userland-soss/loic_review/timeseries_20210806_forKim/'
 #                          'IDTSOSS_clear_noisy_rateints.fits')
noisy_rateints = fits.open(WORKING_DIR + 'IDTSOSS_clear_noisy--rateints.fits')

# Noisy images for extraction
data_noisy = noisy_rateints[1].data  # Images of flux [adu/s]
data_noisy = data_noisy.astype('float64', copy=False)
#delta_noisy = noisy_rateints[2].data  # Errors [adu/s]
#delta_noisy = delta_noisy.astype('float64', copy=False)
dq = noisy_rateints[3].data  # Data quality
i = np.where(dq % 2 != 0)  # Odd values of dq = DO NOT USE these pixels
data_noisy[i[0], i[1], i[2]] = 0.
#delta_noisy[i[0], i[1], i[2]] = 0.

# NOISELESS
clear = fits.open('/genesis/jwst/userland-soss/loic_review/timeseries_20210806_forKim/IDTSOSS_clear.fits')

data_clear = clear[1].data[:,-1]  # Images of flux [adu/s]  (Last image of each integration (5th group)
# Convert data from fits files to float (fits precision is 1e-8)
data_noiseless = [im_ord.astype('float64') for im_ord in data_clear]

"""
####################################################################################
# TIKHONOV

# List of orders to consider in the extraction
order_list = [1, 2]

####################################################################################
#### Wavelength solution ####
wave_maps = []
wave_maps.append(fits.getdata("/genesis/jwst/userland-soss/loic_review/cartewaveGJT.fits"))
wave_maps.append(fits.getdata("/genesis/jwst/userland-soss/loic_review/cartewaveGJT_order2.fits"))
wave_maps[1] = np.where(np.isnan(wave_maps[1]), np.nanmin(wave_maps[1]), wave_maps[1])

#### Spatial profiles ####
spat_pros = fits.getdata("/genesis/jwst/userland-soss/loic_review/2DTrace_native_nopadding_20210729.fits")
spat_pros = spat_pros[:2]

# Convert data from fits files to float (fits precision is 1e-8)
wave_maps = [wv.astype('float64') for wv in wave_maps]
spat_pros = [p_ord.astype('float64') for p_ord in spat_pros]

####################################################################################
#### Throughputs ####
thrpt_list = [ThroughputSOSS(order) for order in order_list]

#### Convolution kernels ####
ker_list = [WebbKer(wv_map) for wv_map in wave_maps]

# Put all inputs from reference files in a list
ref_files_args = [spat_pros, wave_maps, thrpt_list, ker_list]

####################################################################################
# EXTRACTION PARAMETERS
params = {}

# Map of expected noise (standard deviation)
bkgd_noise = 20.  # In counts?

# Wavelength extraction grid oversampling
params["n_os"] = 5

# Threshold on the spatial profile
# Only pixels above this threshold will be used for extraction (for at least one order)
params["thresh"] = 1e-4  # Same units as the spatial profiles

# List of orders considered
params["orders"] = order_list

data = data_clear[0]

####################################################################################
# INITIATE EXTRACTION OBJECT
# (This needs to be done only once unless the oversampling (n_os) changes.)
extract = TrpzOverlap(*ref_files_args, **params)  # Precalculate matrices

# Find the best tikhonov factor
# This takes some time, so it's better to do it once if the exposures are part of a time series observation,
# i.e. observations of the same object at similar SNR
# Determine which factors to tests.
factors = np.logspace(-25, -12, 14)   # (-25, 5, 20)

# Noise estimate to weigh the pixels
# Poisson noise + background noise
sig = np.sqrt(data + bkgd_noise**2) * 0 + 1.

# Tests all these factors
tests = extract.get_tikho_tests(factors, data=data, sig=sig)

# Find the best factor
plt.figure()
best_fac = extract.best_tikho_factor(tests=tests, i_plot=True)

# Refine the grid (span 4 orders of magnitude).
best_fac = np.log10(best_fac)
factors = np.logspace(best_fac-2, best_fac+2, 20)

plt.figure()
tests = extract.get_tikho_tests(factors, data=data, sig=sig)
best_fac = extract.best_tikho_factor(tests=tests, i_plot=True)

####################################################################################
lam_bin_array = np.zeros(shape=(np.shape(data_clear)[0], np.shape(data_clear)[2]), dtype='float')
f_bin_array = np.zeros(shape=(np.shape(data_clear)[0], np.shape(data_clear)[2]), dtype='float')

for t in range(np.shape(data_clear)[0]):  # For each image of the timeseries
    data = data_clear[t]

    # Extract the oversampled spectrum 𝑓𝑘
    # Extract the spectrum
    f_k = extract.extract(data=data, sig=sig, tikhonov=True, factor=best_fac)

    if False:
        # Plot the extracted spectrum.
        plt.plot(extract.lam_grid, f_k)
        plt.xlabel("Wavelength [$\mu m$]")
        plt.ylabel("Oversampled Spectrum $f_k$ [energy$\cdot s^{-1} \cdot \mu m^{-1}$]")
        # For now, arbitrairy units, but it should be the flux that hits the detector, so energy/time/wavelength
        plt.tight_layout()
        plt.show()
    
    ####################################################################################
    # Bin to pixel native sampling
    # To get a result comparable to typical extraction methods, we need to integrate the oversampled spectrum (𝑓𝑘)
    # to a grid representative of the native pixel sampling (for each order).
    # This integration is done according to the equation bin𝑖=∫𝜆+𝑛𝑖𝜆−𝑛𝑖𝑇𝑛(𝜆)𝑓̃ 𝑛(𝜆)𝜆𝑑𝜆,
    # where 𝑛 is a given order, 𝑇𝑛 is the throughput of the order and 𝑓̃ 𝑛 is the underlying flux convolved to the order 𝑛
    # resolution. The result of this integral will be in fake counts (it is not directly the sum of the counts
    # so that's why I call it fake).
    # One could directly extract the integrated flux by setting the throughput to 𝑇𝑛(𝜆)=1
    # (see second example). The result would then be in flux units instead of counts.

    # Bin in counts
    # Save the output in a list for different orders
    f_bin_list = []  # Integrated flux
    lam_bin_list = []  # Wavelength grid

    for i_ord in range(extract.n_ord):

        # Integrate
        lam_bin, f_bin = extract.bin_to_pixel(f_k=f_k, i_ord=i_ord)

        # Save
        f_bin_list.append(f_bin)
        lam_bin_list.append(lam_bin)

        if False:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    
            for i_ord in range(extract.n_ord):
                label = extract.orders[i_ord]
                ax[i_ord].plot(lam_bin_list[i_ord], f_bin_list[i_ord], label=label)
    
            ax[0].set_ylabel("Extracted signal [counts]")
    
            ax[1].set_xlabel("Wavelength [$\mu m$]")
            ax[1].set_ylabel("Extracted signal [counts]")
    
            plt.tight_layout()
            plt.show()
        
        if False:        
            # Bin in flux units
            # Set throughput to 1
    
            def throughput(x):
                return np.ones_like(x)
        
            # Save the output in a list for different orders.
            ftik_bin_list = []  # Integrated flux
            lam_bin_list = []  # Wavelength grid
    
            for i_ord in range(extract.n_ord):
    
                # Integrate
                lam_bin, f_bin = extract.bin_to_pixel(f_k=f_k, i_ord=i_ord, throughput=throughput)
    
                # Save.
                ftik_bin_list.append(f_bin)
                lam_bin_list.append(lam_bin)
    
                
                # Plot
                label = extract.orders[i_ord]
                plt.plot(lam_bin_list[i_ord], f_bin_list[i_ord], label=label)
    
            plt.ylabel(r"Convolved flux $\tilde{f_k}$ [energy$\cdot s^{-1} \cdot \mu m^{-1}$]")
            plt.xlabel("Wavelength [$\mu m$]")
    
            plt.tight_layout()
            plt.legend(title="Order")
            plt.show()
        

    lam_bin_array[t] = lam_bin_list[0]  # Only save order 1
    f_bin_array[t] = f_bin_list[0]

    print('t = {}: Done'.format(t))

hdu_lam = fits.PrimaryHDU(lam_bin_array)
hdu_lam.writeto(WORKING_DIR + "timeseries/lam_bin_array_clear.fits", overwrite=True)
hdu_f = fits.PrimaryHDU(f_bin_array)
hdu_f.writeto(WORKING_DIR + "timeseries/f_bin_array_clear.fits", overwrite=True)

"""
####################################################################################
# BOX EXTRACTION

# Position of trace for box extraction
x, y, w = box_kim.readtrace(os=1)

# Extractions
fbox_noiseless = np.zeros(shape=(np.shape(data_noiseless)[0], np.shape(data_noiseless)[2]), dtype='float')
fbox_noisy = np.zeros(shape=(np.shape(data_noisy)[0], np.shape(data_noisy)[2]), dtype='float')

radius_pixel = 30  # Radius for box extraction
oversample = 4.  # Oversampling of the simulation

for t in range(np.shape(data_noiseless)[0]):  # For each image of the timeseries
    fbox_noiseless[t] = box_kim.flambda_adu(x, data_noiseless[t], y, radius_pixel=radius_pixel) / oversample   # [adu/s]
    fbox_noisy[t] = box_kim.flambda_adu(x, data_noisy[t], y, radius_pixel=radius_pixel) / oversample  # [adu/s]

####################################################################################
def normalization(f_lambda, t1, t4):
    """
    Normalize transit light curve by mean during out of transit.
    """
    hors_t = np.concatenate((f_lambda[: t1 + 1], f_lambda[t4:]))
    mean = np.mean(hors_t)
    return f_lambda / mean

def transit_depth(f_lambda, t2, t3):
    """
    Calculates mean flux value during transit.
    """
    return np.mean(f_lambda[t2: t3+1])

def fit_resFunc(coeff, x, y):
    """
    Function to minimize in least square fit for polynomial retrieval.
    """
    p = np.poly1d(coeff)
    return (p(x)) - y

####################################################################################
# NOISELESS
data = data_noiseless

if False:  # Tikhonov extraction
    with fits.open(WORKING_DIR + "timeseries/lam_bin_array_clear.fits") as hdulist:
        lam_array = hdulist[0].data   # [um]
        lam_array = lam_array[0]   # [um]

    with fits.open(WORKING_DIR + "timeseries/f_bin_array_clear.fits") as hdulist:
        f_array = hdulist[0].data * gain   # [é/s]
if True:   # Box extraction
    lam_array = w
    f_array = fbox_noiseless

start = 0   # To avoid the first deviating points
# t1 = 97   # * 5. * 5.491 / 60  # [min]
# t2 = 163   # * 5. * 5.491 / 60  # [min]
# t3 = 274   # * 5. * 5.491 / 60  # [min]
# t4 = 341   # * 5. * 5.491 / 60  # [min]
t1 = 95   # * 5. * 5.491 / 60  # [min]
t2 = 155   # * 5. * 5.491 / 60  # [min]
t3 = 283   # * 5. * 5.491 / 60  # [min]
t4 = 343   # * 5. * 5.491 / 60  # [min]

time = np.arange(f_array.shape[0])
time_min = time * 5. * 5.491 / 60.  # [min]
time_fit = np.concatenate((time[start:t1], time[t4:]))

f_array = np.nan_to_num(f_array)

# White light curve
f_white_sum = np.sum(data, axis=(1, 2))  # Sum all detector

f_white = np.sum(f_array, axis=1)  # Real white light curve

# Deviations
hors_t_white = np.concatenate((f_white[start:t1], f_white[t4:]))
hors_t_white = np.nan_to_num(hors_t_white)
std_dev = np.std(hors_t_white)   # Standard deviation
photon_noise = np.sqrt(hors_t_white)
rapport_white = std_dev / photon_noise

# Normalize white light curve
new_f_white = normalization(f_white, t1, t4)


# For each wavelength
new_f_array = np.copy(f_array)
rapport_array = np.zeros(shape=(len(time_fit), f_array.shape[1]))

for n in range(f_array.shape[1]):   # For each wavelength
    # Deviations
    hors_t = np.concatenate((f_array[start:t1, n], f_array[t4:, n]))
    hors_t = np.nan_to_num(hors_t)
    std_dev = np.std(hors_t)   # Standard deviation
    photon_noise = np.sqrt(hors_t)
    rapport_array[:, n] = std_dev / photon_noise

    # Normalize each light curve
    new_f_array[:, n] = normalization(f_array[:, n], t1, t4)

# Transit depth
depth = []
for i in range(f_array.shape[1]):   # For each wavelength
    depth_i = transit_depth(new_f_array[:, i], t2, t3)
    depth.append(1 - depth_i)

# Save arrays for the noiseless case
f_array_clear = np.copy(f_array)
new_f_array_clear = np.copy(new_f_array)
f_white_clear = np.copy(f_white)
new_f_white_clear = np.copy(new_f_white)
rapport_array_clear = np.copy(rapport_array)
rapport_white_clear = np.copy(rapport_white)
lam_array_clear = np.copy(lam_array)
depth_clear = np.copy(depth)

####################################################################################
# NOISY
data = data_noisy

if False:   # Tikhonov extraction
    with fits.open(WORKING_DIR + "timeseries/lam_bin_array.fits") as hdulist:
        lam_array = hdulist[0].data   # [um]
        lam_array = lam_array[0]   # [um]

    with fits.open(WORKING_DIR + "timeseries/f_bin_array.fits") as hdulist:
        f_array = hdulist[0].data * gain   # [é/s]
if True:   # Box extraction
    lam_array = w
    f_array = fbox_noisy

f_array = np.nan_to_num(f_array)

# White light curve
f_white_sum = np.sum(data, axis=(1, 2))   # Sum all detector
f_white_sum = normalization(f_white_sum, t1, t4)

f_white = np.sum(f_array, axis=1)   # Real white light curve
f_white_norm = normalization(f_white, t1, t4)   # Normalize it for polynomial fit

if False:
    # Fit a polynomial curve
    f_fit = np.concatenate((f_white_norm[start:t1], f_white_norm[t4:]))   # Flux to consider in fit
    # A third degree polynomial is the best fit
    p3, p2, p1, p0 = box_kim.robust_polyfit(fit_resFunc, time_fit, f_fit, [-2e-10, 2e-7, -5e-5, 1.005])   # [-0.05, 10., -1000., 1.1e7]
    poly_fit = np.poly1d([p3, p2, p1, p0])
    poly_white = poly_fit(time)
    # print(p3, p2, p1, p0)
    # Apply correction
    new_f_white = f_white / poly_white

if False:
    # Deviations
    hors_t_white = np.concatenate((new_f_white[start:t1], new_f_white[t4:]))
    hors_t_white = np.nan_to_num(hors_t_white)
    std_dev = np.std(hors_t_white)   # Standard deviation
    photon_noise = np.sqrt(hors_t_white)
    rapport_white = std_dev / photon_noise

# Normalize the corrected white light curve
new_f_white = normalization(new_f_white, t1, t4)


# For each wavelength
new_f_array = np.copy(f_array)
f_array_norm = np.zeros_like(f_array, dtype='float')
rapport_array = np.zeros(shape=(len(time_fit), f_array.shape[1]), dtype='float')
rapport_array2 = np.zeros(shape=(f_array.shape[1]), dtype='float')

for n in range(f_array.shape[1]):   # For each wavelength
    if False:
        f_fit = normalization(f_array[:, n], t1, t4)   # Flux to consider in fit
        f_fit = np.concatenate((f_fit[start:t1], f_fit[t4:]))
        f_fit = np.nan_to_num(f_fit)
        p3_n, p2_n, p1_n, p0_n = box_kim.robust_polyfit(fit_resFunc, time_fit, f_fit, [p3, p2, p1, p0])
        poly_fit = np.poly1d([p3_n, p2_n, p1_n, p0_n])
        poly_f_n = poly_fit(time)
        # Apply correction
        new_f_array[:, n] /= poly_f_n
    if False:
        # Deviations
        hors_t = np.concatenate((new_f_array[start:t1, n], new_f_array[t4:, n]))
        hors_t = np.nan_to_num(hors_t)
        std_dev = np.std(hors_t)
        photon_noise = np.sqrt(hors_t)
        #photon_noise = np.sqrt(np.mean(hors_t))
        rapport_array[:, n] = std_dev / photon_noise
        rapport_array2[n] = np.mean(rapport_array[:, n])

    # Normalize corrected light curve
    f_array_norm[:, n] = normalization(f_array[:, n], t1, t4)
    new_f_array[:, n] = normalization(new_f_array[:, n], t1, t4)


# Transit depth
depth = []
for i in range(f_array.shape[1]):   # For each wavelength
    depth_i = transit_depth(new_f_array[:, i], t2, t3)
    depth.append(1 - depth_i)

# Save arrays for the noisy case
f_array_noisy = np.copy(f_array)
new_f_array_noisy = np.copy(new_f_array)
f_white_noisy = np.copy(f_white)
new_f_white_noisy = np.copy(new_f_white)
f_white_norm_noisy = np.copy(f_white_norm)
rapport_array_noisy = np.copy(rapport_array)
rapport_white_noisy = np.copy(rapport_white)
lam_array_noisy = np.copy(lam_array)
depth_noisy = np.copy(depth)

####################################################################################
"""
hdu = fits.PrimaryHDU(time_min)
hdu.writeto(WORKING_DIR + "timeseries/time_min.fits", overwrite=True)

hdu = fits.PrimaryHDU(new_f_array_clear)
hdu.writeto(WORKING_DIR + "timeseries/new_f_array_clear.fits", overwrite=True)
"""
####################################################################################
# Relative difference
relatDiff_depth = box_kim.relative_difference(depth_noisy, depth_clear)

####################################################################################
# # Model  (from William's notebook convolve_PMOD.ipynb)
# with fits.open(WORKING_DIR + "timeseries/wavs_mod_interp.fits") as hdulist:
#     wavs_mod_interp = hdulist[0].data   # [um]
#
# with fits.open(WORKING_DIR + "timeseries/convdata.fits") as hdulist:
#     convdata = hdulist[0].data   # [ppm]

####################################################################################
# PLOT
if True:
    plt.figure()
    plt.plot(time_min, f_white_noisy, '.', color='b')
    #plt.plot(time_min[start:], poly_white[start:], '--', lw=2,  color='r', label='Polynomial fit')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Normalized flux')
    plt.legend()
    print(np.shape(f_white_noisy))
if True:
    i = 1000
    plt.figure()
    plt.plot(time_min[:], new_f_array_noisy[:, i], '.', color='b', label=r'$\lambda$ = {} $\mu$m'.format(np.around(lam_array_noisy[i], decimals=3)))    # label='Avant'
    #plt.plot(time_min[start:], new_f_array_noisy[start:, i], '.', color='b', label='Après')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Normalized flux [$adu/s$]')
    plt.legend()

if False:
    plt.figure()
    plt.plot(time_min[start:], new_f_white_noisy[start:], '.', color='b')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Normalized flux')

if False:
    plt.figure()
    plt.plot(lam_array_noisy, rapport_array2, '.', color='b')
    plt.xlabel(r"Wavelength [$\mu m$]")
    plt.ylabel('Std dev. / photon noise')
    #plt.legend()

if False:
    plt.figure()
    plt.plot(lam_array_clear, depth_clear * 1e6, color='b', label='Data (noiseless)')
    #plt.plot(lam_array_noisy, depth_noisy * 1e6, color='b', label='Noisy')
    plt.plot(wavs_mod_interp, convdata, color='Orange', label='Convolved model')
    plt.xlabel(r"Wavelength [$\mu m$]")
    plt.ylabel(r'$(R_p/R_s)²$ [ppm]')
    plt.xlim(np.min(lam_array_clear), np.max(lam_array_clear))
    #plt.title('Transit spectrum')
    plt.legend()

if False:
    plt.figure()
    plt.plot(lam_array_noisy,relatDiff_depth * 1e6, color='b')
    #plt.scatter(lam_array[4], depth[4], color='r')
    plt.xlabel(r"Wavelength [$\mu m$]")
    plt.ylabel(r'Relative difference [ppm]')   # ($(R_p/R_s)²$)


plt.show()

