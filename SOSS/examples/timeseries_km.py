import SOSS.trace.tracepol as tp
import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
from sys import path

# Imports from the extraction
from extract.overlap import TrpzOverlap
from extract.throughput import ThroughputSOSS
from extract.convolution import WebbKer

# Imports for plots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For displaying of FITS images.

import box_kim

# Matplotlib defaults
plt.rc('figure', figsize=(13,7))
plt.rc('font', size=14)
plt.rc('image', cmap='inferno')
plt.rc('lines', lw=2)

####################################################################################
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/'

sys.path.insert(0,"/genesis/jwst/jwst-ref-soss/fortran_lib/")

# Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_kim.txt')
pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              # Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars)    # Read in parameter file

####################################################################################
# List of orders to consider in the extraction
order_list = [1, 2]

"""
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
"""
####################################################################################
# LOAD SIMULATIONS
noisy_rateints = fits.open('/genesis/jwst/userland-soss/loic_review/timeseries_20210806_forKim/'
                           'IDTSOSS_clear_noisy_rateints.fits')

#with fits.open('/genesis/jwst/userland-soss/loic_review/timeseries_20210806_forKim/'
 #                          'IDTSOSS_clear_noisy_rateints.fits') as hdulist:
  #  keys = hdulist[0].header.keys

# Noisy images for extraction
data_noisy = noisy_rateints[1].data  # Images of flux [adu/s]
data_noisy = data_noisy.astype('float64', copy=False)
delta_noisy = noisy_rateints[2].data  # Errors [adu/s]
delta_noisy = delta_noisy.astype('float64', copy=False)
dq = noisy_rateints[3].data  # Data quality
i = np.where(dq % 2 != 0)  # Odd values of dq = DO NOT USE these pixels
data_noisy[i[0], i[1], i[2]] = 0.
delta_noisy[i[0], i[1], i[2]] = 0.
"""
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

data = data_noisy[0]

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
sig = np.sqrt(data + bkgd_noise**2)

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
lam_bin_array = np.zeros(shape=(np.shape(data_noisy)[0], np.shape(data_noisy)[2]), dtype='float')
f_bin_array = np.zeros(shape=(np.shape(data_noisy)[0], np.shape(data_noisy)[2]), dtype='float')

for t in range(np.shape(data_noisy)[0]):  # For each image of the timeseries
    data = data_noisy[t]

    # Extract the oversampled spectrum 𝑓𝑘
    # Extract the spectrum
    f_k = extract.extract(data=data, sig=sig, tikhonov=True, factor=best_fac)

    
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

        
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

        for i_ord in range(extract.n_ord):
            label = extract.orders[i_ord]
            ax[i_ord].plot(lam_bin_list[i_ord], f_bin_list[i_ord], label=label)

        ax[0].set_ylabel("Extracted signal [counts]")

        ax[1].set_xlabel("Wavelength [$\mu m$]")
        ax[1].set_ylabel("Extracted signal [counts]")

        plt.tight_layout()
        plt.show()
        

        
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
        

    lam_bin_array[t] = lam_bin_list[0]
    f_bin_array[t] = f_bin_list[0]

    print('t = {}: Done'.format(t))

hdu_lam = fits.PrimaryHDU(lam_bin_array)
hdu_lam.writeto(WORKING_DIR + "timeseries/lam_bin_array.fits", overwrite=True)
hdu_f = fits.PrimaryHDU(f_bin_array)
hdu_f.writeto(WORKING_DIR + "timeseries/f_bin_array.fits", overwrite=True)
"""

####################################################################################
# PLOT
def normalization(f_lambda, t1, t4):
    hors_t = np.concatenate((f_lambda[: t1 + 1], f_lambda[t4:]))
    mean = np.mean(hors_t)
    return f_lambda / mean

def fit_resFunc(coeff, x, y):
    p = np.poly1d(coeff)
    return (p(x)) - y


with fits.open(WORKING_DIR + "timeseries/lam_bin_array.fits") as hdulist:
    lam_array = hdulist[0].data
    lam_array = lam_array[0]

with fits.open(WORKING_DIR + "timeseries/f_bin_array.fits") as hdulist:
    f_array = hdulist[0].data * 1.6   # [é/s]

f_white_sum = np.sum(data_noisy, axis=(1, 2))

time = np.arange(f_array.shape[0]) * 5. * 5.491 / 60.  # [min]

t1 = 100   # * 5. * 5.491 / 60  # [min]
t2 = 160   # Might be bigger than reality
t3 = 285   # Might be less than reality
t4 = 340   # * 5. * 5.491 / 60  # [min]

i = np.where(np.abs(lam_array - 1.) == np.min(np.abs(lam_array - 1.)))[0][0]

f_i = f_array[:, i]
f_array_nonan = np.nan_to_num(f_array)
f_white = np.sum(f_array_nonan, axis=1)

# f_white = normalization(f_white, t1, t4)

# Fit a linear curve
p3_coeff = []
p2_coeff = []
p1_coeff = []
p0_coeff = []

time_fit = np.concatenate((time[4:t1], time[t4:]))
f_fit = np.concatenate((f_white[4:t1], f_white[t4:]))

p3, p2, p1, p0 = box_kim.robust_polyfit(fit_resFunc, time_fit, f_fit, [-0.05, 10., -1000., 1.1e7])
poly_fit = np.poly1d([p3, p2, p1, p0])
fit_y = poly_fit(time)
p3_coeff.append(p3), p2_coeff.append(p2), p1_coeff.append(p1), p0_coeff.append(p0)

print(p3_coeff, p2_coeff, p1_coeff, p0_coeff)

new_f_white = f_white / fit_y

hors_t = np.concatenate((f_i[:t1+1], f_i[t4:]))
np.nan_to_num(hors_t, copy=False)
std_dev = np.std(hors_t)
photon_noise = np.sqrt(hors_t)
rapport = std_dev / photon_noise

plt.figure()
plt.plot(time, f_white, '.', color='b')
plt.plot(time, fit_y, '--', color='r')
plt.xlabel('Time [min]')
plt.ylabel('Relative flux')

plt.figure()
plt.plot(time[5:], new_f_white[5:], '.', color='b')

plt.show()

