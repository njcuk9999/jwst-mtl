import SOSS.trace.tracepol as tp
import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
import scipy.constants as sc_cst
from scipy.interpolate import interp1d
from sys import path
from Fake_data.simu_utils import load_simu

# Imports from the extraction
from extract.overlap import TrpzOverlap
from extract.throughput import ThroughputSOSS
from extract.convolution import WebbKer

# Imports for plots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For displaying of FITS images.

import box_kim

# MATPLOTLIB DEFAULTS
plt.rc('figure', figsize=(13,7))
plt.rc('font', size=14)
plt.rc('image', cmap='inferno')
plt.rc('lines', lw=1)

# CONSTANTS
h = sc_cst.Planck
c = sc_cst.speed_of_light
gain = 1.6
area = 25.
ng = 3   # NGROUP
t_read = 5.49   # Reading time [s]
tint = (ng - 1) * t_read   # Integration time [s]

radius_pixel = 36

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

# CHOOSE ORDER (for box extraction)   !!!
m_order = 1  # For now, only option is 1

# CHOOSE ORDER(S) TO EXTRACT (ADB)  !!!
only_order_1 = True

# CHOOSE noiseless or noisy !!!
if only_order_1 is False:
    noisy = False
else:
    noisy = False

# SAVE FIGS? !!!
save = False

####################################################################################
os_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

####################################################################################
# Read relevant files
#### Wavelength solution ####
# _adb : Antoine's files
wave_maps_adb = []
wave_maps_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/wavelengths_m1.fits"))
wave_maps_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/wavelengths_m2.fits"))
i_zero = np.where(wave_maps_adb[1][0] == 0.)[0][0]  # Where Antoine put 0 in his 2nd order wl map

# Loic's files
wave_la = fits.getdata("/genesis/jwst/userland-soss/loic_review/refs/map_wave_2D_native.fits")
wave_la[1, :, i_zero:] = 0.  # Also set to 0 the same points as Antoine in Loic's 2nd order wl map
wave_maps_la = [wave_la[0]] if only_order_1 else wave_la[:2]

# _clear
wave_clear = fits.getdata(WORKING_DIR + "with_peaks/oversampling_1/wave_map2D.fits".format(os))
wave_maps_clear = [wave_clear[0]] if only_order_1 else wave_clear[:2]   # Consider only orders 1 & 2

wave_maps = wave_maps_clear

# Convert data from fits files to float (fits precision is 1e-8)
wave_maps = [wv.astype('float64') for wv in wave_maps]

# List of orders to consider in the extraction
order_list = [1] if only_order_1 else [1, 2]

#### Throughputs ####
thrpt_list = [ThroughputSOSS(order) for order in order_list]   # Has been changed to 1 everywhere in throughput.py  K.M.

#### Convolution kernels ####
ker_list = [WebbKer(wv_map) for wv_map in wave_maps]

####################################################################################
# Position of trace for box extraction (TEMPORARY VERSION)
trace_file = "/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv"
pars = tp.get_tracepars(trace_file)  # Gives the middle position of order 1 trace
x, y_not, w = box_kim.readtrace(os=1)  # TODO: Problem with .readtrace
xnew, y, mask = tp.wavelength_to_pix(w, pars, m=1, oversample=1)  # Converts wavelenghths to pixel coordinates  NOT GOOD

####################################################################################
relatdiff_tik_box = np.empty(shape=(len(os_list), len(w)), dtype=float)

plt.figure()
for i in range(len(os_list)):
    os = os_list[i]
    simuPars.noversample = os

    #### Spatial profiles ####
    # _clear : map created with clear000000.fits directly
    new_spat = fits.getdata(WORKING_DIR + "new_map_profile_clear_{}.fits".format(os))
    spat_pros = [new_spat[0]] if only_order_1 else new_spat[:2]
    # Convert data from fits files to float (fits precision is 1e-8)
    spat_pros = [p_ord.astype('float64') for p_ord in spat_pros]

    # Put all inputs from reference files in a list
    ref_files_args = [spat_pros, wave_maps, thrpt_list, ker_list]

    ####################################################################################
    # LOAD SIMULATIONS
    clear_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_000000.fits".format(os))
    noisy_rateints = fits.open(WORKING_DIR + "oversampling_{}/test_clear_noisy_rateints.fits".format(os))

    # Because of x_padding and y_padding
    padd = 10

    if os != 1:
        # Bin to pixel native
        clear_i = soss.rebin(clear_00[0].data, os, flux_method='sum')
        clear_bin = clear_i[:2, padd:-padd, padd:-padd]  # Convolved traces, binned
    else:  # Os = 1
        clear_bin = clear_00[0].data[:2, padd:-padd, padd:-padd]

    ####################################################################################
    # Noiseless images for extraction
    data_conv_bin = clear_bin[0] if only_order_1 else np.sum(clear_bin, axis=0)

    if noisy is True:
        # Noisy images for extraction
        data_noisy = noisy_rateints[1].data[0]  # Image of flux [adu/s]
        delta_noisy = noisy_rateints[2].data[0]  # Error [adu/s]
        dq = noisy_rateints[3].data[0]  # Data quality
        i = np.where(dq % 2 != 0)  # Odd values of dq = DO NOT USE these pixels
        data_noisy[i[0], i[1]] = 0.
        delta_noisy[i[0], i[1]] = 0.
        i_neg = np.where(data_noisy < 0.)
        data_noisy[i_neg[0], i_neg[1]] = 0.

    ####################################################################################
    # BOX EXTRACTIONS
    # EXTRACTED FLUX OF NOISELESS TRACE(S)
    # Convolved trace(s), binned
    if only_order_1 is True:
        fbox_conv_inf_adu_bin = box_kim.flambda_inf_radi_adu(data_conv_bin) / os      # Infinite radius [adu/s]
    fbox_conv_adu_bin = box_kim.flambda_adu(x, data_conv_bin, y, radius_pixel=radius_pixel) / os      # [adu/s]
    fbox_conv_energy_bin = box_kim.f_lambda(x, data_conv_bin, w, y, radius_pixel=radius_pixel) / os   # [J/s/m²/um]

    # EXTRACTED FLUX OF NOISY TRACE(S)
    if noisy is True:
        fbox_noisy_adu = box_kim.flambda_adu(x, data_noisy, y, radius_pixel=radius_pixel) / os      # [adu/s]
        fbox_noisy_energy = box_kim.f_lambda(x, data_noisy, w, y, radius_pixel=radius_pixel) / os   # [J/s/m²/um]

    ####################################################################################
    # TIKHONOV EXTRACTION  (done on convolved, binned traces)
    data = data_noisy if noisy else data_conv_bin

    params = {}

    # Map of expected noise (standard deviation).
    bkgd_noise = 20.  # In counts?

    # Wavelength extraction grid oversampling.
    params["n_os"] = 5

    # Threshold on the spatial profile.
    # Only pixels above this threshold will be used for extraction. (for at least one order) # Will create a mask
    params["thresh"] = 1e-4  # Same units as the spatial profiles

    # List of orders considered
    params["orders"] = order_list

    # Initiate extraction object
    # (This needs to be done only once unless the oversampling (n_os) changes.)
    extract = TrpzOverlap(*ref_files_args, **params)  # Precalculate matrices

    # Find the best tikhonov factor
    # This takes some time, so it's better to do it once if the exposures are part of a time series observation,
    # i.e. observations of the same object at similar SNR
    # Determine which factors to tests.
    factors = np.logspace(-25, -12, 14)

    # Noise estimate to weight the pixels
    sig = np.sqrt(data + bkgd_noise ** 2)   # Poisson noise + background noise

    # Tests all these factors.
    tests = extract.get_tikho_tests(factors, data=data, sig=sig)

    # Find the best factor.
    best_fac = extract.best_tikho_factor(tests=tests, i_plot=False)

    # Refine the grid (span 4 orders of magnitude).
    best_fac = np.log10(best_fac)
    factors = np.logspace(best_fac - 2, best_fac + 2, 20)

    tests = extract.get_tikho_tests(factors, data=data, sig=sig)
    best_fac = extract.best_tikho_factor(tests=tests, i_plot=False)

    # Extract the oversampled spectrum 𝑓𝑘
    # Can be done in a loop for a timeseries and/or iteratively for different estimates of the reference files.
    f_k = extract.extract(data=data, sig=sig, tikhonov=True, factor=best_fac) / os

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
    # Integrate
    lam_bin, ftik_bin = extract.bin_to_pixel(f_k=f_k, i_ord=0)  # Only extract order 1

    # For comparison:
    # Because w and lam_bin are not the same
    f = interp1d(lam_bin, ftik_bin, fill_value='extrapolate')
    ftik_bin_interp = f(w)   # Extracted flux by Tikhonov interpolated on my wl grid

    # Comparison
    if only_order_1 is True:
        relatdiff_tik_box[i] = box_kim.relative_difference(ftik_bin_interp, fbox_conv_inf_adu_bin)  # Tikhonov vs noiseless
        print('Radius pixel = infinite (for fbox_conv_ in relative differences)')
    else:
        ref_data = fbox_noisy_adu if noisy else fbox_conv_adu_bin
        relatdiff_tik_box[i] = box_kim.relative_difference(ftik_bin_interp, ref_data)
        print("Radius pixel = ", radius_pixel, '(for fbox_conv_ in relative differences)')

    plt.plot(w[5:-5], relatdiff_tik_box[i, 5:-5] * 1e6, label='os = {}'.format(os))

    print('Os = {} : Done'.format(os))

# Apply median filter on all relative differences
relatdiff_median = np.median(relatdiff_tik_box, axis=0)
relatdiff_norm = relatdiff_tik_box - relatdiff_median
relatdiff_std = np.std(relatdiff_norm[:, 5:-5], axis=1)

print('Standard deviations: ', relatdiff_std)

plt.plot(w[5:-5], relatdiff_median[5:-5] * 1e6, label='Median')
title_noise = 'noisy' if noisy else 'noiseless'
plt.title("Relative difference, Tikhonov vs box extraction, {}".format(title_noise))
plt.xlabel("Wavelength [$\mu m$]")
plt.ylabel("Relative difference [ppm]")
plt.legend(bbox_to_anchor=(0.95,1))
if save is True:
    if only_order_1 is True:
        plt.savefig(WORKING_DIR + 'relatdiff_tik_box_order1.png')
    else:
        if noisy is True:
            plt.savefig(WORKING_DIR + 'relatdiff_tik_box_noisy.png')
        else:
            plt.savefig(WORKING_DIR + 'relatdiff_tik_box.png')

if only_order_1 is True:
    print("Order 1 only")
else:
    print('Noisy') if noisy else print('Orders 1 & 2 summed')

plt.show()

plt.figure()
for i in range(len(os_list)):
    plt.plot(w[5 :-5], relatdiff_norm[i, 5 :-5] * 1e6, label=os_list[i])
plt.xlabel("Wavelength [$\mu m$]")
plt.ylabel("Difference [ppm]")
plt.title('Relative difference - median filter')
plt.legend(title='Oversampling')
if save is True:
    if only_order_1 is True:
        plt.savefig(WORKING_DIR + 'relatdiff_tik_box_norm_order1.png')
    else:
        if noisy is True:
            plt.savefig(WORKING_DIR + 'relatdiff_tik_box_norm_noisy.png')
        else:
            plt.savefig(WORKING_DIR + 'relatdiff_tik_box_norm.png')
plt.show()

plt.figure()
plt.scatter(os_list, relatdiff_std * 1e6, color='r')
plt.xticks(os_list, os_list)
plt.xlabel('Oversampling')
plt.ylabel('Standard deviation [ppm]')
plt.title('Standard deviations from difference')
if save is True:
    if only_order_1 is True:
        plt.savefig(WORKING_DIR + 'relatdiff_tik_box_std_order1.png')
    else:
        if noisy is True:
            plt.savefig(WORKING_DIR + 'relatdiff_tik_box_std_noisy.png')
        else:
            plt.savefig(WORKING_DIR + 'relatdiff_tik_box_std.png')
plt.show()
