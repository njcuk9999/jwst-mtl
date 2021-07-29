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
ng = 3   # NGROUP
t_read = 5.49   # Reading time [s]
tint = (ng - 1) * t_read   # Integration time [s]

radius_pixel = 34

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
# CHOOSE ORDER(S) TO EXTRACT (ADB)  !!!
only_order_1 = True

# CHOOSE noiseless or noisy !!!
if only_order_1 is False:
    noisy = False
else:
    noisy = False

# CHOOSE WHICH MAPS !!!
map_la = True  # False gives the new clear map

# SAVE FIGS? !!!
save = False

####################################################################################
os_list = [4]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

####################################################################################
if map_la:
    SAVE_DIR = WORKING_DIR + "with_old_loic_wl_map2D/"
else:
    SAVE_DIR = WORKING_DIR + "with_new_wl_map2D/"

# Read relevant files
#### Wavelength solution ####
# _adb : Antoine's files
wave_maps_adb = []
wave_maps_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/wavelengths_m1.fits"))
wave_maps_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/wavelengths_m2.fits"))
i_zero = np.where(wave_maps_adb[1][0] == 0.)[0][0]  # Where Antoine put 0 in his 2nd order wl map

if map_la:
    # Loic's files
    #wave = fits.getdata("/genesis/jwst/userland-soss/loic_review/refs/map_wave_2D_native.fits")
    #wave = fits.getdata("/genesis/jwst/userland-soss/loic_review/2DWave_native_nopadding_20210729.fits")
    #wave = fits.getdata("/genesis/jwst/userland-soss/loic_review/cartewaveGJT.fits")
    #wave_maps = [wave]
    wave = fits.getdata(WORKING_DIR + "with_peaks/wave_map2D.fits")   # TO REMOVE
else:
    # _clear
    wave = fits.getdata(WORKING_DIR + "with_peaks/wave_map2D.fits")
#wave[1, :, i_zero:] = 0.  # Also set to 0 the same points as Antoine in clear's 2nd order wl map
wave_maps = [wave[0]] if only_order_1 else wave[:2]   # Consider only orders 1 & 2
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
relatdiff_tik_box_ref = np.empty(shape=(len(os_list), len(w)), dtype=float)
lam_bin_array = np.empty(shape=(len(os_list), len(w)), dtype=float)
ftik_bin_array = np.empty(shape=(len(os_list), len(w)), dtype=float)

fig1, ax1 = plt.subplots(2, 1, sharex=True)
fig2, ax2 = plt.subplots(1, 1)
fig3, ax3 = plt.subplots(2, 1, sharex=True)

for i in range(len(os_list)):
    os = os_list[i]
    simuPars.noversample = os

    #### Spatial profiles ####
    if map_la:
        # Loic's files
        new_spat = fits.getdata("/genesis/jwst/userland-soss/loic_review/2DTrace_native_nopadding_20210729.fits")
        #new_spat = fits.getdata(WORKING_DIR + "new_map_profile_clear_{}.fits".format(os))   # TO REMOVE
    else:
        # _clear : map created with clear000000.fits directly
        new_spat = fits.getdata(WORKING_DIR + "new_map_profile_clear_{}.fits".format(os))
    spat_pros = [new_spat[0]] if only_order_1 else new_spat[:2]

    # Convert data from fits files to float (fits precision is 1e-8)
    spat_pros = [p_ord.astype('float64') for p_ord in spat_pros]

    # Put all inputs from reference files in a list
    ref_files_args = [spat_pros, wave_maps, thrpt_list, ker_list]

    ####################################################################################
    # LOAD SIMULATIONS
    #clear_tr_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_trace_000000.fits".format(os))
    clear_tr_00 = fits.open('/genesis/jwst/userland-soss/loic_review/tmp/clear_trace_000000.fits')   # timeseries_20210729_forGJ/nint1/clear_trace_000000.fits
    #clear_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_000000.fits".format(os))
    clear_00 = fits.open('/genesis/jwst/userland-soss/loic_review/tmp/clear_000000_constantflambda.fits')   # timeseries_20210729_forGJ/nint1/clear_000000.fits
    noisy_rateints = fits.open(WORKING_DIR + "oversampling_{}/test_clear_noisy_rateints.fits".format(os))

    # Because of x_padding and y_padding
    #padd = 10
    padd = 100

    if os != 1:
        # Bin to pixel native
        ref_i = soss.rebin(clear_tr_00[0].data, os, flux_method='sum')
        clear_ref_bin = ref_i[:2, padd:-padd, padd:-padd]  # Reference thin trace, binned
        clear_i = soss.rebin(clear_00[0].data, os, flux_method='sum')
        clear_bin = clear_i[:2, padd:-padd, padd:-padd]  # Convolved traces, binned
    else:  # Os = 1
        clear_ref_bin = clear_tr_00[0].data[:2, padd:-padd, padd:-padd]
        clear_bin = clear_00[0].data[:2, padd:-padd, padd:-padd]

    ####################################################################################
    # Noiseless images for extraction
    data_conv_bin = clear_bin[0] if only_order_1 else np.sum(clear_bin, axis=0)
    data_ref_bin = clear_ref_bin[0] if only_order_1 else np.sum(clear_ref_bin, axis=0)

    if noisy:
        # Noisy images for extraction
        data_noisy = noisy_rateints[1].data[0]  # Image of flux [adu/s]
        delta_noisy = noisy_rateints[2].data[0]  # Error [adu/s]
        dq = noisy_rateints[3].data[0]  # Data quality
        i = np.where(dq % 2 != 0)  # Odd values of dq = DO NOT USE these pixels
        data_noisy[i[0], i[1]] = 0.
        delta_noisy[i[0], i[1]] = 0.

    ####################################################################################
    # BOX EXTRACTIONS
    # EXTRACTED FLUX OF NOISELESS TRACE(S)
    # Reference thin trace:
    fbox_ref_adu_bin = box_kim.flambda_adu(x, data_ref_bin, y, radius_pixel=4) / os  # Binned [adu/s]

    # Convolved trace(s), binned
    if only_order_1:
        fbox_conv_inf_adu_bin = box_kim.flambda_inf_radi_adu(data_conv_bin) / os      # Infinite radius [adu/s]
    fbox_conv_adu_bin = box_kim.flambda_adu(x, data_conv_bin, y, radius_pixel=radius_pixel) / os      # [adu/s]
    fbox_conv_energy_bin = box_kim.f_lambda(x, data_conv_bin, w, y, radius_pixel=radius_pixel) / os   # [J/s/m²/um]

    # EXTRACTED FLUX OF NOISY TRACE(S)
    if noisy:
        fbox_noisy_adu = box_kim.flambda_adu(x, data_noisy, y, radius_pixel=radius_pixel) / os      # [adu/s]
        fbox_noisy_energy = box_kim.f_lambda(x, data_noisy, w, y, radius_pixel=radius_pixel) / os   # [J/s/m²/um]

    if only_order_1:
        ax3[0].plot(w, box_kim.relative_difference(fbox_conv_inf_adu_bin, fbox_ref_adu_bin) * 1e6, label='os = {}'.format(os))
        ax3[1].plot(w, fbox_conv_inf_adu_bin, label='os = {}'.format(os))
    else:
        ax3[0].plot(w, box_kim.relative_difference(fbox_conv_adu_bin, fbox_ref_adu_bin) * 1e6, label='os = {}'.format(os))
        ax3[1].plot(w, fbox_conv_adu_bin, label='os = {}'.format(os))
    ax3[0].set_title('Relative difference of extracted fluxes : convolved vs ref. thin trace (binned)')
    ax3[1].set_title('Extracted flux from convolved trace (binned)')
    ax3[1].legend(bbox_to_anchor=(0.99, 1.5))
    ax3[1].set_xlabel("Wavelength [$\mu m$]")
    ax3[0].set_ylabel("Relative difference [ppm]"), ax3[1].set_ylabel("Extracted flux [adu/s]")

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
    sig = np.sqrt(np.abs(data) + bkgd_noise ** 2) * 0 + 1e-20

    # Tests all these factors.
    tests = extract.get_tikho_tests(factors, data=data, sig=sig)

    # Find the best factor.
    best_fac = extract.best_tikho_factor(tests=tests, i_plot=True)

    # Refine the grid (span 4 orders of magnitude).
    best_fac = np.log10(best_fac)
    factors = np.logspace(best_fac - 2, best_fac + 2, 20)

    tests = extract.get_tikho_tests(factors, data=data, sig=sig)
    best_fac = extract.best_tikho_factor(tests=tests, i_plot=True)

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
    lam_bin_array[i] = lam_bin
    ftik_bin_array[i] = ftik_bin

    # For comparison:
    # Because w and lam_bin are not the same
    f = interp1d(lam_bin, ftik_bin, fill_value='extrapolate')
    ftik_bin_interp = f(w)   # Extracted flux by Tikhonov interpolated on my wl grid   # w
    if only_order_1:
        ff = interp1d(w, fbox_conv_inf_adu_bin, fill_value='extrapolate')
        fbox_conv_inf_adu_bin_interp = ff(lam_bin)
    else:
        ff = interp1d(w, fbox_conv_adu_bin, fill_value='extrapolate')
        fbox_conv_adu_bin_interp = ff(lam_bin)
    fff = interp1d(w, fbox_ref_adu_bin, fill_value='extrapolate')
    fbox_ref_adu_bin_interp = fff(lam_bin)

    # Comparison
    if only_order_1:
        relatdiff_tik_box[i] = box_kim.relative_difference(ftik_bin, fbox_conv_inf_adu_bin_interp)  # Tikhonov vs noiseless   # np.flip(ftik_bin)
        print('Radius pixel = infinite (for fbox_conv_ in relative differences)')
    else:
        ref_data = fbox_noisy_adu if noisy else fbox_conv_adu_bin_interp
        relatdiff_tik_box[i] = box_kim.relative_difference(ftik_bin, ref_data)
        print("Radius pixel = ", radius_pixel, '(for fbox_conv_ in relative differences)')

    relatdiff_tik_box_ref[i] = box_kim.relative_difference(ftik_bin, fbox_ref_adu_bin_interp)   # Tikhonov vs noiseless

    ax1[0].plot(lam_bin[5:-5], relatdiff_tik_box[i, 5:-5] * 1e6, label='os = {}'.format(os))
    ax1[1].plot(lam_bin[5:-5], relatdiff_tik_box_ref[i, 5:-5] * 1e6, label='os = {}'.format(os))

    if os == 4:
        ax2.plot(lam_bin[5:-5], ftik_bin[5:-5], label='tikhonov, os = {}'.format(os))
        ax2.plot(np.flip(lam_bin[5:-5]), fbox_ref_adu_bin[5:-5], label='box')

    print('Os = {} : Done'.format(os))

    # Rebuild the detector
    rebuilt = extract.rebuild(f_k * os)

    plt.subplot(111, aspect='equal')
    plt.pcolormesh((rebuilt - data)/data * 1e6)   #
    plt.colorbar(label="(rebuilt - data)/data [ppm]", orientation='horizontal', aspect=40)   #/data [ppm]
    plt.tight_layout()

title_noise = 'noisy' if noisy else 'noiseless'

ax2.set_title("Extracted flux, Tikhonov vs box extraction, {}".format(title_noise))
ax2.set_xlabel(r"Wavelength [$\mu m$]")
ax2.set_ylabel("Extracted flux [adu/s]")
ax2.legend(bbox_to_anchor=(0.95,1))

# Apply median filter on all relative differences
relatdiff_median = np.median(relatdiff_tik_box, axis=0)
diff_norm = relatdiff_tik_box - relatdiff_median
diff_std = np.std(diff_norm[:, 5:-5], axis=1)
print('Standard deviations: ', diff_std)

relatdiff_median_ref = np.median(relatdiff_tik_box_ref, axis=0)
diff_norm_ref = relatdiff_tik_box_ref - relatdiff_median_ref
diff_std_ref = np.std(diff_norm_ref[:, 5:-5], axis=1)

ax1[0].plot(lam_bin[5:-5], relatdiff_median[5:-5] * 1e6, label='Median')
ax1[1].plot(lam_bin[5:-5], relatdiff_median_ref[5:-5] * 1e6, label='Median')
ax1[0].set_title("Relative difference, Tikhonov vs box extraction, convolved, {}".format(title_noise))
ax1[1].set_title("Relative difference, Tikhonov vs box extraction, thin trace, {}".format(title_noise))
ax1[1].set_xlabel("Wavelength [$\mu m$]")
ax1[0].set_ylabel("Relative difference [ppm]"), ax1[1].set_ylabel("Relative difference [ppm]")
ax1[1].legend(bbox_to_anchor=(0.96,1.5))
if save:
    if only_order_1:
        fig1.savefig(SAVE_DIR + 'relatdiff_tik_box_order1.png')
        fig3.savefig(WORKING_DIR + 'relatdiff_box_convVSref_order1.png')
    else:
        if noisy:
            fig1.savefig(SAVE_DIR + 'relatdiff_tik_box_noisy.png')
        else:
            fig1.savefig(SAVE_DIR + 'relatdiff_tik_box_12.png')
            fig3.savefig(WORKING_DIR + 'relatdiff_box_convVSref_12.png')

if only_order_1:
    print("Order 1 only")
else:
    print('Noisy') if noisy else print('Orders 1 & 2 summed')
plt.show()

fig, ax = plt.subplots(2, 1, sharex=True)
for i in range(len(os_list)):
    ax[0].plot(lam_bin[5:-5], diff_norm[i, 5:-5] * 1e6, label=os_list[i])
    ax[1].plot(lam_bin[5:-5], diff_norm_ref[i, 5:-5] * 1e6, label=os_list[i])
ax[1].set_xlabel(r"Wavelength [$\mu m$]")
ax[0].set_ylabel("Difference [ppm]"), ax[1].set_ylabel("Difference [ppm]")
ax[0].set_title('Relative difference - median filter (convolved)')
ax[1].set_title('Relative difference - median filter (thin trace)')
ax[1].legend(title='Oversampling', bbox_to_anchor=(0.96,1.5))
if save:
    if only_order_1:
        fig.savefig(SAVE_DIR + 'relatdiff_tik_box_norm_order1.png')
    else:
        if noisy:
            fig.savefig(SAVE_DIR + 'relatdiff_tik_box_norm_noisy.png')
        else:
            fig.savefig(SAVE_DIR + 'relatdiff_tik_box_norm_12.png')
plt.show()

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].scatter(os_list, diff_std * 1e6, color='r')
ax[1].scatter(os_list, diff_std_ref * 1e6, color='r')
ax[0].set_xticks(os_list, os_list), ax[1].set_xticks(os_list, os_list)
ax[1].set_xlabel('Oversampling')
ax[0].set_ylabel('Standard deviation [ppm]'), ax[1].set_ylabel('Standard deviation [ppm]')
ax[0].set_title('Standard deviations from difference, convolved')
ax[1].set_title('Standard deviations from difference, thin trace')
if save:
    if only_order_1:
        fig.savefig(SAVE_DIR + 'relatdiff_tik_box_std_order1.png')
    else:
        if noisy:
            fig.savefig(SAVE_DIR + 'relatdiff_tik_box_std_noisy.png')
        else:
            fig.savefig(SAVE_DIR + 'relatdiff_tik_box_std_12.png')
plt.show()

if save:
    hdu_ftik = fits.PrimaryHDU(ftik_bin_array)
    hdu_lam = fits.PrimaryHDU(lam_bin_array)
    o = 1 if only_order_1 else 12
    if map_la:
        hdu_ftik.writeto(WORKING_DIR + "ftik_bin_array_la_{}.fits".format(o), overwrite=True)
        hdu_lam.writeto(WORKING_DIR + "lam_bin_array_la_{}.fits".format(o), overwrite=True)
    else:
        hdu_ftik.writeto(WORKING_DIR + "ftik_bin_array_clear_{}.fits".format(o), overwrite=True)
        hdu_lam.writeto(WORKING_DIR + "lam_bin_array_clear_{}.fits".format(o), overwrite=True)

sys.exit()

ftik_bin_array_la = fits.getdata(WORKING_DIR + "ftik_bin_array_la_{}.fits".format(o))
lam_bin_array_la = fits.getdata(WORKING_DIR + "lam_bin_array_la_{}.fits".format(o))
ftik_bin_array_clear = fits.getdata(WORKING_DIR + "ftik_bin_array_clear_{}.fits".format(o))
lam_bin_array_clear = fits.getdata(WORKING_DIR + "lam_bin_array_clear_{}.fits".format(o))

plt.figure()
for i in range(ftik_bin_array_la.shape[0]):
    fct = interp1d(lam_bin_array_clear[i], ftik_bin_array_clear[i], fill_value='extrapolate')
    ftik_bin_clear_interp = fct(lam_bin_array_la[i])
    relatdiff_ftik = box_kim.relative_difference(ftik_bin_clear_interp, ftik_bin_array_la[i])
    plt.plot(lam_bin_array_la[i], relatdiff_ftik * 1e6, label='os = {}'.format(os_list[i]))
plt.ylabel("Relative difference [ppm]")
plt.xlabel(r"Wavelength [$\mu m$]")
plt.title('Relative difference between extracted fluxes (tikho.) with new wave_map2D vs old one')
plt.legend(title='Oversampling', bbox_to_anchor=(0.96,1))
if False:   # save
    if only_order_1:
        plt.savefig(WORKING_DIR + 'relatdiff_ftik_order1.png')
    else:
        plt.savefig(WORKING_DIR + 'relatdiff_ftik_12.png')
plt.show()

