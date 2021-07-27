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


# Matplotlib defaults
plt.rc('figure', figsize=(12,7))
plt.rc('font', size=14)
plt.rc('image', cmap='inferno')
plt.rc('lines', lw=2)

# Constants
h = sc_cst.Planck
c = sc_cst.speed_of_light
gain = 1.6
area = 25.
ng = 3   # NGROUP
t_read = 5.49   # Reading time [s]
tint = (ng - 1) * t_read   # Integration time [s]

radius_pixel = 36


WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/'

sys.path.insert(0,"/genesis/jwst/jwst-ref-soss/fortran_lib/")

# Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_kim.txt')
pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              #Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars) #read in parameter file


#####################################################
# CHOOSE ORDER (for box extraction)   !!!
m_order = 1  # For now, only option is 1.

# CHOOSE OVERSAMPLE  !!!
simuPars.noversample = 5
os = simuPars.noversample

# CHOOSE ORDER(S) TO EXTRACT (ADB)  !!!
only_order_1 = False

# CHOOSE noiseless or noisy !!!
if only_order_1 is False:
    noisy = False
else:
    noisy = False

# SAVE FIGS? !!!
save = True
#####################################################

# Position of trace for box extraction
trace_file = "/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv"
pars = tp.get_tracepars(trace_file)   # Gives the middle position of order 1 trace
#x = np.arange(2048)    # Array of pixels
#w, tmp = tp.specpix_to_wavelength(x, pars, m=1)   # Returns wavelength for each x, order 1
#xnew, y, mask = tp.wavelength_to_pix(w, pars, m=1)   # Converts wavelenghths to pixel coordinates  NOT GOOD

#x_os = np.arange(2048 * os)    # Array of pixels with os
#w_os, tmp_os = tp.specpix_to_wavelength(x_os, pars, m=1, oversample=os)  # Returns wavelength for each x, order 1, os
#xnew_os, y_os, mask_os = tp.wavelength_to_pix(w_os , pars, m=1, oversample=os)  # Converts wavelenghths to pixel coordinates, os  NOT GOOD

x, y_not, w = box_kim.readtrace(os=1)   # TODO: Problem with .readtrace
x_os, y_os_not, w_os = box_kim.readtrace(os=os)

xnew, y, mask = tp.wavelength_to_pix(w, pars, m=1, oversample=1)   # Converts wavelenghths to pixel coordinates  NOT GOOD
xnew_os, y_os, mask_os = tp.wavelength_to_pix(w_os , pars, m=1, oversample=os)  # Converts wavelenghths to pixel coordinates, os  NOT GOOD

#####################################################

# List of orders to consider in the extraction
order_list = [1] if only_order_1 else [1, 2]

#####################################################

# Read relevant files
#### Wavelength solution ####
# _adb : Antoine's files
wave_maps_adb = []
wave_maps_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/wavelengths_m1.fits"))
wave_maps_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/wavelengths_m2.fits"))
i_zero = np.where(wave_maps_adb[1][0] == 0.)[0][0]   # Where Antoine put 0 in his 2nd order wl map

# _la : Loic's files
wave_la = fits.getdata("/genesis/jwst/userland-soss/loic_review/refs/map_wave_2D_native.fits")
wave_la[1,:,i_zero:] = 0.  # Also set to 0 the same points as Antoine in Loic's 2nd order wl map
wave_maps_la = wave_la[:2]   # Consider only orders 1 & 2

# _la : NEW Loic's files
wave_la = fits.getdata("/genesis/jwst/userland-soss/loic_review/2D")
wave_la[1,:,i_zero:] = 0.  # Also set to 0 the same points as Antoine in Loic's 2nd order wl map
wave_maps_la = wave_la[:2]   # Consider only orders 1 & 2

# _clear
wave_clear = fits.getdata(WORKING_DIR + "with_peaks/oversampling_1/wave_map2D.fits")
wave_clear[1,:,i_zero:] = 0.  # Also set to 0 the same points as Antoine in clear's 2nd order wl map
wave_maps_clear = wave_clear[:2]   # Consider only orders 1 & 2

diff_wave_map = (wave_la - wave_clear) #/ wave_clear

if False:
    fig, ax = plt.subplots(3, 1)
    for i in range(diff_wave_map.shape[0]):
        im = ax[i].imshow(diff_wave_map[i], origin='lower')
        ax[i].set_title('Order {}'.format(i+1))
        fig.colorbar(im, ax=ax[i])
    if save is True:
        hdu = fits.PrimaryHDU(diff_wave_map)
        hdu.writeto(WORKING_DIR + "with_peaks/oversampling_1/wave_map2D_diff_laVSclear.fits", overwrite=True)
    plt.show()

#### Spatial profiles ####
# _adb : Antoine's files
#spat_pros_adb = []
#spat_pros_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/"
 #                                 "spat_profile_m1.fits").squeeze())
#spat_pros_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/"
  #                                "spat_profile_m2.fits").squeeze())

# _la : Loic's files
spat = fits.getdata("/genesis/jwst/userland-soss/loic_review/refs/map_profile_2D_native.fits").squeeze()
spat_pros_la = spat[:2, -256:]   # Consider only order 1 & 2 and set to same size as data
# Shift Loic's trace image to fit with the simulation (data)
new_spat_la = np.zeros_like(spat_pros_la, dtype=float)
new_spat_la[:, :162] = spat_pros_la[:, 94:]
hdu = fits.PrimaryHDU(new_spat_la)
hdu.writeto(WORKING_DIR + "new_spat_pros.fits", overwrite = True)

# _clear : map created with clear000000.fits directly
new_spat = fits.getdata(WORKING_DIR + "new_map_profile_clear_{}.fits".format(os))
spat_pros_clear = new_spat[:2]


# CHOOSE between Loic's and Antoine's maps
wave_maps = wave_maps_clear # or wave_maps_adb
spat_pros = spat_pros_clear  # or new_spat_la   # or spat_pros_adb
if only_order_1 is True:
    wave_maps = [wave_maps[0]]
    spat_pros = [spat_pros[0]]

# Convert data from fits files to float (fits precision is 1e-8)
wave_maps = [wv.astype('float64') for wv in wave_maps]
spat_pros = [p_ord.astype('float64') for p_ord in spat_pros]

#####################################################
#### Throughputs ####
thrpt_list = [ThroughputSOSS(order) for order in order_list]   # Has been changed to 1 everywhere in throughput.py  K.M.

#### Convolution kernels ####
ker_list = [WebbKer(wv_map) for wv_map in wave_maps]

# Put all inputs from reference files in a list
ref_files_args = [spat_pros, wave_maps, thrpt_list, ker_list]

#####################################################
# LOAD SIMULATIONS

# LA
clear_tr_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_trace_000000.fits".format(os))
clear_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_000000.fits".format(os))
noisy_rateints = fits.open(WORKING_DIR + "oversampling_{}/test_clear_noisy_rateints.fits".format(os))

# Because of x_padding and y_padding
padd = 10
padd_os = padd * os

clear_tr_ref = clear_tr_00[0].data[:2, padd_os:-padd_os, padd_os:-padd_os]   # Reference thin trace, not binned
clear_conv = clear_00[0].data[:2, padd_os:-padd_os, padd_os:-padd_os]        # Convolved traces, not binned
if os != 1:
    # Bin to pixel native
    ref_i = soss.rebin(clear_tr_00[0].data, os, flux_method='sum')
    clear_ref_bin = ref_i[:2, padd:-padd, padd:-padd]   # Reference thin trace, binned
    clear_i = soss.rebin(clear_00[0].data, os, flux_method='sum')
    clear_bin = clear_i[:2, padd:-padd, padd:-padd]     # Convolved traces, binned
else:  # Os = 1
    clear_ref_bin = np.copy(clear_tr_ref)
    clear_bin = np.copy(clear_conv)

#####################################################
# Noiseless images for extraction
if only_order_1 is True:
    # Order 1 only
    data_ref = clear_tr_ref[0]         # Reference thin trace, not binned [adu/s]
    data_ref_bin = clear_ref_bin[0]    # Reference thin trace, binned [adu/s]
    data_conv = clear_conv[0]          # Convolved, not binned [adu/s]
    data_conv_bin = clear_bin[0]       # Convolved, binned [adu/s]
else:
    # Sum traces 1 & 2
    data_ref = np.sum(clear_tr_ref, axis=0)         # Reference thin trace, not binned [adu/s]
    data_ref_bin = np.sum(clear_ref_bin, axis=0)    # Reference thin trace, binned [adu/s]
    data_conv = np.sum(clear_conv, axis=0)          # Convolved, not binned [adu/s]
    data_conv_bin = np.sum(clear_bin, axis=0)       # Convolved, binned [adu/s]
    if noisy is True:
        # Noisy images for extraction
        data_noisy = noisy_rateints[1].data[0]    # Image of flux [adu/s]
        delta_noisy = noisy_rateints[2].data[0]   # Error [adu/s]
        dq = noisy_rateints[3].data[0]            # Data quality
        i = np.where(dq %2 != 0)   # Odd values of dq = DO NOT USE these pixels
        data_noisy[i[0], i[1]] = 0.
        delta_noisy[i[0], i[1]] = 0.

#####################################################
# BOX EXTRACTIONS
# EXTRACTED FLUX OF NOISELESS TRACE(S)
# Reference thin trace:
fbox_ref_adu = box_kim.flambda_adu(x_os, data_ref, y_os, radius_pixel=5)           # Not binned [adu/s]
fbox_ref_adu_bin = box_kim.flambda_adu(x, data_ref_bin, y, radius_pixel=5) / os    # Binned [adu/s]

# Convolved trace(s), not binned
if only_order_1 is True:
    fbox_conv_inf_adu = box_kim.flambda_inf_radi_adu(data_conv)              # Infinite radius [adu/s]
fbox_conv_adu = box_kim.flambda_adu(x_os, data_conv, y_os, radius_pixel=radius_pixel * os)   # [adu/s]

# Convolved trace(s), binned
if only_order_1 is True:
    fbox_conv_inf_adu_bin = box_kim.flambda_inf_radi_adu(data_conv_bin) / os       # Infinite radius [adu/s]
fbox_conv_adu_bin = box_kim.flambda_adu(x, data_conv_bin, y, radius_pixel=radius_pixel) / os       # [adu/s]
#fbox_conv_elec_bin = box_kim.flambda_elec(x, data_conv_bin, y, radius_pixel=radius_pixel) / os     # [e⁻]
fbox_conv_energy_bin = box_kim.f_lambda(x, data_conv_bin, w, y, radius_pixel=radius_pixel) / os    # [J/s/m²/um]

# EXTRACTED FLUX OF NOISY TRACE(S)
if noisy is True:
    fbox_noisy_adu = box_kim.flambda_adu(x, data_noisy, y, radius_pixel=radius_pixel) / os      # [adu/s]
    #fbox_noisy_elec = box_kim.flambda_elec(x, data_noisy, y, radius_pixel=radius_pixel) / os    # [e⁻]
    fbox_noisy_energy = box_kim.f_lambda(x, data_noisy, w, y, radius_pixel=radius_pixel) / os   # [J/s/m²/um]

#####################################################
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
sig = np.sqrt(np.abs(data) + bkgd_noise**2)   # Poisson noise + background noise

# Tests all these factors.
tests = extract.get_tikho_tests(factors, data=data, sig=sig)

# Find the best factor.
best_fac = extract.best_tikho_factor(tests=tests, i_plot=False)

# Refine the grid (span 4 orders of magnitude).
best_fac = np.log10(best_fac)
factors = np.logspace(best_fac-2, best_fac+2, 20)

tests = extract.get_tikho_tests(factors, data=data, sig=sig)
best_fac = extract.best_tikho_factor(tests=tests, i_plot=False)

# Extract the oversampled spectrum 𝑓𝑘
# Can be done in a loop for a timeseries and/or iteratively for different estimates of the reference files.
f_k = extract.extract(data=data, sig=sig, tikhonov=True, factor=best_fac) / os


# Plot the extracted spectrum.
plt.figure()
plt.plot(extract.lam_grid, f_k, color='Blue')
plt.xlabel("Wavelength [$\mu m$]")
plt.ylabel("Oversampled Spectrum $f_k$ [energy$\cdot s^{-1} \cdot \mu m^{-1}$]")
# For now, arbitrairy units, but it should be the flux that hits the detector, so energy/time/wavelength
plt.tight_layout()
if save is True:
    if only_order_1 is True:
        plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/oversampled_spectrum_tik_order1.png".format(os))   #
    else:
        if noisy is True:
            plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/oversampled_spectrum_tik_noisy.png".format(os))   # /with_newclear_wave_map2D
        else:
            plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/oversampled_spectrum_tik.png".format(os))   # /with_newclear_wave_map2D

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
# Save the output in a list for different orders.
ftik_bin_list = []  # Integrated flux
lam_bin_list = []  # Wavelength grid

for i_ord in range(extract.n_ord):

    # Integrate.
    lam_bin, f_bin = extract.bin_to_pixel(f_k=f_k, i_ord=i_ord)

    # Save.
    ftik_bin_list.append(f_bin)
    lam_bin_list.append(lam_bin)

if only_order_1 is True:
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 6))

    # For comparison:
    # Because w and lam_bin_list[0] are not the same
    #f = interp1d(lam_bin_list[0], ftik_bin_list[0] , fill_value='extrapolate')
    #ftik_bin_interp = f(w)    # Extracted flux by Tikhonov interpolated on my wl grid
    ff = interp1d(w, fbox_conv_inf_adu_bin, fill_value='extrapolate')
    fbox_conv_inf_adu_bin_interp = ff(lam_bin_list[0])
    fff = interp1d(w, fbox_ref_adu_bin, fill_value='extrapolate')
    fbox_ref_adu_bin_interp = fff(lam_bin_list[0])
    ax.plot(w, fbox_conv_inf_adu_bin, label='Box extraction, radius = inf', color='DeepPink')
    ax.plot(lam_bin_list[0], ftik_bin_list[0], ls='--', label="Tikhonov extraction", color='Blue')

    ax.set_ylabel("Extracted signal [counts (adu/s)]")
    ax.set_xlabel("Wavelength [$\mu m$]")
    plt.legend()

else:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    for i_ord in range(extract.n_ord):
        label = extract.orders[i_ord]

        if i_ord == 0:
            # For comparison:
            # Because w and lam_bin_list[0] are not the same
            #f = interp1d(lam_bin_list[i_ord], ftik_bin_list[i_ord], fill_value='extrapolate')
            #ftik_bin_interp = f(w)   # Extracted flux by Tikhonov interpolated on my wl grid
            ff = interp1d(w, fbox_conv_adu_bin, fill_value='extrapolate')
            fbox_conv_adu_bin_interp = ff(lam_bin_list[0])
            ffff = interp1d(w, fbox_ref_adu_bin, fill_value='extrapolate')
            fbox_ref_adu_bin_interp = ffff(lam_bin_list[0])
            if noisy is False:
                ax[i_ord].plot(w, fbox_conv_adu_bin, color='DeepPink', label='Box extraction, order {}'.format(label))
            else:
                fff = interp1d(w, fbox_noisy_adu, fill_value='extrapolate')
                fbox_noisy_adu_interp = fff(lam_bin_list[0])
                ax[i_ord].plot(w, fbox_noisy_adu, color='DeepPink', label='Box extraction, order {}'.format(label))

        ax[i_ord].plot(lam_bin_list[i_ord], ftik_bin_list[i_ord], color='Blue',
                       label='Tikhonov, order {}'.format(label))
        ax[i_ord].set_ylabel("Extracted signal [counts]")
        ax[i_ord].legend()
    ax[1].set_xlabel(r"Wavelength [$\mu m$]")

plt.tight_layout()
if save is True:
    if only_order_1 is True:
        plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/extracted_signal_tik_box_order1.png".format(os))   #/with_newclear_wave_map2D
    else:
        if noisy is True:
            plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/extracted_signal_tik_box_noisy.png".format(os))   # /with_newclear_wave_map2D
        else:
            plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/extracted_signal_tik_box.png".format(os))   #/with_newclear_wave_map2D


# Comparison
if only_order_1 == True:
    relatdiff_box_tik = box_kim.relative_difference(ftik_bin_list[0], fbox_conv_inf_adu_bin_interp)  # Tikhonov vs noiseless

    relatdiff_conv_ref = box_kim.relative_difference(fbox_conv_inf_adu, fbox_ref_adu)    # Convolved vs ref, not binned
    relatdiff_conv_ref_bin = box_kim.relative_difference(fbox_conv_inf_adu_bin, fbox_ref_adu_bin)  # Convolved vs ref, binned
    relatdiff_tik_ref = box_kim.relative_difference(ftik_bin_list[0], fbox_ref_adu_bin_interp)             # Tikhonov vs ref
    print('Radius pixel = infinite (for fbox_conv_ in relative differences)')

else:
    if noisy is False:
        relatdiff_box_tik = box_kim.relative_difference(ftik_bin_list[0], fbox_conv_adu_bin_interp)  # Tikhonov vs noiseless
    elif noisy is True:
        relatdiff_box_tik = box_kim.relative_difference(ftik_bin_list[0], fbox_noisy_adu_interp)     # Tikhonov vs noisy

    relatdiff_conv_ref = box_kim.relative_difference(fbox_conv_adu, fbox_ref_adu)        # Convolved vs ref, not binned
    relatdiff_conv_ref_bin = box_kim.relative_difference(fbox_conv_adu_bin, fbox_ref_adu_bin)  # Convolved vs ref, binned
    relatdiff_tik_ref = box_kim.relative_difference(ftik_bin_list[0], fbox_ref_adu_bin_interp)         # Tikhonov vs ref
    print("Radius pixel = ", radius_pixel, '(for fbox_conv_ in relative differences)')


plt.figure()
plt.plot(w_os, relatdiff_conv_ref * 1e6, color='Indigo', label='Noiseless convolved vs ref, not binned, os={}'.format(os))
plt.plot(w, relatdiff_conv_ref_bin * 1e6, color='Green', label='Noiseless convolved vs ref, binned')
plt.plot(lam_bin_list[0], relatdiff_tik_ref * 1e6, color='Crimson', label='Tikhonov vs ref, binned')
plt.xlabel("Wavelength [$\mu m$]")
plt.ylabel("Relative difference [ppm]")
plt.title("Relative differences, compared to reference thin trace")
plt.legend()
if save is True:
    if only_order_1 is True:
        plt.savefig(WORKING_DIR + 'oversampling_{}/with_newclear_wave_map2D/relatdiff_vs_ref_order1.png'.format(os))   #/with_newclear_wave_map2D
    else:
        if noisy is True:
            plt.savefig(WORKING_DIR + 'oversampling_{}/with_newclear_wave_map2D/relatdiff_vs_ref_noisy.png'.format(os))  #/with_newclear_wave_map2D
        else:
            plt.savefig(WORKING_DIR + 'oversampling_{}/with_newclear_wave_map2D/relatdiff_vs_ref.png'.format(os))   # /with_newclear_wave_map2D


plt.figure()
plt.plot(lam_bin_list[0][5:-5], relatdiff_box_tik[5:-5] * 1e6, color='DarkOrchid')
plt.xlabel("Wavelength [$\mu m$]")
plt.ylabel("Relative difference [ppm]")
title_noise = 'noisy' if noisy else 'noiseless'
plt.title("Relative difference, Tikhonov vs box extraction, {}".format(title_noise))
if save is True:
    if only_order_1 is True:
        plt.savefig(WORKING_DIR + 'oversampling_{}/with_newclear_wave_map2D/relatdiff_box_tik_order1.png'.format(os))   #/with_newclear_wave_map2D
    else:
        if noisy is True:
            plt.savefig(WORKING_DIR + 'oversampling_{}/with_newclear_wave_map2D/relatdiff_box_tik_noisy.png'.format(os))   #/with_newclear_wave_map2D
        else:
            plt.savefig(WORKING_DIR + 'oversampling_{}/with_newclear_wave_map2D/relatdiff_box_tik.png'.format(os))   #/with_newclear_wave_map2D
plt.show()


# Bin in flux units
# Set throughput to 1
def throughput(x):
    return np.ones_like(x)

plt.figure()
for i_ord in range(extract.n_ord):

    # Integrate.
    lam_bin, ftik_bin = extract.bin_to_pixel(f_k=f_k, i_ord=i_ord, throughput=throughput)
                            # APPEND ? K.M.

    # Plot
    label = extract.orders[i_ord]
    plt.plot(lam_bin, ftik_bin, label=label)

plt.ylabel(r"Convolved flux $\tilde{f_k}$ [energy$\cdot s^{-1} \cdot \mu m^{-1}$]")
plt.xlabel("Wavelength [$\mu m$]")
plt.tight_layout()
plt.legend(title="Order")
if save is True:
    if only_order_1 is True:
        plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/convolved_flux_tik_order1.png".format(os))   #/with_newclear_wave_map2D
    else:
        if noisy is True:
            plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/convolved_flux_tik_noisy.png".format(os))   #/with_newclear_wave_map2D
        else:
            plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/convolved_flux_tik.png".format(os))   # /with_newclear_wave_map2D

# Quality estimate
# Rebuild the detector
rebuilt = extract.rebuild(f_k*os)

plt.subplot(111, aspect='equal')
plt.pcolormesh((rebuilt - data)/sig, vmin=-3, vmax=3)
plt.colorbar(label="Error relative to noise", orientation='horizontal', aspect=40)
plt.tight_layout()
if save is True:
    if only_order_1 is True:
        plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/rebuild_tik_order1.png".format(os))   #/with_newclear_wave_map2D
    else:
        if noisy is True:
            plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/rebuild_tik_noisy.png".format(os))   #/with_newclear_wave_map2D
        else:
            plt.savefig(WORKING_DIR + "oversampling_{}/with_newclear_wave_map2D/rebuild_tik.png".format(os))   #/with_newclear_wave_map2D
plt.show()
# (We can see that we are very close to the photon noise limit in this case. There are some small
# structures in the 2nd order in the overlap region, but the extracted spectrum is dominated by the
# 1st order in this wavelength region anyway, due to the higher throughput.) - ADB simu


##########################
# GRAPHICS FOR BOX EXTRACTION
# Images of traces
plt.figure()
if noisy is False:
    plt.imshow(data_conv, vmin=0, origin="lower")
    plt.title("clear_000000.fits, os={}".format(os))
else:
    plt.imshow(data_noisy, vmin=0, origin="lower")
    plt.title("test_clear_noisy_rateints.fits, os={}".format(os))
#plt.plot(x, y_not, color="Lime", lw=1, label="Order 1 trace's position l.a.")   # Middle position of order 1 trace
plt.plot(x_os, y_os, color="r", lw=1, label="Order 1 trace's position")   # Middle position of order 1 trace
plt.colorbar(label="[adu/s]", orientation='horizontal')
plt.legend()


plt.figure()
plt.plot(w_os, fbox_ref_adu, color='DeepPink', label='Reference thin trace, not binned')
plt.plot(w_os, fbox_conv_adu, color='Green', label='Convolved, not binned')
plt.ylabel(r"Extracted flux [adu/s]")
plt.xlabel(r"Wavelength [$\mu$m]")   # plt.xlabel(r"x")
plt.title('os = {}'.format(os))
plt.legend()
if save is True:
    if only_order_1 is True:
        plt.savefig(WORKING_DIR + "oversampling_{}/extracted_flux_ref_vs_conv_order1.png".format(os))
    else:
        plt.savefig(WORKING_DIR + "oversampling_{}/extracted_flux_ref_vs_conv.png".format(os))


# Extracted flux [J/s/m²/um]
start, end = 5, -5   # To avoid problems with the extremities

# Extracted flux
if noisy is True:
    plt.figure()
    plt.plot(w[start:end], fbox_noisy_adu[start:end], color="HotPink")
    plt.ylabel(r"Extracted flux [adu/s]")
    plt.xlabel(r"Wavelength [$\mu$m]")
    plt.title(r"Extracted flux of order 1 trace from noisy traces")

    plt.figure()
    plt.plot(w[start:end], fbox_noisy_energy[start:end], color="HotPink")
    plt.xlabel(r"Wavelength [$\mu$m]")
    plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
    plt.title("Extracted flux of order 1 from noisy traces")
else:
    plt.figure()
    plt.plot(w[start:end], fbox_conv_energy_bin[start:end], color='MediumVioletRed')
    plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
    plt.xlabel(r"Wavelength [$\mu$m]")
    plt.title("Extracted flux of noiseless, convolved binned trace")
plt.show()


print("Oversample = {}".format(os))
if only_order_1 is True:
    print("Order 1 only")
else:
    print('Noisy') if noisy else print('Orders 1 & 2 summed')