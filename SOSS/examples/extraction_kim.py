# Box extraction
# All extractions are for order 1

import SOSS.trace.tracepol as tp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
import scipy.constants as sc_cst
from scipy.interpolate import interp1d

import box_kim

# Constants
h = sc_cst.Planck
c = sc_cst.speed_of_light
gain = 1.6
area = 25.
ng = 3   # NGROUP
t_read = 5.49   # Reading time [s]
tint = (ng - 1) * t_read   # Integration time [s]

radius_pixel = 30
length_med = 85    # Length of window for median filter  # For oscillations

start, end = 5, -5   # To avoid problems with the extremities

#########################################################
WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/'

#sys.path.insert(0,"/home/kmorel/ongenesis/github/jwst-mtl/SOSS/specgen/utils/")
sys.path.insert(0, "/genesis/jwst/jwst-ref-soss/fortran_lib/")

# Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_kim.txt')
pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              #Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars) #read in parameter file

#########################################################
# CHOOSE ORDER   !!!
m_order = 1  # For now, only option is 1.

# CHOOSE OVERSAMPLE  !!!
simuPars.noversample = 4
os = simuPars.noversample

# SAVE FIGS? !!!
save = True

#########################################################
# Generate or read the star atmosphere model
starmodel_angstrom, starmodel_flambda, ld_coeff = soss.starmodel(simuPars, pathPars)

#########################################################
# Position of trace
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

xnew, y, mask = tp.wavelength_to_pix(w, pars, m=1)   # Converts wavelenghths to pixel coordinates  NOT GOOD
xnew_os, y_os, mask_os = tp.wavelength_to_pix(w_os , pars, m=1, oversample=os)  # Converts wavelenghths to pixel coordinates, os  NOT GOOD

#########################################################
# LOADING IMAGES
clear_tr_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_trace_000000.fits".format(os))
clear_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_000000.fits".format(os))
noisy_rateints = fits.open(WORKING_DIR + "oversampling_{}/test_clear_noisy_rateints.fits".format(os))

# Because of x_padding and y_padding
padd = 10
padd_os = padd * os

clear_tr_ref = clear_tr_00[0].data[:2, padd_os:-padd_os, padd_os:-padd_os]   # Reference thin trace, not binned
clear_conv = clear_00[0].data[:, padd_os:-padd_os, padd_os:-padd_os]    # Convolved traces, not binned
if os != 1:
    # Bin to pixel native
    ref_i = soss.rebin(clear_tr_00[0].data, os, flux_method='sum')
    clear_ref_bin = ref_i[:2, padd:-padd, padd:-padd]    # Reference thin trace, binned
    clear_i = soss.rebin(clear_00[0].data, os, flux_method='sum')
    clear_bin = clear_i[:2, padd:-padd, padd:-padd]      # Convolved traces, binned
else:  # Os = 1
    clear_ref_bin = np.copy(clear_tr_ref)
    clear_bin = np.copy(clear_conv)


# IMAGES ARRAYS
# Noiseless
# Order 1 only
data_ref1 = clear_tr_ref[m_order-1]         # Reference thin trace, not binned [adu/s]
data_ref1_bin = clear_ref_bin[m_order-1]    # Reference thin trace, binned [adu/s]
data_conv1 = clear_conv[m_order-1]          # Convolved, not binned [adu/s]
data_conv1_bin = clear_bin[m_order-1]       # Convolved, binned [adu/s]
# Orders 1 & 2 summed
data_ref12 = np.sum(clear_tr_ref, axis=0)        # Reference thin trace, not binned [adu/s]
data_ref12_bin = np.sum(clear_ref_bin, axis=0)   # Reference thin trace, binned [adu/s]
data_conv12 = np.sum(clear_conv, axis=0)         # Convolved, not binned [adu/s]
data_conv12_bin = np.sum(clear_bin, axis=0)      # Convolved, binned [adu/s]

# Noisy
data_noisy = noisy_rateints[1].data[0]   # Image of flux [adu/s]
delta_noisy = noisy_rateints[2].data[0]  # Error [adu/s]
dq = noisy_rateints[3].data[0]           # Data quality
i = np.where(dq %2 != 0)   # Odd values of dq = DO NOT USE these pixels
data_noisy[i[0], i[1]] = 0
delta_noisy[i[0], i[1]] = 0

#########################################################
# EXTRACTIONS

# EXTRACTED FLUX OF NOISELESS TRACE, ORDER 1 ONLY
# Reference thin trace:
fbox_ref1_adu = box_kim.flambda_adu(x_os, data_ref1, y_os, radius_pixel=5)       # Not binned [adu/s]
fbox_ref1_adu_bin = box_kim.flambda_adu(x, data_ref1_bin, y, radius_pixel=5) / os    # Binned [adu/s]

# Convolved trace, not binned
fbox_conv1_inf_adu = box_kim.flambda_inf_radi_adu(data_conv1)                  # Infinite radius [adu/s]
fbox_conv1_adu = box_kim.flambda_adu(x_os, data_conv1, y_os, radius_pixel=radius_pixel * os)   # [adu/s]

# Convolved trace, binned
fbox_conv1_inf_adu_bin = box_kim.flambda_inf_radi_adu(data_conv1_bin) / os           # Infinite radius [adu/s]
fbox_conv1_adu_bin = box_kim.flambda_adu(x, data_conv1_bin, y, radius_pixel=radius_pixel) / os       # [adu/s]
fbox_conv1_elec_bin = box_kim.flambda_elec(x, data_conv1_bin, y, radius_pixel=radius_pixel) / os     # [e⁻]
fbox_conv1_energy_bin = box_kim.f_lambda(x, data_conv1_bin, w, y, radius_pixel=radius_pixel) / os    # [J/s/m²/um]
fbox_conv1_inf_ener_bin = box_kim.flambda_inf_radi_ener(data_conv1_bin, w)           # Infinite radius [J/s/m²/um]


# EXTRACTED FLUX OF NOISELESS TRACES, ORDERS 1 & 2 SUMMED
# Reference thin trace:
fbox_ref12_adu = box_kim.flambda_adu(x_os, data_ref12, y_os, radius_pixel=5)       # Not binned [adu/s]
fbox_ref12_adu_bin = box_kim.flambda_adu(x, data_ref12_bin, y, radius_pixel=5) / os    # Binned [adu/s]

# Convolved traces, not binned
fbox_conv12_adu = box_kim.flambda_adu(x_os, data_conv12, y_os, radius_pixel=radius_pixel * os)   # [adu/s]

# Convolved traces, binned
fbox_conv12_adu_bin = box_kim.flambda_adu(x, data_conv12_bin, y, radius_pixel=radius_pixel) / os        # [adu/s]
fbox_conv12_elec_bin = box_kim.flambda_elec(x, data_conv12_bin, y, radius_pixel=radius_pixel) / os      # [e⁻]
fbox_conv12_energy_bin = box_kim.f_lambda(x, data_conv12_bin, w, y, radius_pixel=radius_pixel) / os     # [J/s/m²/um]


# EXTRACTED FLUX OF NOISY TRACES
fbox_noisy_adu = box_kim.flambda_adu(x, data_noisy, y, radius_pixel=radius_pixel) / os       # [adu/s]
fbox_noisy_elec = box_kim.flambda_elec(x, data_noisy, y, radius_pixel=radius_pixel) / os     # [e⁻]
fbox_noisy_energy = box_kim.f_lambda(x, data_noisy, w, y, radius_pixel=radius_pixel) / os    # [J/s/m²/um]

# Error on extracted flux
#sigma_fbox_noisy_ener = box_kim.sigma_flambda(x, delta_noisy, w, y, radius_pixel=radius_pixel) # [J/s/m²/um]

#########################################################
"""
# WHEN THERE WERE OSCILLATIONS...

# Array of wavelengths when using filter [um]
w_median = box_kim.wl_filter(w, x, length=length_med)
w_median_os = box_kim.wl_filter(w_os, x_os, length=length_med)

# MEDIAN FILTER
print('With median filter:')
# Reference thin trace, not binned [adu/s]
fbox_ref1_adu_med = box_kim.median_filter(fbox_ref1_adu, length=length_med)  # Application of median filter [adu/s]
fbox_ref1_med_norm = fbox_ref1_adu[length_med//2:-length_med//2] / fbox_ref1_adu_med   # Normalization with median
std_ref1_med_norm = np.std(fbox_ref1_med_norm)                                         # Standard deviation
print("Standard deviation (reference thin trace [adu/s], not binned, normalized) =", std_ref1_med_norm)

# Convolved trace 1, not binned [adu/s]
fbox_conv1_inf_adu_med = box_kim.median_filter(fbox_conv1_inf_adu, length=length_med)  # Application of median filter [adu/s]
fbox_conv1_inf_med_norm = fbox_conv1_inf_adu[length_med//2:-length_med//2] / fbox_conv1_inf_adu_med   # Normalization with median
std_conv1_inf_med_norm = np.std(fbox_conv1_inf_med_norm[start*os*3:])                                   # Standard deviation
print("Standard deviation (noiseless convolved, not binned order 1 [adu/s], infinite radius, normalized) =", std_conv1_inf_med_norm)

# Convolved trace 1, binned [adu/s]
fbox_conv1_adu_bin_med = box_kim.median_filter(fbox_conv1_adu_bin, length=length_med)  # Application of median filter [adu/s]
fbox_conv1_bin_med_norm = fbox_conv1_adu_bin[length_med//2:-length_med//2] / fbox_conv1_adu_bin_med   # Normalization with median
std_conv1_bin_med_norm = np.std(fbox_conv1_bin_med_norm[start:])                                       # Standard deviation
print("Standard deviation (noiseless convolved, binned order 1 [adu/s], radius=",radius_pixel,", normalized) =", std_conv1_bin_med_norm)

# Convolved trace 1, binned [J/s/m²/um]
fbox_conv1_inf_ener_bin_med = box_kim.median_filter(fbox_conv1_inf_ener_bin, length=length_med)  # Application of median filter [J/s/m²/um]
fbox_conv1_inf_ener_bin_med_norm = fbox_conv1_inf_ener_bin[length_med//2:-length_med//2] / fbox_conv1_inf_ener_bin_med   # Normalization with median
std_conv1_inf_ener_bin_med_norm = np.std(fbox_conv1_inf_ener_bin_med_norm[start:])                                       # Standard deviation
print("Standard deviation (noiseless convolved, binned order 1 [J/s/m²/um], normalized) =", std_conv1_inf_ener_bin_med_norm)

# Convolved traces 1 & 2, binned
fbox_conv12_adu_bin_med = box_kim.median_filter(fbox_conv12_adu_bin, length=length_med)  # Application of median filter [adu]
fbox_conv12_bin_med_norm = fbox_conv12_adu_bin[length_med//2:-length_med//2] / fbox_conv12_adu_bin_med   # Normalization with median
std_conv12_bin_med_norm = np.std(fbox_conv12_bin_med_norm[start:])                                    # Standard deviation
print("Standard deviation (noiseless orders 1 & 2 summed, normalized) =", std_conv12_bin_med_norm)

# Noisy traces
fbox_noisy_adu_med = box_kim.median_filter(fbox_noisy_adu, length=length_med)     # Application of median filter [adu]
fbox_noisy_med_norm = fbox_noisy_adu[length_med//2:-length_med//2] / fbox_noisy_adu_med   # Normalization
std_noisy_med_norm = np.std(fbox_noisy_med_norm)                                      # Standard deviation
print("Standard deviation (noisy, normalized) =", std_noisy_med_norm)

#########################################################
"""

"""
#########################################################
# ESTIMATED PHOTON NOISES ON CONVOLVED, BINNED TRACE(S)
dw = box_kim.dispersion(w)             # Dispersion [um]
phot_ener = box_kim.photon_energy(w)   # Energy of photons [J]

# Convolved, binned order 1
sigma_noise_conv1_bin = np.sqrt(fbox_conv1_elec_bin)                                # Photon noise [e⁻]
sigma_noise_conv1_bin_ener = sigma_noise_conv1_bin * phot_ener / dw / area / tint   # Photon noise [J/s/m²/um]

# Convolved, binned orders 1 & 2 summed
sigma_noise_conv12_bin = np.sqrt(fbox_conv12_elec_bin)                                # Photon noise [e⁻]
sigma_noise_conv12_bin_ener = sigma_noise_conv12_bin * phot_ener / dw / area / tint   # Photon noise [J/s/m²/um]

# For plots
fbox_conv1_energy_bin_minus = fbox_conv1_energy_bin - sigma_noise_conv1_bin_ener
fbox_conv1_energy_bin_plus = fbox_conv1_energy_bin + sigma_noise_conv1_bin_ener
fbox_conv12_energy_bin_minus = fbox_conv12_energy_bin - sigma_noise_conv12_bin_ener
fbox_conv12_energy_bin_plus = fbox_conv12_energy_bin + sigma_noise_conv12_bin_ener
#########################################################
"""

#########################################################
# COMPARISONS
# Relative difference
relatdiff_conv_energy_bin = box_kim.relative_difference(fbox_conv12_energy_bin, fbox_conv1_energy_bin)
relatdiff_noisy_energy = box_kim.relative_difference(fbox_noisy_energy, fbox_conv1_energy_bin)

#########################################################
print("Oversample = {}".format(os))
print('Radius pixel = ', radius_pixel)

#########################################################
# GRAPHICS

# Matplotlib defaults
plt.rc('figure', figsize=(12, 7))
plt.rc('font', size=14)
plt.rc('lines', lw=2)

#"""
# Images of traces
plt.figure()
plt.imshow(data_noisy, vmin=0, origin="lower")
plt.plot(x, y, color="r", lw=1, label="Order 1 trace's position")
plt.colorbar(label="[adu/s]", orientation='horizontal')
plt.title("test_clear_noisy_rateints.fits")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/noisy_rateints.png".format(os))
plt.show()

plt.figure()
plt.imshow(data_conv1_bin, origin="lower")
plt.plot(x, y, color="r", lw=1, label="Order 1 trace's position")
plt.colorbar(label="[adu/s]", orientation='horizontal')
plt.title("clear_000000.fits, order 1")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/clear1_000000.png".format(os))
plt.show()

plt.figure()
plt.imshow(data_conv12_bin, origin="lower")
plt.plot(x, y, color="r", lw=1, label="Order 1 trace's position")
plt.colorbar(label="[adu/s]", orientation='horizontal')
plt.title("clear_000000.fits, orders 1 & 2")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/clear12_000000.png".format(os))
plt.show()
#"""

# Extracted flux [J/s/m²/um]

plt.figure()
plt.plot(w[start:end], fbox_noisy_energy[start:end], color="HotPink")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.title("Extracted flux of order 1 from noisy traces")
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_noisy_energy.png".format(os))
plt.show()

plt.figure()
plt.plot(w[start:end], fbox_conv1_energy_bin[start:end], color='r', label = "Order 1 only")
plt.plot(w[start:end], fbox_conv12_energy_bin[start:end], color='b', label="Orders 1 & 2 summed")
#plt.fill_between(w[start:end], fbox_conv1_energy_bin_minus[start:end], fbox_conv1_energy_bin_plus[start:end],
#                 alpha=0.4, color='r')
#plt.fill_between(w[start:end], fbox_conv12_energy_bin_minus[start:end], fbox_conv12_energy_bin_plus[start:end],
#                 alpha=0.4, color='b')
#plt.errorbar(w[start:end], fbox_conv1_energy_bin[start:end], yerr=sigma_noise_conv1_bin_ener[start:end], lw=1,
#             elinewidth=1, color="r", ecolor='r', label = "Order 1 only")
#plt.errorbar(w[start:end], fbox_conv12_energy_bin[start:end], yerr=sigma_noise_conv12_bin_ener[start:end], lw=1,
#             elinewidth=1, color="b", ecolor='b', label="Orders 1 & 2 summed")
plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title('Extracted flux of order 1 from noiseless trace(s)')
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_conv_energy_bin.png".format(os))
plt.show()

plt.figure()
plt.plot(w[start:end], fbox_conv1_inf_ener_bin[start:end], color="Purple")
plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title("Extracted flux with infinite radius of noiseless order 1 trace")
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_conv1_inf_ener_bin.png".format(os))
plt.show()

"""
# Model
plt.figure()
plt.plot(starmodel_angstrom, starmodel_flambda, color='Aqua')
plt.xlabel(r"Wavelength [angstrom]")
plt.ylabel(r"Flux")
plt.title("Model")
plt.show()
"""

# Extracted flux [adu/s]
# Median filter
plt.figure()
plt.plot(w_os[start:end], fbox_ref1_adu[start:end], color="HotPink", label="Data", zorder=0)
#plt.plot(w_median_os, fbox_ref1_adu_med, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Extracted flux [adu/s]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux of reference thin trace, order 1 not binned")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_ref1_adu.png".format(os))
plt.show()

plt.figure()
plt.plot(w_os[start:end], fbox_conv1_inf_adu[start:end], color="HotPink", label="", zorder=0)
plt.plot(w_os[start:end], fbox_ref1_adu[start:end], color="HotPink", label="Ref", zorder=0)
#plt.plot(w_median_os, fbox_conv1_inf_adu_med, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Extracted flux [adu/s]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux of noiseless, convolved, not binned order 1 trace (rad. = inf)")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_conv1_inf_adu.png".format(os))
plt.show()

plt.figure()
plt.plot(w[start:end], fbox_conv1_adu_bin[start:end], color="HotPink", label="Data", zorder=0)
#plt.plot(w_median, fbox_conv1_adu_bin_med, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Extracted flux [adu/s]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux of noiseless, convolved, binned order 1 trace")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_conv1_bin_adu.png".format(os))
plt.show()

"""
# When there were oscillations
plt.figure()
plt.plot(w_median_os, fbox_ref1_med_norm, color="Blue", label='Reference thin trace, order 1 not binned', zorder=0)
plt.plot(w_median_os[start*os*3:], fbox_conv1_inf_med_norm[start*os*3:], ls='--',color="LimeGreen",
         label='Noiseless, convolved, not binned order 1 trace (rad.=inf)', zorder=5)
plt.plot(w_median[start:], fbox_conv1_bin_med_norm[start:], ls='--', color="OrangeRed",
         label='Noiseless, convolved, binned order 1 trace (rad.={})'.format(radius_pixel), zorder=10)
plt.ylabel(r"Normalized flux")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux normalized by median filter")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_1_med_norm.png".format(os))
plt.show()
"""

plt.figure()
plt.plot(w[start:end], fbox_conv1_inf_ener_bin[start:end], color="HotPink", label="Data", zorder=0)
#plt.plot(w_median, fbox_conv1_inf_ener_bin_med, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Extracted flux [J/s/m²/um]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux of noiseless, convolved, binned order 1 trace")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_conv1_ener_bin.png".format(os))
plt.show()

"""
# When there were oscillations
plt.figure()
plt.plot(w_median[start:], fbox_conv1_inf_ener_bin_med_norm[start:], color="Lime")
plt.ylabel(r"Normalized flux")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux normalized by median filter of noiseless, convolved, binned order 1 trace")
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_conv1_ener_bin_med_norm.png".format(os))
plt.show()
"""

"""
plt.figure()
plt.plot(w[start:end], fbox_conv12_adu_bin[start:end], color="HotPink", label="Data", zorder=0)
#plt.plot(w_filter, fbox_conv12_adu_bin_med, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Extracted flux [adu/s]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux of noiseless, convolved, binned orders 1 & 2 traces summed")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_conv12_adu_bin.png".format(os))
plt.show()

# When there were oscillations
plt.figure()
plt.plot(w_filter[start:], fbox_conv12_bin_med_norm[start:], color="Lime")
plt.ylabel(r"Normalized flux")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux normalized by median filter of noiseless, convolved, binned orders 1 & 2 traces summed")
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_conv12_bin_med_norm.png".format(os))
plt.show()
"""

plt.figure()
plt.plot(w[start:end], fbox_noisy_adu[start:end], color="HotPink", label="Data", zorder=0)
#plt.plot(w_median, fbox_noisy_adu_med, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Extracted flux [adu/s]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux from noisy traces")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_noisy_adu.png".format(os))
plt.show()

"""
# When there were oscillations
plt.figure()
plt.plot(w_median, fbox_noisy_med_norm, color="Lime")
plt.ylabel(r"Normalized flux")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux normalized by median filter of noisy traces")
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/fbox_noisy_med_norm.png".format(os))
plt.show()
"""

# Comparisons
plt.figure()
plt.plot(w[start:end], relatdiff_conv_energy_bin[start:end], color="HotPink")
plt.ylabel("Relative difference")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title('Relative difference between extracted flux for convolved, binned traces: \nOrders 1 & 2 '
          'summed vs order 1, noiseless')
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/relatdiff_conv_energy_bin.png".format(os))
plt.show()

plt.figure()
plt.plot(w[start:end], relatdiff_noisy_energy[start:end], color="Green")
plt.ylabel("Relative difference")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title('Relative difference between extracted flux for convolved, binned traces: \nNoisy vs order 1 noiseless')
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/relatdiff_noisy_energy.png".format(os))
plt.show()