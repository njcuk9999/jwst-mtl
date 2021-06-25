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

# Imports from the extraction.
from extract.overlap import TrpzOverlap
from extract.throughput import ThroughputSOSS
from extract.convolution import WebbKer

# Imports for plots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For displaying of FITS images.

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

radius_pixel = 30
length = 240   # Length of window for median filter  # For oscillations

def photon_energy(wl):
    """
    wl: Wavelength in microns
    return: Photon energy in J
    """
    return h * c / (wl * 1e-6)

def dispersion(wl):
    """
    wl: Wavelengths array [um]
    return: Dispersion [um]
    """
    dw = np.zeros_like(wl)
    for i in range(len(wl)):
        if i == 0:
            dw[i] = wl[0] - wl[1]   #The wl array has to be reversed
        else:
            dw[i] = wl[i - 1] - wl[i]
    return dw

def f_lambda(pixels, im_test, wl, y_trace, radius_pixel=radius_pixel, area=area, gain=gain):
    """
    pixels: Array of pixels
    im_test: Trace's image [adu/s]
    wl: Array of wavelengths (same size as pixels)  [um]
    y_trace: Array for the positions of the center of the trace for each column
    radius_pixel: Radius of extraction box [pixels]
    area: Area of photons collection surface [m²]
    gain: Gain [e⁻/adu]
    return: Extracted flux [J/s/m²/um]
    """
    flux = np.zeros_like(pixels)  # Array for extracted spectrum
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        flux[x_i] = im_test[int(first), x_i] * (1 - first % int(first)) + np.sum(
            im_test[int(first) + 1:int(last) + 1, x_i]) + im_test[int(last) + 1, x_i] * (last % int(last))

    # Calculate the flux in J/s/m²/um
    phot_ener = photon_energy(wl)  # Energy of each photon [J/photon]
    dw = dispersion(wl)   #Dispersion [um]

    return flux * gain * phot_ener / area / dw

def flambda_elec(pixels, im_test, y_trace, radius_pixel=radius_pixel, gain=gain):
    """
    pixels: Array of pixels
    im_test: Trace's image [adu/s]
    y_trace: Array for the positions of the center of the trace for each column
    radius_pixel: Radius of extraction box [pixels]
    gain: Gain [e⁻/adu]
    return: Extracted flux [e⁻/s/colonne]
    """
    flux = np.zeros_like(pixels)  # Array for extracted spectrum
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        flux[x_i] = im_test[int(first), x_i] * (1 - first % int(first)) + np.sum(
            im_test[int(first) + 1:int(last) + 1, x_i]) + im_test[int(last) + 1, x_i] * (last % int(last))

    return flux * gain


WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/'

sys.path.insert(0,"/genesis/jwst/jwst-ref-soss/fortran_lib/")

# Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_kim.txt')
pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              #Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars) #read in parameter file


#####################################
# CHOOSE ORDER (for box extraction)   !!!
m_order = 1  # For now, only option is 1.

# CHOOSE OVERSAMPLE  !!!
simuPars.noversample = 4
#####################################


# Read relevant files
# List of orders to consider in the extraction
order_list = [1]  #,2]

#### Wavelength solution ####
# _adb : Antoine's files
wave_maps_adb = []
wave_maps_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/wavelengths_m1.fits"))
wave_maps_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/wavelengths_m2.fits"))
i_zero = np.where(wave_maps_adb[1][0] == 0.)[0][0]   # Where Antoine put 0 in his 2nd order wl map
# _la : Loic's files
wave = fits.getdata("/genesis/jwst/userland-soss/loic_review/refs/map_wave_2D_native.fits")
wave[1,:,i_zero:] = 0.  # Also set to 0 the same points as Antoine in Loic's 2nd order wl map
wave_maps_la = wave[:2]   # Consider only order 1 & 2


#### Spatial profiles ####
# _adb : Antoine's files
spat_pros_adb = []
spat_pros_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/"
                                  "spat_profile_m1.fits").squeeze())
spat_pros_adb.append(fits.getdata("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/extract/Ref_files/"
                                  "spat_profile_m2.fits").squeeze())
# _la : Loic's files
spat = fits.getdata("/genesis/jwst/userland-soss/loic_review/refs/map_profile_2D_native.fits").squeeze()
spat_pros_la = spat[:2,-256:]   # Consider only order 1 & 2 and set to same size as data
# Shift Loic's trace image to fit with the simulation (data)
new_spat_la = np.zeros_like(spat_pros_la)
new_spat_la[:,:162] = spat_pros_la[:,94:]
hdu = fits.PrimaryHDU(new_spat_la)
hdu.writeto(WORKING_DIR + "new_spat_pros.fits", overwrite = True)
# _clear : map created with clear000000.fits directly
new_spat = fits.getdata(WORKING_DIR + "new_map_profile_clear_{}.fits".format(simuPars.noversample))
spat_pros_clear = new_spat[:2]

# Choose between Loic's and Antoine's maps
wave_maps = wave_maps_la[0]   # or wave_maps_adb
spat_pros = spat_pros_clear[0]  # or new_spat_la    # or spat_pros_adb

# Convert data from fits files to float (fits precision is 1e-8)
wave_maps = [wv.astype('float64') for wv in wave_maps]
spat_pros = [p_ord.astype('float64') for p_ord in spat_pros]

#### Throughputs ####
thrpt_list = [ThroughputSOSS(order) for order in order_list]   # Has been changed to 1 everywhere in throughput.py  K.M.

#### Convolution kernels ####
ker_list = [WebbKer(wv_map) for wv_map in wave_maps]

# Put all inputs from reference files in a list
ref_files_args = [spat_pros, wave_maps, thrpt_list, ker_list]

# Load simulations
path.append("Fake_data")

# ADB
#simu = load_simu("/home/kmorel/ongenesis/github/jwst-mtl/SOSS/Fake_data/phoenix_teff_02300_scale_1.0e+02.fits")
#data = simu["data"]
# LA
clear_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_000000.fits".format(simuPars.noversample))
if simuPars.noversample == 1:
    data_clear = clear_00[0].data[:2, 10:266, 10:2058]  # Because of x_padding and y_padding
else:
    data_clear = np.empty(shape=(2,256,2048))
    for i in range(len(clear_00[0].data)-1):
        clear_i = soss.rebin(clear_00[0].data[i], simuPars.noversample)
        data_clear[i] = clear_i[10:266, 10:2058]  # Because of x_padding and y_padding

# CHOOSE between order 1 only or sum of orders 1 & 2
data = data_clear[0]   # Order 1 only [adu/s]
#data = np.sum(data_clear, axis=0)   # Sum all traces 1 & 2 [adu/s]

plt.figure()
plt.imshow(data, origin="lower")
plt.colorbar()
plt.title('Data')
plt.show()

# Extraction
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

# Noise estimate to weigh the pixels.
# Poisson noise + background noise.
sig = np.sqrt(data + bkgd_noise**2)

# Tests all these factors.
tests = extract.get_tikho_tests(factors, data=data, sig=sig)

# Find the best factor.
best_fac = extract.best_tikho_factor(tests=tests, i_plot=True)

# Refine the grid (span 4 orders of magnitude).
best_fac = np.log10(best_fac)
factors = np.logspace(best_fac-2, best_fac+2, 20)

tests = extract.get_tikho_tests(factors, data=data, sig=sig)
best_fac = extract.best_tikho_factor(tests=tests, i_plot=True)

# Extract the oversampled spectrum 𝑓𝑘
# Can be done in a loop for a timeseries and/or iteratively for different estimates of the reference files.
# Extract the spectrum.
f_k = extract.extract(data=data, sig=sig, tikhonov=True, factor=best_fac)
# Could we make change this method to __call__?  # Very good idea!

# Plot the extracted spectrum.
plt.figure()
plt.plot(extract.lam_grid, f_k)
plt.xlabel("Wavelength [$\mu m$]")
plt.ylabel("Oversampled Spectrum $f_k$ [energy$\cdot s^{-1} \cdot \mu m^{-1}$]")
# For now, arbitrairy units, but it should be the flux that hits the detector, so energy/time/wavelength
plt.tight_layout()
plt.savefig(WORKING_DIR + "oversampling_{}/oversampled_spectrum_adb_order1.png".format(simuPars.noversample))
plt.show()

# Bin to pixel native sampling
#
# To get a result comparable to typical extraction methods, we need to integrate the oversampled spectrum (𝑓𝑘
# ) to a grid representative of the native pixel sampling (for each order). This integration is done according to the equation
# bin𝑖=∫𝜆+𝑛𝑖𝜆−𝑛𝑖𝑇𝑛(𝜆)𝑓̃ 𝑛(𝜆)𝜆𝑑𝜆,
# where 𝑛 is a given order, 𝑇𝑛 is the throughput of the order and 𝑓̃ 𝑛 is the underlying flux convolved to the order 𝑛
#
# resolution. The result of this integral will be in fake counts (it is not directly the sum of the counts
# so that's why I call it fake).
#
# One could directly extract the integrated flux by setting the throughput to 𝑇𝑛(𝜆)=1
# (see second example). The result would then be in flux units instead of counts.

# Bin in counts
# Save the output in a list for different orders.

f_bin_list = []  # Integrated flux.
lam_bin_list = []  # Wavelength grid.

for i_ord in range(extract.n_ord):

    # Integrate.
    lam_bin, f_bin = extract.bin_to_pixel(f_k=f_k, i_ord=i_ord)

    # Save.
    f_bin_list.append(f_bin)
    lam_bin_list.append(lam_bin)

#fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 6))

for i_ord in range(extract.n_ord):
    label = extract.orders[i_ord]
    ax[i_ord].plot(lam_bin_list[i_ord], f_bin_list[i_ord], label=label)
#ax.plot(lam_bin_list[0], f_bin_list[0])

ax[0].set_ylabel("Extracted signal [counts]")
ax[0].set_xlabel("Wavelength [$\mu m$]")
#ax[1].set_xlabel("Wavelength [$\mu m$]")
#ax[1].set_ylabel("Extracted signal [counts]")

plt.tight_layout()
plt.savefig(WORKING_DIR + "oversampling_{}/extracted_signal_adb_order1.png".format(simuPars.noversample))
plt.show()

# Bin in flux units
# Set throughput to 1
def throughput(x):
    return np.ones_like(x)

plt.figure()
for i_ord in range(extract.n_ord):

    # Integrate.
    lam_bin, f_bin = extract.bin_to_pixel(f_k=f_k, i_ord=i_ord, throughput=throughput)

    # Plot
    label = extract.orders[i_ord]
    plt.plot(lam_bin_list[i_ord], f_bin_list[i_ord], label=label)

plt.ylabel(r"Convolved flux $\tilde{f_k}$ [energy$\cdot s^{-1} \cdot \mu m^{-1}$]")
plt.xlabel("Wavelength [$\mu m$]")
plt.tight_layout()
plt.legend(title="Order")
plt.savefig(WORKING_DIR + "oversampling_{}/convolved_flux_adb_order1.png".format(simuPars.noversample))
plt.show()

# Quality estimate
# Rebuild the detector
rebuilt = extract.rebuild(f_k)

plt.subplot(111, aspect='equal')
plt.pcolormesh((rebuilt - data)/sig, vmin=-3, vmax=3)
plt.colorbar(label="Error relative to noise", orientation='horizontal', aspect=40)
plt.tight_layout()
plt.savefig(WORKING_DIR + "oversampling_{}/rebuild_adb_order1.png".format(simuPars.noversample))
plt.show()
# We can see that we are very close to the photon noise limit in this case. There are some small
# structures in the 2nd order in the overlap region, but the extracted spectrum is dominated by the
# 1st order in this wavelength region anyway, due to the higher throughput.