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
from simu_utils import load_simu

# Imports from the extraction.
from extract.overlap import TrpzOverlap
from extract.throughput import ThroughputSOSS
from extract.convolution import WebbKer

# Imports for plots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For displaying of FITS images.
# TODO astropy has some nice functions for colorbars scaling of astronomical data, might be worth looking into.

plt.rc('figure', figsize=(13,8))
plt.rc('font', size=16)
plt.rc('image', cmap='inferno')
plt.rc('lines', lw=2)

# Read relevant files

# List of orders to consider in the extraction
order_list = [1, 2]

#### Wavelength solution ####
wave_maps = []
wave = fits.getdata("/genesis/jwst/userland-soss/loic_review/refs/map_wave_2D_native.fits")
wave_maps.append(wave[0])
wave_maps.append(wave[1])

#### Spatial profiles ####
spat_pros = []
spat = fits.getdata("/genesis/jwst/userland-soss/loic_review/refs/map_profile_2D_native.fits").squeeze()
spat_pros.append(spat[0,-256:])
spat_pros.append(spat[1,-256:])

# Convert data from fits files to float (fits precision is 1e-8)
wave_maps = [wv.astype('float64') for wv in wave_maps]
spat_pros = [p_ord.astype('float64') for p_ord in spat_pros]

#### Throughputs ####
thrpt_list = [ThroughputSOSS(order) for order in order_list]   # Has been changed to 1 everywhere in throughput.py

#### Convolution kernels ####
ker_list = [WebbKer(wv_map) for wv_map in wave_maps]

# Put all inputs from reference files in a list
ref_files_args = [spat_pros, wave_maps, thrpt_list, ker_list]

# Load simulations
#path.append("Fake_data")

# Load a simulation
#simu = load_simu("Fake_data/phoenix_teff_02300_scale_1.0e+02.fits")
clear_00 = fits.open("/home/kmorel/ongenesis/jwst-user-soss/tmp/oversampling_2/clear_000000.fits")
data = np.empty(shape=(3,256,2048))
for i in range(len(clear_00[0].data)):
    data[i] = soss.rebin(clear_00[0].data[i],simuPars.noversample)
#data = simu["data"]

# Extraction
params = {}

# Map of expected noise (standard deviation).
bkgd_noise = 20.  # In counts?

# Wavelength extraction grid oversampling.
params["n_os"] = 5  # TODO explain a bit more how the grid is determined?
# Answer: I was thinking of explaining all inputs in another notebook or text?
#         Since this parameter is needed for every extraction, I didn't want
#         to re-explain it in all examples. What do you think?

# Threshold on the spatial profile.
# Only pixels above this threshold will be used for extraction.
# (for at least one order)
params["thresh"] = 1e-4  # Same units as the spatial profiles

# Initiate extraction object
# (This needs to be done only once unless the oversampling (n_os) changes.)
extract = TrpzOverlap(*ref_files_args, **params)

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
# TODO sig is the uncertainty on the date here so it might be good to call it that?

# Find the best factor.
best_fac = extract.best_tikho_factor(tests=tests, i_plot=True)

# Refine the grid (span 4 orders of magnitude).
best_fac = np.log10(best_fac)
factors = np.logspace(best_fac-2, best_fac+2, 20)

# No need to specify `data` and `sig` again.
# TODO: why not? Wouldn't it be better to require that to avoid confusion?
# Answer: When a reference file or science file is specified, the class keeps it
#         as an attribute. When an extraction is called, it is updated if specified.
#         It is done to save some text when iterating on the spatial profile, for
#         example, and to save time (some matrix multiplications don't need to be
#         re-computed). But I'm open to discuss it!
tests = extract.get_tikho_tests(factors, data=data, sig=sig)
best_fac = extract.best_tikho_factor(tests=tests, i_plot=True)

# Extract the oversampled spectrum 𝑓𝑘
# Can be done in a loop for a timeseries and/or iteratively for different estimates of the reference files.
# Extract the spectrum.
f_k = extract.extract(data=data, sig=sig, tikhonov=True, factor=best_fac)
# Could we make change this method to __call__?  # Very good idea!

# Plot the extracted spectrum.
plt.plot(extract.lam_grid, f_k)
plt.xlabel("Wavelength [$\mu m$]")
plt.ylabel("Oversampled Spectrum $f_k$ [energy$\cdot s^{-1} \cdot \mu m^{-1}$]")
# For now, arbitrairy units, but it should be the flux that hits the detector, so energy/time/wavelength
plt.tight_layout()
plt.show()

# Bin to pixel native sampling
#
# To get a result comparable to typical extraction methods, we need to integrate the oversampled spectrum (𝑓𝑘
# ) to a grid representative of the native pixel sampling (for each order). This integration is done according to the equation
# bin𝑖=∫𝜆+𝑛𝑖𝜆−𝑛𝑖𝑇𝑛(𝜆)𝑓̃ 𝑛(𝜆)𝜆𝑑𝜆,
# where 𝑛 is a given order, 𝑇𝑛 is the throughput of the order and 𝑓̃ 𝑛 is the underlying flux convolved to the order 𝑛
#
# resolution. The result of this integral will be in fake counts (it is not directly the sum of the counts so that's why I call it fake).
#
# One could directly extract the integrated flux by setting the throughput to 𝑇𝑛(𝜆)=1
# (see second example). The result would then be in flux units instead of counts.

# Bin in counts
# Save the output in a list for different orders.

f_bin_list = []  # Integrated flux.
lam_bin_list = []  # Wavelength grid.

for i_ord in range(extract.n_ord):
    # TODO I think we can make it so we just get the order m=1,2 and never have to deal with an index as well.

    # Integrate.
    lam_bin, f_bin = extract.bin_to_pixel(f_k=f_k, i_ord=i_ord)

    # Save.
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


for i_ord in range(extract.n_ord):
    # TODO I think we can make it so we just get the order m=1,2 and never have to deal with an index as well.

    # Integrate.
    lam_bin, f_bin = extract.bin_to_pixel(f_k=f_k, i_ord=i_ord, throughput=throughput)

    # Plot
    label = extract.orders[i_ord]
    plt.plot(lam_bin_list[i_ord], f_bin_list[i_ord], label=label)

plt.ylabel(r"Convolved flux $\tilde{f_k}$ [energy$\cdot s^{-1} \cdot \mu m^{-1}$]")
plt.xlabel("Wavelength [$\mu m$]")
plt.tight_layout()
plt.legend(title="Order")
plt.show()

# Quality estimate
# Rebuild the detector
rebuilt = extract.rebuild(f_k)

plt.subplot(111, aspect='equal')
plt.pcolormesh((rebuilt - data)/sig, vmin=-3, vmax=3)
plt.colorbar(label="Error relative to noise", orientation='horizontal', aspect=40)
plt.tight_layout()
plt.show()
# We can see that we are very close to the photon noise limit in this case. There are some small
# structures in the 2nd order in the overlap region, but the extracted spectrum is dominated by the
# 1st order in this wavelength region anyway, due to the higher throughput.