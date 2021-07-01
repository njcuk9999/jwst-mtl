# Box extraction

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
length = 240   # Length of window for median filter  # For oscillations


def median_window(flux, start, length):
    """
    flux: Extracted flux
    start: Start point of window for which to compute median
    length: Length of window for which to compute median
    return: Median
    """
    list = flux[start : start + length]
    return np.median(list)

def median_filter(flux, length = length):
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

def wl_median(wl, pixels, length=length):
    """
    wl: Wavelengths array
    pixels: Pixels array (same size as wl)
    length: Length of window for which to compute median
    return: New wavelengths array matching median filtered flux
    """
    if length%2==1:
        w_med = wl[length//2:-length//2]
    elif length%2==0:
        pixels_new = pixels - 0.5
        pixels_new[0] = 0
        f = interp1d(pixels,wl)
        w_median = f(pixels_new)
        w_med = w_median[length//2:-length//2]
    return w_med


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


#####################################
# CHOOSE ORDER   !!!
m_order = 1  # For now, only option is 1.

# CHOOSE OVERSAMPLE  !!!
simuPars.noversample = 2
os = simuPars.noversample

# SAVE FIGS? !!!
save = False
#####################################


# Generate or read the star atmosphere model
starmodel_angstrom, starmodel_flambda, ld_coeff = soss.starmodel(simuPars, pathPars)


# Position of trace
trace_file = "/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv"
x = np.linspace(0, 2047, 2048)    # Array of pixels
pars = tp.get_tracepars(trace_file)   # Gives the middle position of oorder 1 trace
w, tmp = tp.specpix_to_wavelength(x,pars,m=1)   # Returns wavelength for each x, order 1
xnew, y, mask = tp.wavelength_to_pix(w, pars, m=1)   # Converts wavelenghths to pixel coordinates

### x and xnew should be the same!...
#ynew = np.interp(x, xnew, y)    #Interpolation of y at integer values of x
#ynew = np.around(ynew)


# Loading images
padd = 10
padd_os = padd * os   # Because of x_padding and y_padding
noisy_rateints = fits.open(WORKING_DIR + "oversampling_{}/test_clear_noisy_rateints.fits".format(os))
clear_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_000000.fits".format(os))

clear_conv = clear_00[0].data[:, padd_os:-padd_os, padd_os:-padd_os]    # Convolved traces, not binned
if os != 1:
    # Bin to pixel native
    clear_i = soss.rebin(clear_00[0].data, os)
    clear = clear_i[:, padd:-padd, padd:-padd]
else:
    clear = np.copy(clear_conv)


# Images
# With noise
im_adu_noisy = noisy_rateints[1].data[0]  # Image of flux [adu/s]
delta = noisy_rateints[2].data[0]   # Error [adu/s]
dq = noisy_rateints[3].data[0]  # Data quality
# Odd values of dq = DO NOT USE these pixels
i = np.where(dq %2 != 0)
im_adu_noisy[i[0], i[1]] = 0
delta[i[0], i[1]] = 0

# Without noise
# Clear order 1 only
m1_clear_adu = clear[m_order-1]  # [adu/s]

# All clear traces added
tot_clear_adu = np.sum(clear, axis=0)   # Sum all traces [adu/s]


# EXTRACTIONS
# Extracted flux of noisy image
flamb_noisy_energy = box_kim.f_lambda(x, im_adu_noisy, w, y, radius_pixel=radius_pixel)   # Extracted flux [J/s/m²/um]
sigma_flamb_noisy_ener = box_kim.sigma_flambda(x, delta, w, y, radius_pixel=radius_pixel)   # Error on extracted flux [J/s/m²/um]
flamb_noisy_elec = box_kim.flambda_elec(x, im_adu_noisy, y, radius_pixel=radius_pixel) * tint  # Extracted flux in electrons [e⁻/colonne]

# Extracted flux of clear order 1 trace only
flamb_m1clear_energy = box_kim.f_lambda(x, m1_clear_adu, w, y, radius_pixel=radius_pixel)   # Extracted flux [J/s/m²/um]
flamb_m1_inf_radi_ener = box_kim.flambda_inf_radi_ener(m1_clear_adu, w)   # Extracted flux with infinite radius [J/s/m²/um]
flamb_m1clear_elec = box_kim.flambda_elec(x, m1_clear_adu, y, radius_pixel=radius_pixel) * tint  # Extracted flux in electrons [e⁻/colonne]
elec_noise_m1 = np.sqrt(flamb_m1clear_elec)   # Photon noise [e⁻/colonne]

# Extracted flux of order 1 from sum of clear traces
flamb_totclear_energy = box_kim.f_lambda(x, tot_clear_adu, w, y, radius_pixel=radius_pixel)   # Extracted flux [J/s/m²/micron]
flamb_totclear_elec = box_kim.flambda_elec(x, tot_clear_adu, y, radius_pixel=radius_pixel) * tint  # Extracted flux in electrons [e⁻/colonne]
elec_noise_tot = np.sqrt(flamb_totclear_elec)   # Photon noise [e⁻/colonne]

"""
# When there were oscillations :
w_median = wl_median(w, x)  # Array of wavelengths when using median filter [um]

flamb_noisy_median = median_filter(flamb_noisy_elec)  # Application of median filter [e⁻/colonne]
flamb_noisy_normal = flamb_noisy_elec[length//2:-length//2] / flamb_noisy_median   # Normalization
std_noisy_normal = np.std(flamb_noisy_normal)   # Standard deviation
print("Standard deviation (noisy, normalized) =", std_noisy_normal)

flamb_m1clear_median = median_filter(flamb_m1clear_elec)  # Application of median filter [e⁻/colonne]
flamb_m1clear_normal = flamb_m1clear_elec[length//2:-length//2] / flamb_m1clear_median   # Normalization with median
std_m1clear_normal = np.std(flamb_m1clear_normal)   # Standard deviation
print("Standard deviation (clear order 1 only) =", std_m1clear_normal)

flamb_totclear_median = median_filter(flamb_totclear_elec)  # Application of median filter [e⁻/colonne]
flamb_totclear_normal = flamb_totclear_elec[length//2:-length//2] / flamb_totclear_median   # Normalization with median
std_totclear_normal = np.std(flamb_totclear_normal)   # Standard deviation
print("Standard deviation (order 1 from total clear traces) =", std_totclear_normal)

# Standard deviations
std_noisy_os1 = 0.026461379942256263
std_m1clear_os1 = 0.00019035284694652238
std_totclear_os1 = 0.0001903628002432131
std_noisy_os2 = 0.02686251435682784
std_m1clear_os2 = 0.0002453873999317287
std_totclear_os2 = 0.00024538627484775767
std_noisy_os4 = 0.0269058766168865
std_m1clear_os4 = 0.0001430331995948216
std_totclear_os4 = 0.00014305414511618802
std_noisy_os10 = 0.027189003044306014
std_m1clear_os10 = 0.00027169484098796135
std_totclear_os10 = 0.00027168470248692937
os = np.array([1,2,4,10])
std_noisy = np.array([std_noisy_os1, std_noisy_os2, std_noisy_os4, std_noisy_os10])
std_m1clear = np.array([std_m1clear_os1, std_m1clear_os2, std_m1clear_os4, std_m1clear_os10])
std_totclear = np.array([std_totclear_os1, std_totclear_os2, std_totclear_os4, std_totclear_os10])
"""

# Estimated photon noises
dw = box_kim.dispersion(w)   # Dispersion [um]
phot_ener = box_kim.photon_energy(w)   # Energy of photon [J]

# Order 1 trace only
photon_noise_m1 = flamb_m1clear_energy * tint * dw * area / phot_ener
sigma_noise_m1 = np.sqrt(photon_noise_m1)   # Photon noise [photons/colonne]
sigma_noisem1_ener = sigma_noise_m1 * phot_ener / dw / area / tint   # Photon noise [J/s/m²/micron]

# Order 1 of total clear traces
photon_noise_tot = flamb_totclear_energy * tint * dw * area / phot_ener
sigma_noise_tot = np.sqrt(photon_noise_tot)   # Photon noise [photons/colonne]
sigma_noisetot_ener = sigma_noise_tot * phot_ener / dw / area / tint   # Photon noise [J/s/m²/micron]
"""
# For plots
flamb_m1_clear_minus = flamb_m1clear_energy - sigma_noisem1_ener
flamb_m1_clear_plus = flamb_m1clear_energy + sigma_noisem1_ener
flamb_tot_clear_minus = flamb_totclear_energy - sigma_noisetot_ener
flamb_tot_clear_plus = flamb_totclear_energy + sigma_noisetot_ener
"""

"""
# Comparisons
diff_m1 = flamb_totclear_energy - flamb_m1clear_energy  # Difference between order 1 only and 
                                                        # order 1 of all traces [J/s/m²/micron]
sigma_diff_m1 = np.sqrt(sigma_noisetot_ener**2 + sigma_noisem1_ener**2) # Sigma of the difference [J/s/m²/micron]

# Relative flux
relat_flux_clear = diff_m1 / flamb_m1clear_energy  
sigma_relatflux_clear = np.sqrt( (sigma_noisetot_ener / flamb_m1_clear)**2 + 
                                 (sigma_noisem1_ener * flamb_totclear_energy / flamb_m1_clear**2)**2 )

diff_noisy = flamb_totclear_energy - flamb_noisy_energy   # Difference between order 1 of all traces and 
                                                          # order 1 of noisy traces
sigma_diff_noisy = np.sqrt(sigma_noisetot_ener**2 + sigma_flamb_noisy_ener**2)  # Sigma of the difference [J/s/m²/micron]

# Relative flux
relat_flux_noisy = diff_noisy / flamb_noisy_energy  
sigma_relatflux_noisy = np.sqrt( (sigma_noisetot_ener / flamb_noisy_energy)**2 + 
                                 (sigma_flamb_noisy_ener * flamb_totclear_energy / flamb_noisy_energy**2)**2 )
"""

print("Oversample = {}".format(os))

##########################
# GRAPHICS

# Matplotlib defaults
plt.rc('figure', figsize=(12, 7))
plt.rc('font', size=14)

# Images of traces
plt.figure()
plt.imshow(im_adu_noisy, vmin=0, origin="lower")   # Image of noisy traces
plt.plot(x, y, color="r", label="Order 1 trace's position")   # Middle position of order 1 trace
plt.colorbar(label="[adu/s]", orientation='horizontal')
plt.title("test_clear_noisy_rateints.fits")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/noisy_rateints_imshow.png".format(os))
plt.show()

plt.figure()
plt.imshow(m1_clear_adu, origin="lower")   # Image of clear order 1 trace
plt.plot(x, y, color="r", label="Order 1 trace's position")   # Middle position of order 1 trace
plt.colorbar(label="[adu/s]", orientation='horizontal')
plt.title("clear_000000.fits, order 1 only")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/clearm1_imshow.png".format(os))
plt.show()

plt.figure()
plt.imshow(tot_clear_adu, origin="lower")   # Image of clear order 1 traces
plt.plot(x, y, color="r", label="Order 1 trace's position")   # Middle position of order 1 trace
plt.colorbar(label="[adu/s]", orientation='horizontal')
plt.title("clear_000000.fits, all orders")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/cleartot_imshow.png".format(os))
plt.show()


# Extracted flux [J/s/m²/um]
start, end = 5, -5   # To avoid problems with the extremities

plt.figure()
plt.errorbar(w[start:end], flamb_noisy_energy[start:end], yerr=sigma_flamb_noisy_ener[start:end], lw=1, elinewidth=1,
             color="HotPink", ecolor='r')
plt.xlabel(r"Wavelength [$\mu$m]")
plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.title("Extracted flux of order 1 from noisy traces")
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/flamb_noisy_energy.png".format(os))
plt.show()

plt.figure()
plt.plot(w[start:end], flamb_m1clear_energy[start:end], lw=1, color='r',
         label = "Extracted flux of clear order 1 trace")
plt.plot(w[start:end], flamb_totclear_energy[start:end], lw=1, color='b',
         label="Extracted flux of order 1 from all clear traces")
#plt.fill_between(w[start:end], flamb_m1_clear_minus[start:end], flamb_m1_clear_plus[start:end], alpha=0.4, color='r')
#plt.fill_between(w[start:end], flamb_tot_clear_minus[start:end], flamb_tot_clear_plus[start:end], alpha=0.4, color='b')
#plt.errorbar(w[start:end], flamb_m1clear_energy[start:end], yerr=sigma_noisem1_ener[start:end], lw=1, elinewidth=1, color="r",
             #ecolor='r', label = "Extracted flux of clear order 1 trace")
#plt.errorbar(w[start:end], flamb_totclear_energy[start:end], yerr=sigma_noisetot_ener[start:end], lw=1, elinewidth=1, color="b",
             #ecolor='b', label="Extracted flux of order 1 from all clear traces")
plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/flamb_clear_energy.png".format(os))
plt.show()

plt.figure()
plt.plot(w[start:end], flamb_m1_inf_radi_ener[start:end], lw=2, color="Purple")
plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title("Extracted flux with infinite radius of clear order 1 trace")
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/flamb_m1_inf_energy.png".format(os))
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

# Extracted flux [e⁻/colonne]
plt.figure()
plt.errorbar(w[start:end], flamb_m1clear_elec[start:end], yerr=elec_noise_m1[start:end], lw=1, elinewidth=1,
             color="HotPink", ecolor='r', label="Data", zorder=0)
#plt.plot(w_median, flamb_m1clear_median, lw=1, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Extracted flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux of clear order 1 trace")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/flamb_m1clear_elec.png".format(os))
plt.show()

"""
# When there were oscillations..
plt.figure()
plt.plot(w_median[start:], flamb_m1clear_normal[start:], lw=1, color="Lime")
plt.ylabel(r"Extracted flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux normalized by median filter of clear order 1 trace")
plt.show()
"""

plt.figure()
plt.errorbar(w[start:end], flamb_totclear_elec[start:end], yerr=elec_noise_tot[start:end], lw=1, elinewidth=1,
             color="HotPink", ecolor='r', label="Data", zorder=0)
#plt.plot(w_median, flamb_totclear_median, lw=1, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Extracted flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux of order 1 trace from all clear traces")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/flamb_totclear_elec.png".format(os))
plt.show()

"""
# When there were oscillations...
plt.figure()
plt.plot(w_median[start:], flamb_totclear_normal[start:], lw=1, color="Lime")
plt.ylabel(r"Extracted flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux normalized by median filter \nof order 1 trace from all clear traces")
plt.show()
"""

plt.figure()
plt.plot(w[start:end], flamb_noisy_elec[start:end], lw=1, color="HotPink", label="Data", zorder=0)
#plt.plot(w_median, flamb_noisy_median, lw=1, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Extracted flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux of order 1 trace from noisy traces")
plt.legend()
if save is True:
    plt.savefig(WORKING_DIR + "oversampling_{}/flamb_noisy_elec.png".format(os))
plt.show()

"""
# When there were oscillations...
plt.figure()
plt.plot(w_median, flamb_noisy_normal, lw=1, color="Lime")
plt.ylabel(r"Extracted flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Extracted flux normalized by median filter, noisy traces")
plt.show()

plt.figure()
plt.scatter(os, std_m1clear, color='b')
#plt.scatter(os, std_totclear, s=5, color='r', label = "Order 1 from total clear traces")
plt.xlabel("Oversample")
plt.ylabel("Standard deviation")
plt.title("Oscillations amplitude of clear trace, order 1 only (standard deviations)")
plt.show()

plt.figure()
plt.scatter(os, std_noisy, color='b')
plt.xlabel("Oversample")
plt.ylabel("Standard deviation")
plt.title("Oscillations amplitude of noisy traces (standard deviations)")
plt.show()
"""

"""
# Comparisons

plt.figure()
plt.plot(w[start:end], diff_m1[start:end], lw=1, color="HotPink", label="Difference")
#plt.errorbar(w[start:end], diff_m1[start:end], yerr=sigma_diff_m1[start:end], elinewidth=1, color="HotPink", ecolor='r',
             #label="Difference")
plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.legend(), plt.show()

plt.figure()
plt.plot(w[start:end], relat_flux_clear[start:end], lw=1, color="HotPink", label="Relative flux (clear)")
#plt.errorbar(w[start:end], relat_flux_clear[start:end], yerr=sigma_relatflux_clear[start:end], elinewidth=1, color="HotPink",
            # ecolor='r', label="Relative flux (clear)")
plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.legend(), plt.show()

plt.figure()
plt.errorbar(w[start:end], diff_noisy[start:end], yerr=sigma_diff_noisy[start:end], lw=1, elinewidth=1, color='HotPink',
             ecolor='r')
plt.title("Difference between order 1 extracted from clear \ntraces and order 1 extracted from noisy traces")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.show()

plt.figure()
plt.errorbar(w[start:end], relat_flux_noisy[start:end], yerr=sigma_relatflux_noisy[start:end], lw=1, elinewidth=1,
             color='HotPink', ecolor='r')
plt.ylabel(r"Extracted flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title("Relative flux (noisy)")
plt.show()
"""
