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

#Constants
h = sc_cst.Planck
c = sc_cst.speed_of_light
gain = 1.6
area = 25.
radius_pixel = 13
length = 240
ng = 3   # NGROUP
t_read = 5.49   # Reading time [s]
tint = (ng - 1) * t_read   # Integration time [s]

def photon_energy(wl):
    """
    :param wl: Wavelength in microns
    :return: Photon energy in J
    """
    return h * c / (wl * 1e-6)

def dispersion(wl):
    """
    :param wl: Wavelengths array [microns]
    :return: Dispersion [microns]
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
    :param pixels: Array of pixels
    :param im_test: Trace's image [adu/s]
    :param wl: Array of wavelengths (same size as pixels)  [microns]
    :param y_trace: Array for the positions of the center of the trace for each column
    :param radius_pixel: Radius of extraction box [pixels]
    :param area: Area of photons collection surface [m²]
    :param gain: Gain [e⁻/adu]
    :return: Extracted flux [J/s/m²/micron]
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
    dw = dispersion(wl)   #Dispersion [microns]

    return flux * gain * phot_ener / area / dw

def flambda_elec(pixels, im_test, y_trace, radius_pixel=radius_pixel, gain=gain):
    """
    :param pixels: Array of pixels
    :param im_test: Trace's image [adu/s]
    :param y_trace: Array for the positions of the center of the trace for each column
    :param radius_pixel: Radius of extraction box [pixels]
    :param gain: Gain [e⁻/adu]
    :return: Extracted flux [e⁻/s/colonne]
    """
    flux = np.zeros_like(pixels)  # Array for extracted spectrum
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        flux[x_i] = im_test[int(first), x_i] * (1 - first % int(first)) + np.sum(
            im_test[int(first) + 1:int(last) + 1, x_i]) + im_test[int(last) + 1, x_i] * (last % int(last))

    return flux #* gain

def flambda_inf_radi(im_test, wl, area=area, gain=gain):
    """
    :param im_test: Trace's image [adu/s]
    :param wl: Array of wavelengths (same size as pixels)  [um]
    :param area: Area of photons collection surface [m²]
    :param gain: Gain [e⁻/adu]
    :return: Extracted flux for infinite radius [J/s/m²/um]
    """
    flux = np.sum(im_test,axis=0)

    # Calculate the flux in J/s/m²/um
    phot_ener = photon_energy(wl)  # Energy of each photon [J/photon]
    dw = dispersion(wl)   #Dispersion [um]

    return flux * gain * phot_ener / area / dw

def sigma_flambda(pixels, error, wl, y_trace, radius_pixel=radius_pixel, area=area, gain=gain):
    """
    :param pixels: Array of pixels
    :param variance: Variance of pixels [adu/s]
    :param wl: Array of wavelengths (same size as pixels)  [microns]
    :param y_trace: Array for the positions of the center of the trace for each column
    :param radius_pixel: Radius of extraction box [pixels]
    :param area: Area of photons collection surface [m²]
    :param gain: Gain [e⁻/adu]
    :return: Variance of extracted flux [J/s/m²/micron]
    """
    variance = error ** 2  # Variance of each pixel [adu²/s²]

    vari = np.zeros_like(pixels)  # Array for variances
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        vari[x_i] = variance[int(first), x_i] * (1 - first % int(first)) + np.sum(
            variance[int(first) + 1:int(last) + 1, x_i]) + variance[int(last) + 1, x_i] * (last % int(last))

    # Calculate the total error in J/s/m²/um
    delta_flambda = np.sqrt(vari)   #Sigma of column [adu/s]

    phot_ener = photon_energy(wl)  # Energy of each photon [J/photon]
    dw = dispersion(wl)  # Dispersion [microns]

    return delta_flambda * gain * phot_ener / area / dw

def median_window(flux, start, length):
    list = flux[start : start + length]
    return np.median(list)

def median_filter(flux, length = length):
    m = []
    start = 0
    while start + length < len(flux):
        m.append(median_window(flux, start, length))
        start += 1
    return m

def wl_median(wl, pixels, length=length):
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
sys.path.insert(0,"/genesis/jwst/jwst-ref-soss/fortran_lib/")

# Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_kim.txt')
pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              #Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars) #read in parameter file

# Generate or read the star atmosphere model
starmodel_angstrom, starmodel_flambda, ld_coeff = soss.starmodel(simuPars, pathPars)

# Position of trace
trace_file = "/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv"
x = np.linspace(0,2047,2048)    # Array of pixels
pars = tp.get_tracepars(trace_file)   # Gives the middle position of trace
w, tmp = tp.specpix_to_wavelength(x,pars,m=1)   # Returns wavelength for each x, order 1
xnew, y, mask = tp.wavelength_to_pix(w, pars, m=1)   # Converts wavelenghths to pixel coordinates
### x and xnew should be the same!...
#ynew = np.interp(x, xnew, y)    #Interpolation of y at integer values of x
#ynew = np.around(ynew)

m_order = 0  # Order - 1

# CHOOSE oversample : Comment files not used
"""
#oversample = 1
#noisy_rateints = fits.open("/home/kmorel/ongenesis/jwst-user-soss/oversampling_1/test_clear_noisy_rateints.fits")
clear = fits.open("/home/kmorel/ongenesis/jwst-user-soss/tmp/oversampling_1/clear_000000.fits")
clear = clear[0].data
"""
#"""
#oversample = 10
#noisy_rateints = fits.open("/home/kmorel/ongenesis/jwst-user-soss/oversampling_10/test_clear_noisy_rateints.fits")
clear_00 = fits.open("/home/kmorel/ongenesis/jwst-user-soss/tmp/oversampling_10/clear_000000.fits")
#oversample = 4
#noisy_rateints = fits.open("/home/kmorel/ongenesis/jwst-user-soss/oversampling_4/test_clear_noisy_rateints.fits")
#clear_00 = fits.open("/home/kmorel/ongenesis/jwst-user-soss/tmp/oversampling_4/clear_000000.fits")
#oversample = 2
#noisy_rateints = fits.open("/home/kmorel/ongenesis/jwst-user-soss/oversampling_2/test_clear_noisy_rateints.fits")
#clear_00 = fits.open("/home/kmorel/ongenesis/jwst-user-soss/tmp/oversampling_2/clear_000000.fits")
clear = np.empty(shape=(3,256,2048))
for i in range(len(clear_00[0].data)):
    clear[i] = soss.rebin(clear_00[0].data[i],simuPars.noversample)
#"""

# Images
# With noise
"""
im_adu_noisy = noisy_rateints[1].data[m_order]  # Image of flux [adu/s]
delta = noisy_rateints[2].data[m_order]   # Error [adu/s]
dq = noisy_rateints[3].data[m_order]  # Data quality
# Odd values of dq = DO NOT USE these pixels
i = np.where(dq %2 != 0)
im_adu_noisy[i[0],i[1]] = 0
delta[i[0],i[1]] = 0
"""
# Without noise
# Order 1 only
m1_clear_adu = clear[m_order]  # [adu/s]
#m1_clear_adu = np.flipud(m1_clear_adu)  # Flip image

# Order 1 from all orders added
tot_clear_adu = np.sum(clear,axis=0)   # Sum all traces [adu/s]
#tot_clear_adu = np.flipud(tot_clear_adu)  # Flip image


# EXTRACTIONS
# Extracted flux of noisy image
"""
flamb_noisy_energy = f_lambda(x, im_adu_noisy, w, y)   # [J/s/m²/um]
sigma_flamb_noisy_ener = sigma_flambda(x, delta, w, y)   # [J/s/m²/um]
flamb_noisy_elec = flambda_elec(x, im_adu_noisy, y) * tint  # [e⁻/colonne]
flamb_noisy_median = median_filter(flamb_noisy_elec)  # Application of median filter [e⁻/colonne]
flamb_noisy_normal = flamb_noisy_elec[length//2:-length//2] / flamb_noisy_median   # Normalization
std_noisy_normal = np.std(flamb_noisy_normal)   # Standard deviation
print("Standard deviation (noisy, normalized) =", std_noisy_normal)
"""
w_median = wl_median(w, x)  # Array of wavelengths when using median filter [um]
# Extracted flux of clear, order 1 trace only
flamb_m1clear_energy = f_lambda(x, m1_clear_adu, w, y)   # [J/s/m²/um]
flamb_m1_inf_radi_ener = flambda_inf_radi(m1_clear_adu, w)   # With infinite radius [J/s/m²/um]
flamb_m1clear_elec = flambda_elec(x, m1_clear_adu, y) * tint  # [e⁻/colonne]
elec_noise_m1 = np.sqrt(flamb_m1clear_elec)   # [e⁻/colonne]
flamb_m1clear_median = median_filter(flamb_m1clear_elec)  # Application of median filter [e⁻/colonne]
#p = np.polyfit(w, flamb_m1clear_elec, 50)  # Coefficients of polynomial
#flamb_m1clear_poly = np.polyval(p, w)   # Compute result of polynomial
flamb_m1clear_normal = flamb_m1clear_elec[length//2:-length//2] / flamb_m1clear_median   # Normalization with median
#flamb_m1clear_normal= flamb_m1clear_elec / flamb_m1clear_poly   # Normalization with polynomial
std_m1clear_normal = np.std(flamb_m1clear_normal)   # Standard deviation
print("Standard deviation (clear order 1 only) =", std_m1clear_normal)

# Extracted flux of order 1 of total clear traces
flamb_totclear_energy = f_lambda(x,tot_clear_adu,w,y)   # [J/s/m²/micron]
flamb_totclear_elec = flambda_elec(x,tot_clear_adu,y) * tint  # [e⁻/colonne]
elec_noise_tot = np.sqrt(flamb_totclear_elec)   # [e⁻/colonne]
flamb_totclear_median = median_filter(flamb_totclear_elec)  # Application of median filter [e⁻/colonne]
flamb_totclear_normal = flamb_totclear_elec[length//2:-length//2] / flamb_totclear_median   # Normalization with median
std_totclear_normal = np.std(flamb_totclear_normal)   # Standard deviation
print("Standard deviation (order 1 from total clear traces) =", std_totclear_normal)
"""
# Estimated photon noises
dw = dispersion(w)   # Dispersion [microns]
phot_ener = photon_energy(w)   # Energy of photon [J]

photon_noise_m1 = flamb_m1_clear * tint * dw * area / phot_ener
sigma_noise_m1 = np.sqrt(photon_noise_m1)   # Photon noise for order 1 only trace [photons/colonne]
sigma_noisem1_ener = sigma_noise_m1 * phot_ener / dw / area / tint   # Photon noise [J/s/m²/micron]

photon_noise_tot = flamb_tot_clear * tint * dw * area / phot_ener
sigma_noise_tot = np.sqrt(photon_noise_tot)   # Photon noise for order 1 of sum of traces [photons]
sigma_noisetot_ener = sigma_noise_tot * phot_ener / dw / area / tint   # Photon noise [J/s/m²/micron]

#Comparisons
diff_m1 = flamb_tot_clear - flamb_m1_clear   # Difference between order 1 only and order 1 of sum of traces
sigma_diff_m1 = np.sqrt(sigma_noisetot_ener**2 + sigma_noisem1_ener**2)

relat_flux_clear = diff_m1 / flamb_m1_clear  # Relative flux
sigma_relatflux_clear = np.sqrt( (sigma_noisetot_ener / flamb_m1_clear)**2 + (sigma_noisem1_ener * flamb_tot_clear /
                                                                      flamb_m1_clear**2)**2 )

diff_noisy = flamb_tot_clear - flamb_noisy_energy   # Difference between order 1 of sum of traces and order 1 of noisy traces
sigma_diff_noisy = np.sqrt(sigma_noisetot_ener**2 + sigma_flamb_im**2)

relat_flux_noisy = diff_noisy / flamb_noisy_energy  # Relative flux
sigma_relatflux_noisy = np.sqrt( (sigma_noisetot_ener / flamb_noisy_energy)**2 + (sigma_flamb_noisy_ener * flamb_tot_clear /
                                                                      flamb_noisy_energy**2)**2 )

flamb_m1_clear_minus = flamb_m1_clear - sigma_noisem1_ener
flamb_m1_clear_plus = flamb_m1_clear + sigma_noisem1_ener
flamb_tot_clear_minus = flamb_tot_clear - sigma_noisetot_ener
flamb_tot_clear_plus = flamb_tot_clear + sigma_noisetot_ener
"""
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

##########################
# GRAPHICS

# Images of traces
"""
plt.figure(1)
plt.plot(x, y, color="r", label="Order 1 trace's position")   # Middle position of order 1 trace
plt.imshow(im_adu_noisy, vmin=0, vmax=1000, origin="lower")   # Image of traces
plt.title("test_clear_noisy_rateints.fits")
plt.legend(), plt.show()
"""
plt.figure(2)
plt.imshow(m1_clear_adu, vmin=0, vmax=3000, origin="lower")
plt.plot(x, y, color="r", label="Order 1 trace's position")
plt.title("clear_000000.fits, order 1 only")
plt.legend(), plt.show()

plt.figure(3)
plt.imshow(tot_clear_adu, vmin=0, vmax=1000, origin="lower")
plt.plot(x, y, color="r", label="Order 1 trace's position")
plt.title("clear_000000.fits")
plt.legend(), plt.show()


# Extracted flux [J/s/m²/um]
start, end = 7, -7   # To avoid problems with the extremities
"""
plt.figure(4)
plt.errorbar(w[start:end], flamb_noisy_energy[start:end], yerr=sigma_flamb_noisy_ener[start:end], lw=1, elinewidth=1,
             color="HotPink", ecolor='r')
plt.xlabel(r"Wavelength [$\mu$m]")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.title("Extracted flux of order 1 from noisy traces")
plt.show()
"""
plt.figure(5)
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
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.legend(), plt.show()
"""
plt.figure(6)
plt.plot(w[start:end], flamb_m1_inf_radi_ener[start:end], lw=1, color="Violet")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title("Flux with infinite radius (clear_000000.fits, order 1 only)")
plt.show()


# Model
plt.figure(7)
plt.plot(starmodel_angstrom, starmodel_flambda, color='Aqua')
plt.xlabel(r"Wavelength [angstrom]")
plt.ylabel(r"Flux")
plt.title("Model")
plt.show()
"""

# Extracted flux [e⁻/colonne]
plt.figure(8)
plt.errorbar(w[start:end], flamb_m1clear_elec[start:end], yerr=elec_noise_m1[start:end], lw=1, elinewidth=1,
             color="HotPink", ecolor='r', label="Data", zorder=0)
plt.plot(w_median, flamb_m1clear_median, lw=1, color="Lime", label="Median filter applied", zorder=5)
#plt.plot(w[start:end], flamb_m1clear_poly[start:end], lw=1, color="Lime", zorder=5, label="Polynomial")
plt.ylabel(r"Flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Flux in e$^-$/colonne (clear_000000.fits, order 1 only)")
plt.legend(), plt.show()

plt.figure(9)
plt.plot(w_median[start:], flamb_m1clear_normal[start:], lw=1, color="Lime")
plt.ylabel(r"Flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Flux in e$^-$/colonne normalized by median filter (clear_000000.fits, order 1 only)")
plt.show()

plt.figure(10)
plt.errorbar(w[start:end], flamb_totclear_elec[start:end], yerr=elec_noise_tot[start:end], lw=1, elinewidth=1,
             color="HotPink", ecolor='r', label="Data", zorder=0)
plt.plot(w_median, flamb_totclear_median, lw=1, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Flux in e$^-$/colonne (clear_000000.fits, order 1 from total)")
plt.legend(), plt.show()

plt.figure(11)
plt.plot(w_median[start:], flamb_totclear_normal[start:], lw=1, color="Lime")
plt.ylabel(r"Flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Flux in e$^-$/colonne normalized by median filter \n(clear_000000.fits, order 1 from total)")
plt.show()
"""
plt.figure(12)
plt.plot(w[start:end], flamb_noisy_elec[start:end], lw=1, color="HotPink", label="Data", zorder=0)
plt.plot(w_median, flamb_noisy_median, lw=1, color="Lime", label="Median filter applied", zorder=5)
plt.ylabel(r"Flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Flux in e$^-$/colonne (test_clear_noisy_rateints.fits)")
plt.legend(), plt.show()

plt.figure(13)
plt.plot(w_median, flamb_noisy_normal, lw=1, color="Lime")
plt.ylabel(r"Flux [e$^-$/colonne]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title(r"Flux in e$^-$/colonne normalized by median filter (test_clear_noisy_rateints.fits)")
plt.show()

plt.figure(14)
plt.scatter(os, std_m1clear, color='b')
#plt.scatter(os, std_totclear, s=5, color='r', label = "Order 1 from total clear traces")
plt.xlabel("Oversample")
plt.ylabel("Standard deviation")
plt.title("Oscillations amplitude for clear trace, order 1 only (standard deviations)")
plt.show()

plt.figure(15)
plt.scatter(os, std_noisy, color='b')
plt.xlabel("Oversample")
plt.ylabel("Standard deviation")
plt.title("Oscillations amplitude for noisy traces (standard deviations)")
plt.show()


plt.figure(7)
plt.plot(w[start:end], diff_m1[start:end], lw=1, color="HotPink", label="Difference")
#plt.errorbar(w[start:end], diff_m1[start:end], yerr=sigma_diff_m1[start:end], elinewidth=1, color="HotPink", ecolor='r',
             #label="Difference")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.legend(), plt.show()

plt.figure(8)
plt.plot(w[start:end], relat_flux_clear[start:end], lw=1, color="HotPink", label="Relative flux (clear)")
#plt.errorbar(w[start:end], relat_flux_clear[start:end], yerr=sigma_relatflux_clear[start:end], elinewidth=1, color="HotPink",
            # ecolor='r', label="Relative flux (clear)")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.legend(), plt.show()

plt.figure(9)
plt.errorbar(w[start:end], diff_noisy[start:end], yerr=sigma_diff_noisy[start:end], lw=1, elinewidth=1, color='HotPink',
             ecolor='r')
plt.title("Difference between order 1 extracted from clear \ntraces and order 1 extracted from noisy traces")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.show()

plt.figure(10)
plt.errorbar(w[start:end], relat_flux_noisy[start:end], yerr=sigma_relatflux_noisy[start:end], lw=1, elinewidth=1,
             color='HotPink', ecolor='r')
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title("Relative flux (noisy)")
plt.show()
"""
