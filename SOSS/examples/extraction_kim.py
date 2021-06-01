import SOSS.trace.tracepol as tp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
import scipy.constants as sc_cst

#Constants
h = sc_cst.Planck
c = sc_cst.speed_of_light
gain = 1.6
area = 25.
radius_pixel = 14

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
        print(x_i, y_i, first, last)
    # Calculate the flux in J/s/m²/um
    phot_ener = photon_energy(wl)  # Energy of each photon [J/photon]
    dw = dispersion(wl)   #Dispersion [microns]

    return flux * gain * phot_ener / area / dw

def flambda_inf_radi(im_test, wl, area=area, gain=gain):
    flux = np.sum(im_test,axis=0)

    # Calculate the flux in J/s/m²/um
    phot_ener = photon_energy(wl)  # Energy of each photon [J/photon]
    dw = dispersion(wl)   #Dispersion [microns]

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
x = np.linspace(0,20479,20480)    # Array of pixels
pars = tp.get_tracepars(trace_file)   # Gives the middle position of trace
w, tmp = tp.specpix_to_wavelength(x,pars,m=1)   # Returns wavelength for each x, order 1

xnew, y, mask = tp.wavelength_to_pix(w, pars, m=1)   # Converts wavelenghths to pixel coordinates
### x and xnew should be the same!...

m_order = 0  # Order - 1
"""
noisy_rateints = fits.open("/home/kmorel/ongenesis/jwst-user-soss/test_clear_noisy_rateints.fits")
im = noisy_rateints[1].data[m_order]  # Image of flux [adu/s]
delta = noisy_rateints[2].data[m_order]   # Error [adu/s]
dq = noisy_rateints[3].data[m_order]  # Data quality

# Odd values of dq = DO NOT USE these pixels
i = np.where(dq %2 != 0)
im[i[0],i[1]] = 0
delta[i[0],i[1]] = 0

#ynew = np.interp(x, xnew, y)    #Interpolation of y at integer values of x
#ynew = np.around(ynew)

#EXTRACTIONS
#Extracted flux of im
flamb_im = f_lambda(x,im,w,y)
sigma_flamb_im = sigma_flambda(x,delta,w,y)
"""
clear = fits.open("/home/kmorel/ongenesis/jwst-user-soss/tmp/clear_000000.fits")

#Extract flux of clear, order 1 trace only
m1_clear = clear[0].data[m_order]  # [adu/s]
m1_clear = np.flipud(m1_clear)  # Flip image
#m1_clear[i[0],i[1]] = 0  # Data quality   #TO DO???
"""
flamb_m1_clear = f_lambda(x,m1_clear,w,y)   # Extracted flux [J/s/m²/micron]
m1_clear_inf_radi = flambda_inf_radi(m1_clear,w)   # With infinite radius

#Extract flux of order 1 of clear, all orders added
tot_clear = np.sum(clear[0].data,axis=0)   # Sum all traces
tot_clear = np.flipud(tot_clear)  # Flip image
#tot_clear[i[0],i[1]] = 0   # Data quality   #TO DO???
flamb_tot_clear = f_lambda(x,tot_clear,w,y)   # Extracted flux [J/s/m²/micron]

# Estimated photon noises
ng = 3   # NGROUP
t_read = 5.49   # Reading time [s]
tint = (ng - 1) * t_read   # Integration time [s]
dw = dispersion(w)   # Dispersion [microns]
phot_ener = photon_energy(w)   # Energy of photon [J]

photon_noise_m1 = flamb_m1_clear * tint * dw * area / phot_ener
sigma_noise_m1 = np.sqrt(photon_noise_m1)   # Photon noise for order 1 only trace [photons]
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

diff_noisy = flamb_tot_clear - flamb_im   # Difference between order 1 of sum of traces and order 1 of noisy traces
sigma_diff_noisy = np.sqrt(sigma_noisetot_ener**2 + sigma_flamb_im**2)

relat_flux_noisy = diff_noisy / flamb_im  # Relative flux
sigma_relatflux_noisy = np.sqrt( (sigma_noisetot_ener / flamb_im)**2 + (sigma_flamb_im * flamb_tot_clear /
                                                                      flamb_im**2)**2 )

flamb_m1_clear_minus = flamb_m1_clear - sigma_noisem1_ener
flamb_m1_clear_plus = flamb_m1_clear + sigma_noisem1_ener
flamb_tot_clear_minus = flamb_tot_clear - sigma_noisetot_ener
flamb_tot_clear_plus = flamb_tot_clear + sigma_noisetot_ener
"""
# GRAPHICS
# To avoid problems with the extremities :
beg = 5
end = -5
""""
plt.figure(1)
plt.imshow(im, vmin=0, vmax=1000, origin="lower")   # Image of traces
plt.plot(x, y, color="r", label="Order 1 trace's position")   # Middle position of order 1 trace
plt.title("test_clear_noisy_rateints.fits")
plt.legend(), plt.show()

plt.figure(2)
plt.errorbar(w[beg:end], flamb_im[beg:end], yerr=sigma_flamb_im[beg:end], lw=1, elinewidth=1, color="HotPink", ecolor='r')
plt.xlabel(r"Wavelength [$\mu$m]")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.title("Extracted flux of order 1 from noisy traces")
plt.show()

plt.figure(4)
plt.plot(starmodel_angstrom, starmodel_flambda, color='Aqua')
plt.xlabel(r"Wavelength [angstrom]")
plt.ylabel(r"Flux")
plt.title("Model")
plt.show()
"""
plt.figure(5)
plt.imshow(m1_clear, vmin=0, vmax=3000, origin="lower")
#plt.imshow(tot_clear, vmin=0, vmax=1000, origin="lower")
plt.plot(x, y, color="r", label="Order 1 trace's position")
plt.title("clear_000000.fits")
plt.legend(), plt.show()

plt.figure(11)
plt.plot(w[beg:end], m1_clear_inf_radi[beg:end], lw=1, color="Violet")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title("Infinite radius, order 1 only")
plt.legend(), plt.show()

plt.figure(6)
plt.plot(w[beg:end], flamb_m1_clear[beg:end], lw=1, color='r', label = "Extracted flux of clear order 1 trace")
plt.plot(w[beg:end], flamb_tot_clear[beg:end], lw=1, color='b', label="Extracted flux of order 1 from all clear traces")
#plt.fill_between(w[beg:end], flamb_m1_clear_minus[beg:end], flamb_m1_clear_plus[beg:end], alpha=0.4, color='r')
#plt.fill_between(w[beg:end], flamb_tot_clear_minus[beg:end], flamb_tot_clear_plus[beg:end], alpha=0.4, color='b')
#plt.errorbar(w[beg:end], flamb_m1_clear[beg:end], yerr=sigma_noisem1_ener[beg:end], lw=1, elinewidth=1, color="r",
             #ecolor='r', label = "Extracted flux of clear order 1 trace")
#plt.errorbar(w[beg:end], flamb_tot_clear[beg:end], yerr=sigma_noisetot_ener[beg:end], lw=1, elinewidth=1, color="b",
             #ecolor='b', label="Extracted flux of order 1 from all clear traces")
#plt.ylim(2.82e-12, 2.95e-12)
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.legend(), plt.show()

plt.figure(7)
plt.plot(w[beg:end], diff_m1[beg:end], lw=1, color="HotPink", label="Difference")
#plt.errorbar(w[beg:end], diff_m1[beg:end], yerr=sigma_diff_m1[beg:end], elinewidth=1, color="HotPink", ecolor='r',
             #label="Difference")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.legend(), plt.show()

plt.figure(8)
plt.plot(w[beg:end], relat_flux_clear[beg:end], lw=1, color="HotPink", label="Relative flux (clear)")
#plt.errorbar(w[beg:end], relat_flux_clear[beg:end], yerr=sigma_relatflux_clear[beg:end], elinewidth=1, color="HotPink",
            # ecolor='r', label="Relative flux (clear)")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.legend(), plt.show()
"""
plt.figure(9)
plt.errorbar(w[beg:end], diff_noisy[beg:end], yerr=sigma_diff_noisy[beg:end], lw=1, elinewidth=1, color='HotPink',
             ecolor='r')
plt.title("Difference between order 1 extracted from clear \ntraces and order 1 extracted from noisy traces")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.show()

plt.figure(10)
plt.errorbar(w[beg:end], relat_flux_noisy[beg:end], yerr=sigma_relatflux_noisy[beg:end], lw=1, elinewidth=1,
             color='HotPink', ecolor='r')
plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.xlabel(r"Wavelength [$\mu$m]")
plt.title("Relative flux (noisy)")
plt.show()
"""