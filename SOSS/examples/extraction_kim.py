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

def f_lambda(pixels, im_test, wl, y_trace, radius_pixel=14, area=25., gain=1.6):
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
    flux = np.zeros_like(pixels)  # Array extracted spectrum
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        flux[x_i] = im_test[int(first), x_i] * (1 - first % int(first)) + np.sum(
            im_test[int(first) + 1:int(last) + 1, x_i]) + im_test[int(last) + 1, x_i] * (last % int(last))

    # Calculate the flux in J/s/m²/um
    phot_ener = h * c / (wl * 1e-6)  # Energy of photons [J/s]

    dw = np.zeros_like(wl)  # Dispersion [microns]
    for i in range(len(wl)):
        if i == 0:
            dw[i] = wl[0] - wl[1]
        else:
            dw[i] = wl[i-1] - wl[i]

    # Flux
    return flux * gain * phot_ener / area / dw


def var_flambda(pixels, variance, wl, y_trace, radius_pixel=14, area=25., gain=1.6):

    vari = np.zeros_like(pixels)  # Array for variances
    for x_i in pixels:
        x_i = int(x_i)
        y_i = y_trace[x_i]
        first = y_i - radius_pixel
        last = y_i + radius_pixel
        vari[x_i] = variance[int(first), x_i] * (1 - first % int(first)) + np.sum(
            variance[int(first) + 1:int(last) + 1, x_i]) + variance[int(last) + 1, x_i] * (last % int(last))

    # Calculate the total variance in J/s/m²/um
    phot_ener = h * c / (wl * 1e-6)  # Energy of photons [J/s]

    dw = np.zeros_like(wl)  # Dispersion [microns]
    for i in range(len(wl)):
        if i == 0:
            dw[i] = wl[0] - wl[1]
        else:
            dw[i] = wl[i-1] - wl[i]

    # Variance of flux
    return vari * gain * phot_ener / area / dw


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

#Position of trace :
trace_file = "/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv"
x = np.linspace(0,2047,2048)    #Array of pixels
pars = tp.get_tracepars(trace_file)   #Gives the middle position of trace
w, tmp = tp.specpix_to_wavelength(x,pars,m=1)   #Returns wavelength for each x, order 1

xnew, y, mask = tp.wavelength_to_pix(w, pars, m=1)   #Converts wavelenghths to pixel coordinate
### x and xnew should be the same!...

m_order = 0  #Order - 1
a = fits.open("/home/kmorel/ongenesis/jwst-user-soss/test_clear_noisy_rateints.fits")
im = a[1].data[m_order]  #adu/s
dq = a[3].data[m_order]  #Data quality
delta = a[2].data   #Error
var = delta[m_order]**2   #Variance

#VALEUR IMPAIRES DE DQ = PIXELS DO NOT USE
i = np.where(dq %2 != 0)
im[i[0],i[1]] = 0
var[i[0],i[1]] = 0

#ynew = np.interp(x, xnew, y)    #Interpolation of y at integer values of x
#ynew = np.around(ynew)

#Extracted flux of im
f_lamb_im = f_lambda(x,im,w,y)
var_flamb_im = var_flambda(x,var,w,y)   #Its variance


b = fits.open("/home/kmorel/ongenesis/jwst-user-soss/tmp/clear_000000.fits")
m1_clear = b[0].data[m_order]  #adu/s
m1_clear = np.flipud(m1_clear)  #Flip image

#Extracted flux of m1_clear, order 1 only
f_lamb_m1_clear = f_lambda(x,m1_clear,w,y)


plt.figure(1)
plt.imshow(im, vmin=0, vmax=1000, origin="lower")
plt.plot(x, y, color="red")
plt.show()

plt.figure(2)
plt.plot(w, f_lamb_im, color="HotPink")
#plt.errorbar()
plt.xlabel(r"Wavelength [$\mu$m]"), plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.title("Extracted flux of noisy spectrum, order 1 from all orders")
plt.show()

plt.figure(4)
plt.imshow(m1_clear, vmin=0, vmax=1000, origin="lower")
plt.plot(x, y, color="red")
plt.show()

plt.figure(2)
plt.plot(w, f_lamb_m1_clear, color="HotPink")
plt.xlabel(r"Wavelength [$\mu$m]"), plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.title("Extracted flux of clear spectrum, order 1 only")
plt.show()

plt.figure(3)
plt.plot(starmodel_angstrom, starmodel_flambda,color='Aqua')
plt.xlabel(r"Wavelength [angstrom]"), plt.ylabel(r"Flux")
plt.title("Model")
plt.show()