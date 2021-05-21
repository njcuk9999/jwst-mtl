import SOSS.trace.tracepol as tp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
import scipy.constants as sc_cst

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

trace_file = "/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv"
x = np.linspace(0,2047,2048)    #Array of pixels
pars = tp.get_tracepars(trace_file)   #Gives the middle position of trace
w, tmp = tp.specpix_to_wavelength(x,pars,m=1)   #Returns wavelength for each x, order 1

xnew, y, mask = tp.wavelength_to_pix(w, pars, m=1)   #Converts wavelenghths to pixel coordinate

cube = fits.getdata("/home/kmorel/ongenesis/jwst-user-soss/test_noisy_rateints.fits")
im = cube[0,:,:]

a = fits.open("/home/kmorel/ongenesis/jwst-user-soss/test_noisy_rateints.fits")
dq = a[3].data  #Data quality
dq = dq[0,:,:]

#VALEUR IMPAIRES DE DQ = PIXELS À NE PAS UTILISER
i = np.where(dq %2 != 0)
im[i[0],i[1]] = 0

ynew = np.interp(x, xnew, y)    #Interpolation of y at integer values of x
ynew = np.around(ynew)

radius_pixel = 15   #"Rayon" de la boîte
flux = np.zeros_like(x)   #Array extracted spectrum

for x_i in x:
    x_i = int(x_i)
    i = int(ynew[x_i])
    flux[x_i] = np.sum(im[i-radius_pixel:i+radius_pixel,x_i])

gain = 1.6  #e⁻/adu
phot_ener = sc_cst.Planck*sc_cst.speed_of_light/(w*10**(-6))
area = 25  #m²

dw = np.zeros_like(w)

for i in range(len(w)):
    if i == 0:
        dw[i] = w[0]-w[1]
    else:
        dw[i] = w[i-1]-w[i]

f_lambda = flux*gain*phot_ener/area/dw  #Flux

plt.figure(1)
plt.imshow(im, vmin=0, vmax=1000, origin="lower")
plt.plot(x, y, color="red")
plt.show()

plt.figure(2)
plt.plot(w, f_lambda, color="HotPink")
plt.xlabel(r"Wavelength [$\mu$m]"), plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.show()

plt.figure(3)
plt.plot(starmodel_angstrom, starmodel_flambda,color='Aqua')
plt.xlabel(r"Wavelength [angstrom]"), plt.ylabel(r"Flux [J s⁻¹ m⁻² $\mu$m⁻¹]")
plt.show()

print(w)