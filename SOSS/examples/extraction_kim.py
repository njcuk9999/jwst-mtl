import SOSS.trace.tracepol as tp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen

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
x = np.linspace(0,2047,2048)
pars = tp.get_tracepars(trace_file)
w, tmp = tp.specpix_to_wavelength(x,pars,m=1)   #Donne mes wl pour chaque x

xnew, y, mask = tp.wavelength_to_pix(w, pars, m=1)

cube = fits.getdata("/home/kmorel/ongenesis/jwst-user-soss/test_noisy_rateints.fits")
im = cube[0,:,:]

plt.figure()
plt.imshow(im,vmin=0,vmax=1000,origin="lower")
plt.plot(x,y,color="red")
plt.show()