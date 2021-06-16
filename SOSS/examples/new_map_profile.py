import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
import matplotlib.pyplot as plt

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

clear_00 = fits.open("/home/kmorel/ongenesis/jwst-user-soss/tmp/oversampling_2/clear_000000.fits")
clear = np.empty(shape=(3,256,2048))
map_clear = np.empty_like(clear)
for i in range(len(clear_00[0].data)):
    clear[i] = soss.rebin(clear_00[0].data[i],simuPars.noversample)
    sum_col = np.sum(clear[i], axis=0)
    map_clear[i] = clear[i] / sum_col
    map_clear[i] = np.flipud(map_clear[i])  # Flip image

# Save map_profile
hdu = fits.PrimaryHDU(map_clear)
hdu.writeto("/home/kmorel/ongenesis/jwst-user-soss/new_map_profile_clear.fits", overwrite = True)

plt.figure()
plt.imshow(map_clear[0], origin="lower")
plt.colorbar()
plt.show()


