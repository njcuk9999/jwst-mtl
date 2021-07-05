# Creates a map profile from clear_000000.fits image

import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
import matplotlib.pyplot as plt

WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/'

sys.path.insert(0, "/genesis/jwst/jwst-ref-soss/fortran_lib/")

# Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_kim.txt')
pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              # Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars)   # Read in parameter file


###############################
# CHOOSE OVERSAMPLE  !!!
simuPars.noversample = 8
###############################


clear_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_000000.fits".format(simuPars.noversample))
clear = np.empty(shape=(3, 256, 2048), dtype=float)
map_clear = np.empty_like(clear, dtype=float)

for i in range(len(clear_00[0].data)):
    if simuPars.noversample == 1:
        clear[i] = clear_00[0].data[i, 10:-10, 10:-10]  # Because of x_padding and y_padding
    else:
        clear_i = soss.rebin(clear_00[0].data[i], simuPars.noversample)
        clear[i] = clear_i[10:-10, 10:-10]   # Because of x_padding and y_padding
    sum_col = np.sum(clear[i], axis=0)
    map_clear[i] = clear[i] / sum_col

map_clear[1, :, 1790:] = 0  # Problem with end of order 2 trace


# Save map_profile
hdu = fits.PrimaryHDU(map_clear)
hdu.writeto(WORKING_DIR + "new_map_profile_clear_{}.fits".format(simuPars.noversample), overwrite=True)
