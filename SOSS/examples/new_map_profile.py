# Creates a map profile from clear_000000.fits image

import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
import matplotlib.pyplot as plt
import box_kim

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
simuPars.noversample = 11
os = simuPars.noversample
###############################


clear_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_000000.fits".format(os))
clear = np.empty(shape=(3, 256, 2048), dtype=float)
map_clear = np.empty_like(clear, dtype=float)

padd = 10   # Because of x_padding and y_padding

for i in range(len(clear_00[0].data)):
    if os == 1:
        clear[i] = clear_00[0].data[i, padd:-padd, padd:-padd]
    else:
        clear_i = soss.rebin(clear_00[0].data[i], os, flux_method='sum')
        clear[i] = clear_i[padd:-padd, padd:-padd]
    sum_col = np.sum(clear[i], axis=0)
    map_clear[i] = clear[i] / sum_col

map_clear[1, :, 1790:] = 0  # Problem with end of order 2 trace

# Save map_profile
#hdu = fits.PrimaryHDU(map_clear)
#hdu.writeto(WORKING_DIR + "new_map_profile_clear_{}.fits".format(os), overwrite=True)


#####################################################################
# New wavelength map
wave = box_kim.create_wave(R=65000, w_min=0.6, w_max=3.)
comb, peaks = box_kim.make_comb(wave, peak_spacing=0.005, peak_width=0.001)

plt.figure()
plt.plot(wave, comb)
plt.xlabel('Wavelength [um]')
plt.ylabel('Flux')
plt.show()

print(peaks.shape)
print(wave.shape, comb.shape)