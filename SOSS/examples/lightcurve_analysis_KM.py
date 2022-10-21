import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from astropy.io import fits

import utils_KM as utils

import juliet

# Extracted spectra directory
spec_dir = '/home/kmorel/ongenesis/jwst-user-soss/master/WASP_96/pipeline_outputs_directory/Stage3/'
spec_filename = spec_dir + 'WASP-96_box_spectra_fullres.fits'

# Integrations of ingress and egress.
baseline_ints = [107, -70]


with fits.open(spec_filename) as hdulist:
    # Order 1
    print('Data = ', hdulist[3].header['EXTNAME'])
    spec_ord1 = hdulist[3].data   # Extracted flux [time, wl]
    err_ord1 = hdulist[4].data    # Error
    # Convert data from fits files to float (fits precision is 1e-8)
    spec_ord1 = spec_ord1.astype('float64', copy=False)
    err_ord1 = err_ord1.astype('float64', copy=False)

    # Order 2
    print('Data = ', hdulist[7].header['EXTNAME'])
    spec_ord2= hdulist[7].data   # Extracted flux [time, wl]
    err_ord2 = hdulist[8].data    # Error
    # Convert data from fits files to float (fits precision is 1e-8)
    spec_ord2 = spec_ord2.astype('float64', copy=False)
    err_ord2 = err_ord2.astype('float64', copy=False)

    # Time
    # print('Data = ', hdulist[9].header['EXTNAME'])
    # time = hdulist[9].data  # Time
    # print(np.shape(time))
    # # Convert data from fits files to float (fits precision is 1e-8)
    # time = time.astype('float64', copy=False)

# White light curve, order 1
wlc = np.sum(spec_ord1, axis=1)
# Normalization
wlc_norm = utils.normalization(wlc, baseline_ints, 'transit')

plt.figure()
plt.plot(time, wlc_norm)
plt.savefig(spec_dir + 'wlc_norm')