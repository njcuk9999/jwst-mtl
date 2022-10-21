import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from astropy.io import fits

import utils_KM as utils

spec_dir = '/home/kmorel/ongenesis/jwst-user-soss/master/WASP_96/pipeline_outputs_directory/Stage3/'
spec_filename = spec_dir + 'WASP-96_box_spectra_fullres.fits'

# Integrations of ingress and egress.
baseline_ints = [107, -70]


with fits.open(spec_filename) as hdulist:
    print(hdulist[3].header['EXTNAME'])
    spectra = hdulist[3].data   # Shape = (time, wl)

    # Convert data from fits files to float (fits precision is 1e-8)
    spectra = spectra.astype('float64', copy=False)

wlc = np.sum(spectra, axis=1)

# Normalization
wlc_norm = utils.normalization(wlc, baseline_ints, 'transit')

plt.figure()
plt.plot(wlc_norm)
plt.savefig(spec_dir + 'wlc_norm')