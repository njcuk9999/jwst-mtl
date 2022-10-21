import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from astropy.io import fits

spec_dir = '/home/kmorel/ongenesis/jwst-user-soss/master/WASP_96/pipeline_outputs_directory/Stage3/'
spec_filename = spec_dir + 'WASP-96_box_spectra_fullres.fits'

with fits.open(spec_filename) as hdulist:
    print(hdulist[3].header['EXTNAME'])
    spectrum = hdulist[3].data   # Shape = (time, wl)

    # Convert data from fits files to float (fits precision is 1e-8)
    spectrum = spectrum.astype('float64', copy=False)

wlc = np.sum(spectrum, axis=1)
print(np.shape(wlc))
plt.figure()
plt.plot(wlc)
plt.savefig(spec_dir + 'spectrum')