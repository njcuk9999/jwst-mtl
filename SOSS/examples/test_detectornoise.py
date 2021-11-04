import numpy as np

import detector.detector as detector

from astropy.io import fits

files = '/genesis/jwst/userland-soss/loic_review/IDTSOSS_clear.fits'
a = detector.add_noise(files, '/genesis/jwst/jwst-ref-soss/noise_files/',
                       outputfilename='/genesis/jwst/userland-soss/loic_review/toto.fits',
                       readout=False,
                       zodibackg=False,
                       photon=False,
                       superbias=False,
                       flatfield=False,
                       nonlinearity=False,
                       oneoverf=False,
                       darkcurrent=True)

#fits.writeto(a, '/genesis/jwst/userland-soss/loic_review/toto.fits', overwrite=True)









