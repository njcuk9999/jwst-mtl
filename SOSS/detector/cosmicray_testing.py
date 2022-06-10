import sys
import astropy.io.fits as pyfits
import numpy as np
from crsim import *
from timeseries import *

test_obs = TimeSeries('/home/plamontagne/ongenesis/userland-soss/FULL/IDTSOSS_clear_fixed.fits', '/genesis/jwst/jwst-ref-soss/noise_files/')

test_obs.add_cosmic_rays(sun_activity='SUNMAX')



