import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits


class ThroughputSOSS(interp1d):

    filename = "NIRISS_Throughput_STScI.fits"
    path = "../Ref_files/"

    def __init__(self, order=1):

        hdu = fits.open(self.path + self.filename)

        # Get transmission
        key = 'SOSS_order{}'.format(order)
        tr = hdu[1].data[key].squeeze()

        # Get wavelength
        wv = hdu[1].data['LAMBDA'].squeeze()
        # nm to microns
        wv /= 1000.

        # Interpolate
        super().__init__(wv, tr, kind='cubic',
                         fill_value=0, bounds_error=False)
