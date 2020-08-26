from scipy.interpolate import interp1d
from astropy.io import fits

# Default file parameters
FILE_SOSS = "NIRISS_Throughput_STScI.fits"
DEF_PATH = "Ref_files/"


class ThroughputSOSS(interp1d):
    """
    Callable Throughput of SOSS mode for a given order.
    Function oof wavelength in microns.
    """
    filename = FILE_SOSS
    path = DEF_PATH

    def __init__(self, order=1):
        """
        Parameter:
        order: int
            which order do you want? Default is the first order (1)
        """
        # Open file
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
