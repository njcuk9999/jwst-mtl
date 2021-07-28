from scipy.interpolate import interp1d
from astropy.io import fits

###############################################
# Hack to get the path of module. To be changed.
from os.path import abspath, dirname


def get_module_path(file):

    dir_path = abspath(file)
    dir_path = dirname(dir_path) + '/'

    return dir_path
###############################################

# Default file parameters
FILE_SOSS = "NIRISS_Throughput_STScI.fits"
FILE_SOSS = 'NIRISS_Throughput_20210318.fits'

DEF_PATH = get_module_path(__file__) + "Ref_files/"
DEF_PATH = '/genesis/jwst/jwst-ref-soss/trace_model/'



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
        tr = tr*0 + 1     # KIM

        # Get wavelength
        wv = hdu[1].data['LAMBDA'].squeeze()
        # nm to microns
        wv /= 1000.

        print(tr)

        # Interpolate
        super().__init__(wv, tr, kind='cubic',
                         fill_value=0, bounds_error=False)
