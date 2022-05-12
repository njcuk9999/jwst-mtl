
import numpy as np
from astropy.io import fits
from jwst import datamodels

def stack(cube):
    deepstack = np.nanmedian(cube, axis=0)
    rms = np.nanstd(cube, axis=0)
    return deepstack, rms

def makemask(stack, rms):
    # Exact strategy TBD
    return

def correct1overf(uncal):

    ngroup = uncal.meta.exposure.ngroups
    nint = uncal.meta.exposure.nints
    dimx = np.shape(uncal.data)[-1]

    deepstack, rms = stack(uncal.data)

    hdu = fits.PrimaryHDU(deepstack)
    hdu.writeto('/genesis/jwst/userland-soss/loic_review/oneoverf/deepstack1.fits', overwrite=True)
    hdu = fits.PrimaryHDU(rms)
    hdu.writeto('/genesis/jwst/userland-soss/loic_review/oneoverf/rms1.fits', overwrite=True)

    # Weighted average to determine the 1/F DC level
    w = 1/rms # weight
    print(np.shape(w))
    print(np.shape(w * uncal.data[0]))

    dcmap = np.copy(uncal.data)
    subcorr = np.copy(uncal.data)
    for i in range(nint):
        sub = uncal.data[i] - deepstack
        hdu = fits.PrimaryHDU(sub)
        hdu.writeto('/genesis/jwst/userland-soss/loic_review/oneoverf/sub.fits', overwrite=True)
        dc = np.nansum(w * sub, axis=1) / np.nansum(w, axis=1)
        # dc is 2-dimensional - expand to the 3rd (columns) dimension
        dcmap[i,:,:,:] = np.repeat(dc, 256).reshape((ngroup, 2048, 256)).swapaxes(1,2)
        subcorr[i, :, :, :] = sub - dcmap[i, :, :, :]

    hdu = fits.PrimaryHDU(subcorr)
    hdu.writeto('/genesis/jwst/userland-soss/loic_review/oneoverf/subcorr.fits', overwrite=True)
    hdu = fits.PrimaryHDU(dcmap)
    hdu.writeto('/genesis/jwst/userland-soss/loic_review/oneoverf/toto.fits', overwrite=True)

    corrected = uncal.data - dcmap
    #hdu = fits.PrimaryHDU(corrected)
    #hdu.writeto('/genesis/jwst/userland-soss/loic_review/oneoverf/corrected.fits', overwrite=True)

    datamodel_corr = uncal.copy()
    datamodel_corr.data = corrected

    return datamodel_corr


# Open the uncal time series that needs 1/f correction
uncal = datamodels.open('/genesis/jwst/userland-soss/loic_review/GTO/wasp52b/IDTSOSS_clear_noisy.fits')
#uncal = datamodels.open('/genesis/jwst/userland-soss/loic_review/CAP_rehearsal/twa33_varyteff/3000k/IDTSOSS_clear_noisy.fits')

# Run the 1/f correction step
map = correct1overf(uncal)

# Free up the input time series
uncal.close()

# Write down the output corrected time series
map.write('/genesis/jwst/userland-soss/loic_review/oneoverf/uncal_corrected.fits')
