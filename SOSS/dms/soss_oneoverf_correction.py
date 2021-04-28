#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from astropy.io import fits
from astropy.stats import SigmaClip


def make_background_mask(deepstack, width=20):
    """Build a mask of the pixels considered to contain the majority of the 
    flux, and should therefore not be used to compute the background.

    :param deepstack: a deep image of the trace constructed by combining
        individual integrations of the observation.
    :param width: the width of the trace is used to set the fraction of pixels
        to exclude with the mask (i.e. width/256 for a SUBSTRIP256 observation).

    :type deepstack: array[float]
    :type width: int

    :returns: bkg_mask - Masks pixels in the trace based on the deepstack or
        non-finite in the image.
    :rtype: array[bool]
    """

    # Get the dimensions of the input image.
    nrows, ncols = np.shape(deepstack)

    # Set the appropriate quantile for masking based on the subarray size.
    if nrows == 96:  # SUBSTRIP96
        quantile = 100*(1 - width/96)  # Mask 1 order worth of pixels.
    elif nrows == 256:  # SUBSTRIP256
        quantile = 100*(1 - 2*width/256)  # Mask 2 orders worth of pixels.
    elif nrows == 2048:  # FULL
        quantile = 100*(1 - 2*width/2048)  # Mask 2 orders worth of pixels.
    else:
        msg = ('Unexpected image dimensions, expected nrows = 96, 256 or 2048, '
               'got nrows = {}.')
        raise ValueError(msg.format(nrows))
        
    # Mask pixels above the threshold determined by the quantile.
    threshold = np.nanpercentile(deepstack, quantile)
    bkg_mask = (deepstack > threshold)

    return bkg_mask
    

def soss_oneoverf_correction(scidata, scimask, deepstack, bkg_mask=None,
                             zero_bias=False):
    """Compute a columnwise correction to the 1/f noise on the difference image
    of an inidividual SOSS integration (i.e. an individual integration - a deep
    image of the same observation).
    
    :param scidata: the image of the SOSS trace.
    :param scimask: a boolean mask of pixels to be excluded based on the DQ
        values.
    :param deepstack: a deep image of the trace constructed by combining
        individual integrations of the observation.
    :param bkg_mask:
        a boolean mask of pixels to be exluded because they are in the trace, if
        not provided a mask will be constructed based on the deepstack.
    :param zero_bias: if True the corrections to individual columns will be
        adjusted so that their mean is zero.

    :type scidata: array[float]
    :type scimask: array[bool]
    :type deepstack: array[float]
    :type bkg_mask: array[bool]
    :type zero_bias: bool

    :returns: scidata_cor, col_cor, npix_cor, bias - The 1/f corrected image,
        columnwise correction values, number of pixels used in each column, and
        the net change to the image if zero_bias was False.
    :rtype: Tuple(array[float], array[float], array[float], float)
    """

    # Check the validity of the input.
    data_shape = scidata.shape

    if scimask.shape != data_shape:
        msg = 'scidata and scimask must have the same shape.'
        raise ValueError(msg)

    if deepstack.shape != data_shape:
        msg = 'scidata and deepstack must have the same shape.'
        raise ValueError(msg)
    
    # Subtract the deep stack from the image.
    diffimage = scidata - deepstack

    # If not given build a background mask based on the deepstack.
    if bkg_mask is None:
        bkg_mask = make_background_mask(deepstack)
    elif bkg_mask.shape != data_shape:
        msg = 'scidata and bkg_mask must have the same shape.'
        raise ValueError(msg)

    # Combine the masks and create a masked array.
    mask = (scimask | bkg_mask) | ~np.isfinite(deepstack)  # TODO invalid values in deepstack?
    diffimage_masked = np.ma.array(diffimage, mask=mask)

    # Mask additional pixels using sigma-clipping.
    sigclip = SigmaClip(sigma=3, maxiters=None, cenfunc='mean')
    diffimage_clipped = sigclip(diffimage_masked, axis=0)

    # Compute the mean for each column and record the number of pixels used.
    col_cor = diffimage_clipped.mean(axis=0)
    npix_cor = (~diffimage_clipped.mask).sum(axis=0)

    # Compute the net change to the image.
    bias = np.nanmean(col_cor)

    # Set the net bias to zero.
    if zero_bias:
        col_cor = col_cor - bias

    # Apply the 1/f correction to the image.
    scidata_cor = scidata - col_cor

    return scidata_cor, col_cor, npix_cor, bias


def main():
    # test the thing

    # Geert Jan, I've put the fits files used as input here:
    # http://www.astro.umontreal.ca/~albert/jwst/

    import matplotlib.pyplot as plt

    a = fits.open('/home/talens-irex/Downloads/mask.fits')
    mask = a[0].data

    a = fits.open('/home/talens-irex/Downloads/deepstack.fits')
    deepstack = a[0].data
    deepstack = np.rot90(deepstack)

    a = fits.open('/home/talens-irex/Downloads/cds_256_ng3.fits')
    cube = a[0].data
    image = np.rot90(cube[10, :, :])

    imagecorr = np.zeros((2, 50, 256, 2048))
    allDC = np.zeros(50)
    for i in range(1):
        image = np.rot90(cube[i, :, :])
        mask = ~np.isfinite(image)
        # call the function with an already created mask. But in reality, you would want to
        # first combine the DMS DQ map plus the background mask.
        imagecorr[0, i, :, :], col_cor, npix_cor, allDC[i] = soss_oneoverf_correction(image, mask, deepstack)

        plt.subplot(211)
        plt.plot(col_cor)
        plt.subplot(212)
        plt.plot(npix_cor)
        plt.show()

        imagecorr[1, i, :, :] = imagecorr[0, i, :, :] - allDC[i]
        # print(i, allDC[i])

    plt.subplot(211)
    plt.imshow(image, vmin=-50, vmax=500)
    plt.subplot(212)
    plt.imshow(imagecorr[0, i], vmin=-50, vmax=500)
    plt.show()

    print(allDC)

    hdu = fits.PrimaryHDU()
    hdu.data = imagecorr
    hdu.writeto('imagecorr.fits', overwrite=True)

    return


if __name__ == '__main__':
    main()
