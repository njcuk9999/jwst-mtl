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

    # Compute the threshold value above which a pixel should be masked.
    if nrows == 96:
        quantile = 100*(1 - width/96)  # Mask 1 order worth of pixels.
        threshold = np.nanpercentile(deepstack, quantile)
    elif nrows >= 256:
        quantile = 100*(1 - 2*width/256)  # Mask 2 orders worth of pixels. TODO does not make sense when nrows>256?
        threshold = np.nanpercentile(deepstack, quantile)
    else:
        msg = ('Unexpected image dimensions, expected nrows = 96 or '
               'nrows >= 256, got nrows = {}.')
        raise ValueError(msg.format(nrows))
        
    # Mask pixels above the threshold in the background mask.
    bkg_mask = (deepstack > threshold)

    return bkg_mask
    

def soss_oneoverf_correction(image, deepstack, backgroundmask=None, 
                             make_net_bias_zero=False):
    """This function does a column by column subtraction of a "difference"
    images (a SOSS integration - a deep stack of the same time series) to 
    remove the 1/f noise from each individual integration.
    
    image : Image for a single integration. The flux units have to be the same
            as that of the deep stack to allow for a proper subtraction.
    deepstack : Image obtained by stacking a large number of single integra-
            tions in the same observing sequence as the input image. The idea
            is for deepstack to be of significantly higher SNR than the image.
    errormap : Uncertainty associated with the input image. That will be used
            set thresholds for masking pixels having significant star flux. If
            no errormap is passed then an estimate of the photon noise + read
            out noise will be calculated based on the input image alone.
    backgroundmask : A mask of the pixels considered to be mostly free of
            star flux. Those with light contamination are masked out. If no
            such map is passed then one will be constructed.
    make_net_bias_zero : Controls whether or not we want the net overall flux
            (avaraged over the full image) to remain as it was before the
            column correction was applied. By dedault, make_net_bias_zero is
            False and we let the bias float.
            
    returns : 1) corrected image. i.e. image - correctionmap
              2) the overall DC offset produced by that correction
    
    """
    
    # Check that the deep stack image is of same dimensions as input image to
    # correct 1/f for.
    # TO DO:
    # Get the dimensions of the input image and output mask
    nrow, ncol = np.shape(image)
    
    # Subtract the deep stack from the image
    diffimage = image - deepstack
    
    '''
    Construct a map of backgroud pixels, i.e. pixels containing no or very
    little star flux.
    '''
    if backgroundmask is None:
        backgroundmask = make_background_pixel_mask(image, deepstack)
    hdu = fits.PrimaryHDU()
    hdu.data = backgroundmask
    hdu.writeto('mask.fits', overwrite=True)
    
    '''
    Calculate the correction map
    '''
    
    # Initialize a correction map
    correctionmap = np.zeros((nrow, ncol))
    
    # Work on the mask of pixels considered to be devoid of star flux
    diffimage_masked = diffimage * backgroundmask
    
    # Compute the average level of each column. The correction is simply
    # the average of each column. No linear or higher order fit.
    sigclip = SigmaClip(sigma_lower=3, sigma_upper=3, maxiters=None, cenfunc='mean')
    for col in range(ncol):
        # Extract one column
        column = diffimage_masked[:, col]
        # Retain elements whose deviations respect +/- 3 sigma
        column = sigclip(column)
        correctionmap[:, col] = np.ones(nrow) * column.mean()
    
    # Column levels, once corrected, have a net effect on the whole image, i.e.
    # the mean overall flux may change by some DC level. Here, we compute that
    # bias offset over the whole image. What you do with the information 
    # depends on the previous processing that went on, i.e. was their a prior
    # reference pixels correction made in the level 1 DMS pipeline or is the 
    # astrophysical scene variable? If it is not variable then do not apply
    # this image-wide DC offset.
    bias = np.mean(correctionmap)
    if make_net_bias_zero:
        correctionmap = correctionmap - bias

    print('correctionmap mean = {:}'.format(bias))

    hdu = fits.PrimaryHDU()
    hdu.data = correctionmap
    hdu.writeto('corr.fits', overwrite=True)
    
    # Return the image minus the levels of all "difference" image columns.
    # Also return the overall DC bias correction.
    return image - correctionmap, bias


def main():
    # test the thing

    # Geert Jan, I've put the fits files used as input here:
    # http://www.astro.umontreal.ca/~albert/jwst/

    a = fits.open('mask.fits')
    mask = a[0].data

    a = fits.open('/Users/albert/NIRISS/CV3/myanalysis/rehearsal/deepstack.fits')
    deepstack = a[0].data
    deepstack = np.rot90(deepstack)

    a = fits.open('/Users/albert/NIRISS/CV3/myanalysis/rehearsal/cds_256_ng3.fits')
    cube = a[0].data
    image = np.rot90(cube[10, :, :])

    imagecorr = np.zeros((2, 50, 256, 2048))
    allDC = np.zeros(50)
    for i in range(50):
        image = np.rot90(cube[i, :, :])
        # call the function with an already created mask. But in reality, you would want to
        # first combine the DMS DQ map plus the background mask.
        imagecorr[0, i, :, :], allDC[i] = soss_oneoverf_correction(image, deepstack, backgroundmask=mask)
        imagecorr[1, i, :, :] = imagecorr[0, i, :, :] - allDC[i]
        print(i, allDC[i])

    print(allDC)

    hdu = fits.PrimaryHDU()
    hdu.data = imagecorr
    hdu.writeto('imagecorr.fits', overwrite=True)

    return


if __name__ == '__main__':
    main()
