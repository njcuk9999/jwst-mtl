#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from astropy.io import fits
from astropy.stats import SigmaClip


def make_background_pixel_mask(image, deepstack):
    """This function creates a mask of the pixels considered to be mostly
    devoid of star flux.
    """

    # Get the dimensions of the input image and output mask
    nrow, ncol = np.shape(image)
    
    # Definition of masked or not masked pixels
    masked_value = np.nan
    notmasked_value = 1
    
    # Initialize the background mask to 1 everywhere
    bkgd_mask = np.ones((nrow, ncol)) * notmasked_value

    # Set to zero all pixels that are
    bkgd_mask[~np.isfinite(image)] = masked_value
    
    # On the deepstack, identify pixels mostly devoid of star flux, i.e. pixels
    # belonging to a low percentile threshold. Assuming a trace is 20 pixels
    # wide, for SUBSTRIP96 with only order 1, 20/96 ~ 20%. For SUBSTRIP256 or
    # FF, then traces of order 1 and 2 are present. It is a bit counter-intuit
    # ive here: making the trace width smaller ususally helps the fit.
    width = 20
    if nrow == 96:
        flux_threshold = np.nanpercentile(deepstack, 100*(1-width/96.0))
    if nrow >= 256:
        flux_threshold = np.nanpercentile(deepstack, 100*(1-2*width/256.0))
        
    # Mask those pixels in the background mask
    bkgd_mask[deepstack > flux_threshold] = masked_value

    return bkgd_mask
    

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
