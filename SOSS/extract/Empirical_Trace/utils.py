#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Mar 11 14:35 2020

@author: MCR

Miscellaneous utility functions for the empirical trace construction.
"""

from astropy.io import fits
from datetime import datetime
import numpy as np
from scipy.optimize import least_squares


def _gen_ImageHDU_header(hdu, order, pad, oversample):
    '''Generate the appropriate fits header for reference file image HDUs.
    '''

    hdu.header['ORDER'] = order
    hdu.header.comments['ORDER'] = 'Spectral order'
    hdu.header['OVERSAMP'] = oversample
    hdu.header.comments['OVERSAMP'] = 'Pixel oversampling'
    hdu.header['PADDING'] = pad
    hdu.header.comments['PADDING'] = 'Native pixel-size padding around the image'
    hdu.header['EXTNAME'] = 'ORDER   '
    hdu.header['EXTVER'] = order

    return hdu


def _gen_PrimaryHDU_header(hdu, subarray, filename):
    '''Generate the appropriate header for the reference file primary HDU.
    '''

    hdu.header['DATE'] = str(datetime.utcnow())
    hdu.header.comments['DATE'] = 'Date this file was created (UTC)'
    hdu.header['ORIGIN'] = 'SOSS Team MTL'
    hdu.header.comments['ORIGIN'] = 'Orginazation responsible for creating file'
    hdu.header['TELESCOP'] = 'JWST    '
    hdu.header.comments['TELESCOP'] = 'Telescope used to acquire the data'
    hdu.header['INSTRUME'] = 'NIRISS  '
    hdu.header.comments['INSTRUME'] = 'Instrument used to acquire the data'
    hdu.header['SUBARRAY'] = subarray
    hdu.header.comments['SUBARRAY'] = 'Subarray used'
    hdu.header['FILENAME'] = filename
    hdu.header.comments['FILENAME'] = 'Name of the file'
    hdu.header['REFTYPE'] = 'SPECPROFILE'
    hdu.header.comments['REFTYPE'] = 'Reference file type'
    hdu.header['PEDIGREE'] = 'GROUND  '
    hdu.header.comments['PEDIGREE'] = 'The pedigree of the refernce file'
    hdu.header['DESCRIP'] = '2D trace profile'
    hdu.header.comments['DESCRIP'] = 'Desription of the reference file'
    hdu.header['AUTHOR'] = 'Michael Radica'
    hdu.header.comments['AUTHOR'] = 'Author of the reference file'
    hdu.header['USEAFTER'] = '2000-01-01T00:00:00'
    hdu.header.comments['USEAFTER'] = 'Use after date of the reference file'
    hdu.header['EXP_TYPE'] = 'NIS_SOSS'
    hdu.header.comments['EXP_TYPE'] = 'Type of data in the exposure'

    return hdu


def _lik(k, data, model):
    '''Utility likelihood function for flux rescaling. Esssentially a Chi^2
    multiplied by the data such that wing values don't carry too much weight.
    '''
    return np.nansum((data - k*model)**2)


def _local_mean(array, step):
    '''Calculate the mean of an array in chunks of 2*step.
    '''
    running_means = []
    for i in range(-step, step):
        if i == 0:
            continue
        running_means.append(np.roll(array, i))
    loc_mean = np.mean(running_means, axis=0)

    return loc_mean


def _robust_polyfit(x, y, p0):
    '''Wrapper around scipy's least_squares fitting routine implementing the
     Huber loss function - to be more resistant to outliers.

    Parameters
    ----------
    x : list
        Data describing dependant variable.
    y : list
        Data describing independant variable.
    p0 : tuple
        Initial guess straight line parameters. The length of p0 determines the
        polynomial order to be fit - i.e. a length 2 tuple will fit a 1st order
        polynomial, etc.

    Returns
    -------
    res.x : list
        Best fitting parameters of the desired polynomial order.
    '''

    def poly_res(p, x, y):
        '''Residuals from a polynomial'''
        return np.polyval(p, x) - y

    # Preform outlier resistant fitting.
    res = least_squares(poly_res, p0, loss='huber', f_scale=0.1, args=(x, y))
    return res.x


def _verbose_to_bool(verbose):
    '''Convert integer verbose to bool to disable or enable progress bars.
    '''

    if verbose in [2, 3]:
        verbose_bool = False
    else:
        verbose_bool = True

    return verbose_bool


def _write_to_file(order1, order2, subarray, filename, pad, oversample):
    '''Utility function to write the 2D trace profile to disk. Data will be
    saved as a multi-extension fits file.

    Parameters
    ----------
    order1 : np.ndarray (2D)
        Uncontaminated first order data frame.
    order2 : np.ndarray (2D)
        Uncontaminated first order data frame.
    subarray : str
        Subarray used.
    filename : str
        Name of the file to which to write the data.
    pad : int
        Amount of padding in native pixels.
    oversample : int
        Oversampling factor.
    '''

    # Generate the primary HDU with appropriate header keywords.
    hdu_p = fits.PrimaryHDU()
    hdu_p = _gen_PrimaryHDU_header(hdu_p, subarray, filename)
    hdulist = [hdu_p]
    # Generate an ImageHDU for the first order profile.
    hdu_1 = fits.ImageHDU(data=order1)
    hdu_1 = _gen_ImageHDU_header(hdu_1, 1, pad, oversample)
    hdulist.append(hdu_1)
    # Generate an ImageHDU for the second order profile.
    hdu_2 = fits.ImageHDU(data=order2)
    hdu_2 = _gen_ImageHDU_header(hdu_2, 2, pad, oversample)
    hdulist.append(hdu_2)

    # Write the file to disk.
    hdu = fits.HDUList(hdulist)
    hdu.writeto(filename, overwrite=True)
