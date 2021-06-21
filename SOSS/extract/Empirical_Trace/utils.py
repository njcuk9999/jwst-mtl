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
import pandas as pd
from scipy.optimize import least_squares
import warnings
from SOSS.extract.empirical_trace import _calc_interp_coefs

# Local path to reference files.
path = '/Users/michaelradica/Documents/GitHub/jwst-mtl/SOSS/extract/empirical_trace/'


def _gen_ImageHDU_header(hdu, order, pad, oversample):
    '''Generate the appropriate fits header for reference file image HDUs.

    Parameters
    ----------
    hdu : HDU object
        Image HDU object.
    order : int
        Diffraction order.
    pad : int
        Amount of padding in native pixels.
    oversample : int
        Oversampling factor.

    Returns
    -------
    hdu : HDU object
        Image HDU object with appropriate header added.
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

    Parameters
    ----------
    hdu : HDU object
        Primary HDU object.
    subarray : str
        Subarray identifier.
    filname : str
        Output filename.

    Returns
    -------
    hdu : HDU object
        Primary HDU object with appropriate header added.
    '''

    hdu.header['DATE'] = str(datetime.utcnow())
    hdu.header.comments['DATE'] = 'Date this file was created (UTC)'
    hdu.header['ORIGIN'] = 'SOSS Team MTL'
    hdu.header.comments['ORIGIN'] = 'Orginization responsible for creating file'
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


def _read_interp_coefs(F277W=True, verbose=0):
    '''Read the interpolation coefficients from the appropriate reference file.
    If the reference file does not exist, or the correct coefficients cannot be
    found, they will be reclaculated.

    Parameters
    ----------
    F277W : bool
        If True, selects the coefficients with a 2.45µm red anchor.
    verbose : int
        Level of verbosity.

    Returns
    -------
    coef_b : np.ndarray
        Blue anchor coefficients.
    coef_r : np.ndarray
        Red anchor coefficients.
    '''

    # Attemot to read interpolation coefficients from reference file.
    try:
        df = pd.read_csv(path+'Ref_files/interpolation_coefficients.csv')
        # If there is an F277W exposure, get the coefficients to 2.45µm.
        if F277W is True:
            coef_b = np.array(df['F_blue'])
            coef_r = np.array(df['F_red'])
        # For no F277W exposure, get the coefficients out to 2.9µm.
        else:
            coef_b = np.array(df['NF_blue'])
            coef_r = np.array(df['NF_red'])
    # If the reference file does not exists, or the appropriate coefficients
    # have not yet been generated, call the _calc_interp_coefs function to
    #  calculate them.
    except (FileNotFoundError, KeyError):
        print('No interpolation coefficients found. \
They will be calculated now.')
        coef_b, coef_r = _calc_interp_coefs.calc_interp_coefs(F277W=F277W,
                                                              verbose=verbose)

    return coef_b, coef_r


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


def _poly_res(p, x, y):
    '''Residuals from a polynomial.
    '''
    return np.polyval(p, x) - y


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

    # Preform outlier resistant fitting.
    res = least_squares(_poly_res, p0, loss='huber', f_scale=0.1, args=(x, y))
    return res.x


def _validate_inputs(etrace):
    '''Validate the input parameters for the empirical trace construction
    module, and determine the correct subarray for the data.

    Parameters
    ----------
    etrace : Empirical_Trace instance
        Instance of an Empirical_Trace object.

    Returns
    -------
    subarray : str
        The correct NIRISS/SOSS subarray identifier corresponding to the CLEAR
        dataframe.
    '''

    # Ensure F277W and CLEAR have the same dimensions.
    if etrace.F277W is not None:
        if np.shape(etrace.F277W) != np.shape(etrace.CLEAR):
            msg = 'F277W and CLEAR dataframes must be the same shape.'
            raise ValueError(msg)
    # Ensure bad pixel mask and clear have the same dimensions.
    if np.shape(etrace.CLEAR) != np.shape(etrace.badpix_mask):
        raise ValueError('Bad pixel mask must be the same shape as the data.')
    # Ensure padding and oversampling are integers.
    if type(etrace.pad) != tuple and len(etrace.pad) != 2:
        raise ValueError('Padding must be a length 2 tuple.')
    if type(etrace.oversample) != int:
        raise ValueError('Oversampling factor must be an integer.')
    # Ensure verbose is the correct format.
    if etrace.verbose not in [0, 1, 2, 3]:
        raise ValueError('Verbose argument must be in the range 0 to 3.')

    # Determine correct subarray dimensions.
    dimy, dimx = np.shape(etrace.CLEAR)
    if dimy == 96:
        subarray = 'SUBSTRIP96'
        # Fail if user wants to create reference files with a SUBSTRIP96
        # exposure. Use a SUBSTRIP256 for this.
        if etrace.oversample != 1 or etrace.pad != (0, 0):
            errmsg = 'The creation of reference files is not supported for \
SUBSTRIP96. Please use a SUBSTRIP256 observation instead.'
            raise NotImplementedError(errmsg)
        # Warn the user that only the first pass, first order profile can be
        # generated for SUBSTRIP96 data.
        warnmsg = 'Only a first order 2D profile can be generated for \
SUBSTRIP96.\nPlease use a reference file for the second order.'
        warnings.warn(warnmsg)
    elif dimy == 256:
        subarray = 'SUBSTRIP256'
    elif dimy == 2048:
        subarray = 'FULL'
    else:
        raise ValueError('Unrecognized subarray: {}x{}.'.format(dimy, dimx))

    return subarray


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
        Uncontaminated first order data frame. Pass None to only write the
        first order profile to file.
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
    if order2 is not None:
        hdu_2 = fits.ImageHDU(data=order2)
        hdu_2 = _gen_ImageHDU_header(hdu_2, 2, pad, oversample)
        hdulist.append(hdu_2)

    # Write the file to disk.
    hdu = fits.HDUList(hdulist)
    hdu.writeto(filename, overwrite=True)
