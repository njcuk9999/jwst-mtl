#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Mar 11 14:35 2020

@author: MCR

Miscellaneous utility functions for APPLESOSS.
"""

from astropy.io import fits
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm

from SOSS.extract.applesoss import _calibrations
from SOSS.extract.applesoss import plotting


def _gen_imagehdu_header(hdu, order, pad, oversample):
    """Generate the appropriate fits header for reference file image HDUs.

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
    """

    hdu.header['ORDER'] = order
    hdu.header.comments['ORDER'] = 'Spectral order'
    hdu.header['OVERSAMP'] = oversample
    hdu.header.comments['OVERSAMP'] = 'Pixel oversampling'
    hdu.header['PADDING'] = pad
    hdu.header.comments['PADDING'] = 'Native pixel-size padding around the image'
    hdu.header['EXTNAME'] = 'ORDER   '
    hdu.header['EXTVER'] = order

    return hdu


def _gen_primaryhdu_header(hdu, subarray, filename):
    """Generate the appropriate header for the reference file primary HDU.

    Parameters
    ----------
    hdu : HDU object
        Primary HDU object.
    subarray : str
        Subarray identifier.
    filename : str
        Output filename.

    Returns
    -------
    hdu : HDU object
        Primary HDU object with appropriate header added.
    """

    hdu.header['DATE'] = str(datetime.utcnow())
    hdu.header.comments['DATE'] = 'Date this file was created (UTC)'
    hdu.header['ORIGIN'] = 'SOSS Team MTL'
    hdu.header.comments['ORIGIN'] = 'Organization responsible for creating file'
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
    hdu.header.comments['PEDIGREE'] = 'The pedigree of the reference file'
    hdu.header['DESCRIP'] = '2D trace profile'
    hdu.header.comments['DESCRIP'] = 'Description of the reference file'
    hdu.header['AUTHOR'] = 'Michael Radica'
    hdu.header.comments['AUTHOR'] = 'Author of the reference file'
    hdu.header['USEAFTER'] = '2000-01-01T00:00:00'
    hdu.header.comments['USEAFTER'] = 'Use after date of the reference file'
    hdu.header['EXP_TYPE'] = 'NIS_SOSS'
    hdu.header.comments['EXP_TYPE'] = 'Type of data in the exposure'

    return hdu


def get_wave_solution(order):
    """Extract wavelength calibration information from the wavelength solution
    reference file.

    Parameters
    ----------
    order : int
        Diffraction order.

    Returns
    -------
    wavecal_x : np.array
        X pixel coordinate.
    wavecal_w : np.array
        Wavelength value.
    """

    # Get wavelength calibration reference file.
    wave_soln = 'Ref_files/jwst_niriss_wavemap_0013.fits'
    wavemap = fits.getdata(wave_soln, order)
    header = fits.getheader(wave_soln, order)
    ovs = header['OVERSAMP']
    pad = header['PADDING']

    # Bin the map down to native resolution and remove padding.
    nrows, ncols = wavemap.shape
    trans_map = wavemap.reshape((nrows // ovs), ovs, (ncols // ovs), ovs)
    trans_map = trans_map.mean(1).mean(-1)
    trans_map = trans_map[pad:-pad, pad:-pad]
    dimy, dimx = np.shape(trans_map)
    # Collapse over the spatial dimension.
    wavecal_w = np.nanmean(trans_map, axis=0)
    wavecal_x = np.arange(dimx)

    return wavecal_x, wavecal_w


def lik(k, data, model):
    """Utility likelihood function for flux rescaling. Essentially a Chi^2
    multiplied by the data such that wing values don't carry too much weight.
    """
    return np.nansum((data - k*model)**2)


def local_mean(array, step):
    """Calculate the mean of an array in chunks of 2*step.
    """
    running_means = []
    for i in range(-step, step):
        if i == 0:
            continue
        running_means.append(np.roll(array, i))
    loc_mean = np.mean(running_means, axis=0)

    return loc_mean


def _poly_res(p, x, y):
    """Residuals from a polynomial.
    """
    return np.polyval(p, x) - y


def read_interp_coefs(f277w=True, verbose=0):
    """Read the interpolation coefficients from the appropriate reference file.
    If the reference file does not exist, or the correct coefficients cannot be
    found, they will be recalculated.

    Parameters
    ----------
    f277w : bool
        If True, selects the coefficients with a 2.45µm red anchor.
    verbose : int
        Level of verbosity.

    Returns
    -------
    coef_b : np.array
        Blue anchor coefficients.
    coef_r : np.array
        Red anchor coefficients.
    """

    # Attempt to read interpolation coefficients from reference file.
    try:
        df = pd.read_csv('Ref_files/interpolation_coefficients.csv')
        # If there is an F277W exposure, get the coefficients to 2.45µm.
        if f277w is True:
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
        print('No interpolation coefficients found. They will be calculated now.')
        coef_b, coef_r = _calibrations.calc_interp_coefs(f277w=f277w,
                                                         verbose=verbose)

    return coef_b, coef_r


def read_width_coefs(verbose=0):
    """Read the width coefficients from the appropriate reference file.
    If the reference file does not exist, the coefficients will be
    recalculated.

    Parameters
    ----------
    verbose : int
        Level of verbosity.

    Returns
    -------
    wc : np.array
        Width calbration polynomial coefficients.
    """

    # First try to read the width calibration file, if it exists.
    try:
        coef_file = pd.read_csv('Ref_files/width_coefficients.csv')
        wc = np.array(coef_file['width_coefs'])
    # If file does not exist, redo the width calibration.
    except FileNotFoundError:
        print('No width coefficients found. They will be calculated now.')
        wc = _calibrations.derive_width_relations(verbose=verbose)

    return wc


def replace_badpix(clear, thresh=5, box_size=10, verbose=0):
    """Replace bad pixels with the median of the pixels values of a box
    centered on the bad pixel.

    Parameters
    ----------
    clear : np.array
        Data frame.
    thresh : int
        Threshold in standard deviations to flag a bad pixel.
    box_size : int
        Box side half-length to use.
    verbose : int
        level of verbosity.

    Returns
    -------
    clear_r : np.array
        Data frame with bad pixels interpolated.
    """

    # Initial setup of arrays and variables
    disable = verbose_to_bool(verbose)
    clear_r = clear.copy()
    counts = 0
    dimy, dimx = np.shape(clear)

    def get_interp_box(data, box_size, i, j, dimx, dimy):
        """ Get median and standard deviation of a box centered on a specified
        pixel.
        """

        # Get the box limits.
        low_x = np.max([i - box_size, 0])
        up_x = np.min([i + box_size, dimx - 1])
        low_y = np.max([j - box_size, 0])
        up_y = np.min([j + box_size, dimy - 1])

        # Calculate median and std deviation of box.
        median = np.nanmedian(data[low_y:up_y, low_x:up_x])
        stddev = np.nanstd(data[low_y:up_y, low_x:up_x])

        # Pack into array.
        box_properties = np.array([median, stddev])

        return box_properties

    # Loop over all pixels and interpolate those that deviate by more than the
    # threshold from the surrounding median.
    for i in tqdm(range(dimx), disable=disable):
        for j in range(dimy):
            box_size_i = box_size
            box_prop = get_interp_box(clear_r, box_size_i, i, j, dimx, dimy)

            # Ensure that the median and std dev we extract are good.
            # If not, increase the box size until they are.
            while np.any(np.isnan(box_prop)) or np.any(box_prop < 0):
                box_size_i += 1
                box_prop = get_interp_box(clear_r, box_size_i, i, j, dimx,
                                          dimy)
            med, std = box_prop[0], box_prop[1]

            # If central pixel is too deviant (or nan, or negative) replace it.
            if np.abs(clear_r[j, i] - med) > thresh*std or np.isnan(clear_r[j, i]) or clear_r[j, i] < 0:
                clear_r[j, i] = med
                counts += 1

    # Print statistics and show debugging plot if necessary.
    if verbose != 0:
        print('{:.2f}% of pixels interpolated.'.format(counts/dimx/dimy*100))
        if verbose == 3:
            plotting.plot_badpix(clear, clear_r)

    return clear_r


def robust_polyfit(x, y, p0):
    """Wrapper around scipy's least_squares fitting routine implementing the
     Huber loss function - to be more resistant to outliers.

    Parameters
    ----------
    x : list, np.array
        Data describing dependant variable.
    y : list, np.array
        Data describing independent variable.
    p0 : tuple
        Initial guess straight line parameters. The length of p0 determines the
        polynomial order to be fit - i.e. a length 2 tuple will fit a 1st order
        polynomial, etc.

    Returns
    -------
    res.x : list
        Best fitting parameters of the desired polynomial order.
    """

    # Preform outlier resistant fitting.
    res = least_squares(_poly_res, p0, loss='huber', f_scale=0.1, args=(x, y))
    return res.x


def sigma_clip(xdata, ydata, thresh=5):
    """Perform rough sigma clipping on data to remove outliers.

    Parameters
    ----------
    xdata : list, np.array
        Independent variable.
    ydata : list, np.array
        Dependent variable.
    thresh : int
        Sigma threshold at which to clip.

    Returns
    -------
    xdata : np.array
        Independent variable; sigma clipped.
    ydata : np.array
        Dependent variable; sigma clipped.
    """

    xdata, ydata = np.atleast_1d(xdata), np.atleast_1d(ydata)
    # Get mean and standard deviation.
    mean = np.mean(ydata)
    std = np.std(ydata)
    # Points which are >thresh-sigma deviant.
    inds = np.where(np.abs(ydata - mean) < thresh*std)

    return xdata[inds], ydata[inds]


def validate_inputs(etrace):
    """Validate the input parameters for the empirical trace construction
    module, and determine the correct subarray for the data.

    Parameters
    ----------
    etrace : EmpiricalTrace instance
        Instance of an EmpiricalTrace object.

    Returns
    -------
    subarray : str
        The correct NIRISS/SOSS subarray identifier corresponding to the CLEAR
        dataframe.
    """

    # Ensure F277 exposure has same shapse as CLEAR.
    if etrace.f277w is not None:
        if np.shape(etrace.f277w) != np.shape(etrace.clear):
            msg = 'F277W and CLEAR frames must be the same shape.'
            raise ValueError(msg)
    # Ensure padding and oversampling are integers.
    if type(etrace.pad) != tuple and len(etrace.pad) != 2:
        raise ValueError('Padding must be a length 2 tuple.')
    if type(etrace.oversample) != int:
        raise ValueError('Oversampling factor must be an integer.')
    # Ensure verbose is the correct format.
    if etrace.verbose not in [0, 1, 2, 3]:
        raise ValueError('Verbose argument must be in the range 0 to 3.')

    # Determine correct subarray dimensions.
    dimy, dimx = np.shape(etrace.clear)
    if dimy == 96:
        # Fail if user wants to use a SUBSTRIP96 exposure
        msg = 'SUBSTRIP96 is currently not supported.'
        raise NotImplementedError(msg)
    elif dimy == 256:
        subarray = 'SUBSTRIP256'
    elif dimy == 2048:
        subarray = 'FULL'
    else:
        raise ValueError('Unrecognized subarray: {}x{}.'.format(dimy, dimx))

    return subarray


def verbose_to_bool(verbose):
    """Convert integer verbose to bool to disable or enable progress bars.
    """

    if verbose in [2, 3]:
        verbose_bool = False
    else:
        verbose_bool = True

    return verbose_bool


def write_to_file(order1, order2, subarray, filename, pad, oversample):
    """Utility function to write the 2D trace profile to disk. Data will be
    saved as a multi-extension fits file.

    Parameters
    ----------
    order1 : np.array (2D)
        Uncontaminated first order data frame.
    order2 : np.array (2D)
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
    """

    # Generate the primary HDU with appropriate header keywords.
    hdu_p = fits.PrimaryHDU()
    hdu_p = _gen_primaryhdu_header(hdu_p, subarray, filename)
    hdulist = [hdu_p]
    # Generate an ImageHDU for the first order profile.
    hdu_1 = fits.ImageHDU(data=order1)
    hdu_1 = _gen_imagehdu_header(hdu_1, 1, pad, oversample)
    hdulist.append(hdu_1)
    # Generate an ImageHDU for the second order profile.
    if order2 is not None:
        hdu_2 = fits.ImageHDU(data=order2)
        hdu_2 = _gen_imagehdu_header(hdu_2, 2, pad, oversample)
        hdulist.append(hdu_2)

    # Write the file to disk.
    hdu = fits.HDUList(hdulist)
    hdu.writeto(filename, overwrite=True)
