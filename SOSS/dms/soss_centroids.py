#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings

import numpy as np

from astropy.io import fits

from soss_utils import zero_roll, robust_polyfit, get_image_dim

from matplotlib import colors
import matplotlib.pyplot as plt

PATH = '/home/talens-irex/Dropbox/SOSS_Ref_Files'


def _plot_centroid(image, xtrace, ytrace):
    """Overplot the extracted trace positions on the image."""

    nrows, ncols = image.shape

    plt.figure(figsize=(ncols/128, nrows/128))

    plt.imshow(image, origin='lower', cmap='inferno', norm=colors.LogNorm())
    plt.plot(xtrace, ytrace, lw=2, c='black')

    plt.show()
    plt.close()

    return


def center_of_mass(column, ypos, halfwidth):
    """Compute a windowed center-of-mass along a column.

    :param column: The column on which to compute the windowed center of mass.
    :param ypos: The position along the column to center the window on.
    :param halfwidth: The half-size of the window in pixels.

    :type column: array[float]
    :type ypos: float
    :type halfwidth: int

    :returns: ycom - the centerof-mass of the pixels withn the window.
    :rtype: float
    """

    # Get the column shape and create a corresponding array of positions.
    dimy, = column.shape
    ypix = np.arange(dimy)

    # Find the indices of the window.
    miny = np.int(np.fmax(np.around(ypos - halfwidth), 0))
    maxy = np.int(np.fmin(np.around(ypos + halfwidth + 1), dimy))

    # Compute the center of mass on the window.
    with np.errstate(invalid='ignore'):
        ycom = np.nansum(column[miny:maxy]*ypix[miny:maxy])/np.nansum(column[miny:maxy])

    return ycom


def get_uncontam_centroids(image, header=None, mask=None, poly_order=11, verbose=False):
    """Determine the x, y coordinates of the trace using a center-of-mass analysis.
    Works for either order if there is no contamination, or for order 1 on a detector
    where the two orders are overlapping.

    :param image: A 2D image of the detector.
    :param header: The header from one of the SOSS reference files.
    :param mask: A boolean array of the same shape as stack. Pixels corresponding to True values will be masked.
    :param poly_order: Order of the polynomial to fit to the extracted trace positions.
    :param verbose: If set True some diagnostic plots will be made.

    :type image: array[float]
    :type header: astropy.io.fits.Header
    :type mask: array[bool]
    :type poly_order: int
    :type verbose: bool

    :returns: xtrace, ytrace, param - The x, y coordinates of trace as computed from the best fit polynomial
    and the best-fit polynomial parameters.
    :rtype: Tuple(array[float], array[float], array[float])
    """

    # If no mask was given use all pixels.
    if mask is None:
        mask = np.zeros_like(image, dtype='bool')

    # Call the script that determines the dimensions of the stack.
    dimx, dimy, xos, yos, xnative, ynative, padding, refpix_mask = \
        get_image_dim(image, header=header, verbose=verbose)

    # Replace masked pixel values with NaNs.
    image_masked = np.where(mask | ~refpix_mask, np.nan, image)

    # Compute and subtract the background level of each column.
    col_bkg = np.nanpercentile(image_masked, 10, axis=0)
    image_masked_bkg = image_masked - col_bkg

    # Find centroid - first pass, use all pixels in the column.
    # Normalize each column
    with np.errstate(invalid='ignore'):
        image_norm = image_masked_bkg / np.nanmax(image_masked_bkg, axis=0)

    # Create 2D Array of pixel positions.
    xpix = np.arange(dimx)
    ypix = np.arange(dimy)
    _, ygrid = np.meshgrid(xpix, ypix)

    # CoM analysis to find initial positions using all rows.
    with np.errstate(invalid='ignore'):
        ytrace = np.nansum(image_norm*ygrid, axis=0)/np.nansum(image_norm, axis=0)

    # Second pass - use a windowed CoM at the previous position.
    halfwidth = 30 * yos
    for icol in range(dimx):

        ycom = center_of_mass(image_norm[:, icol], ytrace[icol], halfwidth)

        # If NaN was returned we are done.
        if not np.isfinite(ycom):
            ytrace[icol] = np.nan
            continue

        # If the pixel at the centroid is below the local mean we are likely mid-way between orders and
        # we should shift the window downward to get a reliable centroid for order 1.
        irow = np.int(np.around(ycom))
        miny = np.int(np.fmax(np.around(ycom) - halfwidth, 0))
        maxy = np.int(np.fmin(np.around(ycom) + halfwidth + 1, dimy))
        if image_norm[irow, icol] < np.nanmean(image_norm[miny:maxy, icol]):
            ycom = center_of_mass(image_norm[:, icol], ycom - halfwidth, halfwidth)

        # If NaN was returned or the position is too close to the array edge, use NaN.
        if not np.isfinite(ycom) or (ycom <= 5 * yos) or (ycom >= (ynative - 6) * yos):
            ytrace[icol] = np.nan
            continue

        # Update the position if the above checks were succesfull.
        ytrace[icol] = ycom

    # Third pass - fine tuning using a smaller window.
    halfwidth = 16 * yos
    for icol in range(dimx):

        ytrace[icol] = center_of_mass(image_norm[:, icol], ytrace[icol], halfwidth)

    # Fit the y-positions with a polynomial and use the result as the true y-positions.
    xtrace = np.arange(dimx)
    mask = np.isfinite(ytrace)

    # For padded arrays ignore padding for consistency with real data
    if padding != 0:
        mask = mask & (xtrace >= xos*padding) & (xtrace < (dimx - xos*padding))

    param = robust_polyfit(xtrace[mask], ytrace[mask], poly_order)
    ytrace = np.polyval(param, xtrace)

    # If verbose visualize the result.
    if verbose is True:
        _plot_centroid(image_masked, xtrace, ytrace)

    return xtrace, ytrace, param


def test_uncontam_centroids():
    """"""

    filename = os.path.join(PATH, 'SOSS_ref_2D_profile_SUBSTRIP256.fits.gz')

    image, header = fits.getdata(filename, ext=2, header=True)
    xtrace, ytrace, param = get_uncontam_centroids(image, header=header, verbose=True)

    return


def edge_trigger(image, halfwidth=5, yos=1, verbose=False):
    """Detect the edges and center of the trace based on the minima and maxima of the derivate
     of the columns, which is computed in a running window along the columns of the detector image

     :param image: A 2D image of the detector.
     :param halfwidth: the size of the window used when computing the derivatives.
     :param yos: the oversampling factor of the image array along the y-direction.
     :param verbose: If set True some diagnostic plots will be made.

     :type image: array[float]
     :type halfwidth: int
     :type yos: int
     :type verbose: bool

     :returns: ytrace_max, ytrace_min, ytrace_comb - The upper edge, lower edge and center of the trace.
     :rtype: Tuple(array[float], array[float], array[float])
     """

    dimy, dimx = image.shape
    halfwidth = halfwidth * yos

    # Create coordinate arrays.
    xpix = np.arange(dimx)
    ypix = np.arange(dimy)
    _, ygrid = np.meshgrid(xpix, ypix)

    # Compute windowed slopes over the columns.
    slopevals = np.zeros_like(image)
    for irow in range(halfwidth, dimy-halfwidth):

        # Compute the window indices.
        ymin = irow - halfwidth
        ymax = irow + halfwidth + 1

        # Get the x and y data to find the slope to.
        datay = image[ymin:ymax, :]
        mask = np.isfinite(datay)
        datax = np.where(mask, ygrid[ymin:ymax, :], np.nan)  # Need to set values NaN in y to NaN in x.

        # Compute the slope.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xmean = np.nanmean(datax, axis=0, keepdims=True)
            ymean = np.nanmean(datay, axis=0, keepdims=True)

        with np.errstate(invalid='ignore'):
            slope = np.nansum((datax - xmean) * (datay - ymean), axis=0) / np.nansum((datax - xmean) ** 2, axis=0)

        # Set slopes computed from < 3 datapoints to NaN.
        slopevals[irow, :] = np.where(np.sum(mask, axis=0) >= 3, slope, 0.)

    # Find the upper and lower bounds on the trace.
    args = np.nanargmax(slopevals, axis=0)
    vals = np.nanmax(slopevals, axis=0)
    ytrace_max = np.where(vals != 0, ypix[args], np.nan)

    args = np.nanargmin(slopevals, axis=0)
    vals = np.nanmin(slopevals, axis=0)
    ytrace_min = np.where(vals != 0, ypix[args], np.nan)

    # Scan through a range of trace widths.
    slopes_best = np.zeros_like(xpix)
    ytrace_best = np.zeros_like(xpix)
    widths_best = np.zeros_like(xpix)
    for width in range(18*yos, 27*yos):

        # Add the slope and its offset negative.
        comb = slopevals - zero_roll(slopevals, -width)

        # Find the maximum resulting slope.
        args = np.nanargmax(comb, axis=0)
        vals = np.nanmax(comb, axis=0)

        # Update the best values.
        mask = (vals > slopes_best)
        slopes_best = np.where(mask, vals, slopes_best)
        ytrace_best = np.where(mask, ypix[args], ytrace_best)
        widths_best = np.where(mask, width, widths_best)

    # Set the y position to NaN if the best slope was zero.
    ytrace_best = np.where(slopes_best != 0, ytrace_best + widths_best/2., np.nan)

    if verbose:

        plt.imshow(image, origin='lower', cmap='inferno', norm=colors.LogNorm())
        plt.plot(ytrace_min, lw=2, ls='--', c='black')
        plt.plot(ytrace_max, lw=2, ls='--', c='black')
        plt.plot(ytrace_best, lw=2, c='black')

        plt.show()
        plt.close()

    return ytrace_max, ytrace_min, ytrace_best


def get_uncontam_centroids_edgetrig(image, header=None, mask=None, poly_order=11,
                                    halfwidth=5, mode='combined', verbose=False):
    """Determine the x, y coordinates of the trace using the derivatives along the y-axis.
    Works for either order if there is no contamination.

    :param image: A 2D image of the detector.
    :param header: The header from one of the SOSS reference files.
    :param mask: A boolean array of the same shape as stack. Pixels corresponding to True values will be masked.
    :param poly_order: Order of the polynomial to fit to the extracted trace positions.
    :param halfwidth: the size of the window used when computing the derivatives.
    :param mode: Which trace values to use. Can be 'maxedge', 'minedge', 'mean' or 'combined'.
    :param verbose: If set True some diagnostic plots will be made.

    :type image: array[float]
    :type header: astropy.io.fits.Header
    :type mask: array[bool]
    :type poly_order: int
    :type halfwidth: int
    :type mode: str
    :type verbose: bool

    :returns: xtrace, ytrace, tracewidth, param - The x, y coordinates of trace as computed from the best fit polynomial
    and the best-fit polynomial parameters.
    :rtype: Tuple(array[float], array[float], array[float])
    """

    # If no mask was given use all pixels.
    if mask is None:
        mask = np.zeros_like(image, dtype='bool')

    # Call the script that determines the dimensions of the stack.
    dimx, dimy, xos, yos, xnative, ynative, padding, refpix_mask = \
        get_image_dim(image, header=header, verbose=verbose)

    # Replace masked pixel values with NaNs.
    image_masked = np.where(mask | ~refpix_mask, np.nan, image)

    # Use edge trigger to compute the edges and center of the trace.
    ytrace_max, ytrace_min, ytrace_comb = edge_trigger(image_masked, halfwidth=halfwidth, yos=yos, verbose=verbose)

    # Compute an estimate of the trace width.
    tracewidth = np.abs(ytrace_min - ytrace_max)

    # Use different y-positions depending on the mode parameter.
    if mode == 'maxedge':
        ytrace = ytrace_max
    elif mode == 'minedge':
        ytrace = ytrace_min
    elif mode == 'mean':
        ytrace = (ytrace_min + ytrace_max)/2.
    elif mode == 'combined':
        ytrace = ytrace_comb
    else:
        raise ValueError('Unknow mode: {}'.format(mode))

    # Fit the y-positions with a polynomial and use the result as the true y-positions.
    xtrace = np.arange(dimx)
    mask = np.isfinite(ytrace)

    param = robust_polyfit(xtrace[mask], ytrace[mask], poly_order)
    ytrace = np.polyval(param, xtrace)

    # If verbose visualize the result.
    if verbose is True:
        _plot_centroid(image_masked, xtrace, ytrace)

    return xtrace, ytrace, tracewidth, param


def test_uncontam_centroids_edgetrig():
    """"""

    filename = os.path.join(PATH, 'SOSS_ref_2D_profile_SUBSTRIP256.fits.gz')

    image, header = fits.getdata(filename, ext=2, header=True)
    xtrace, ytrace, width, param = get_uncontam_centroids_edgetrig(image, header=header, verbose=True)

    filename = '/home/talens-irex/Downloads/stack_256_ng3_DMS.fits'

    image = fits.getdata(filename)
    xtrace, ytrace, width, param = get_uncontam_centroids_edgetrig(image, verbose=True)

    return


def main():

    test_uncontam_centroids()
    test_uncontam_centroids_edgetrig()

    return


if __name__ == '__main__':
    main()
