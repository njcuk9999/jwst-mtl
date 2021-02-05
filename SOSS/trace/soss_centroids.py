#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt

PATH = '/home/talens-irex/Dropbox/SOSS_Ref_Files'


def determine_stack_dimensions(stack, header=None, verbose=False):
    """Determine the size of a stack array. Will be called by
    get_uncontam_centroids and make_trace_mask.

    Parameters
    ----------
    stack : array of floats (2D)
        Data frame. Assumes DMS orientation.
        This array could be a native pixel sized SOSS subarray or FF.
        Could also be a 2D trace reference file in which case padding exists
        around the edges, and the pixels may be oversampled by an integer
        factor.
    header : fits header
        Header associated to the stack array.
        If the header is None then some assumptions will be made regarding the
        stack array. If the header is passed, then specific keywords will be
        read in it to assess what the stack array is. This ensures that a 2D
        Trace Reference file will be digested properly.
    verbose : bool
        Control verbosity.

    Returns
    -------
    dimx, dimy : int
        The dimensions of the stack array.
    xos, yos : int
        The oversampling factor of the stack array.
    xnative, ynative : int
        The dimensions of the stack image, in native pixels.
    padding : int
        Amount of padding all around the image, in native pixels.
    working_pixel_bool : array of bool (2D)
        2D array of the same size as stack with boolean values of False where
        pixels are not light sensitive (the reference pixels). True elsewhere.

    Raises
    ------
    ValueError
        If the data dimensions are non-standard, or the header information
        does not match up with the data shape.
    """

    # Dimensions of the subarray.
    dimy, dimx = np.shape(stack)

    # Determine what is the input stack based either on its dimensions or on
    # the header if passed. Construct a mask of working pixels in case the
    # stack contains reference pixels.
    if header is None:
        # No header passed - Assume stack is valid SOSS subarray or FF, i.e.
        # 2048x256 or 2040x252 (working pixels) or multiple of if oversampled
        # 2048x96 or 2040x96 (working pixels) or multiple of
        # 2048x2048 or 2040x2040 (working pixels) or multiple of
        if (dimx % 2048) == 0:
            # stack is a multiple of native pixels.
            xnative = 2048
            # The x-axis oversampling is:
            xos = int(dimx / 2048)
        elif (dimx % 2040) == 0:
            # stack is a multiple of native *** working *** pixels.
            xnative = 2040
            # The y-axis oversampling is:
            xos = int(dimx / 2040)
        else:
            # stack x dimension has unrecognized size.
            raise ValueError('Stack X dimension has unrecognized size of {:}. Accepts 2048, 2040 or multiple of.'.format(dimx))
        # Check if the Y axis is consistent with the X axis.
        acceptable_ydim = [96, 256, 252, 2040, 2048]
        yaxis_consistent = False
        for accdim in acceptable_ydim:
            if dimy / (accdim*xos) == 1:
                # Found the acceptable dimension
                yos = np.copy(xos)
                ynative = np.copy(accdim)
                yaxis_consistent = True
        if yaxis_consistent is False:
            # stack y dimension is inconsistent with the x dimension.
            raise ValueError('Stack Y dimension ({:}) is inconsistent with X dimension ({:}) for acceptable SOSS arrays'.format(dimy, dimx))
        # Construct a boolean mask (true or false) of working pixels
        working_pixel_bool = np.full((dimy, dimx), True)

        # For dimensions where reference pixels would have been included in
        # stack, mask those reference pixels out.
        # Sizes 96, 252 and 2040 should not contain any reference pixel.
        if xnative == 2048:
            # Mask out the left and right columns of reference pixels
            working_pixel_bool[:, 0:xos * 4] = False
            working_pixel_bool[:, -xos * 4:] = False
        if ynative == 2048:
            # Mask out the top and bottom rows of reference pixels
            working_pixel_bool[0:yos * 4, :] = False
            working_pixel_bool[-yos * 4:, :] = False
        if ynative == 256:
            # Mask the top rows of reference pixels
            working_pixel_bool[-yos * 4:, :] = False

        # Initialize padding to zero in this case because it is not a 2D Trace
        # ref file.
        padding = int(0)

    else:
        # header was passed
        # Read in the relevant keywords
        xos, yos = int(header['OVERSAMP']), int(header['OVERSAMP'])
        padding = int(header['PADDING'])
        # The 2D Trace profile is for FULL FRAME so 2048x2048
        xnative, ynative = int(2048), int(2048)
        # Check that the stack respects its intended format
        if dimx != ((xnative+2*padding)*xos):
            # Problem
            raise ValueError('The header passed is inconsistent with the X dimension of the stack.')
        if dimy != ((ynative+2*padding)*yos):
            # Problem
            raise ValueError('The header passed is inconsistent with the Y dimension of the stack.')
        # Construct a mask of working pixels. The 2D Trace REFERENCE file does
        # not contain any reference pixel. So all are True.
        working_pixel_bool = np.full((dimy, dimx), True)

    # For debugging purposes...
    if verbose is True:
        print('Data dimensions:\n')
        print('dimx={:}, dimy={:}, xos={:}, yos={:}, xnative={:}, ynative={:}'.format(dimx, dimy, xos, yos, xnative, ynative))

    return dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool


def _plot_centroid(image, x, y):
    """Overplot the extracted trace positions on the image."""

    nrows, ncols = image.shape

    plt.figure(figsize=(ncols/128, nrows/128))

    plt.imshow(image, origin='lower', cmap='inferno')
    plt.plot(x, y, lw=2, c='white')

    plt.show()
    plt.close()

    return


def center_of_mass(column, ypos, halfwidth):
    """"""

    dimy, = column.shape
    ypix = np.arange(dimy)

    miny = np.int(np.fmax(np.around(ypos - halfwidth), 0))
    maxy = np.int(np.fmin(np.around(ypos + halfwidth), dimy))

    com = np.nansum(column[miny:maxy]*ypix[miny:maxy])/np.nansum(column[miny:maxy])

    return com


# TODO add widths as parameters.
def get_uncontam_centroids(stack, header=None, mask=None, poly_order=11, verbose=False):
    """Determine the x, y positions of the trace centroids from an
    exposure using a center-of-mass analysis. Works for either order if there
    is no contamination, or for order 1 on a detector where the two orders
    are overlapping.
    This is Loïc's update to my adaptation of Loïc's get_order1_centroids.

    Parameters
    ----------
    stack : array of floats (2D)
        Data frame. Assumes DMS orientation.
        This array could be a native pixel size SOSS subarray or FF.
        It could also be a 2D trace reference file in which case padding exists
        around the edges, and the pixels may be oversampled by some integer
        factor.
    header : fits header
        Header associated to the stack array.
        If the header is None then some assumptions will be made regarding the
        stack array. If the header is passed, then specific keywords will be
        read in it to assess what the stack array is. This ensures that a 2D
        Trace Reference file will be digested properly.
    badpix : array of floats (2D)
        Anything different than zero indicates a bad pixel. Optional input bad
        pixel mask to apply to the stack. Should be of the same dimensions as
        the stack.
    tracemask : array of floats (2D)
        Anything different than zero indicates a masked out pixel. The spirit
        is to have zeros along one spectral order with a certain width.
    poly_order : int
        Order of the polynomial fit to the extracted positions.
    verbose : bool
        Control verbosity.

    Returns
    -------
    tracexbest : np.array
        Best estimate data x centroids.
    traceybest : np.array
        Best estimate data y centroids.
    """

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subaaray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header, verbose=verbose)

    if mask is None:
        mask = np.zeros_like(stack, dtype='bool')

    # Replace masked pixel values with NaNs.
    stackm = np.where(mask | ~working_pixel_bool, np.nan, stack)

    # Identify the floor level of all 2040 working cols to subtract it first.
    floorlevel = np.nanpercentile(stackm, 10, axis=0)
    backsub = stackm - floorlevel

    # Find centroid - first pass, use all pixels in the column.
    # Normalize each column
    norm = backsub / np.nanmax(backsub, axis=0)

    # Create 2D Array of pixel positions.
    xpix = np.arange(dimx)
    ypix = np.arange(dimy)
    _, ypix = np.meshgrid(xpix, ypix)

    # CoM analysis to find centroid using all rows.
    com = np.nansum(norm*ypix, axis=0)/np.nansum(norm, axis=0)

    # Adopt these trace values as best
    tracey_best = np.copy(com)

    # Second pass, use a windowed CoM at the previous position.
    tracey = np.full_like(tracey_best, fill_value=np.nan)
    halfwidth = 30 * yos
    for icol in range(dimx):

        com = center_of_mass(backsub[:, icol], tracey_best[icol], halfwidth)

        # Ensure that the centroid position is not getting too close to an edge
        # such that it is biased.
        if (not np.isfinite(com)) or (com <= 5*yos) or (com >= (ynative-6)*yos):
            continue

        # For a bright second order, it is likely that the centroid at this
        # point will be somewhere in between the first and second order.
        # If this is the case (i.e. the pixel value of the centroid is very low
        # compared to the column average), restrict the range of pixels
        # considered to be above the current centroid.
        if backsub[int(com), icol] < np.nanmean(backsub[int(com) - halfwidth:int(com) + halfwidth, icol]):

            com = center_of_mass(backsub[:, icol], com - halfwidth, halfwidth)

        tracey[icol] = com

    # Adopt these trace values as best.
    tracey_best = np.copy(tracey)

    # Third pass - fine tuning.
    tracey = np.full_like(tracey_best, fill_value=np.nan)
    halfwidth = 16 * yos
    for icol in range(dimx):

        com = center_of_mass(backsub[:, icol], tracey_best[icol], halfwidth)

        tracey[icol] = np.copy(com)

    # Adopt these trace values as best.
    tracey_best = np.copy(tracey)

    # Final pass : Fitting a polynomial to the measured (noisy) positions
    tracex_best = np.arange(dimx)

    if padding == 0:
        # Only use the non NaN pixels.
        induse = np.isfinite(tracex_best) & np.isfinite(tracey_best)
    else:
        # Important steps in the case of the 2D Trace reference file.
        # Mask out the padded pixels from the fit so it is rigorously the
        # same as for regular science images.
        induse = np.isfinite(tracex_best) & np.isfinite(tracey_best) & \
                 (tracex_best >= xos*padding) & (tracex_best < (dimx - xos*padding))

    # Fit the CoM y-positions with a polynomial and use the result as the true y-positions.
    param = np.polyfit(tracex_best[induse], tracey_best[induse], poly_order)
    tracey_best = np.polyval(param, tracex_best)

    if verbose is True:
        _plot_centroid(stackm, tracex_best, tracey_best)

    return tracex_best, tracey_best, param


def test_uncontam_centroids():
    """"""

    filename = os.path.join(PATH, 'SOSS_ref_2D_profile.fits.gz')

    image, header = fits.getdata(filename, ext=2, header=True)
    xtrace, ytrace, param = get_uncontam_centroids(image, header=header, verbose=True)

    return xtrace, ytrace


def robust_polyfit(x, y, order, maxiter=5, nstd=3.):
    """"""

    mask = np.ones_like(x, dtype='bool')
    for niter in range(maxiter):

        # Fit the data and evaluate the best-fit model.
        param = np.polyfit(x[mask], y[mask], order)
        yfit = np.polyval(param, x)

        # Compute residuals and mask ouliers.
        res = y - yfit
        stddev = np.std(res)
        mask = np.abs(res) <= nstd*stddev

    return param


def shift(xs, n):
    """Like np.roll but the wrapped around part is set to zero"""

    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = 0.0
        e[n:] = xs[:-n]
    else:
        e[n:] = 0.0
        e[:n] = xs[-n:]

    return e


def edge_trigger(image, triggerscale=2, yos=1, verbose=False):
    """"""

    dimy, dimx = image.shape
    halfwidth = triggerscale * yos

    # Create coordinate arrays.
    xpix = np.arange(dimx)
    ypix = np.arange(dimy)
    _, ygrid = np.meshgrid(xpix, ypix)

    # Compute windowed slopes over the columns.
    slopevals = np.full_like(image, fill_value=np.nan)
    for irow in range(halfwidth, dimy-halfwidth):

        # Compute the window indices.
        ymin = irow - halfwidth
        ymax = irow + halfwidth + 1

        # Get the x and y data to find the slope to.
        datay = image[ymin:ymax, :]
        mask = np.isfinite(datay)
        datax = np.where(mask, ygrid[ymin:ymax, :], np.nan)  # Need to set values NaN in y to NaN in x.

        # Compute the slope.
        xmean = np.nanmean(datax, axis=0, keepdims=True)
        ymean = np.nanmean(datay, axis=0, keepdims=True)
        slope = np.nansum((datax - xmean) * (datay - ymean), axis=0) / np.nansum((datax - xmean) ** 2, axis=0)

        # Set slopes computed from < 3 datapoints to NaN.
        slopevals[irow, :] = np.where(np.sum(mask, axis=0) >= 3, slope, np.nan)

    # Find the upper and lower bounds on the trace.
    args = np.nanargmax(slopevals, axis=0)
    vals = np.nanmax(slopevals, axis=0)
    ytrace_max = np.where(vals != 0, ypix[args], np.nan)

    args = np.nanargmin(slopevals, axis=0)
    vals = np.nanmin(slopevals, axis=0)
    ytrace_min = np.where(vals != 0, ypix[args], np.nan)

    # Scan through a range of trace widths.
    slopes_best = np.zeros_like(ypix)
    ytrace_best = np.zeros_like(ypix)
    widths_best = np.zeros_like(ypix)
    widthrange = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    for width in widthrange:

        # Add the slope and its offset negative.
        comb = slopevals - shift(slopevals, -yos*width)

        # Find the maximum resulting slope.
        args = np.nanargmax(comb, axis=0)
        vals = np.nanmax(comb, axis=0)

        # Update the best values.
        mask = (vals > slopes_best)
        slopes_best = np.where(mask, vals, slopes_best)
        ytrace_best = np.where(mask, ypix[args], ytrace_best)
        widths_best = np.where(mask, yos*width, widths_best)

    # Set the y position to NaN if the best slope was zero.
    ytrace_best = np.where(slopes_best != 0, ytrace_best + widths_best/2., np.nan)

    if verbose:

        plt.imshow(image, origin='lower')
        plt.plot(ytrace_min)
        plt.plot(ytrace_max)
        plt.plot(ytrace_best)

        plt.show()
        plt.close()

    return ytrace_max, ytrace_min, ytrace_best


def get_uncontam_centroids_edgetrig(stack, header=None, mask=None, poly_order=11,
                                    triggerscale=5, mode='combined', verbose=False):
    """ Determine the x, y positions of the trace centroids from an exposure
    using the two edges and the width of the traces. This should be performed on a very high SNR
    stack.
    INPUTS
    stack : a fits image, preferably a high SNR stack with 2 dimensions. Not for raw images
        with 4 dimensions. The best stack here is from the 2D Trace reference file. You call
        this for each order.
    OPTIONAL INPUTS
    header :
        In the case that the input stack comes from the 2D Trace reference file, pass its
        header which contains important info regrading the padding, for example.
    badpix : Can provide a bad pixel mask, it is assumed to be of same dimensions as the stack
    mask : Can provide a mask that will be applied on top of the stack, assumed same dimensions as stack
    poly_order : For the fit to the trace positions.
    triggerscale : The number of pixels to median spatially when calculating the column slopes to identify edges.
        Default 5. Has to be an odd number. Should not play too much with that.
    verbose : Will output stuff and make plots if set to True, default False.
    return_what : What to return. Either x,y positions or polynomial parameters,
        either for one of the edges or for the mean of both edges (i.e. trace center)
        'edgemean_param' : polynomial fit parameters to the mean of both edges, i.e. trace center
        'edgeman_xy' : x, y values for the mean of both edges
        'rededge_param' : polynomial fit parameters to the red edge (spatially)
        'rededge_xy' : x, y values for the red edge
        'blueedge_param' : polynomial fit parameters to the blue edge (spatially)
        'blueedge_xy' : x, y values for the blue edge
        'edgecomb_param' : when both edges are detected simultaneously (one inverted with a trace width offset)
        'edgecomb_xy' : x, y values
        'tracewidth' : scalar representing the median of (red edge - blue edge) y values
    """

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subarray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header)

    if mask is None:
        mask = np.zeros_like(stack)

    # Replace masked pixel values with NaNs.
    stackm = np.where(mask | ~working_pixel_bool, np.nan, stack)

    # Use edge trigger to compute the edges and center of the trace.
    ytrace_max, ytrace_min, ytrace_comb = edge_trigger(stackm, triggerscale=triggerscale, yos=yos, verbose=verbose)

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
    param = np.polyfit(xtrace[mask], ytrace[mask], poly_order)
    ytrace = np.polyval(param, xtrace)

    if verbose is True:
        _plot_centroid(stackm, xtrace, ytrace)

    # Compute an estimate of the trace width.
    tracewidth = np.nanmedian(np.abs(ytrace_max - ytrace_min))

    return xtrace, ytrace, tracewidth, param


def test_uncontam_centroids_edgetrig():
    """"""

    filename = os.path.join(PATH, 'SOSS_ref_2D_profile.fits.gz')

    image, header = fits.getdata(filename, ext=2, header=True)
    xtrace, ytrace, width, param = get_uncontam_centroids_edgetrig(image, header=header, verbose=True)

    return xtrace, ytrace


def main():

    test_uncontam_centroids()
    test_uncontam_centroids_edgetrig()

    return


if __name__ == '__main__':
    main()
