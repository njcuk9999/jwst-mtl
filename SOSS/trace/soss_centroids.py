#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt

PATH = '/home/talens-irex/Dropbox/SOSS_Ref_Files'


def get_image_dim(image, header=None, verbose=False):
    """Determine the properties of the image array.

    :param image: A 2D image of the detector.
    :param header: The header from one of the SOSS reference files.
    :param verbose: If set True some diagnostic plots will be made.

    :type image: array[float]
    :type header: astropy.io.fits.Header
    :type verbose: bool

    :returns:
    dimx, dimy
        The dimensions of the stack array.
    xos, yos
        The oversampling factors of the stack array.
    xnative, ynative
        The dimensions of the stack image, in native pixels.
    padding
        Amount of padding around the image, in native pixels.
    refpix_mask
        Boolean array indicating which pixels are lightsensitive (True) and which are reference pixels (False).

    :rtype: Tuple(int, int, int, int, int, int, int, array[bool])
    """

    # Dimensions of the subarray.
    dimy, dimx = np.shape(image)

    # If no header was passed we have to check all possible sizes.
    if header is None:

        # Initialize padding to zero in this case because it is not a reference file.
        padding = 0

        # Assume the stack is a valid SOSS subarray.
        # FULL: 2048x2048 or 2040x2040 (working pixels) or multiple if oversampled.
        # SUBSTRIP96: 2048x96 or 2040x96 (working pixels) or multiple if oversampled.
        # SUBSTRIP256: 2048x256 or 2040x252 (working pixels) or multiple if oversampled.

        # Check if the size of the x-axis is valid.
        if (dimx % 2048) == 0:
            xnative = 2048
            xos = int(dimx // 2048)

        elif (dimx % 2040) == 0:
            xnative = 2040
            xos = int(dimx // 2040)

        else:
            raise ValueError('Stack X dimension has unrecognized size of {:}. Accepts 2048, 2040 or multiple of.'.format(dimx))

        # Check if the y-axis is consistent with the x-axis.
        acceptable_ydim = [96, 256, 252, 2040, 2048]
        yaxis_consistent = False
        for accdim in acceptable_ydim:  # TODO See if this can be done more similarly to the x-axis.

            if dimy / (accdim*xos) == 1:

                # Found the acceptable dimension.
                yos = np.copy(xos)
                ynative = np.copy(accdim)
                yaxis_consistent = True

        if not yaxis_consistent:
            raise ValueError('Stack Y dimension ({:}) is inconsistent with X dimension ({:}) for acceptable SOSS arrays'.format(dimy, dimx))

        # Create a boolean mask indicating which pixels are not reference pixels.
        refpix_mask = np.ones_like(image, dtype='bool')
        if xnative == 2048:
            # Mask out the left and right columns of reference pixels.
            refpix_mask[:, :xos * 4] = False
            refpix_mask[:, -xos * 4:] = False

        if ynative == 2048:
            # Mask out the top and bottom rows of reference pixels.
            refpix_mask[:yos * 4, :] = False
            refpix_mask[-yos * 4:, :] = False

        if ynative == 256:
            # Mask the top rows of reference pixels.
            refpix_mask[-yos * 4:, :] = False

    else:
        # Read the oversampling and padding from the header.
        padding = int(header['PADDING'])
        xos, yos = int(header['OVERSAMP']), int(header['OVERSAMP'])

        # The 2D Trace profile is for FULL FRAME so 2048x2048.
        xnative, ynative = 2048, 2048

        # Check that the stack respects its intended format.
        if dimx != (xnative + 2*padding)*xos:
            raise ValueError('The header passed is inconsistent with the X dimension of the stack.')

        if dimy != (ynative + 2*padding)*yos:
            raise ValueError('The header passed is inconsistent with the Y dimension of the stack.')

        # The trace file contains no reference pixels so all pixels are good.
        refpix_mask = np.ones_like(image, dtype='bool')

    # If verbose print the output.
    if verbose:
        print('Data dimensions:\n')
        print('dimx={:}, dimy={:}, xos={:}, yos={:}, xnative={:}, ynative={:}'.format(dimx, dimy, xos, yos, xnative, ynative))

    return dimx, dimy, xos, yos, xnative, ynative, padding, refpix_mask


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

    # Find the indices of the window.
    miny = np.int(np.fmax(np.around(ypos - halfwidth), 0))
    maxy = np.int(np.fmin(np.around(ypos + halfwidth + 1), dimy))

    # Compute the center of mass on the window.
    ycom = np.nansum(column[miny:maxy]*ypix[miny:maxy])/np.nansum(column[miny:maxy])

    return ycom


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

    # If no mask was given use all pixels.
    if mask is None:
        mask = np.zeros_like(stack, dtype='bool')

    # Call the script that determines the dimensions of the stack.
    dimx, dimy, xos, yos, xnative, ynative, padding, refpix_mask = \
        get_image_dim(stack, header=header, verbose=verbose)

    # Replace masked pixel values with NaNs.
    stackm = np.where(mask | ~refpix_mask, np.nan, stack)  # TODO rename variable

    # Identify the floor level of all 2040 working cols to subtract it first.
    floorlevel = np.nanpercentile(stackm, 10, axis=0)
    backsub = stackm - floorlevel

    # Find centroid - first pass, use all pixels in the column.
    # Normalize each column
    norm = backsub / np.nanmax(backsub, axis=0)

    # Create 2D Array of pixel positions.
    xpix = np.arange(dimx)
    ypix = np.arange(dimy)
    _, ygrid = np.meshgrid(xpix, ypix)

    # CoM analysis to find initial positions using all rows.
    ytrace_best = np.nansum(norm*ygrid, axis=0)/np.nansum(norm, axis=0)

    # Second pass - use a windowed CoM at the previous position.
    halfwidth = 30 * yos
    ytrace = np.full_like(ytrace_best, fill_value=np.nan)
    for icol in range(dimx):

        com = center_of_mass(backsub[:, icol], ytrace_best[icol], halfwidth)

        # Ensure that the centroid position is not getting too close to an edge
        # such that it is biased.
        if (not np.isfinite(com)) or (com <= 5*yos) or (com >= (ynative-6)*yos):
            continue

        # For a bright second order, it is likely that the centroid at this
        # point will be somewhere in between the first and second order.
        # If this is the case (i.e. the pixel value of the centroid is very low
        # compared to the column average), restrict the range of pixels
        # considered to be above the current centroid.
        if backsub[int(com), icol] < np.nanmean(backsub[int(com) - halfwidth:int(com) + halfwidth + 1, icol]):

            com = center_of_mass(backsub[:, icol], com - halfwidth, halfwidth)

        ytrace[icol] = com

    # Adopt these trace values as best.
    ytrace_best = np.copy(ytrace)

    # Third pass - fine tuning using a smaller window.
    halfwidth = 16 * yos
    ytrace = np.full_like(ytrace_best, fill_value=np.nan)
    for icol in range(dimx):

        ytrace[icol] = center_of_mass(backsub[:, icol], ytrace_best[icol], halfwidth)

    # Adopt these trace values as best.
    ytrace_best = np.copy(ytrace)

    # Fit the y-positions with a polynomial and use the result as the true y-positions.
    xtrace = np.arange(dimx)
    mask = np.isfinite(ytrace_best)

    # For padded arrays ignore padding for consistency with real data
    if padding != 0:
        mask = mask & (xtrace >= xos*padding) & (xtrace < (dimx - xos*padding))

    param = robust_polyfit(xtrace[mask], ytrace_best[mask], poly_order)
    ytrace_best = np.polyval(param, xtrace)

    # If verbose visualize the result.
    if verbose is True:
        _plot_centroid(stackm, xtrace, ytrace_best)

    return xtrace, ytrace_best, param


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

    # If no mask was given use all pixels.
    if mask is None:
        mask = np.zeros_like(stack, dtype='bool')

    # Call the script that determines the dimensions of the stack.
    dimx, dimy, xos, yos, xnative, ynative, padding, refpix_mask = \
        get_image_dim(stack, header=header, verbose=verbose)

    # Replace masked pixel values with NaNs.
    stackm = np.where(mask | ~refpix_mask, np.nan, stack)

    # Use edge trigger to compute the edges and center of the trace.
    ytrace_max, ytrace_min, ytrace_comb = edge_trigger(stackm, triggerscale=triggerscale, yos=yos, verbose=verbose)

    # Compute an estimate of the trace width.
    tracewidth = np.nanmedian(np.abs(ytrace_max - ytrace_min))

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
        _plot_centroid(stackm, xtrace, ytrace)

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
