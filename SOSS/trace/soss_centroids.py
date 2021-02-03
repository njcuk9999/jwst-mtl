#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt


def determine_stack_dimensions(stack, header=None, verbose=False):
    '''Determine the size of a stack array. Will be called by
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
    '''

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
            raise ValueError('Stack Y dimension ({:}) is inconsistent with X dimension ({:}) for acceptable SOSS arrays'.format(dimy,dimx))
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

    return(dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool)


def _plot_centroid(clear, cen_x, cen_y):
    '''Utility function to overplot the trace centroids extracted from
    the data over the data itself to verify accuracy.
    '''
    plt.figure(figsize=(15, 3))
    plt.imshow(np.log10(clear), origin='lower', cmap='jet')
    plt.plot(cen_x, cen_y, c='black')
    plt.show()


def get_uncontam_centroids(stack, header=None, badpix=None, tracemask=None,
                           verbose=False):
    '''Determine the x, y positions of the trace centroids from an
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
    specpix_bounds : native spectral pixel bounds to consider in fitting the
        trace. Most likely used for the 2nd and 3rd orders, not for the 1st.

    Returns
    -------
    tracexbest : np.array
        Best estimate data x centroids.
    traceybest : np.array
        Best estimate data y centroids.
    '''

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subaaray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header, verbose=verbose)

    # Make a numpy mask array of the working pixels
    working_pixel_mask = np.ma.array(np.ones((dimy, dimx)), mask=np.invert(working_pixel_bool))
    # Fill the working pixel mask with NaN
    working_pixel_mask = np.ma.filled(working_pixel_mask, np.nan)

    # Check for the optional input badpix and create a bad pixel numpy mask
    if badpix is not None:
        # 1) Check the dimension is the same as stack
        # TODO:
        # 2) Create the numpy.ma array with it
        # The bad pixel mask has values of 'one' for valid pixels.
        badpix_mask = np.ma.array(np.ones((dimy, dimx)), mask=(badpix != 0))
    else:
        # Create a mask with all valid pixels (all ones)
        badpix_mask = np.ma.array(np.ones((dimy, dimx)))
    # Fill the bad pixels with NaN
    badpix_mask = np.ma.filled(badpix_mask, np.nan)

    # Check for the optional input tracemask and create a trace numpy mask
    if tracemask is not None:
        # 1) Check the dimension is the same as stack
        # TODO:
        # 2) Create the numpy.ma array with it
        # The trace mask has values of 'one' for valid pixels.
        trace_mask = np.ma.array(np.ones((dimy, dimx)), mask=(tracemask == 0))
    else:
        # Create a mask with all pixels in the trace (all ones)
        trace_mask = np.ma.array(np.ones((dimy, dimx)))
    # Fill the trace mask with NaN
    trace_mask = np.ma.filled(trace_mask, np.nan)

    # Multiply working pixel mask, bad pixel mask and trace mask
    # The stack image with embedded numpy mask is stackm
    stackm = stack * badpix_mask * working_pixel_mask * trace_mask

    # Identify the floor level of all 2040 working cols to subtract it first.
    floorlevel = np.nanpercentile(stackm, 10, axis=0)
    backsub = stackm - floorlevel
    # Find centroid - first pass, use all pixels in the column.
    # Normalize each column
    norm = backsub / np.nanmax(backsub, axis=0)
    # Create 2D Array of pixel positions
    rows = (np.ones((dimx, dimy)) * np.arange(dimy)).T
    # CoM analysis to find centroid
    com = (np.nansum(norm * rows, axis=0) / np.nansum(norm, axis=0)).data
    # Adopt these trace values as best
    tracex_best = np.arange(dimx)
    tracey_best = np.copy(com)
    # Second pass, find centroid on a subset of pixels
    # from an area around the centroid determined earlier.
    tracex = np.arange(dimx)
    tracey = np.zeros(dimx)*np.nan
    row = np.arange(dimy)
    w = 30 * yos
    for i in range(dimx):
        miny = np.int(np.nanmax([np.around(tracey_best[i] - w), 0]))
        maxy = np.int(np.nanmin([np.around(tracey_best[i] + w), dimy - 1]))
        val = backsub[miny:maxy, i] / np.nanmax(backsub[:, i])
        ind = np.where(np.isfinite(val))
        thisrow = (row[miny:maxy])[ind]
        thisval = val[ind]
        com = np.sum(thisrow * thisval) / np.sum(thisval)
        # Ensure that the centroid position is not getting too close to an edge
        # such that it is biased.
        if (not np.isfinite(com)) or (com <= 5*yos) or (com >= (ynative-6)*yos):
            continue
        # For a bright second order, it is likely that the centroid at this
        # point will be somewhere in between the first and second order.
        # If this is the case (i.e. the pixel value of the centroid is very low
        # compared to the column average), restrict the range of pixels
        # considered to be above the current centroid.
        if backsub[int(com)][i] < np.nanmean(backsub[(int(com) - w):(int(com) + w), i]):
            miny = np.int(np.nanmax([np.around(com) - 2*w, 0]))
            maxy = np.int(np.nanmin([np.around(com), dimy - 1]))
            val = backsub[miny:maxy, i] / np.nanmax(backsub[:, i])
            ind = np.where(np.isfinite(val))
            thisrow = (row[miny:maxy])[ind]
            thisval = val[ind]
            com = np.sum(thisrow * thisval) / np.sum(thisval)
        tracey[i] = com
    # Adopt these trace values as best.
    tracex_best = np.copy(tracex)
    tracey_best = np.copy(tracey)

    # Third pass - fine tuning.
    tracex = np.arange(dimx)
    tracey = np.zeros(dimx) * np.nan
    row = np.arange(dimy)
    w = 16 * yos
    for i in range(len(tracex_best)):
        miny = np.int(np.nanmax([np.around(tracey_best[i] - w), 0]))
        maxy = np.int(np.nanmin([np.around(tracey_best[i] + w), dimy - 1]))
        val = backsub[miny:maxy, i] / np.nanmax(backsub[:, i])
        ind = np.where(np.isfinite(val))
        thisrow = (row[miny:maxy])[ind]
        thisval = val[ind]
        com = np.sum(thisrow * thisval) / np.sum(thisval)
        tracex[i] = np.copy(tracex_best[i])
        tracey[i] = np.copy(com)
    # Update with the best estimates
    tracex_best = np.copy(tracex)
    tracey_best = np.copy(tracey)

    # Final pass : Fitting a polynomial to the measured (noisy) positions
    if padding == 0:
        # Only use the non NaN pixels.
        induse = np.isfinite(tracex_best) & np.isfinite(tracey_best)
    else:
        # Important steps in the case of the 2D Trace reference file.
        # Mask out the padded pixels from the fit so it is rigorously the
        # same as for regular science images.
        induse = np.isfinite(tracex_best) & np.isfinite(tracey_best) & \
                 (tracex_best >= xos*padding) & (tracex_best < (dimx-xos*padding))

    # Use a *** fixed *** polynomial order of 11 to keep results consistent
    # from data set to data set. Any systematics would remain fixed.
    polyorder = 11
    param = np.polyfit(tracex_best[induse], tracey_best[induse], polyorder)
    tracey_best = np.polyval(param, tracex_best)

    if verbose is True:
        _plot_centroid(stack, tracex_best, tracey_best)

    return tracex_best, tracey_best


def test_uncontam_centroids():

    # Read in the 2D trace reference file (each extension has an isolated
    # trace). When it exists, make sure to pass the deader as well in the call
    # to the get_edge_centroids function. For now, we are missing that file so
    # use the CV3 stack instead.
    #
    # im = read.the.2Dtrace.ref.file
    # hdr = is.its.header
    a = fits.open('/genesis/jwst/userland-soss/loic_review/stack_256_ng3_DMS.fits')
    im = a[0].data

    # x_o1, y_o1 = get_centerofmass_centroids(im, header=hdr, badpix=False, verbose=verbose)
    x_o1, y_o1 = get_uncontam_centroids(im, badpix=False, verbose=False)

    return x_o1, y_o1


def polyfit_sigmaclip(x, y, order, sigma=4):

    polyorder = np.copy(order)
    ind = np.where(y == y)
    for n in range(5):
        #if n == 4: polyorder = order + 1
        param = np.polyfit(x[ind], y[ind], polyorder)
        yfit = np.polyval(param, x)
        dev = np.abs(yfit - y)
        ind = np.where(dev <= sigma)

    return param


def shift(xs, n):
    # Like np.roll but the wrapped around part is set to zero
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = 0.0
        e[n:] = xs[:-n]
    else:
        e[n:] = 0.0
        e[:n] = xs[-n:]
    return e


def edge_trigger(column, triggerscale=2, verbose=False, yos=1):
    # triggerscale must be an integer, expressed in native pixel size

    # dimension of the column array
    dim, = np.shape(column)
    #halftrig = int((triggerscale-1)/2)
    half = triggerscale*yos
    # positions along that column where the full triggerscale is accessible
    #ic = halftrig + np.arange(dim-triggerscale*yos)
    ic = half + np.arange(dim - 2*half)
    # slope of the flux at position datax
    slope = []
    datax = np.arange(2*half+1)
    # For each position, grab current n pixels, exclude NaN, fit a slope
    for i in ic:
        data = column[i-half:i+half+1]
        ind = np.where(np.isfinite(data))
        if np.size(ind) >=3:
            param = np.polyfit(datax[ind],data[ind],1)
            slope.append(param[0])
        else:
            slope.append(np.nan)
    slope = np.array(slope)

    # Determine which x sees the maximum slope
    indmax = np.argwhere((slope == np.nanmax(slope)) & (slope !=0))
    edgemax = np.nan # default value because ref pixels produce no slope
    if indmax.size > 0: edgemax = ic[indmax[0][0]]

    # Determine which x sees the minimum slope
    indmin = np.argwhere((slope == np.nanmin(slope)) & (slope !=0))
    edgemin = np.nan
    if indmin.size > 0: edgemin = ic[indmin[0][0]]

    # Determine which (x,x+width) pair sees the maximum and minimum slopes simultaneously.
    # This methods is the most robust. It scans the combined slope peak for 12 different
    # trace widths 15 to 27 native pixels and triggers on the maximum combined slope.
    # Initialize the combined slope maximum value (valmax), x index (indcomb) and trace width
    valmax, indcomb, widthcomb = 0, -1, 0
    widthrange = [18,19,20,21,22,23,24,25,26,27]
    # Scan for 9 trace widths 18 to 27 native pixels
    for width in widthrange:
        # add the slope and its offsetted negative
        #comb = slope - np.roll(slope, -yos*width)
        comb = slope - shift(slope, -yos*width)
        # Find the maximum resulting slope
        indcurr = np.argwhere((comb == np.nanmax(comb)) & (comb != 0))
        valcurr = -1
        if indcurr.size > 0: valcurr = comb[indcurr[0][0]]
        # Keep that as the optimal if it is among all trace widths
        if valcurr > valmax:
            valmax = np.copy(valcurr)
            indcomb = np.copy(indcurr[0][0])
            widthcomb = yos*width
    edgecomb = np.nan
    if np.size(indcomb) > 0:
        if indcomb != -1: edgecomb = ic[indcomb] + widthcomb/2.


    # Make a plot if verbose is True
    if verbose == True:
        plt.plot(ic,slope)
        plt.plot(ic,comb)
        plt.show()

    return edgemax, edgemin, edgecomb


def get_edge_centroids(stack, header=None, badpix=None, mask=None, verbose=False,
                       return_what='edgecomb_param', polynomial_order=2,
                       triggerscale=5):
    ''' Determine the x, y positions of the trace centroids from an exposure
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
    polynomial_order : For the fit to the trace positions.
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
    '''

    # Call the script that determines the dimensions of the stack. It handles
    # regular science images of various subarray sizes, with or without the
    # reference pixels, oversampled or not. It also handles the 2D Trace
    # Reference File.
    dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool = \
        determine_stack_dimensions(stack, header=header)

    if mask == None: mask = np.ones((dimy,dimx))

    edge1, edge2, edgecomb = [], [], []
    for i in range(dimx):
        if (i % 100 == 0) & (verbose == True):
            y1, y2, ycomb = edge_trigger(stack[:, i] * mask[:, i], triggerscale=triggerscale, verbose=True, yos=yos)
        else:
            y1, y2, ycomb = edge_trigger(stack[:, i] * mask[:, i], triggerscale=triggerscale, verbose=False, yos=yos)
        # print(i, y1, y2)
        edge1.append(y1)
        edge2.append(y2)
        edgecomb.append(ycomb)
    edge1 = np.array(edge1)
    edge2 = np.array(edge2)
    edgecomb = np.array(edgecomb)

    # Fit the red edge
    x_red = np.arange(dimx)
    ind = np.where(np.isfinite(edge1))
    param_red = polyfit_sigmaclip(x_red[ind], edge1[ind], polynomial_order)
    y_red = np.polyval(param_red, x_red)
    # Fit the blue edge
    x_blue = np.arange(dimx)
    ind = np.where(np.isfinite(edge2))
    param_blue = polyfit_sigmaclip(x_blue[ind], edge2[ind], polynomial_order)
    y_blue = np.polyval(param_blue, x_blue)
    # Fit the combined edges simultaneously
    x_comb = np.arange(dimx)
    ind = np.where(np.isfinite(edgecomb))
    param_comb = polyfit_sigmaclip(x_comb[ind], edgecomb[ind], polynomial_order)
    y_comb = np.polyval(param_comb, x_comb)
    # Fit the mean of both edges
    x_both = np.arange(dimx)
    both = (edge1+edge2)/2.
    ind = np.where(np.isfinite(both))
    param_both = polyfit_sigmaclip(x_both[ind], both[ind], polynomial_order)
    y_both = np.polyval(param_both, x_both)

    # Plot the edge position as a function of x
    if True:
        plt.scatter(x_red, edge1, marker=",", s=0.8, label='RAW Rising edge')
        plt.scatter(x_blue, edge2, marker=",", s=0.8, label='RAW Falling edge')
        plt.scatter(x_comb, edgecomb, marker=",", s=0.8, label='RAW Combined+inverted+offset rising edges')
        plt.scatter(x_both, both, marker=",", s=0.8, label='RAW Rising + falling edges average')
        plt.plot(x_red, y_red, label='FIT Rising edge')
        plt.plot(x_blue, y_blue, label='FIT Falling edge')
        plt.plot(x_comb, y_comb, label='FIT Combined+inverted+offset rising edges')
        plt.plot(x_both, y_both, label='FIT Rising + falling edges average')
        plt.scatter(x_red, np.abs(edge1-edge2), marker=",", s=0.8, label='RAW Trace width')
        plt.legend()
        plt.show()

    # Trace width
    tracewidth = np.nanmedian(np.abs(edge1 - edge2))

    if return_what == 'edgemean_param' : return param_both
    if return_what == 'rededge_param' : return param_red
    if return_what == 'blueedge_param' : return param_blue
    if return_what == 'edgecomb_param' : return param_comb
    if return_what == 'edgemean_xy' : return x_both, y_both
    if return_what == 'rededge_xy' : return x_red, y_red
    if return_what == 'blueedge_xy' : return x_blue, y_blue
    if return_what == 'edgecomb_xy' : return x_comb, y_comb
    if return_what == 'tracewidth' : return tracewidth


def get_uncontam_centroids_edgetrig():
    # Here is how to call the code

    # Read in the 2D trace reference file (each extension has an isolated
    # trace). When it exists, make sure to pass the deader as well in the call
    # to the get_edge_centroids function. For now, we are missing that file so
    # use the CV3 stack instead.
    #
    # im = read.the.2Dtrace.ref.file
    # hdr = is.its.header
    a = fits.open('/genesis/jwst/userland-soss/loic_review/stack_256_ng3_DMS.fits')
    im = a[0].data
    a = fits.open('/genesis/jwst/userland-soss/loic_review/simu_o1only.fits')
    im = a[1].data
    im = im[0, 0, :, :]
    # Try on the first ever 2D Trace reference file
    a = fits.open('/genesis/jwst/userland-soss/loic_review/SOSS_ref_2D_profile.fits.gz')
    im = a[2].data
    hdr = a[2].header
    #im = im[0, :, :]

    # Triggers on the rising and declining edges of the trace. Make a polynomial
    # fit to those and return the x,y fit. Alternatively, the parameters of that
    # fit coudl be returned by using return_what='edgemean_param'.
    x_o1, y_o1 = get_edge_centroids(im, header=hdr, return_what='edgecomb_xy',
                                    polynomial_order=10, verbose=False)
    #x_o1, y_o1 = get_edge_centroids(im, return_what='edgecomb_xy',
    #                                polynomial_order=10, verbose=False)

    return x_o1, y_o1


def main():
    return


if __name__ == '__main__':
    main()
