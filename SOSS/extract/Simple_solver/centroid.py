#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:39 2021

@author: MCR

All functions associated with the determining the centroid positions of the
order 1 SOSS spectral trace.
"""

import numpy as np
import emcee
import warnings
import matplotlib.pyplot as plt
from SOSS.trace import tracepol as tp
from SOSS.extract.simple_solver import plotting as plotting

warnings.simplefilter(action='ignore', category=FutureWarning)

# hack to get around the fact that relative paths are constantly messing up atm
path = '/Users/michaelradica/Documents/GitHub/jwst-mtl/SOSS/'


def determine_stack_dimensions(stack, header=None, verbose=False):
    '''Determine the size of the stack array. Will be called by
    get_uncontam_centroids and make_trace_mask.

    Parameters
    ----------
    stack : array of floats (2D)
        Data frame. Assumes DMS orientation.
        This array could be a native pixel size SOSS subarray or FF.
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
    dimx, dimy : The dimensions of the stack array.
    xos, yos : The oversampling factor (integer) of the stack array.
    xnative, ynative : The dimensions of the stack image, expressed in native
    pixels.
    padding : the size of padding all around the image, in units of native
    pixels.
    working_pixel_bool : a 2D array of the same size as stack with boolean
    values of False where pixels are not light sensitive (the reference
    pixels). True elsewhere.
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
        print('dimx={:}, dimy={:}, xos={:}, yos={:}, xnative={:}, ynative={:}'.format(dimx, dimy, xos, yos, xnative, ynative))

    return(dimx, dimy, xos, yos, xnative, ynative, padding, working_pixel_bool)


def _do_emcee(xref, yref, xdat, ydat, subarray='substrip256',
              showprogress=False):
    '''Calls the emcee package to preform an MCMC determination of the best
    fitting rotation angle and offsets to map the reference centroids onto the
    data for the first order.

    Parameters
    ----------
    xref, yref : array of float
        X and Y trace centroids respectively to be used as a reference point,
        for example: as returned by get_om_centroids.
    xdat, ydat : array of float
        X and Y trace centroids determined from the data, for example: as
        returned by get_uncontam_centroids.
    subarray : str
        Identifier for the correct subarray.
    showprogress: bool
        If True, show the emcee progress bar.

    Returns
    -------
    sampler : emcee EnsembleSampler object
        MCMC fitting results.
    '''

    # Set up the MCMC run.
    initial = np.array([0, 0, 0])  # Initial guess parameters
    pos = initial + 0.5*np.random.randn(32, 3)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_probability,
                                    args=[xref, yref, xdat, ydat, subarray])
    # Run the MCMC for 5000 steps - it has generally converged
    # within ~3000 steps in trial runs.
    sampler.run_mcmc(pos, 5000, progress=showprogress)

    return sampler


def get_uncontam_centroids(stack, header=None, badpix=None, tracemask=None,
                           verbose=False):
    '''Determine the x, y positions of the trace centroids from an
    exposure using a center-of-mass analysis. Works for either order if there
    is no contamination, or for order 1 on a detector where the two orders
    are overlapping.
    This is an adaptation of Loïc's get_order1_centroids which can better
    deal with a bright second order.

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
    badpix : array of floats (2D) with anything different than zero meaning a bad pixel.
        Optional input bad pixel mask to apply to the stack. Should be of
        the same dimensions as the stack.
    tracemask : array of floats (2D) with anything different than zero meaning
        a masked out pixel. The spirit is to have zeros along one spectral
        order with a certain width.
    specpix_bounds : native spectral pixel bounds to consider in fitting the
        trace. Most likely used for the 2nd and 3rd orders, not for the 1st.

    Returns
    -------
    tracexbest : np.array
        Best estimate data x centroid.
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

    if verbose is True:
        plt.figure()
        plt.plot(tracex_best, tracey_best)

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
        plt.plot(tracex_best, tracey_best, color='r')
        plt.show()

    return tracex_best, tracey_best


def _log_likelihood(theta, xmod, ymod, xdat, ydat, subarray):
    '''Definition of the log likelihood. Called by _do_emcee.
    xmod/ymod should extend past the edges of the detector.
    '''
    ang, xshift, yshift = theta
    # Calculate rotated model
    modelx, modely = rot_centroids(ang, xshift, yshift, xmod, ymod, bound=True,
                                   subarray=subarray)
    # Interpolate rotated model onto same x scale as data
    modely = np.interp(xdat, modelx, modely)

    return -0.5 * np.sum((ydat - modely)**2 - 0.5 * np.log(2 * np.pi * 1))


def _log_prior(theta):
    '''Definition of the priors. Called by _do_emcee.
    Angle within +/- 5 deg (one motor step is 0.15deg).
    X-shift within +/- 100 pixels, TA shoukd be accurate to within 1 pixel.
    Y-shift to within +/- 50 pixels.
    '''
    ang, xshift, yshift = theta

    if -5 <= ang < 5 and -100 <= xshift < 100 and -50 <= yshift < 50:
        return -1
    else:
        return -np.inf


def _log_probability(theta, xmod, ymod, xdat, ydat, subarray):
    '''Definition of the final probability. Called by _do_emcee.
    '''
    lp = _log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp + _log_likelihood(theta, xmod, ymod, xdat, ydat, subarray)


def rot_centroids(ang, xshift, yshift, xpix, ypix, bound=True, atthesex=None,
                  cenx=1024, ceny=50, subarray='substrip256'):
    '''Apply a rotation and shift to the trace centroids positions. This
    assumes that the trace centroids are already in the CV3 coordinate system.

    Parameters
    ----------
    ang : float
        The rotation angle in degrees CCW.
    xshift : float
        Offset in the X direction to be rigidly applied after rotation.
    yshift : float
        Offset in the Y direction to be rigidly applied after rotation.
    xpix : float or np.array of float
        Centroid pixel X values.
    ypix : float or np.array of float]
        Centroid pixel Y values.
    bound : bool
        Whether to trim rotated solutions to fit within the subarray256.
    atthesex : list of float
        Pixel values at which to calculate rotated centroids.
    cenx : int
        X-coordinate in pixels of the rotation center.
    ceny : int
        Y-coordinate in pixels of the rotation center.
    subarray : str
        Subarray identifier. One of substrip96, substrip256 or full.

    Returns
    -------
    rot_xpix : np.array of float
        xval after the application of the rotation and translation
        transformations.
    rot_ypix : np.array of float
        yval after the application of the rotation and translation
        transformations.

    Raises
    ------
    ValueError
        If bad subarray identifier is passed.
    '''

    # Convert to numpy arrays
    xpix = np.atleast_1d(xpix)
    ypix = np.atleast_1d(ypix)
    # Required rotation in the detector frame to match the data.
    t = np.deg2rad(ang)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    # Rotation center set to o1 trace centroid halfway along spectral axis.
    points1 = np.array([xpix - cenx, ypix - ceny])
    rot_pix = R @ points1

    rot_pix[0] += cenx
    rot_pix[1] += ceny

    # Apply the offsets
    rot_pix[0] += xshift
    rot_pix[1] += yshift

    if xpix.size >= 10:
        if atthesex is None:
            # Ensure that there are no jumps of >1 pixel.
            min = int(round(np.min(rot_pix[0]), 0))
            max = int(round(np.max(rot_pix[0]), 0))
            # Same range as rotated pixels but with step of 1 pixel.
            atthesex = np.linspace(min, max, max-min+1)
        # Polynomial fit to ensure a centroid at each pixel in atthesex
        pp = np.polyfit(rot_pix[0], rot_pix[1], 5)
        # Warn user if atthesex extends beyond polynomial domain.
        if np.max(atthesex) > np.max(rot_pix[0])+25 or np.min(atthesex) < np.min(rot_pix[0])-25:
            warnings.warn('atthesex extends beyond rot_xpix. Use results with caution.')
        rot_xpix = atthesex
        rot_ypix = np.polyval(pp, rot_xpix)
    else:
        # If too few pixels for fitting, keep rot_pix.
        if atthesex is not None:
            print('Too few pixels for polynomial fitting. Ignoring atthesex.')
        rot_xpix = rot_pix[0]
        rot_ypix = rot_pix[1]

    # Check to ensure all points are on the subarray.
    if bound is True:
        # Get dimensions of the subarray
        if subarray == 'substrip96':
            yend = 96
        elif subarray == 'substrip256':
            yend = 256
        elif subarray == 'full':
            yend = 2048
        else:
            raise ValueError('Bad subarray identifier. Allowed identifiers are "substrip96", "substrip256", or "full".')
        # Reject pixels which are not on the subarray.
        inds = [(rot_ypix >= 0) & (rot_ypix < yend) & (rot_xpix >= 0) &
                (rot_xpix < 2048)]
        rot_xpix = rot_xpix[inds]
        rot_ypix = rot_ypix[inds]

    return rot_xpix, rot_ypix
