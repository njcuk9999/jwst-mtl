#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 9:02 2021

@author: MCR

All functions associated with the determining the centroid positions of the
order 1 and 2 SOSS spectra trace.
"""

import numpy as np
import emcee
import warnings
from SOSS.trace import tracepol as tp
from SOSS.trace import get_uncontam_centroids as uctd
from SOSS.extract.empirical_trace import plotting as plotting

warnings.simplefilter(action='ignore', category=FutureWarning)

# hack to get around the fact that relative paths are constantly messing up atm
path = '/Users/michaelradica/Documents/GitHub/jwst-mtl/SOSS/'


def _do_emcee(xref, yref, xdat, ydat, showprogress=False):
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
                                    args=[xref, yref, xdat, ydat])
    # Run the MCMC for 5000 steps - it has generally converged
    # within ~3000 steps in trial runs.
    sampler.run_mcmc(pos, 5000, progress=showprogress)

    return sampler


def get_contam_centroids(clear, ref_centroids=None, doplot=False,
                         return_orders=[1, 2, 3], bound=True,
                         showprogress=False):
    '''Get the trace centroids for all orders when there is contamination on
    the detector. Fits the first order centroids using the uncontaminated
    method, and determines the second/third order centroids via the
    well-calibrated relationship between all orders.

    Parameters
    ----------
    clear : np.ndarray (2D)
        CLEAR SOSS exposure data frame.
    ref_centroids : np.ndarray (2D)
        Centroids relative to which to determine rotations parameters.
        Must contain lists of x and y centroids for each order to be returned.
        If None, uses the trace table centroids as a reference.
    doplot : bool
        Whether to plot the results of the reference centroids fit to the
        first order.
    return_orders : list
        Orders for which centroid x and y positions will be returned.
    bound : bool
        If True, only returns centroids that fall on the detector after
        polynomial fitting.
    showprogress : bool
        If True, show the emcee progress bar.

    Returns
    -------
    trans_cen : dict
        Dictionary containing x and y trace centroids for each order in
        return_orders.
    rot_pars : tuple
        Tuple containing the required parameters to transform the reference
        centroids to match the dataL: rotation angle, x offset and y offset.

    Raises
    ------
    ValueError
        If ref_centroids does not match return_orders.
    '''

    # Determine reference centroids for all orders.
    ref_cen = {}
    trans_cen = {}
    # If provided as input.
    if ref_centroids is not None:
        # Ensure length is correct.
        if len(ref_centroids) != 2*len(return_orders):
            raise ValueError('Insufficient reference centroids provided.')
        # Repackage into a dictionary
        for i, order in enumerate(return_orders):
            ref_cen['order '+str(order)] = [ref_centroids[2*i], ref_centroids[2*i+1]]
    # Or from the trace table (currently the optics model).
    else:
        for order in return_orders:
            # Extend centroids off of the detector to compensate for shifts.
            xom, yom, tp = get_om_centroids(np.arange(2148)-50, order=order)
            ref_cen['order '+str(order)] = [xom, yom]

    # Get the order 1 centroids from the data.
    xdat_o1, ydat_o1 = uctd.get_uncontam_centroids(clear)
    trans_cen['order 1'] = [xdat_o1, ydat_o1]

    # Fit the reference centroids to the data for order 1.
    fit = _do_emcee(ref_cen['order 1'][0], ref_cen['order 1'][1], xdat_o1,
                    ydat_o1, showprogress=showprogress)
    # Plot MCMC results if requested.
    if doplot is True:
        plotting._plot_corner(fit)
    # Get fitted rotation parameters.
    flat_samples = fit.get_chain(discard=500, thin=15, flat=True)
    ang = np.percentile(flat_samples[:, 0], 50)
    xshift = np.percentile(flat_samples[:, 1], 50)
    yshift = np.percentile(flat_samples[:, 2], 50)
    rot_params = (ang, xshift, yshift)

    # Get rotated centroids for all other orders.
    for order in return_orders:
        if order == 1:
            continue
        xtrans, ytrans = rot_centroids(ang, xshift, yshift,
                                       ref_cen['order '+str(order)][0],
                                       ref_cen['order '+str(order)][1])
        # Ensure that the centroids cover the whole detector.
        pp = np.polyfit(xtrans, ytrans, 5)
        ytrans = np.polyval(pp, np.arange(2048))
        if bound is True:
            inds = np.where((ytrans >= 0) & (ytrans < 256))[0]
            trans_cen['order '+str(order)] = [np.arange(2048)[inds], ytrans[inds]]
        else:
            trans_cen['order '+str(order)] = [np.arange(2048), ytrans]

    return trans_cen, rot_params


# Needs to be updated whenever we decide on how we will interact with
# new reference files
def get_om_centroids(atthesex=None, order=1):
    '''Get trace profile centroids from the NIRISS SOSS optics model.
    These centroids include the standard rotation of 1.489 deg about
    (1514, 486) to transform from the optics model into the CV3 coordinate
    system.

    Parameters
    ----------
    atthesex : list of floats
        Pixel x values at which to evaluate the centroid position.
    order : int
        Diffraction order for which to return the optics model solution.

    Returns
    -------
    xOM : list of floats
        Optics model x centroids.
    yOM : list of floats
        Optics model y centroids.
    tp2 : list of floats
        trace polynomial coefficients.
    '''

    if atthesex is None:
        atthesex = np.linspace(0, 2047, 2048)

    # Derive the trace polynomials.
    tp2 = tp.get_tracepars(filename=path+'/trace/NIRISS_GR700_trace.csv')

    # Evaluate the trace polynomials at the desired coordinates.
    w = tp.specpix_to_wavelength(atthesex, tp2, order, frame='dms', oversample=1)[0]
    xOM, yOM, mas = tp.wavelength_to_pix(w, tp2, order, frame='dms', oversample=1)

    return xOM, yOM, tp2


# depreciated
def get_uncontam_centroids(stack, atthesex=np.arange(2048), fit=True):
    '''Determine the x, y positions of the trace centroids from an
    exposure using a center-of-mass analysis. Works for either order if there
    is no contamination, or for order 1 on a detector where the two orders
    are overlapping.
    This is an adaptation of Loïc's get_order1_centroids which can better
    deal with a bright second order.

    Parameters
    ----------
    stack : array of floats (2D)
        Data frame.
    atthesex : list of floats
        Pixel x values at which to extract the trace centroids.
    fit : bool
        If True, fits a 5th order polynomial to the extracted y-centroids,
        and returns the evaluation of this polynomial at atthesex. If False,
        a y-centroid may not be located for each x-pixel in atthesex.

    Returns
    -------
    tracexbest : np.array
        Best estimate data x centroid.
    traceybest : np.array
        Best estimate data y centroids.
    '''

    # Dimensions of the subarray.
    dimx = len(atthesex)
    dimy = np.shape(stack)[0]

    # Identify the floor level of all 2040 working cols to subtract it first.
    floorlevel = np.nanpercentile(stack, 10, axis=0)
    backsub = stack - floorlevel

    # Find centroid - first pass, use all pixels in the column.
    # Normalize each column
    norm = backsub[:, 4:2044] / np.nanmax(backsub[:, 4:2044], axis=0)
    # Create 2D Array of pixel positions
    rows = (np.ones((2040, 256)) * np.arange(256)).T
    # Mask any nan values
    norm_mask = np.ma.masked_invalid(norm)
    # CoM analysis to find centroid
    cx = (np.nansum(norm_mask * rows, axis=0) / np.nansum(norm, axis=0)).data

    # Adopt these trace values as best
    tracex_best = np.arange(2040)+4
    tracey_best = cx

    # Second pass, find centroid on a subset of pixels
    # from an area around the centroid determined earlier.
    tracex = []
    tracey = []
    row = np.arange(dimy)
    w = 30
    for i in range(dimx - 8):
        miny = np.int(np.nanmax([np.around(tracey_best[i] - w), 0]))
        maxy = np.int(np.nanmax([np.around(tracey_best[i] + w), dimy - 1]))
        val = backsub[miny:maxy, i + 4] / np.nanmax(backsub[:, i + 4])
        ind = np.where(np.isfinite(val))
        thisrow = (row[miny:maxy])[ind]
        thisval = val[ind]
        cx = np.sum(thisrow * thisval) / np.sum(thisval)
        # Ensure that the centroid position is not getting too close to an edge
        # such that it is biased.
        if not np.isfinite(cx) or cx <= 5 or cx >= 250:
            continue
        # For a bright second order, it is likely that the centroid at this
        # point will be somewhere in between the first and second order.
        # If this is the case (i.e. the pixel value of the centroid is very low
        # compared to the column average), restrict the range of pixels
        # considered to be above the current centroid.
        if backsub[int(cx)][i+4] < np.nanmean(backsub[(int(cx) - w):(int(cx)+w), i+4]):
            miny = np.int(np.nanmax([np.around(cx), 0]))
            maxy = np.int(np.nanmin([np.around(cx + 2*w), dimy - 1]))
            val = backsub[miny:maxy, i + 4] / np.nanmax(backsub[:, i + 4])
            ind = np.where(np.isfinite(val))
            thisrow = (row[miny:maxy])[ind]
            thisval = val[ind]
            cx = np.sum(thisrow * thisval) / np.sum(thisval)

        tracex.append(i + 4)
        tracey.append(cx)

    # Adopt these trace values as best.
    tracex_best = np.array(tracex) * 1
    tracey_best = np.array(tracey) * 1

    # Third pass - fine tuning.
    tracex = []
    tracey = []
    row = np.arange(dimy)
    w = 16
    for i in range(len(tracex_best)):
        miny = np.int(np.nanmax([np.around(tracey_best[i] - w), 0]))
        maxy = np.int(np.nanmax([np.around(tracey_best[i] + w), dimy - 1]))
        val = backsub[miny:maxy, i + 4] / np.nanmax(backsub[:, i + 4])
        ind = np.where(np.isfinite(val))
        thisrow = (row[miny:maxy])[ind]
        thisval = val[ind]
        cx = np.sum(thisrow * thisval) / np.sum(thisval)

        tracex.append(tracex_best[i])
        tracey.append(cx)

    tracex_best = np.array(tracex)
    tracey_best = np.array(tracey)

    if fit is True:
        # Fit a polynomial to centroids to ensure there is a centroid at each x
        p_o1 = np.polyfit(tracex_best, tracey_best, 5)
        tracey_best = np.polyval(p_o1, atthesex)
        tracex_best = atthesex

    return tracex_best, tracey_best


def _log_likelihood(theta, xmod, ymod, xdat, ydat):
    '''Definition of the log likelihood. Called by _do_emcee.
    xmod/ymod should extend past the edges of the SUBSTRIP256 detector.
    '''
    ang, xshift, yshift = theta
    # Calculate rotated model
    modelx, modely = rot_centroids(ang, xshift, yshift,
                                   xmod, ymod, bound=True)
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


def _log_probability(theta, xmod, ymod, xdat, ydat):
    '''Definition of the final probability. Called by _do_emcee.
    '''
    lp = _log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp + _log_likelihood(theta, xmod, ymod, xdat, ydat)


def rot_centroids(ang, xshift, yshift, xpix, ypix, bound=True, atthesex=None,
                  cenx=1024, ceny=50, subarray='SUBSTRIP256'):
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
    ypix : float or np.array of float
        Centroid pixel Y values.
    bound : bool
        Whether to trim rotated solutions to fit within the specified subarray.
    atthesex : list of float
        Pixel values at which to calculate rotated centroids.
    cenx : int
        X-coordinate in pixels of the rotation center.
    ceny : int
        Y-coordinate in pixels of the rotation center.
    subarray : str
        Subarray identifier. One of SUBSTRIP96, SUBSTRIP256 or FULL.

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
        if subarray == 'SUBSTRIP96':
            yend = 96
        elif subarray == 'SUBSTRIP256':
            yend = 256
        elif subarray == 'FULL':
            yend = 2048
        else:
            raise ValueError('Unknown subarray. Allowed identifiers are "SUBSTRIP96", "SUBSTRIP256", or "FULL".')
        # Reject pixels which are not on the subarray.
        inds = [(rot_ypix >= 0) & (rot_ypix < yend) & (rot_xpix >= 0) &
                (rot_xpix < 2048)]
        rot_xpix = rot_xpix[inds]
        rot_ypix = rot_ypix[inds]

    return rot_xpix, rot_ypix
