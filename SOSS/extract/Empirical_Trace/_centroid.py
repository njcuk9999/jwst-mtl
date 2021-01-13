#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 9:02 2021

@author: MCR

All functions associated with the determining the centroid positions of the
order 1 and 2 SOSS spectra trace.
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import warnings
from SOSS.trace import tracepol as tp

warnings.simplefilter(action='ignore', category=FutureWarning)

# hack to get around the fact that relative paths are constantly messing up atm
path = '/Users/michaelradica/Documents/GitHub/jwst-mtl/SOSS/'


def _do_emcee(xOM, yOM, xCV, yCV):
    '''Utility function which calls the emcee package to preform
    an MCMC determination of the best fitting rotation angle/center to
    map the OM onto the data.

    Parameters
    ----------
    xOM, yOM : array of floats
        X and Y trace centroids respectively in the optics model system,
        for example: returned by get_om_centroids.
    xCV, yCV : array of floats
        X and Y trace centroids determined from the data, for example:
        returned by get_o1_data_centroids.

    Returns
    -------
    sampler : emcee EnsembleSampler object
        MCMC fitting results.
    '''

    # Set up the MCMC run.
    initial = np.array([1, 1577, 215, 0, 0])  # Initial guess parameters
    pos = initial + 0.5*np.random.randn(32, 5)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_probability,
                                    args=[xOM, yOM, xCV, yCV])
    # Run the MCMC for 5000 steps - it has generally converged
    # within ~3000 steps in trial runs.
    sampler.run_mcmc(pos, 5000, progress=False)

    return sampler


def get_contam_centroids(clear, ref_centroids=None, doplot=False):
    '''Get the trace centroids for both orders when there is
    contaminationof the first order by the second on the detector.
    Fits the first order centroids using the uncontaminated method, and
    determines the second order centroids via the well-calibrated relationship
    between the first and second order profiles.

    Parameters
    ----------
    clear : np.ndarray (2D)
        CLEAR SOSS exposure data frame.
    ref_centroids : np.ndarray (2D)
        Centroids relative to which to determine offset and rotation
        parameters. Array must contain four lists in the following order: x,
        and y centroids for order 1 followed by those of order 2.
    doplot : bool
        Whether to plot the corner plot of the optics model fit to the
        first order centroids.

    Returns
    -------
    atthesex : np.array
        X-centroids for the order 1 trace.
    ycen_o1 : np.array
        y-centroids for the order 1 trace.
    atthesex[inds] : np.array
        x-centroids for the order 2 trace.
    ycen_o2 : np.array
        y-centroids for the order 2 trace.
    ang : float (optional)
        rotation angle for optics model to data transformation.
    xanch : float (optional)
        x coordinate of rotation center for optics model to data transform.
    yanch : float (optional)
        y coordinate of rotation center for optics model to data transform.

    Raises
    ------
    ValueError
        If ref_centroids does not contain four centroids lists.
    '''

    # Determine optics model centroids for both orders
    # as well as order 1 data centroids
    atthesex = np.arange(2048)
    if ref_centroids is not None:
        if len(ref_centroids) != 4:
            raise ValueError('The first dimension of ref_centroids must be 4.')
        xOM1, yOM1 = ref_centroids[0], ref_centroids[1]
        xOM2, yOM2 = ref_centroids[2], ref_centroids[3]
    else:
        xOM1, yOM1, tp1 = get_om_centroids(atthesex)
        xOM2, yOM2, tp2 = get_om_centroids(atthesex, order=2)

    xcen_o1, ycen_o1 = get_uncontam_centroids(clear, atthesex, fit=True)

    # Fit the OM to the data for order 1
    AA = _do_emcee(xOM1, yOM1, atthesex, ycen_o1)

    # Plot MCMC results if required
    if doplot is True:
        _plot_corner(AA)

    # Get fitted rotation parameters
    flat_samples = AA.get_chain(discard=500, thin=15, flat=True)
    ang = np.percentile(flat_samples[:, 0], 50)
    xanch = np.percentile(flat_samples[:, 1], 50)
    yanch = np.percentile(flat_samples[:, 2], 50)
    xshift = np.percentile(flat_samples[:, 3], 50)
    yshift = np.percentile(flat_samples[:, 4], 50)

    # Get rotated OM centroids for order 2
    xcen_o2, ycen_o2 = rot_centroids(ang, xanch, yanch, xshift, yshift,
                                     xOM2, yOM2)
    # Ensure that the second order centroids cover the whole detector
    p_o2 = np.polyfit(xcen_o2, ycen_o2, 5)
    ycen_o2 = np.polyval(p_o2, atthesex)
    inds = np.where((ycen_o2 >= 0) & (ycen_o2 < 256))[0]

    return atthesex, ycen_o1, atthesex[inds], ycen_o2[inds], (ang, xanch, yanch, xshift, yshift)


# Needs to be updated whenever we decide on how we will interact with
# new reference files
def get_om_centroids(atthesex=None, order=1):
    '''Get trace profile centroids from the NIRISS SOSS optics model.
    These centroids include the standard rotation of 1.489 rad about
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
    w = tp.specpix_to_wavelength(atthesex, tp2, order, frame='nat')[0]
    xOM, yOM, mas = tp.wavelength_to_pix(w, tp2, order, frame='nat')

    return xOM, yOM[::-1], tp2


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


def _log_likelihood(theta, xvals, yvals, xCV, yCV):
    '''Definition of the log likelihood. Called by do_emcee.
    '''
    ang, orx, ory, xshift, yshift = theta
    # Calculate rotated model
    modelx, modely = rot_centroids(ang, orx, ory, xshift, yshift,
                                   xvals, yvals, bound=True)
    # Interpolate rotated model onto same x scale as data
    modely = np.interp(xCV, modelx, modely)

    return -0.5 * np.sum((yCV - modely)**2 - 0.5 * np.log(2 * np.pi * 1))


def _log_prior(theta):
    '''Definition of the priors. Called by do_emcee.
    '''
    ang, orx, ory, xshift, yshift = theta

    if -15 <= ang < 15 and 0 <= orx < 2048 and 0 <= ory < 256 and -2048 <= xshift < 2048 and -256 <= yshift < 256:
        return -1
    else:
        return -np.inf


def _log_probability(theta, xvals, yvals, xCV, yCV):
    '''Definition of the final probability. Called by do_emcee.
    '''
    lp = _log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp + _log_likelihood(theta, xvals, yvals, xCV, yCV)


def _plot_corner(sampler):
    '''Utility function to produce the corner plot
    for the results of do_emcee.
    '''
    labels = [r"ang", "cenx", "ceny", "xshift", "yshift"]
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=labels)

    return None


def rot_centroids(ang, cenx, ceny, xshift, yshift, xpix, ypix,
                  bound=True, fill_det=True):
    '''Apply a rotation and shift to the trace centroids positions. This
    assumes that the trace centroids are already in the CV3 coordinate system.

    Parameters
    ----------
    ang : float
        The rotation angle in degrees CCW.
    cenx : float
        The X pixel values to use as the center of rotation.
    ceny : float
        The Y pixel values to use as the center of rotation.
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
    fill_det : bool
        if True, return a transformed centroid position for each pixel on
        the spectral axis.

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
    TypeError
        If fill_det is True, and only one point is passed.
    '''

    # Convert to numpy arrays
    xpix = np.atleast_1d(xpix)
    ypix = np.atleast_1d(ypix)
    # Required rotation in the detector frame to match the data.
    t = np.deg2rad(ang)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    points1 = np.array([xpix - cenx, ypix - ceny])
    rot_pix = R @ points1

    rot_pix[0] += cenx
    rot_pix[1] += ceny

    # Apply the offsets
    rot_pix[0] += xshift
    rot_pix[1] += yshift

    if fill_det is True:
        # Polynomial fit to rotated centroids to ensure there is a centroid at
        # each pixel on the detector
        if xpix.size < 2:
            raise TypeError('fill_det must be False to rotate single points.')
        pp = np.polyfit(rot_pix[0], rot_pix[1], 5)
        rot_xpix = np.arange(2048)
        rot_ypix = np.polyval(pp, rot_xpix)
    else:
        rot_xpix = rot_pix[0]
        rot_ypix = rot_pix[1]

    # Check to ensure all points are on the subarray.
    if bound is True:
        inds = [(rot_ypix >= 0) & (rot_ypix < 256) & (rot_xpix >= 0) &
                (rot_xpix < 2048)]

        return rot_xpix[inds], rot_ypix[inds]
    else:
        return rot_xpix, rot_ypix
