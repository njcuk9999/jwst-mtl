#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:15 2020

@author: MCR

Functions for the 'simple solver' - calculating rotation and offset of
reference order 1 and 2 trace profiles.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import emcee
import corner
import sys
tppath = '/Users/michaelradica/Documents/GitHub/jwst-mtl/SOSS/trace'
sys.path.insert(1, tppath)
import tracepol as tp


def get_uncontam_centroids(stack, atthesex=None):
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

    # Identify the floor level of all 2040 working pixels to subtract it first.
    floorlevel = np.nanpercentile(stack, 10, axis=0)
    backsubtracted = stack*1
    for i in range(dimx-8):
        backsubtracted[:, i] = stack[:, i] - floorlevel[i]

    # Find centroid - first pass, use all pixels in the column.
    tracex = []
    tracey = []
    row = np.arange(dimy)
    for i in range(dimx - 8):
        val = backsubtracted[:, i + 4] / np.nanmax(backsubtracted[:, i + 4])
        ind = np.where(np.isfinite(val))
        thisrow = row[ind]
        thisval = val[ind]
        cx = np.sum(thisrow * thisval) / np.sum(thisval)
        tracex.append(i + 4)
        tracey.append(cx)

    # Adopt these trace values as best
    tracex_best = np.array(tracex) * 1
    tracey_best = np.array(tracey) * 1

    # Second pass, find centroid on a subset of pixels
    # from an area around the centroid determined earlier.
    tracex = []
    tracey = []
    row = np.arange(dimy)
    w = 30
    for i in range(dimx - 8):
        miny = np.int(np.nanmax([np.around(tracey_best[i] - w), 0]))
        maxy = np.int(np.nanmax([np.around(tracey_best[i] + w), dimy - 1]))
        val = backsubtracted[miny:maxy, i + 4] / np.nanmax(backsubtracted[:, i + 4])
        ind = np.where(np.isfinite(val))
        thisrow = (row[miny:maxy])[ind]
        thisval = val[ind]
        cx = np.sum(thisrow * thisval) / np.sum(thisval)

        # For a bright second order, it is likely that the centroid at this
        # point will be somewhere in between the first and second order.
        # If this is the case (i.e. the pixel value of the centroid is very low
        # compared to the column average), restrict the range of pixels
        # considered to be above the current centroid.
        if backsubtracted[int(cx)][i+4] < np.nanmean(backsubtracted[(int(cx) - w):(int(cx)+w), i+4]):
            miny = np.int(np.nanmax([np.around(cx), 0]))
            maxy = np.int(np.nanmin([np.around(cx + 2*w), dimy - 1]))
            val = backsubtracted[miny:maxy, i + 4] / np.nanmax(backsubtracted[:, i + 4])
            ind = np.where(np.isfinite(val))
            thisrow = (row[miny:maxy])[ind]
            thisval = val[ind]
            cx = np.sum(thisrow * thisval) / np.sum(thisval)

            tracex.append(i + 4)
            tracey.append(cx)

        else:
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
    for i in range(dimx - 8):
        miny = np.int(np.nanmax([np.around(tracey_best[i] - w), 0]))
        maxy = np.int(np.nanmax([np.around(tracey_best[i] + w), dimy - 1]))
        val = backsubtracted[miny:maxy, i + 4] / np.nanmax(backsubtracted[:, i + 4])
        ind = np.where(np.isfinite(val))
        thisrow = (row[miny:maxy])[ind]
        thisval = val[ind]
        cx = np.sum(thisrow * thisval) / np.sum(thisval)

        tracex.append(i + 4)
        tracey.append(cx)

    tracex_best = np.array(tracex)
    tracey_best = np.array(tracey)

    return tracex_best, tracey_best


def get_contam_centroids(clear, return_rot_params=False, doplot=False):
    '''Get the trace centroids for both orders when there is
    contaminationof the first order by the second on the detector.
    Fits the first order centroids using the uncontaminated method, and
    determines the second order centroids via the well-calibrated relationship
    between the first and second order profiles in the optics model.

    Parameters
    ----------
    clear : np.ndarray (2D)
        CLEAR SOSS exposure data frame.
    return_rot_params : bool
        Whether to return the rotation angle and anchor point required to
        transform the optics model to match the data.
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
    '''

    # Determine optics model centroids for both orders
    # as well as order 1 data centroids
    atthesex = np.arange(2048)
    xOM1, yOM1, tp1 = get_om_centroids(atthesex)
    xOM2, yOM2, tp2 = get_om_centroids(atthesex, order=2)
    xcen_o1, ycen_o1 = get_uncontam_centroids(clear, atthesex)
    p_o1 = np.polyfit(xcen_o1, ycen_o1, 5)
    ycen_o1 = np.polyval(p_o1, atthesex)

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

    # Get rotated OM centroids for order 2
    xcen_o2, ycen_o2 = rot_om2det(ang, xanch, yanch, xOM2, yOM2, order=2)
    # Ensure that the second order centroids cover the whole detector
    p_o2 = np.polyfit(xcen_o2, ycen_o2, 5)
    ycen_o2 = np.polyval(p_o2, atthesex)
    inds = np.where((ycen_o2 >= 0) & (ycen_o2 < 256))[0]

    # Also return rotation parameters if requested
    if return_rot_params is True:
        return atthesex, ycen_o1, atthesex[inds], ycen_o2[inds], (ang, xanch, yanch)
    else:
        return atthesex, ycen_o1, atthesex[inds], ycen_o2[inds]


def get_om_centroids(atthesex=None, order=1):
    '''Get trace profile centroids from the NIRISS SOSS optics model.

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
    tp2 = tp.get_tracepars(filename='%s/NIRISS_GR700_trace.csv' % tppath)

    # Evaluate the trace polynomials at the desired coordinates.
    w = tp.specpix_to_wavelength(atthesex, tp2, order, frame='nat')[0]
    xOM, yOM, mas = tp.wavelength_to_pix(w, tp2, order, frame='nat')

    return xOM, yOM[::-1], tp2


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
    initial = np.array([1, 1577, 215])  # Initial guess parameters
    pos = initial + 0.5*np.random.randn(32, 3)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_probability,
                                    args=[xOM, yOM, xCV, yCV])
    # Run the MCMC for 5000 steps - it has generally converged
    # within ~3000 steps in trial runs.
    sampler.run_mcmc(pos, 5000, progress=False)

    return sampler


def _log_likelihood(theta, xvals, yvals, xCV, yCV):
    '''Definition of the log likelihood. Called by do_emcee.
    '''
    ang, orx, ory = theta
    # Calculate rotated model
    modelx, modely = rot_om2det(ang, orx, ory, xvals, yvals, bound=True)
    # Interpolate rotated model onto same x scale as data
    modely = np.interp(xCV, modelx, modely)

    return -0.5 * np.sum((yCV - modely)**2 - 0.5 * np.log(2 * np.pi * 1))


def _log_prior(theta):
    '''Definition of the priors. Called by do_emcee.
    '''
    ang, orx, ory = theta

    if -15 <= ang < 15 and 0 < orx < 4048 and 0 < ory < 456:
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
    labels = [r"ang", "cenx", "ceny"]
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=labels)

    return None


def rot_om2det(ang, cenx, ceny, xval, yval, order=1, bound=True):
    '''Utility function to map coordinates in the optics model
    reference frame, onto the detector reference frame, given
    the correct transofmration parameters.

    Parameters
    ----------
    ang : float
        The rotation angle in degrees CCW.
    cenx, ceny : float
        The X and Y pixel values to use as the center of rotation
        in the optics model coordinate system.
    xval, yval : float
        Pixel X and Y values in the optics model coordinate system
        to transform into the detector frame.
    order : int
        Diffraction order.
    bound : bool
        Whether to trim rotated solutions to fit within the subarray256.

    Returns
    -------
    rot_xpix, rot_ypix : float
        xval and yval respectively transformed into the
        detector coordinate system.
    '''

    # Map OM onto detector - the parameters for this transformation
    # are already well known.
    if order == 1:
        t = 1.489*np.pi / 180
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        points1 = np.array([xval - 1514, yval - 456])
        b = R @ points1

        b[0] += 1514
        b[1] += 456

    if order == 2:
        t = 1.84*np.pi / 180
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        points1 = np.array([xval - 1366, yval - 453])
        b = R @ points1

        b[0] += 1366
        b[1] += 453

    # Required rotation in the detector frame to match the data.
    t = (ang+0.95)*np.pi / 180
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    points1 = np.array([b[0] - cenx, b[1] - ceny])
    rot_pix = R @ points1

    rot_pix[0] += cenx
    rot_pix[1] += ceny

    # Polynomial fit to rotated centroids to ensure there is a centroid at
    # each pixel on the detector
    pp = np.polyfit(rot_pix[0], rot_pix[1], 5)
    rot_xpix = np.arange(2048)
    rot_ypix = np.polyval(pp, rot_xpix)

    # Check to ensure all points are on the subarray.
    if bound is True:
        inds = [(rot_ypix >= 0) & (rot_ypix < 256) & (rot_xpix >= 0) &
                (rot_xpix < 2048)]

        return rot_xpix[inds], rot_ypix[inds]
    else:
        return rot_xpix, rot_ypix


def simple_solver(xc, yc, order=1):
    '''Calculate and preform corrections to the reference trace profiles
    due to rotation and vertical/horizontal offsets.

    Parameters
    ----------
    xc : list
        List of data x-centroids for the desired order.
    yc : list
        List of data y-centroids for the desired order
    order : int
        Desired order, either 1 or 2.

    Returns
    -------
    rot_frame : np.ndarray
        Reference trace frame (2D) with rotation and offset corrections
        applied.
    '''

    # Open the reference trace profile for the desired order
    if order == 1:
        ref_frame = fits.open('trace_profile_m1.fits')[0].data[::-1, :]
    if order == 2:
        ref_frame = fits.open('trace_profile_m2.fits')[0].data[::-1, :]

    # Initalize black frame
    rot_frame = np.zeros((256, 2048))

    # Get the optics model centroids (which are the reference trace centroids)
    xcen, ycen, tp = get_om_centroids(atthesex=np.arange(2048), order=order)
    # Convert the y-centroids to ints
    # Scale by 10 to allow for 10x oversampling
    ycen = (np.round(ycen*10, 0)).astype(int)

    # Loop over all columns for which the data centroids fall on the detector
    for newi in range(len(xc)):
        # For the second order, the reference trace is only defined
        # for the first 1710 spectral pixels.
        # If more spectral pixels are required, reuse the 1710 profile.
        if order == 2 and newi > 1710:
            refi = 1710
        else:
            refi = newi

        # 10x oversampled axis accounting for trace profile extending above
        # or below the detector (in spatial direction)
        ax_os = np.linspace(-341, 512, 8531)
        # Oversample the reference trace slice
        slice_os = np.interp(ax_os, np.arange(256), ref_frame[:, refi])
        # Extract the trace spetial profile
        tslice = slice_os[(3411+ycen[refi]-340):(3411+ycen[refi]+351)]

        # New 10x oversampled axis for corrected frame, shifted to
        # spatial position of data
        axis = np.linspace(-34, 34, 690) + yc[newi] - 1
        # Keep only positions where the new axis is on the detector
        inds = np.where((axis < 256) & (axis >= 0))[0]
        # interpolate oversampled trace profile back to native resolution
        newslice = np.interp(np.arange(256), axis[inds], tslice[inds])

        # Add corrected slice to new frame
        rot_frame[:, newi] = newslice

    return rot_frame
