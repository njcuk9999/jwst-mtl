#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Web Oct 28 11:15 2020

@author: MCR

Functions for the 'simple solver' - calculating rotation and offset of
reference order 1 and 2 trace profiles.
"""

import warnings
warnings.filterwarnings('ignore')
import webbpsf
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
    ''' Determine the x, y positions of the order 1 trace centroids from an
    exposure using a center-of-mass analysis.
    This is an adaptation of Loïc's get_order1_centroids which can better
    deal with a bright second order.

    Parameters
    ----------
    stack : numpy array of floats
        Data frame.
    atthesex : list of floats
        Pixel x values at which to extract the trace centroids.

    Returns
    -------
    tracexbest : list of floats
        Best estimate data x centroid.
    traceybest : list of floats
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


def get_contam_centroids(clear, return_rot_params=False):
    ''' Get the order 2 trace centroids via fitting the optics model
    to the first order, and using the known relationship between the
    positions of the first and second orders.

    Parameters
    ----------
    clear : np.ndarray
        CLEAR SOSS exposure data frame.
    return_o1 : bool
        Whether to include the order 1 centroids in the returned value

    Returns
    -------
    xM2 : list
        X-centroids for the order 2 trace.
    yM2 : list
        y-centroids for the order 2 trace.
    xM1 : list (optional)
        x-centroids for the order 1 trace.
    yM1 : list (optional)
        y-centroids for the order 1 trace.
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
    AA = do_emcee(xOM1, yOM1, atthesex, ycen_o1)

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
    ''' Utility function to get order 1 trace profile centroids from the
    JWST NIRISS SOSS optics model.

    Parameters
    ----------
    atthesex : list of floats
        Pixel x values at which to evaluate the centroid position.
    order : int
        Diffraction order for which to return the optics model solution.

    Returns
    -------
    xOM : list of floats
        Optics model x centroid.
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


def do_emcee(xOM, yOM, xCV, yCV):
    ''' Utility function which calls the emcee package to preform
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

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=[xOM, yOM, xCV, yCV])
    # Run the MCMC for 5000 steps - it has generally converged
    # within ~3000 steps in trial runs.
    sampler.run_mcmc(pos, 5000, progress=False)

    return sampler


def log_likelihood(theta, xvals, yvals, xCV, yCV):
    ''' Definition of the log likelihood. Called by do_emcee.
    '''
    ang, orx, ory = theta
    # Calculate rotated model
    modelx, modely = rot_om2det(ang, orx, ory, xvals, yvals, bound=True)
    # Interpolate rotated model onto same x scale as data
    modely = np.interp(xCV, modelx, modely)

    return -0.5 * np.sum((yCV - modely)**2 - 0.5 * np.log(2 * np.pi * 1))


def log_prior(theta):
    ''' Definition of the priors. Called by do_emcee.
    '''
    ang, orx, ory = theta

    if -15 <= ang < 15 and 0 < orx < 4048 and 0 < ory < 456:
        return -1
    else:
        return -np.inf


def log_probability(theta, xvals, yvals, xCV, yCV):
    ''' Definition of the final probability. Called by do_emcee.
    '''
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, xvals, yvals, xCV, yCV)


def plot_corner(sampler):
    ''' Utility function to produce the corner plot
    for the results of do_emcee. Called by makemod.
    '''
    labels = [r"ang", "cenx", "ceny"]
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=labels)

    return None


def rot_om2det(ang, cenx, ceny, xval, yval, order=1, bound=True):
    ''' Utility function to map coordinates in the optics model
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
    rot_pix[0], rot_pix[1] : float
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

    # Check to ensure all points are on the subarray.
    if bound is True:
        inds = [(rot_pix[1] >= 0) & (rot_pix[1] < 256) & (rot_pix[0] >= 0) &
                (rot_pix[0] < 2048)]

        return rot_pix[0][inds], rot_pix[1][inds]
    else:
        return rot_pix[0], rot_pix[1]
