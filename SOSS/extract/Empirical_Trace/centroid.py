#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 9:02 2021

@author: MCR

Functions associated with the determining the centroid positions of the SOSS
spectral traces for the first, second, and third orders.
"""

import numpy as np
from numpy.polynomial import Legendre
import emcee
import warnings
from SOSS.trace import get_uncontam_centroids as uctd
from SOSS.extract import soss_read_refs
from SOSS.extract.empirical_trace import plotting as plotting

warnings.simplefilter(action='ignore', category=FutureWarning)

# hack to get around the fact that relative paths are constantly messing up atm
path = '/Users/michaelradica/Documents/School/Ph.D./Research/SOSS/Extraction/Input_Files/'


def _do_emcee(xref, yref, xdat, ydat, showprogress=False):
    '''Calls the emcee package to preform an MCMC determination of the best
    fitting rotation angle and offsets to map the reference centroids onto the
    data for the first order.

    Parameters
    ----------
    xref, yref : array of float
        X and Y trace centroids respectively to be used as a reference point,
        for example: as returned by get_ref_centroids.
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


def get_contam_centroids(clear, ref_centroids=None, return_orders=[1, 2, 3],
                         bound=True, verbose=False):
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
    return_orders : list
        Orders for which centroid x and y positions will be returned.
    bound : bool
        If True, only returns centroids that fall on the detector after
        polynomial fitting.
    verbose : bool
        If True, show diagnostic plots, and the emcee progress bar.

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
            xref, yref = get_ref_centroids(atthesex=np.arange(2148)-50,
                                           order=order)
            ref_cen['order '+str(order)] = [xref, yref]

    # Get the order 1 centroids from the data.
    xdat_o1, ydat_o1 = uctd.get_uncontam_centroids(clear)
    trans_cen['order 1'] = [xdat_o1, ydat_o1]

    # Fit the reference centroids to the data for order 1.
    fit = _do_emcee(ref_cen['order 1'][0], ref_cen['order 1'][1], xdat_o1,
                    ydat_o1, showprogress=verbose)
    # Plot MCMC results if requested.
    if verbose is True:
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


def get_ref_centroids(atthesex=None, subarray='SUBSTRIP256', order=1):
    '''Get trace centroids from the Trace Table reference file. Uses Legendre
    polynomial fitting to extend the domain of the centroids beyond the
    detector edges if necessary.

    Parameters
    ----------
    atthesex : array of float
        Pixel X-positions at which to return centroids.
    subarray : str
        Subarray to consider. Allowed values are "SUBSTRIP96", "SUBSTRIP256",
        or "FULL".
    order : int
        Diffraction order for which to return centroids. First three
        orders are currently supported.

    Returns
    -------
    xcen_ref : array of float
        Centroid pixel X-coordinates.
    ycen_ref : array of float
        Centroid pixel Y-coordinates.

    Raises
    ------
    ValueError
        If bad subarray identifier is passed.
    '''

    # Verify inputs.
    if subarray not in ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']:
        raise ValueError('Unknown subarray identifier.')

    # Open trace table reference file.
    ttab_file = soss_read_refs.RefTraceTable(path+'SOSS_ref_trace_table.fits')
    # Get first order centroids on subarray.
    xcen_ref = ttab_file('X', subarray=subarray, order=order)[1][::-1]
    ycen_ref = ttab_file('Y', subarray=subarray, order=order)[1][::-1]

    # Use Legendre polynomial fit to extend centroids beyond detector edges.
    inds = np.where((xcen_ref < 2098) & (xcen_ref >= -50))
    pp = Legendre.fit(xcen_ref[inds], ycen_ref[inds], 6-order)
    xcen_ref = atthesex
    ycen_ref = pp(xcen_ref)

    return xcen_ref, ycen_ref


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
        Subarray identifier. One of "SUBSTRIP96", "SUBSTRIP256" or "FULL".

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
