#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 10:41 2020

@author: MCR

Third iteration of functions for the 'simple solver' - calculating rotation
and offset of reference order 1 and 2 trace profiles. This iteration
incorporates offset and rotation to the transformation, and uses a rotation
matrix to preform the rotation transformation instead of the previous
interpolation method.
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import emcee
import corner
import sys
from scipy.ndimage.interpolation import rotate
sys.path.insert(0, '../../trace')
import tracepol as tp

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


def _do_transform(data, rot_ang, x_anch, y_anch, x_shift, y_shift,
                  padding_factor=2, oversampling=1):
    '''Do the rotation (via a rotation matrix) and offset of the reference
    files to match the data. Rotation angle and center, as well as the
    required vertical and horizontal displacements must be calculated
    beforehand.
    This assumes that we have a sufficiently padded reference file, and that
    oversampling is equal in the spatial and spectral directions.
    The reference file is interpolated to the native detector resolution after
    the transformations if oversampled.

    Parameters
    ----------
    data : np.ndarray
        Reference file data.
    rot_ang : float
        Rotation angle in degrees.
    x_anch : float
        x pixel around which to rotate.
    y_anch : float
        y pixel around which to rotate.
    x_shift : float
        Offset in the spectral direction to be applied after rotation.
    y_shift : float
        Offset in the spatial direction to be applied after rotation.
    padding_factor : float
        Factor by which the reference file is padded over the size of
        the SUBSTRIP256 detector.
    oversampling : int
        Factor by which the reference data is oversampled. The
        oversampling is assumed to be equal in both the spectral and
        spatial directions.

    Returns
    -------
    data_sub256_nat : np.ndarray
        Reference file with all transformations applied and interpolated
        to the native detector resolution.
    '''

    # Determine x and y center of the padded dataframe
    pad_xcen = np.int(2048 * padding_factor * oversampling / 2)
    pad_ycen = np.int(256 * padding_factor * oversampling / 2)
    # Find bottom left hand corner of substrip256 in the padded frame
    bleft_x = np.int(oversampling * (1024 * padding_factor - 1024))
    bleft_y = np.int(oversampling * (128 * padding_factor - 128))

    # Shift dataframe such that rotation anchor is in the center of the frame
    data_shift = np.roll(data, (pad_ycen-bleft_y-y_anch, pad_xcen-bleft_x-x_anch), (0, 1))

    # Rotate the shifted dataframe by the required amount
    data_rot = rotate(data_shift, rot_ang, reshape=False)

    # Shift the rotated data back to its original position
    data_shiftback = np.roll(data_rot, (-pad_ycen+bleft_y+y_anch, -pad_xcen+bleft_x+x_anch), (0, 1))

    # Apply vertical and horizontal offsets
    # Add some error if offset value is greater than padding
    data_offset = np.roll(data_shiftback, (y_shift, x_shift), (0, 1))

    # Extract the substrip256 detector - discluding the padding
    data_sub256 = data_offset[bleft_y:bleft_y+256*oversampling,
                              bleft_x:bleft_x+2048*oversampling]

    # Interpolate to native resolution if the reference frame is oversampled
    if oversampling != 1:
        data_sub256_nat = np.empty((256, 2048))
        # Loop over the spectral direction and interpolate the oversampled
        # spatial profile to native resolution
        for i in range(2048*oversampling):
            new_ax = np.arange(256)
            oversamp_ax = np.linspace(0, 256, 256*oversampling, endpoint=False)
            oversamp_prof = data_sub256[:, bleft_x+i]
            data_sub256_nat[:, i] = np.interp(new_ax, oversamp_ax,
                                              oversamp_prof)
    else:
        data_sub256_nat = data_sub256

    return data_sub256_nat


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
    xshift = np.percentile(flat_samples[:, 3], 50)
    yshift = np.percentile(flat_samples[:, 4], 50)

    # Get rotated OM centroids for order 2
    xcen_o2, ycen_o2 = rot_om2det(ang, xanch, yanch, xshift, yshift,
                                  xOM2, yOM2)
    # Ensure that the second order centroids cover the whole detector
    p_o2 = np.polyfit(xcen_o2, ycen_o2, 5)
    ycen_o2 = np.polyval(p_o2, atthesex)
    inds = np.where((ycen_o2 >= 0) & (ycen_o2 < 256))[0]

    # Also return rotation parameters if requested
    if return_rot_params is True:
        return atthesex, ycen_o1, atthesex[inds], ycen_o2[inds], (ang, xanch, yanch, xshift, yshift)
    else:
        return atthesex, ycen_o1, atthesex[inds], ycen_o2[inds]


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


def _log_likelihood(theta, xvals, yvals, xCV, yCV):
    '''Definition of the log likelihood. Called by do_emcee.
    '''
    ang, orx, ory, xshift, yshift = theta
    # Calculate rotated model
    modelx, modely = rot_om2det(ang, orx, ory, xshift, yshift,
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


# Should rename
def rot_om2det(ang, cenx, ceny, xshift, yshift, xval, yval,
               bound=True):
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
    xshift, yshift : float
        Offset in the x and y direction to be rigidly applied to
        the trace model after rotation.
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
    #t = 1.489*np.pi / 180
    #R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    #points1 = np.array([xval - 1514, yval - 456])
    #b = R @ points1

    #b[0] += 1514
    #b[1] += 456
    # Note the major difference here between the old version of the
    # rotation algorithm is that the parameters are identical for the
    # first and second order.

    # Required rotation in the detector frame to match the data.
    #t = (ang+0.95)*np.pi / 180
    t = np.deg2rad(ang)  # CV3 is at a rotation of 0.95 deg
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    points1 = np.array([xval - cenx, yval - ceny])
    rot_pix = R @ points1

    rot_pix[0] += cenx
    rot_pix[1] += ceny

    # Apply the offsets
    rot_pix[0] += xshift
    rot_pix[1] += yshift

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


def simple_solver(clear):
    '''First implementation of the simple_solver algorithm to calculate
    and preform the necessary rotation and offsets to transform the
    reference traces and wavelength maps to match the science data.
    The steps are as follows:
        1. Open the data files and locate the centroids.
        2. Call get_contam_centroids to determine the correct rotation and
        offset parameters.
        3. Open the reference files and determine the appropriate padding
        and oversampling factors.
        4. Call _do_transform to transform the reference files to match the data.
        5. Return the transformed reference traces and wavelength maps.

    Parameters
    ----------
    clear : np.ndarray
        CLEAR science exposure.

    Returns
    -------
    ref_trace_trans : np.ndarray
        2x256x2048 array containing the reference trace profiles transformed
        to match the science data.
    wave_map_trans : np.ndarray
        2x256x2048 array containing the reference trace profiles transformed
        to match the science data.
    '''

    # Get data centroids
    xcen_o1, ycen_o1, xcen_o2, ycen_o2, rot_params = get_contam_centroids(clear, return_rot_params=True)
    rot_ang = rot_params[0]
    x_anch = np.int(rot_params[1])
    y_anch = np.int(rot_params[2])
    x_shift = np.int(rot_params[3])
    y_shift = np.int(rot_params[4])

    ref_trace_trans = np.empty((2, 256, 2048))
    wave_map_trans = np.empty((2, 256, 2048))

    for order in [1, 2]:
        # Load the reference trace and wavelength map for the current order
        if order == 1:
            ref_trace = fits.open(path+'/extract/Ref_files/trace_profile_om1.fits')[0].data[::-1, :]
            # Clean up the reference trace - no zero pixels and add padding
            ref_trace[np.where(ref_trace == 0)] = 1
            ref_trace = np.pad(ref_trace, ((128, 128), (1024, 1024)),
                               constant_values=((1, 1), (1, 1)))
            pad_t = 2
            os_t = 1
            wave_map = fits.open(path+'/extract/Ref_files/wavelengths_m1.fits')[0].data[::-1, :]
            wave_map = np.pad(wave_map, ((128, 128), (1024, 1024)),
                              constant_values=((1, 1), (1, 1)))
            pad_w = 2
            os_w = 1
        if order == 2:
            ref_trace = fits.open(path+'/extract/Ref_files/trace_profile_om2.fits')[0].data[::-1, :]
            # Clean up the reference trace - no zero pixels and add padding
            ref_trace[np.where(ref_trace == 0)] = 1
            ref_trace = np.pad(ref_trace, ((128, 128), (1024, 1024)),
                               constant_values=((1, 1), (1, 1)))
            pad_t = 2
            os_t = 1
            wave_map = fits.open(path+'/extract/Ref_files/wavelengths_m2.fits')[0].data[::-1, :]
            wave_map = np.pad(wave_map, ((128, 128), (1024, 1024)),
                              constant_values=((1, 1), (1, 1)))
            pad_w = 2
            os_w = 1

        ref_trace_trans[order-1, :, :] = _do_transform(ref_trace, rot_ang,
                                                       x_anch, y_anch, x_shift,
                                                       y_shift,
                                                       padding_factor=pad_t,
                                                       oversampling=os_t)
        wave_map_trans[order-1, :, :] = _do_transform(wave_map, rot_ang,
                                                      x_anch, y_anch, x_shift,
                                                      y_shift,
                                                      padding_factor=pad_w,
                                                      oversampling=os_w)

    return ref_trace_trans, wave_map_trans
