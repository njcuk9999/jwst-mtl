#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 10:41 2020

@author: MCR

Functions for the 'simple solver' - calculating rotation and offset of
reference order 1 and 2 trace profiles, as well as wavelength maps. This
iteration incorporates offset and rotation to the transformation, and uses a
rotation matrix to preform the rotation transformation.
"""

import numpy as np
from astropy.io import fits
import re
import warnings
import emcee
from scipy.ndimage.interpolation import rotate
from SOSS.dms import soss_centroids as ctd
from SOSS.extract import soss_read_refs
from SOSS.extract.simple_solver import plotting as plotting

warnings.simplefilter(action='ignore', category=RuntimeWarning)

# local path to reference files.
path = '/Users/michaelradica/Documents/School/Ph.D./Research/SOSS/Extraction/Input_Files/'


def _do_emcee(xref, yref, xdat, ydat, subarray='SUBSTRIP256',
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
        Identifier for the correct subarray. Allowed values are "SUBSTRIP96",
        "SUBSTRIP256", or "FULL".
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
    # Run the MCMC for 5000 steps - it has generally converged well before.
    sampler.run_mcmc(pos, 5000, progress=showprogress)

    return sampler


def _do_transform(data, rot_ang, x_shift, y_shift, pad=0, oversample=1,
                  verbose=False):
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
    x_shift : float
        Offset in the spectral direction to be applied after rotation.
    y_shift : float
        Offset in the spatial direction to be applied after rotation.
    pad : int
        Number of native pixels of padding on each side of the frame.
    oversample : int
        Factor by which the reference data is oversampled. The
        oversampling is assumed to be equal in both the spectral and
        spatial directions.
    verbose : bool
        Whether to do diagnostic plots and prints.

    Returns
    -------
    data_sub256_nat : np.ndarray
        Reference file with all transformations applied and interpolated
        to the native detector resolution.
    '''

    x_shift, y_shift = int(round(x_shift, 0)), int(round(y_shift, 0))
    # Determine x and y center of the padded dataframe
    pad_ydim, pad_xdim = np.shape(data)
    nat_xdim = int(round(pad_xdim / oversample - 2*pad, 0))
    nat_ydim = int(round(pad_ydim / oversample - 2*pad, 0))
    pad_xcen = pad_xdim // 2
    pad_ycen = pad_ydim // 2

    # Rotation anchor is o1 trace centroid halfway along the spectral axis.
    x_anch = int((1024+pad)*oversample)
    y_anch = int((50+pad)*oversample)

    # Shift dataframe such that rotation anchor is in the center of the frame.
    data_shift = np.roll(data, (pad_ycen-y_anch, pad_xcen-x_anch), (0, 1))
    # Rotate the shifted dataframe by the required amount.
    data_rot = rotate(data_shift, rot_ang, reshape=False)
    # Shift the rotated data back to its original position.
    data_shiftback = np.roll(data_rot, (-pad_ycen+y_anch, -pad_xcen+x_anch),
                             (0, 1))
    # Apply vertical and horizontal offsets.
    data_offset = np.roll(data_shiftback, (y_shift*oversample,
                          x_shift*oversample), (0, 1))
    if verbose is True:
        plotting._plot_transformation_steps(data_shift, data_rot,
                                            data_shiftback, data_offset)
    # Remove the padding.
    data_sub = data_offset[(pad*oversample):(-pad*oversample),
                           (pad*oversample):(-pad*oversample)]

    # Interpolate to native resolution if the reference frame is oversampled.
    if oversample != 1:
        data_nat1 = np.ones((nat_ydim, nat_xdim*oversample))
        data_nat = np.ones((nat_ydim, nat_xdim))
        # Loop over the spectral direction and interpolate the oversampled
        # spatial profile to native resolution.
        for i in range(nat_xdim*oversample):
            new_ax = np.arange(nat_ydim)
            oversamp_ax = np.linspace(0, nat_ydim, nat_ydim*oversample,
                                      endpoint=False)
            oversamp_prof = data_sub[:, i]
            data_nat1[:, i] = np.interp(new_ax, oversamp_ax, oversamp_prof)
        # Same for the spectral direction.
        for i in range(nat_ydim):
            new_ax = np.arange(nat_xdim)
            oversamp_ax = np.linspace(0, nat_xdim, nat_xdim*oversample,
                                      endpoint=False)
            oversamp_prof = data_nat1[i, :]
            data_nat[i, :] = np.interp(new_ax, oversamp_ax, oversamp_prof)
    else:
        data_nat = data_sub

    return data_nat


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
    Currently, very broad bounds. Can likely be refined once informed by actual
    observations.
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


def simple_solver(clear, verbose=False, save_to_file=True):
    '''Algorithm to calculate and preform the necessary rotation and offsets to
    transform the reference traces and wavelength maps to match the science
    data.
    The steps are as follows:
        1. Determine the correct subarray for the data.
        2. Get the first order centroids for the refrence trace.
        3. Determine the first order centroids for the data.
        4. Fit the reference centroids to the data to determine the correct
           rotation angle and offset.
        5. Apply this transformation to the reference traces and wavelength
           maps for the first and second order.
        6. Save transformed reference files to disk.

    Parameters
    ----------
    clear : np.ndarray
        CLEAR science exposure.
    verbose : bool
        Whether to do diagnostic prints and plots.
    save_to_file : bool
        If True, write the transformed wavelength map and 2D trace profiles to
        disk in two multi-extension fits files.

    Returns
    -------
    ref_trace_trans : np.ndarray
        2xYx2048 array containing the reference trace profiles transformed
        to match the science data.
    wave_map_trans : np.ndarray
        2xYx2048 array containing the reference trace profiles transformed
        to match the science data.

    Raises
    ------
    ValueError
        If shape of clear input does not match known subarrays.
    '''

    # Open 2D trace profile reference file.
    ref_trace_file = soss_read_refs.Ref2dProfile(path+'SOSS_ref_2D_profile.fits')
    # Open trace table reference file.
    ttab_file = soss_read_refs.RefTraceTable(path+'SOSS_ref_trace_table.fits')
    # Get first order centroids (in DMS coords).
    wavemap_file = soss_read_refs.Ref2dWave(path+'SOSS_ref_2D_wave.fits')

    # Determine correct subarray dimensions and offsets.
    dimy, dimx = np.shape(clear)
    if dimy == 96:
        subarray = 'SUBSTRIP96'
    elif dimy == 256:
        subarray = 'SUBSTRIP256'
    elif dimy == 2048:
        subarray = 'FULL'
    else:
        raise ValueError('Unrecognized subarray shape: {}x{}.'.format(dimy, dimx))

    # Get first order centroids on subarray.
    xcen_ref = ttab_file('X', subarray=subarray)[1]
    # Extend centroids beyond edges of the subarray for more accurate fitting.
    inds = np.where((xcen_ref >= -50) & (xcen_ref < 2098))
    xcen_ref = xcen_ref[inds]
    ycen_ref = ttab_file('Y', subarray=subarray)[1][inds]

    # Get centroids from data.
    xcen_dat, ycen_dat, p = ctd.get_uncontam_centroids(clear, verbose=verbose)

    # Fit the reference file centroids to the data.
    fit = _do_emcee(xcen_ref, ycen_ref, xcen_dat, ycen_dat,
                    subarray=subarray, showprogress=verbose)
    flat_samples = fit.get_chain(discard=500, thin=15, flat=True)
    # Get best fitting rotation angle and pixel offsets.
    rot_ang = np.percentile(flat_samples[:, 0], 50)
    x_shift = np.percentile(flat_samples[:, 1], 50)
    y_shift = np.percentile(flat_samples[:, 2], 50)
    # Plot posteriors if necessary.
    if verbose is True:
        plotting._plot_corner(fit)

    # Transform reference files to match data.
    ref_trace_trans = np.ones((2, dimy, dimx))
    wave_map_trans = np.ones((2, dimy, dimx))
    for order in [1, 2]:
        # Load the reference trace and wavelength map for the current order and
        # correct subarray, as well as padding and oversampling information.
        ref_trace, os_t, pad_t = ref_trace_file(order=order, subarray=subarray,
                                                native=False, only_prof=False)
        ref_wavemap, os_w, pad_w = wavemap_file(order=order, subarray=subarray,
                                                native=False, only_prof=False)
        # Set NaN pixels to zero - the rotation doesn't handle NaNs well.
        ref_trace[np.isnan(ref_trace)] = 0
        ref_wavemap[np.isnan(ref_wavemap)] = 0

        # Do the transformation for the reference 2D trace.
        # Pass negative rot_ang to convert from CCW to CW rotation
        trace_trans = _do_transform(ref_trace, -rot_ang, x_shift, y_shift,
                                    pad=pad_t, oversample=os_t,
                                    verbose=verbose)
        # Renormalize the spatial profile so columns sum to one.
        ref_trace_trans[order-1] = trace_trans / np.nansum(trace_trans, axis=0)
        # Transform the wavelength map.
        wave_map_trans[order-1, :, :] = _do_transform(ref_wavemap, -rot_ang,
                                                      x_shift, y_shift,
                                                      pad=pad_w,
                                                      oversample=os_w,
                                                      verbose=verbose)
    # Write files to disk if requested.
    if save_to_file is True:
        write_to_file(ref_trace_trans, filename='SOSS_ref_2D_profile_simplysolved')
        write_to_file(wave_map_trans, filename='SOSS_ref_2D_wave_simplysolved')

    return ref_trace_trans, wave_map_trans


def write_to_file(stack, filename):
    '''Utility function to write transformed 2D trace profile or wavelength map
    files to disk. Data will be saved as a multi-extension fits file.

    Parameters
    ----------
    stack : np.ndarray (2xYx2048)
        Array containing transformed 2D trace profile or wavelength map data.
        The first dimension must be the spectral order, the second dimension
        the spatial dimension, and the third the spectral dimension.
    filename : str
        Name of the file to which to write the data.
    '''

    hdu_p = fits.PrimaryHDU()
    hdulist = [hdu_p]
    for order in [1, 2]:
        hdu_o = fits.ImageHDU(data=stack[order-1])
        hdu_o.header['ORDER'] = order
        hdu_o.header.comments['ORDER'] = 'Spectral order.'
        hdulist.append(hdu_o)

    hdu = fits.HDUList(hdulist)
    hdu.writeto('{}.fits'.format(filename), overwrite=True)
