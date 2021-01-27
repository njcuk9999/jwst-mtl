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
from scipy.ndimage.interpolation import rotate
from SOSS.extract.simple_solver import centroid as ctd
from SOSS.extract.simple_solver import plotting as plotting

warnings.simplefilter(action='ignore', category=RuntimeWarning)

# local path to reference files.
path = '/Users/michaelradica/Documents/School/Ph.D./Research/SOSS/Extraction/Input_Files/'


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
        plotting.plot_transformation_steps(data_shift, data_rot,
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
        2x256x2048 array containing the reference trace profiles transformed
        to match the science data.
    wave_map_trans : np.ndarray
        2x256x2048 array containing the reference trace profiles transformed
        to match the science data.
    '''

    # Open 2D trace profile reference file.
    ref_trace_file = fits.open(path+'SOSS_ref_2D_profile.fits')
    # Get the first order profile (in DMS coords).
    ref_trace_o1 = ref_trace_file[1].data
    # Open trace table reference file.
    ttab_file = fits.open(path+'SOSS_ref_trace_table.fits')
    # Get first order centroids (in DMS coords).
    centroids_o1 = ttab_file[1].data
    wavemap_file = fits.open(path+'SOSS_ref_2D_wave.fits')

    # Determine correct subarray dimensions and offsets.
    dimy, dimx = np.shape(clear)
    if dimy == 96:
        inds = re.split('\[|:|,|\]', ref_trace_file[1].header['INDEX96'])
        ystart = int(inds[1])
        yend = int(inds[2])
        xstart = int(inds[3])
        xend = int(inds[4])
        dy = ttab_file[1].header['DYSUB96']
        subarray = 'substrip96'
    elif dimy == 256:
        inds = re.split('\[|:|,|\]', ref_trace_file[1].header['INDEX256'])
        ystart = int(inds[1])
        yend = int(inds[2])
        xstart = int(inds[3])
        xend = int(inds[4])
        dy = ttab_file[1].header['DYSUB256']
        subarray = 'substrip256'
    else:
        ystart = 0
        yend = None
        xstart = 0
        xend = None
        dy = 0
        subarray = 'full'

    # Get first order centroids on subarray.
    xcen_ref = centroids_o1['X']
    # Extend centroids beyond edges of the subarray for more accurate fitting.
    inds = np.where((xcen_ref >= -50) & (xcen_ref < 2098))
    xcen_ref = xcen_ref[inds]
    ycen_ref = centroids_o1['Y'][inds]+dy

    # Get centroids from data.
    xcen_dat, ycen_dat = ctd.get_uncontam_centroids(clear)

    # Fit the reference file centroids to the data.
    fit = ctd._do_emcee(xcen_ref, ycen_ref, xcen_dat, ycen_dat,
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
        # Load the reference trace and wavelength map for the current order.
        ref_trace = ref_trace_file[order].data
        ref_wavemap = wavemap_file[order].data
        # Get padding and oversampling information.
        os_t = ref_trace_file[order].header['OVERSAMP']
        pad_t = ref_trace_file[order].header['PADDING']
        os_w = wavemap_file[order].header['OVERSAMP']
        pad_w = wavemap_file[order].header['PADDING']
        # Slice the correct subarray.
        ref_trace = ref_trace[ystart:yend, xstart:xend]
        ref_wavemap = ref_wavemap[ystart:yend, xstart:xend]
        # Set any NaN pixels to zero.
        ref_trace[np.isnan(ref_trace)] = 0
        ref_wavemap[np.isnan(ref_wavemap)] = 0

        # Do the transformation.
        # Pass negative rot_ang to convert from CCW to CW rotation
        ref_trace_trans[order-1, :, :] = _do_transform(ref_trace, -rot_ang,
                                                       x_shift, y_shift,
                                                       pad=pad_t,
                                                       oversample=os_t,
                                                       verbose=verbose)
        wave_map_trans[order-1, :, :] = _do_transform(ref_wavemap, -rot_ang,
                                                      x_shift, y_shift,
                                                      pad=pad_w,
                                                      oversample=os_w,
                                                      verbose=verbose)
    # Write files to disk if necessary.
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
