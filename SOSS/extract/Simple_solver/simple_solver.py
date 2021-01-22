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
from scipy.ndimage.interpolation import rotate
from SOSS.extract.empirical_trace import centroid as ctd

# hack to get around the fact that relative paths are constantly messing up atm
path = '/Users/michaelradica/Documents/GitHub/jwst-mtl/SOSS/'


def _do_transform(data, rot_ang, x_shift, y_shift, padding_factor=2,
                  oversampling=1):
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

    # Rotation anchor is o1 trace centroid halfway along the spectral axis.
    x_anch = 1024
    y_anch = 206

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


def simple_solver(clear):
    '''First implementation of the simple_solver algorithm to calculate
    and preform the necessary rotation and offsets to transform the
    reference traces and wavelength maps to match the science data.
    The steps are as follows:
        1. Locate the reference trace centroids.
        2. Open the data files and locate the centroids.
        3. Call get_contam_centroids to determine the correct rotation and
           offset parameters relative to the reference trace files.
        4. Open the reference files and determine the appropriate padding
           and oversampling factors.
        5. Call _do_transform to transform the reference files to match the
           data.
        6. Return the transformed reference traces and wavelength maps.

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

    # Get reference trace centroids for the first order.
    # May need to be updated when reference traces are oversampled and padded
    ref_trace = fits.open(path+'/extract/Ref_files/trace_profile_om1.fits')[0].data[::-1, :]
    ref_x, ref_y = ctd.get_uncontam_centroids(ref_trace)
    # To ensure accurate fitting, extend centroids beyond detector edges.
    pp = np.polyfit(ref_x, ref_y, 5)
    ref_x_ext = np.arange(2148)-50
    ref_y_ext = np.polyval(pp, ref_x_ext)
    ref_centroids = np.array([ref_x_ext, ref_y_ext])

    # Get data centroids and rotation params relative to the reference trace.
    trans_centroids, rot_pars = ctd.get_contam_centroids(clear,
                                                         ref_centroids=ref_centroids,
                                                         return_orders=[1])
    rot_ang = rot_pars[0]
    x_shift = np.int(rot_pars[1])
    y_shift = np.int(rot_pars[2])

    ref_trace_trans = np.empty((2, 256, 2048))
    wave_map_trans = np.empty((2, 256, 2048))

    for order in [1, 2]:
        # Load the reference trace and wavelength map for the current order
        ref_trace = fits.open(path+'/extract/Ref_files/trace_profile_om%s.fits' % order)[0].data[::-1, :]
        # Clean up the reference trace - no zero pixels and add padding
        ref_trace[np.where(ref_trace == 0)] = 1
        ref_trace = np.pad(ref_trace, ((128, 128), (1024, 1024)),
                           constant_values=((1, 1), (1, 1)))
        pad_t = 2
        os_t = 1
        wave_map = fits.open(path+'/extract/Ref_files/wavelengths_m%s.fits' % order)[0].data[::-1, :]
        wave_map = np.pad(wave_map, ((128, 128), (1024, 1024)),
                          constant_values=((1, 1), (1, 1)))
        pad_w = 2
        os_w = 1

        # Pass negative rot_ang to convert from CCW to CW rotation
        ref_trace_trans[order-1, :, :] = _do_transform(ref_trace, -rot_ang,
                                                       x_shift, y_shift,
                                                       padding_factor=pad_t,
                                                       oversampling=os_t)
        wave_map_trans[order-1, :, :] = _do_transform(wave_map, -rot_ang,
                                                      x_shift, y_shift,
                                                      padding_factor=pad_w,
                                                      oversampling=os_w)

    return ref_trace_trans, wave_map_trans
