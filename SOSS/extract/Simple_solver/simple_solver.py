#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage.interpolation import rotate

from astropy.io import fits

from . import plotting
from SOSS.dms import soss_centroids as ctd  # TODO remove shorthand?

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def rot_centroids(angle, xshift, yshift, xpix, ypix, bound=True, xgrid=None,
                  cenx=1024, ceny=50, subarray='SUBSTRIP256'):
    """Apply a rotation and shift to the trace centroids positions. This
    assumes that the trace centroids are already in the CV3 coordinate system.

    Parameters
    ----------
    angle : float
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
    xgrid : list of float
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
    """

    # Convert to numpy arrays
    xpix = np.atleast_1d(xpix)
    ypix = np.atleast_1d(ypix)

    # Required rotation in the detector frame to match the data.
    t = np.deg2rad(angle)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    # Rotation center set to o1 trace centroid halfway along spectral axis.
    points1 = np.array([xpix - cenx, ypix - ceny])
    rot_pix = R @ points1
    rot_pix[0] += cenx
    rot_pix[1] += ceny

    # Apply the offsets
    rot_pix[0] += xshift
    rot_pix[1] += yshift

    if xgrid is None:

        # Ensure that there are no jumps of >1 pixel.
        minval = int(round(np.amin(rot_pix[0]), 0))
        maxval = int(round(np.amax(rot_pix[0]), 0))

        # Same range as rotated pixels but with step of 1 pixel.
        xgrid = np.linspace(minval, maxval, maxval-minval+1)

    # Polynomial fit to ensure a centroid at each pixel in xgrid. # TODO do we need this for the use case? Might be safer?
    pp = np.polyfit(rot_pix[0], rot_pix[1], 5)

    # Warn user if xgrid extends beyond polynomial domain.
    if np.amax(xgrid) > np.amax(rot_pix[0]) + 25 or np.amin(xgrid) < np.amin(rot_pix[0])-25:
        warnmsg = 'xgrid extends beyond rot_xpix. Use with caution.'
        warnings.warn(warnmsg)

    rot_xpix = xgrid
    rot_ypix = np.polyval(pp, rot_xpix)

    # Check to ensure all points are on the subarray. TODO for the use case we have here this isn't needed?
    if bound is True:

        # Get dimensions of the subarray
        if subarray == 'SUBSTRIP96':
            yend = 96
        elif subarray == 'SUBSTRIP256':
            yend = 256
        elif subarray == 'FULL':
            yend = 2048
        else:
            errmsg = 'Unknown subarray. Allowed identifiers are "SUBSTRIP96",\
             "SUBSTRIP256", or "FULL".'
            raise ValueError(errmsg)

        # Reject pixels which are not on the subarray.
        inds = [(rot_ypix >= 0) & (rot_ypix < yend) & (rot_xpix >= 0) &
                (rot_xpix < 2048)]
        rot_xpix = rot_xpix[inds]
        rot_ypix = rot_ypix[inds]

    return rot_xpix, rot_ypix


def _chi_squared(transform, xmod, ymod, xdat, ydat, subarray):
    """"Definition of a modified Chi squared statistic to fit refrence centroid
    to those extracted from the data.
    """

    angle, xshift, yshift = transform

    # Calculate rotated model.
    modelx, modely = rot_centroids(angle, xshift, yshift, xmod, ymod, bound=True,
                                   subarray=subarray)

    # Interpolate rotated model onto same x scale as data.
    modely = np.interp(xdat, modelx, modely)
    chisq = np.nansum((ydat - modely)**2)

    return chisq


def solve_transform(scidata, scimask, xcen_ref, ycen_ref, subarray,
                    verbose=False):
    """Given a science image, determine the centroids and find the simple
    transformation needed to match xcen_ref and ycen_ref to the image.

    :param scidata: the image of the SOSS trace.
    :param scimask: a boolean mask of pixls to be excluded.
    :param xcen_ref: a priori expectation of the trace x-positions.
    :param ycen_ref: a priori expectation of the trace y-positions.
    :param subarray: the subarray of the observations.
    :param verbose: If set True provide diagnostic information.

    :type scidata: array[float]
    :type scimask: array[bool]
    :type xcen_ref: array[float]
    :type ycen_ref: array[float]
    :type subarray: str
    :type verbose: bool

    :returns: simple_transform - Array containing the angle, x-shift and y-shift
        needed to match xcen_ref and ycen_ref to the image.
    :rtype: array[float]
    """

    # Extend centroids beyond edges of the subarray for more accurate fitting. TODO Actually removes points, no need to do this?
    mask = (xcen_ref >= -50) & (xcen_ref < 2098)
    xcen_ref = xcen_ref[mask]
    ycen_ref = ycen_ref[mask]

    # Get centroids from data.
    centroids = ctd.get_soss_centroids(scidata, mask=scimask,
                                       subarray=subarray, verbose=verbose)

    xcen_dat = centroids['order 1']['X centroid']
    ycen_dat = centroids['order 1']['Y centroid']

    # Fit the reference file centroids to the data.
    guess_transform = np.array([0.15, 1, 1])
    lik_args = (xcen_ref, ycen_ref, xcen_dat, ycen_dat, subarray)
    result = minimize(_chi_squared, guess_transform, args=lik_args)
    simple_transform = result.x

    return simple_transform


def _do_transform(data, rot_ang, x_shift, y_shift, pad=0, oversample=1,
                  verbose=False):
    """Do the rotation (via a rotation matrix) and offset of the reference
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
    verbose : int
        Either 0, 1, or 2. If 2, show all progress prints and diagnostic plots.
        If 1, only show progress prints. If 0, show nothing.

    Returns
    -------
    data_sub256_nat : np.ndarray
        Reference file with all transformations applied and interpolated
        to the native detector resolution.
    """

    x_shift, y_shift = int(round(x_shift, 0)), int(round(y_shift, 0))

    # Determine x and y center of the padded dataframe.
    pad_ydim, pad_xdim = np.shape(data)
    nat_xdim = int(round(pad_xdim / oversample - 2*pad, 0))
    nat_ydim = int(round(pad_ydim / oversample - 2*pad, 0))
    pad_xcen = pad_xdim // 2
    pad_ycen = pad_ydim // 2

    # Rotation anchor is o1 ~trace centroid halfway along the spectral axis.
    x_anch = int((1024 + pad)*oversample)  # TODO hardcoded here, optional in rot_centroids.
    y_anch = int((50 + pad)*oversample)

    # TODO this roll rotate roll construction assumes more padding than we have
    # TODO solve by adding padding (stackoverflow) or use affine_transform()?
    # Shift dataframe such that rotation anchor is in the center of the frame.
    data_shift = np.roll(data, (pad_ycen - y_anch, pad_xcen - x_anch), (0, 1))

    # Rotate the shifted dataframe by the required amount.
    data_rot = rotate(data_shift, rot_ang, reshape=False)

    # Shift the rotated data back to its original position.
    data_shiftback = np.roll(data_rot, (-pad_ycen + y_anch, -pad_xcen + x_anch),
                             (0, 1))

    # Apply vertical and horizontal offsets.
    data_offset = np.roll(data_shiftback, (y_shift*oversample,
                          x_shift*oversample), (0, 1))

    if verbose:
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
        # Can likely be done in a more vectorized way.
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


def apply_transform(simple_transform, ref_maps, oversample, pad, verbose=False):
    """Apply the transformation found by solve_transform() to a 2D reference map.

    :param simple_transform: The transformation parameters returned by
        solve_transform().
    :param ref_maps: Array of reference maps.
    :param oversample: The oversampling factor the reference maps.
    :param pad: The padding (in native pixels) on the reference maps.
    :param verbose: If set True provide diagnostic information.

    :type simple_transform: Tuple, List, Array
    :type ref_maps: array[float]
    :type oversample:
    :type pad:
    :type verbose: bool

    :returns: trans_maps - the ref_maps after having the transformation applied.
    :rtype: array[float]
    """

    # Unpack the transformation.
    rot_ang, x_shift, y_shift = simple_transform

    # Get the dimensions of the reference map.
    norders, dimy, dimx = ref_maps.shape

    trans_maps = np.ones_like(ref_maps)
    for i_ord in range(norders):

        # Set NaN pixels to zero - the rotation doesn't handle NaNs well.
        ref_maps[np.isnan(ref_maps)] = 0

        # Do the transformation for the reference 2D trace.
        # Pass negative rot_ang to convert from CCW to CW rotation
        trans_map = _do_transform(ref_maps[i_ord], -rot_ang, x_shift, y_shift,
                                  pad=pad, oversample=oversample,
                                  verbose=verbose)

        # Renormalize the spatial profile so columns sum to one.
        trans_maps[i_ord] = trans_map/np.nansum(trans_map, axis=0)  # TODO handle this with a switch?

    return trans_maps


def write_to_file(stack, filename):
    """Utility function to write transformed 2D trace profile or wavelength map
    files to disk. Data will be saved as a multi-extension fits file.

    Parameters
    ----------
    stack : np.ndarray (2xYx2048)
        Array containing transformed 2D trace profile or wavelength map data.
        The first dimension must be the spectral order, the second dimension
        the spatial dimension, and the third the spectral dimension.
    filename : str
        Name of the file to which to write the data.
    """

    hdu_p = fits.PrimaryHDU()
    hdulist = [hdu_p]
    for order in [1, 2]:
        hdu_o = fits.ImageHDU(data=stack[order-1])
        hdu_o.header['ORDER'] = order
        hdu_o.header.comments['ORDER'] = 'Spectral order.'
        hdulist.append(hdu_o)

    hdu = fits.HDUList(hdulist)
    hdu.writeto('{}.fits'.format(filename), overwrite=True)

    return


def main():
    """Placeholder for potential multiprocessing."""

    return


if __name__ == '__main__':
    main()
