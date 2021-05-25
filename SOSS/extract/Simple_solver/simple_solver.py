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
    if bound:

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
    """"Compute the chi-squared statistic for fitting the reference positions
    to the true positions.

    :param transform: The transformation parameters.
    :param xmod: The reference x-positions.
    :param ymod: The reference y-positions.
    :param xdat: The data x-positions.
    :param ydat: The data y-positions.
    :param subarray: The subarray.  TODO do we need this?

    :type transform: Tuple, List, Array
    :type xmod: array[float]
    :type ymod: array[float]
    :type xdat: array[float]
    :type ydat: array[float]
    :type subarray: str

    :returns: chisq - The chi-squared value of the model fit.
    :rtype: float
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


def rotate_image(image, angle, origin):
    """Rotate an image around a specific pixel.

    :param image: The image to rotate.
    :param angle: The rotation angle in degrees.
    :param origin: The x, y pixel position around which to rotate.

    :type image: array[float]
    :type angle: float
    :type origin: Tuple, List, Array

    :returns: image_rot - The rotated image.
    :rtype: array[float]
    """

    # Pad image so we can safely rotate around the origin.
    padx = [image.shape[1] - origin[0], origin[0]]
    pady = [image.shape[0] - origin[1], origin[1]]
    image_pad = np.pad(image, [pady, padx], 'constant')

    # Rotate the image.
    image_pad_rot = rotate(image_pad, angle, reshape=False)

    # Remove the padding.
    image_rot = image_pad_rot[pady[0]:-pady[1], padx[0]:-padx[1]]

    return image_rot


def _do_transform(ref_map, angle, xshift, yshift, oversample, pad):
    """Apply the rotation and offset to a 2D reference map, and bin
    the map down the native size and resolution.

    :param ref_map: A 2D reference file array such as a wavelength map or trace
        profile.
    :param angle: The angle by which to rotate the file, in degrees.
    :param xshift: The x-shift to apply in native pixels, will be rounded to the
        nearest (oversampled) pixel.
    :param yshift: The y-shift to apply in native pixels, will be rounded to the
        nearest (oversampled) pixel.
    :param oversample: The oversampling of the reference file array.
    :param pad: The padding, in native pixels, on the reference file array.

    :type ref_map: array[float]
    :type angle: float
    :type xshift: float
    :type yshift: float
    :type oversample: int
    :type pad: int

    :returns: ref_map_nat - The data, after applying the shift and rotation and
        binned down to the native size and resolution.
    :rtype: array[float]
    """

    ovs = oversample

    if (xshift > pad) | (yshift > pad):
        msg = 'The applied shift exceeds the padding.'  # TODO better message.
        raise ValueError(msg)

    # Determine the native shape of the data.
    pad_ydim, pad_xdim = np.shape(ref_map)
    nat_xdim = int(round(pad_xdim/ovs - 2*pad, 0))
    nat_ydim = int(round(pad_ydim/ovs - 2*pad, 0))

    # Rotation anchor is o1 trace centroid halfway along the spectral axis.
    x_anch = int((pad + 1024)*ovs)  # TODO hardcoded here, optional in rot_centroids.
    y_anch = int((pad + 50)*ovs)

    # Rotate the reference map.
    ref_map_rot = rotate_image(ref_map, angle, [x_anch, y_anch])

    # Select the relevant area after shifting.
    minrow = ovs*pad + int(ovs*yshift)
    maxrow = minrow + ovs*nat_ydim
    mincol = ovs*pad + int(ovs*xshift)
    maxcol = mincol + ovs*nat_xdim
    ref_map_sub = ref_map_rot[minrow:maxrow, mincol:maxcol]

    # Bin down to native resolution.
    ref_map_nat = ref_map_sub.reshape(nat_ydim, ovs, nat_xdim, ovs)
    ref_map_nat = ref_map_nat.mean(-1).mean(1)  # TODO needs to be different for wavelength and profile?

    return ref_map_nat


def apply_transform(simple_transform, ref_maps, oversample, pad, norm=False):
    """Apply the transformation found by solve_transform() to a 2D reference map.

    :param simple_transform: The transformation parameters returned by
        solve_transform().
    :param ref_maps: Array of reference maps.
    :param oversample: The oversampling factor the reference maps.
    :param pad: The padding (in native pixels) on the reference maps.
    :param norm: If True normalize columns to 1, used for trace profile
        reference maps.

    :type simple_transform: Tuple, List, Array
    :type ref_maps: array[float]
    :type oversample: int
    :type pad: int
    :type norm: bool

    :returns: trans_maps - the ref_maps after having the transformation applied.
    :rtype: array[float]
    """

    # Unpack the transformation.
    angle, xshift, yshift = simple_transform

    # Get the dimensions of the reference map.
    norders, dimy, dimx = ref_maps.shape

    trans_maps = []
    for i_ord in range(norders):

        # Set NaN pixels to zero - the rotation doesn't handle NaNs well.
        ref_maps[np.isnan(ref_maps)] = 0

        # Do the transformation for the reference 2D trace.
        # Pass negative rot_ang to convert from CCW to CW rotation
        trans_map = _do_transform(ref_maps[i_ord], -angle, xshift, yshift,
                                  pad=pad, oversample=oversample)

        # Renormalize the spatial profile so columns sum to one.
        if norm:
            trans_maps.append(trans_map/np.nansum(trans_map, axis=0))  # TODO handle this with a switch?
        else:
            trans_maps.append(trans_map)

    return np.array(trans_maps)


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
