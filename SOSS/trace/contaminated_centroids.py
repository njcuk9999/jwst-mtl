#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits

from SOSS.extract import soss_read_refs
from SOSS.dms import soss_centroids as cen

import matplotlib.pyplot as plt


def wavelength_calibration(xpos, order=1, subarray='SUBSTRIP256'):
    """Find the wavelengths corresponding to a set of x-positions using the
    trace table reference file.

    :param xpos: the array of x-positions to calibrate.
    :param order: the trace order the x-positions correspond to.
    :param subarray: the subarray the x-positions correspond to.

    :type xpos: array[float]
    :type order: int
    :type subarray: str

    :returns: wavelengths - an array of wavelengths corresponding to xpos.
    :rtype: array[float]
    """

    # Read the wavelength vs x-position relation from the reference file.
    ref = soss_read_refs.RefTraceTable()
    ref_wavelengths, ref_xpos = ref('X', subarray=subarray, order=order)

    # Sort so the reference positions are in ascending order.
    args = np.argsort(ref_xpos)
    ref_xpos, ref_wavelengths = ref_xpos[args], ref_wavelengths[args]

    # Find the wavelengths corresponding to the input array by interpolating.
    wavelengths = np.interp(xpos, ref_xpos, ref_wavelengths)

    return wavelengths


def build_mask_256(subarray='SUBSTRIP256', apex_order1=None):
    """Restrict the analysis to a (N, 2048) section of the image, where N is 256 or less.
    Normally this only applies to the FULL subarray, masking everything but the SUBSTRIP256 region.
    When apex_order1 is given rows from apex_order1 - 40 to apex_order1 + 216 are kept instead.

    :param subarray: The subarray value corresponding to the image.
    :param apex_order1: The y-position of the order1 apex at 1.3 microns, in the given subarray.

    :type subarray: str
    :type apex_order1: float

    :returns: mask_256 - A mask that removes any area not related to the trace of the target.
    :rtype: array[bool]
    """

    dimx = 2048

    # Check the subarray value and set dimy accordingly.
    if subarray == 'FULL':
        dimy = 2048
    elif subarray == 'SUBSTRIP96':
        dimy = 96
    elif subarray == 'SUBSTRIP256':
        dimy = 256
    else:
        msg = 'Unknown subarray: {}'
        raise ValueError(msg.format(subarray))

    # Round the apex value to the nearest integer.
    apex_order1 = int(apex_order1)

    # Prepare the mask array.
    mask_256 = np.ones((dimy, dimx), dtype='bool')

    if apex_order1 is None:

        if subarray == 'FULL':
            # If subarray is FULL keep only the SUBSTRIP256 region.
            mask_256[1792:, :] = False
        else:
            # For all other subarrays keep the entire subarray.
            pass

    else:

        # Keep only the 256 region around the apex_order1 value.
        # In SUBSTRIP256 the apex would be at y ~ 40.
        rowmin = np.maximum(apex_order1 - 40, 0)
        rowmax = np.minimum(apex_order1 + 216, dimy)
        mask_256[rowmin:rowmax, :] = False

    return mask_256


def build_mask_trace(ytrace, subarray='SUBSTRIP256', halfwidth=30,
                     extend_below=False, extend_above=False):
    """Mask out the trace in a given subarray based on the y-positions provided.
    A band of pixels around the trace position of width = 2*halfwidth will be masked.
    Optionally extend_above and extend_below can be used to mask all pixels above
    or below the trace.

    :param ytrace: the trace y-position at each column, must have shape = (2048,).
    :param subarray: the subarray corresponding the the provided positions.
    :param halfwidth: the size of the window to mask around the trace.
    :param extend_below: if True mask all pixels above the trace.
    :param extend_above: if True mask all pixels below the trace.

    :type ytrace: array[float]
    :type subarray: str
    :type halfwidth: float
    :type extend_below: bool
    :type extend_above: bool

    :returns: mask_trace - A mask that removes an area centered on the given trace positions.
    :rtype: array[bool]
    """

    dimx = 2048

    # Check the shape of the y-positions.
    if np.shape(ytrace) != (dimx,):
        msg = 'ytrace must have shape (2048,)'
        raise ValueError(msg)

    # Check the subarray value and set dimy accordingly.
    if subarray == 'FULL':
        dimy = 2048
    elif subarray == 'SUBSTRIP96':
        dimy = 96
    elif subarray == 'SUBSTRIP256':
        dimy = 256
    else:
        msg = 'Unknown subarray: {}'
        raise ValueError(msg.format(subarray))

    # Cannot both be True, that would mask everything.
    if extend_below and extend_above:
        msg = 'Only one of extend_below, extend_above should be used.'
        raise ValueError(msg)

    x = np.arange(dimx)
    y = np.arange(dimy)
    _, ygrid = np.meshgrid(x, y)

    # Mask the pixels within a halfwidth of the trace center.
    mask_trace = np.abs(ygrid - ytrace) < halfwidth

    # If True mask all pixels below the trace center.
    if extend_below:
        mask_below = (ygrid - ytrace) < 0
        mask_trace = mask_trace | mask_below

    # If True mask all pixels above the trace center.
    if extend_above:
        mask_above = (ygrid - ytrace) > 0
        mask_trace = mask_trace | mask_above

    return mask_trace


def build_mask_vertical(subarray='SUBSTRIP256', masked_side='blue',
                        cut_x=1700, mask_between=True, mask_outside=False,
                        verbose=False):
    """Builds a mask where there are two sides: left and right, one being
    masked, the other not. In other words, this masks a region along the
    spectral dispersion axis.

    :param subarray:
    :param masked_side:
    :param cut_x:
    :param mask_between:
    :param mask_outside:
    :param verbose:
    """

    masked_value = True

    if verbose is True:
        print('Going through build_mask_vertical.')

    dimy, dimx = 256, 2048
    if subarray == 'SUBSTRIP96':
        dimy = 96
    if subarray == 'FULL':
        dimy = 2048

    # Initialize a mask
    mask = np.zeros_like(np.zeros((dimy, dimx)), dtype='bool')
    if np.size(cut_x) == 2:
        if mask_between is True:
            mask[:, cut_x[0]:cut_x[1]] = masked_value
        if mask_outside is True:
            mask[:, 0:cut_x[0]] = masked_value
            mask[:, cut_x[1]:] = masked_value
    else:
        if masked_side == 'blue':
            mask[:, cut_x:] = masked_value
        if masked_side == 'red':
            mask[:, 0:cut_x] = masked_value

    return mask


def build_mask_sloped(subarray='SUBSTRIP256', masked_side='blue', pt1=None,
                      pt2=None, verbose=False):
    """Draw a sloped line and mask on one side of it (the side is defined with
    respect to the spectral dispersion axis. Requires the x,y position of two
    points that define the line. The x,y must be given in native size pixels.
    Along the x axis: 0-2047, along the y-axis, it depends on the array size.
    For SUBSTRIP256, y=0-255, for FF, y=0-2047

    :param subarray:
    :param masked_side:
    :param pt1:
    :param pt2:
    :param verbose:
    """

    if pt1 is None:
        pt1 = [0, 0]

    if pt2 is None:
        pt2 = [2048, 0]

    masked_value = True

    dimy, dimx = 256, 2048
    if subarray == 'SUBSTRIP96':
        dimy = 96
    if subarray == 'FULL':
        dimy = 2048

    # Simplify one's life and simply fit the two points
    thex = np.array([pt1[0], pt2[0]])
    they = np.array([pt1[1], pt2[1]])
    param = np.polyfit(thex, they, 1)

    if verbose is True:
        print('line fit param:', param)

    # Initialize a mask
    mask = np.zeros_like(np.zeros((dimy, dimx)), dtype='bool')

    # Compute the position of the line at every x position
    fitx = np.arange(dimx)
    fity = np.polyval(param, fitx)  # round it

    # Make sure negative values in fity get floored to zero, to be able
    # to index in array (below) without wrapping.
    fity[fity < 0] = 0

    # Branch depending on side that needs masking and sign of the slope
    if masked_side == 'blue':
        if param[0] < 0:
            for i in range(dimx):
                mask[int(fity[i]):, i] = masked_value
        else:
            for i in range(dimx):
                mask[0:int(fity[i]), i] = masked_value

    if masked_side == 'red':
        if param[0] < 0:
            for i in range(dimx):
                mask[0:int(fity[i]), i] = masked_value
        else:
            for i in range(dimx):
                mask[int(fity[i]):, i] = masked_value

    return mask


def build_mask_order2_contaminated(x_o1, y_o1, x_o3, y_o3,
                                   subarray='SUBSTRIP256', halfwidth_o1=25,
                                   halfwidth_o3=15, cut_x=150, verbose=False):
    """Function that creates a mask to isolate the uncontaminated part of the
    second order. It masks out all pixels redder than red_cut_x, all pixels
    bluer than blue_cut_x as well as pixels below of the sloped line defined by
    pt1 and pt2.

    :param x_o1: x position of the order 1 centroid
    :param y_o1: y position of the order 1 centroid
    :param x_o3: x position of the order 3 centroid
    :param y_o3: y position of the order 3 centroid
    :param subarray: 'FULL', 'SUBSTRIP96' or 'SUBSTRIP256'
    :param halfwidth_o1:
    :param halfwidth_o3:
    :param cut_x:
    :param verbose:
    """

    # First, the order 1 trace needs to be masked out. Construct a mask
    # that not only covers the order 1 trace but everything below the trace.
    mask_aper_o1 = build_mask_trace(y_o1, subarray=subarray,
                                    halfwidth=halfwidth_o1,
                                    extend_below=True)

    # Do the same to mask out order 3 - this one is fainter so make a
    # narrower mask. Also mask all pixels above the trace.
    mask_aper_o3 = build_mask_trace(y_o3, subarray=subarray,
                                    halfwidth=halfwidth_o3,
                                    extend_above=True)

    # Mask everything to the right of the second order trace where the
    # transmission dip makes the trace disappear.
    mask_blue = build_mask_vertical(subarray=subarray, masked_side='blue',
                                    cut_x=cut_x)

    # Combine masks
    mask_o2_cont = mask_aper_o1 | mask_aper_o3 | mask_blue

    return mask_o2_cont


def build_mask_order2_uncontaminated(x_o1, y_o1, x_o3, y_o3,
                                     subarray='SUBSTRIP256', halfwidth_o1=25,
                                     halfwidth_o3=15, red_cut_x=700,
                                     blue_cut_x=1800, pt1=None,
                                     pt2=None, verbose=False):
    """Function that creates a mask to isolate the uncontaminated part of the
    second order. It masks out all pixels redder than red_cut_x, all pixels
    bluer than blue_cut_x as well as pixels below of the sloped line defined by
    pt1 and pt2.

    :param x_o1: x position of the order 1 centroid
    :param y_o1: y position of the order 1 centroid
    :param x_o3: x position of the order 3 centroid
    :param y_o3: y position of the order 3 centroid
    :param subarray: 'FULL', 'SUBSTRIP96' or 'SUBSTRIP256'
    :param halfwidth_o1:
    :param halfwidth_o3:
    :param red_cut_x:
    :param blue_cut_x:
    :param pt1:
    :param pt2:
    :param verbose:
    """

    if pt1 is None:
        pt1 = [1249, 31]

    if pt2 is None:
        pt2 = [1911, 253]

    # First, the order 1 trace needs to be masked out. Construct a mask
    # that not only covers the order 1 trace but everything below.
    mask_aper_o1 = build_mask_trace(y_o1, subarray=subarray,
                                    halfwidth=halfwidth_o1,
                                    extend_below=True)

    # Do the same to mask out order 3 - this one is fainter so make a
    # narrower mask. Also mask all pixels above (spatially).
    mask_aper_o3 = build_mask_trace(y_o3, subarray=subarray,
                                    halfwidth=halfwidth_o3,
                                    extend_above=True)

    # Mask what is on the left side where orders 1 and 2 are well blended
    mask_red = build_mask_vertical(subarray=subarray, masked_side='red',
                                   cut_x=red_cut_x)

    # Mask everything to the right of where the 2nd order goes out of
    # the image
    mask_blue = build_mask_vertical(subarray=subarray, masked_side='blue',
                                    cut_x=blue_cut_x)

    # Apply a y offset to the points determining the slope to handle
    # different subarray cases
    if subarray == 'FULL':
        pt1[1] = pt1[1] + 1792
        pt2[1] = pt2[1] + 1792

    if subarray == 'SUBSTRIP96':
        pt1[1] = pt1[1] - 150
        pt2[1] = pt2[1] - 150

    # Mask a slope below the order 2 to remove the wings of order 1
    mask_slope = build_mask_sloped(subarray=subarray,
                                   masked_side='blue', pt1=pt1, pt2=pt2)

    # Combine masks
    mask_o2_uncont = mask_aper_o1 | mask_aper_o3 | mask_red | mask_blue \
        | mask_slope

    return mask_o2_uncont


def build_mask_order3(subarray='SUBSTRIP256', apex_order1=40,
                      verbose=False):
    """Builds the mask to isolate the 3rd order trace.

    :param subarray:
    :param apex_order1:
    :param verbose:
    """

    dimy, dimx = 256, 2048
    if subarray == 'SUBSTRIP96':
        dimy = 96
    if subarray == 'FULL':
        dimy = 2048

    # Initialize a mask
    mask = np.zeros_like(np.zeros((dimy, dimx)), dtype='bool')

    if subarray == 'SUBSTRIP96':

        # Nothing to be done because order 3 can not be present.
        print('warning. No mask produced for order 3 when subarray=SUBSTRIP96')
        return mask

    else:

        # As determined on a subarray=256, here are the line parameters to
        # constrain the region for masking.
        # Line to mask redward of order 3: p_1=(0,132) to p_2=(1000,163)
        slope = (163 - 132) / 1000.

        if subarray == 'SUBSTRIP256':
            yintercept = 132.0  # measured on a 256 subarray
            apex_default = 40  # center of order 1 trace at lowest row value

        # Adapt if we are deealing with a FULLFRAME image
        if subarray == 'FULL':
            yintercept = 132.0 + 1792
            apex_default = 40 + 1792

        # vertical line redward of which no order 3 is detected
        maxcolumn = 700

        if apex_order1 is not None:

            # Can handle different Order 1 apex but with limited freedom
            row_offset = apex_order1 - apex_default  # relative to the nominal order 1 position
            yintercept = yintercept + row_offset

            # FALSE # maxcolumn would also move as the order 3 move up or down
            # (scale with slope) maxcolumn = maxcolumn - row_offset / slope
            if verbose:
                print('row_offset = {}, maxcolumn = {}'.format(row_offset, maxcolumn))

            # Check that at least some part of an order 3 trace of width=25
            # (and some slack = 10 pixels) can still be seen
            if yintercept > (dimy-25-10):
                # Too little left. Issue warning.
                msg = 'warning: masking for order 3 with apex_order1={} leaves too little of order 3 to fit position'
                print(msg.format(apex_order1))

        # Column by column, mask pixels below that line
        for i in range(2048):
            y = np.int(np.round(yintercept + slope * i))
            y = np.min([dimy, y])  # Make sure y does not go above array size
            mask[0:y, i] = True

        # Mask redward (spectrally) of the vertical line
        mask[:, maxcolumn:] = True

        return mask


def get_soss_centroids(image, subarray='SUBSTRIP256', apex_order1=None,
                       badpix=None, verbose=False, debug=False):
    """Function that determines the traces positions on a real image (native
    size) with as little assumptions as possible. Those assumptions are:
    1) The brightest order is order 1 and it is also the brightest of all order
    1 traces present on the image.
    2) Order 2 has a minimum in transmission between ~1.0 and ~1.2 microns.
    3) Order 2 widths are the same as order 1 width for the same wavelengths.
    The algorithm to measure trace positions is the 'edge trigger' function.
    :param image: FF, SUBSTRIP96 or SUBSTRIP256 slope image. Expected
    GR700XD+CLEAR. For the GR700XD+F277W case, an optional f277w keyword is
    passed or detected by header if passed.

    :param image:
    :param subarray:
    :param apex_order1: The y position of the apex of the order 1 trace on the
    image. The apex is the row at the center of the trace where the trace
    reaches a minimum on the detector (near 1.3 microns). A rough estimate is
    sufficient as that is only used to mask out rows on a Full-Frame image to
    ensure that the target of interest is detected instead of a field target.
    :param badpix:
    :param verbose:
    :param debug:

    :return:
    """

    # Initialize output dictionary.
    out_dict = dict()

    # Build mask that restrict the analysis to 256 or fewer vertical pixels.
    mask_256 = build_mask_256(subarray=subarray, apex_order1=apex_order1)

    # Combine masks for subsection of ~256 vertical pixels
    if badpix is not None:
        mask_256 = mask_256 | badpix

    # Get the Order 1 position
    x_o1, y_o1, w_o1, par_o1 = cen.get_uncontam_centroids_edgetrig(
            image, mask=mask_256, poly_order=11, halfwidth=2,
            mode='combined', verbose=verbose)

    # Add parameters to output dictionary.
    o1_dict = dict()
    o1_dict['X centroid'] = x_o1
    o1_dict['Y centroid'] = y_o1
    o1_dict['trace widths'] = w_o1
    o1_dict['poly coefs'] = par_o1
    out_dict['order 1'] = o1_dict

    if subarray == 'SUBSTRIP96':
        # Only order 1 can be measured. So return.
        return out_dict

    # Fit the width
    mask = np.isfinite(w_o1) & np.isfinite(x_o1)
    param_o1 = cen.robust_polyfit(x_o1[mask], w_o1[mask], 1)
    w_o1_fit = np.polyval(param_o1, x_o1)

    # Now, one can re-evaluate what the apex of order 1 truly is
    apex_order1_measured = np.round(np.min(y_o1))

    # Make a mask to isolate the 3rd order trace
    mask_o3 = build_mask_order3(subarray=subarray,
                                apex_order1=apex_order1_measured,
                                verbose=verbose)

    # Combine Order 3 mask
    if badpix is not None:
        mask_o3 = mask_o3 | badpix
    if debug is True:
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_o3, np.nan, image)
        hdu.writeto('mask_o3.fits', overwrite=True)

    # Get the centroid position by locking on trace edges and returning mean.
    out = cen.get_uncontam_centroids_edgetrig(image, mask=mask_o3,
                                              poly_order=3, halfwidth=2,
                                              mode='combined', verbose=verbose)
    x_o3, y_o3, w_o3, par_o3 = out

    # Fit the width
    mask = np.isfinite(w_o3) & np.isfinite(x_o3)
    param_o3 = cen.robust_polyfit(x_o3[mask], w_o3[mask], 1)
    w_o3_fit = np.polyval(param_o3, x_o3)

    # Making masks for the second order - split in two measurements:
    # A) Uncontaminated region 700<x<1800 - fit both edges combined (default)
    # B) Contaminated region (x=0-200) - fit only the top edge
    # Build the mask to isolate the uncontaminated part of order 2
    mask_o2_uncont = build_mask_order2_uncontaminated(x_o1, y_o1, x_o3, y_o3,
                                                      subarray=subarray)

    # Add the bad pixel mask to it (including the reference pixels)
    if badpix is not None:
        mask_o2_uncont = mask_o2_uncont | badpix
    if debug is True:
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_o2_uncont, np.nan, image)
        hdu.writeto('mask_o2_uncont.fits', overwrite=True)

    # Build the mask to isolate the contaminated part order 2
    mask_o2_cont = build_mask_order2_contaminated(x_o1, y_o1, x_o3, y_o3,
                                                  subarray=subarray)

    # Combine masks
    if badpix is not None:
        mask_o2_cont = mask_o2_cont | badpix
    if debug is True:
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_o2_cont, np.nan, image)
        hdu.writeto('mask_o2_cont.fits', overwrite=True)

    # For uncontaminated blue part, make the position measurement with the
    # default 'combined' edge method.
    out = cen.get_uncontam_centroids_edgetrig(image,
                                              mask=mask_o2_uncont,
                                              poly_order=4, halfwidth=2,
                                              mode='combined', verbose=verbose)
    x_o2_uncont, y_o2_uncont, w_o2_uncont, par_o2_uncont = out

    # Fit the width
    mask = np.isfinite(w_o2_uncont) & np.isfinite(x_o2_uncont)
    param_o2 = cen.robust_polyfit(x_o2_uncont[mask], w_o2_uncont[mask], 1)
    w_o2_uncont_fit = np.polyval(param_o2, x_o2_uncont)

    # CALIBRATE pixels-->wavelength TO COMPARE TRACE WIDTH BETWEEN ORDERS
    lba_o1 = wavelength_calibration(x_o1, order=1, subarray=subarray)
    lba_o2_uncont = wavelength_calibration(x_o2_uncont, order=2, subarray=subarray)
    lba_o3 = wavelength_calibration(x_o3, order=3, subarray=subarray)

    calibrate_width = False
    if calibrate_width is True:

        # TRACE WIDTH PLOT RELATION
        # Group together data for orders 1 and 2
        w_all = np.concatenate((w_o1, w_o2_uncont), axis=None)
        lba_all = np.concatenate((lba_o1, lba_o2_uncont), axis=None)
        ind = np.argsort(lba_all)
        lba_all, w_all = lba_all[ind], w_all[ind]

        # Fit the width vs wavelength for orders 1 and 2
        fitlog = True
        mask = np.isfinite(w_all) & np.isfinite(lba_all)
        if fitlog is False:

            # Make a linear fit
            param_all = cen.robust_polyfit(lba_all[mask], w_all[mask], 1)
            w_all_fit = np.polyval(param_all, lba_all)

        else:

            # Make a linear fit in the log-log plot - DEFAULT
            param_all = cen.robust_polyfit(np.log(lba_all[mask]),
                                           np.log(w_all[mask]), 1)
            W0, m = param_all[1], param_all[0]  # w = W0 * lba^m
            w_all_fit = np.polyval(param_all, np.log(lba_all))
            w_all_fit = np.exp(w_all_fit)

        # Make a figure of the trace width versus the wavelength
        if debug is True:

            plt.figure(figsize=(6, 6))
            plt.scatter(lba_o1, w_o1, marker=',', s=1, color='red',
                        label='Order 1')
            plt.scatter(lba_o2_uncont, w_o2_uncont+0.05, marker=',', s=1,
                        color='orange', label='Order 2 - Uncontaminated')
            plt.scatter(lba_o3, w_o3+0.15, marker=',', s=1, color='navy',
                        label='Order 3')
            plt.plot(lba_all, w_all_fit, color='black',  linewidth=5,
                     label=r'Order 1 and 2 - Fit:\nwidth = {:6.2F} $\lambda**({:6.4F})$'.format(np.exp(W0), m))
            plt.xlabel('Wavelength (microns)')
            plt.ylabel('Trace Width (pixels)')
            plt.legend()
            plt.show()
    else:
        # Adopt the already computed width relation. The best fit parameters
        # were obtained on the CV3 stack, using halfwidth=2 in the call to
        # get_uncontam_centroids_edgetrig. One should revisit the fit if using
        # a different halfwidth, or different data set.
        param_all = [-0.20711659, 3.16387517]
        W0, m = np.exp(param_all[1]), param_all[0]  # w = W0 * lba^m

    # Apply the width relation on the contaminated second order trace 'top
    # edge' positions to retrieve the trace center.
    out = cen.get_uncontam_centroids_edgetrig(image,
                                              mask=mask_o2_cont,
                                              poly_order=None, halfwidth=2,
                                              mode='minedge', verbose=verbose)
    x_o2_top, y_o2_top, w_o2_top, par_o2_top = out

    # Calibrate the wavelength
    lba_o2_top = wavelength_calibration(x_o2_top, order=2, subarray=subarray)

    # Calibrate the trace width at those wavelengths
    w_o2_cont = W0 * lba_o2_top**m

    # But make sure that unmeasured regions remain so
    w_o2_cont[~np.isfinite(y_o2_top)] = np.nan

    # Retrieve the position of the center of the trace
    y_o2_cont = y_o2_top - w_o2_cont/2.
    x_o2_cont = np.copy(x_o2_top)

    # For the uncontaminated part of second order, make measurements again but
    # return raw measurements rather than the fit.
    out = cen.get_uncontam_centroids_edgetrig(image,
                                              mask=mask_o2_uncont,
                                              poly_order=None, halfwidth=2,
                                              mode='combined', verbose=verbose)
    x_o2_uncont, y_o2_uncont, w_o2_uncont, par_o2_uncont = out

    # For the final order 2 solution, merge the contaminated and the
    # uncontaminated measurements
    y_o2 = np.nanmean([y_o2_uncont, y_o2_cont], axis=0)
    x_o2 = np.nanmean([x_o2_uncont, x_o2_cont], axis=0)
    w_o2 = np.nanmean([w_o2_uncont, w_o2_cont], axis=0)

    # Fit the width
    mask = np.isfinite(x_o2) & np.isfinite(y_o2)
    par_o2 = cen.robust_polyfit(x_o2[mask], y_o2[mask], 5)
    y_o2 = np.polyval(par_o2, x_o2)

    if debug is True:
        plt.figure(figsize=(8, 8))
        plt.ylim((0, 256))
        plt.imshow(np.log10(image), vmin=0.7, vmax=3, origin='lower',
                   aspect='auto')
        plt.plot(x_o2_cont, y_o2_cont, color='red', label='Contaminated')
        plt.plot(x_o2_uncont, y_o2_uncont, color='navy',
                 label='Uncontaminated')
        plt.plot(x_o2, y_o2, color='black', label='Merged')
        plt.legend()
        plt.show()

    # Add parameters to output dictionary.
    o2_dict, o3_dict = {}, {}
    o2_dict['X centroid'] = x_o2
    o2_dict['Y centroid'] = y_o2
    o2_dict['trace widths'] = w_o2
    o2_dict['poly coefs'] = par_o2
    out_dict['order 2'] = o2_dict
    o3_dict['X centroid'] = x_o3
    o3_dict['Y centroid'] = y_o3
    o3_dict['trace widths'] = w_o3
    o3_dict['poly coefs'] = par_o3
    out_dict['order 3'] = o3_dict

    return out_dict


# TODO - test function can be removed.
def test_soss_trace_position(im, bad):

    # Read the CV3 deep stack and bad pixel mask
    # image is the dataframe, bad the bad pixel map
    badpix = np.zeros_like(bad, dtype='bool')
    badpix[~np.isfinite(bad)] = True

    # Modify the image to test the 3 subarray cases
    try96 = True
    tryfull = False
    subarray = 'SUBSTRIP256'
    dimy = 256
    if try96 is True:
        badpix = badpix[10:106, :]
        im = im[10:106, :]
        subarray = 'SUBSTRIP96'
        dimy = 96
    if tryfull is True:
        badpixtmp = np.zeros_like(np.zeros((2048, 2048)), dtype='bool')
        badpixtmp[1792:, :] = badpix
        badpix = badpixtmp
        imtmp = np.zeros((2048, 2048))
        imtmp[1792:, :] = im
        im = imtmp
        subarray = 'FULL'
        dimy = 2048

    # Example for the call
    centroids = get_soss_centroids(im, subarray=subarray, apex_order1=None,
                                   badpix=badpix, verbose=False)

    # Figure to show the positions for all 3 orders
    plt.figure(figsize=(10, 10))
    plt.ylim((0, dimy))
    plt.imshow(np.log10(im), vmin=0.7, vmax=3, origin='lower', aspect='auto')

    tmp = centroids['order 1']
    plt.plot(tmp['X centroid'], tmp['Y centroid'], color='orange', label='Order 1')
    plt.plot(tmp['X centroid'], tmp['Y centroid'] - tmp['trace widths'] / 2, color='orange')
    plt.plot(tmp['X centroid'], tmp['Y centroid'] + tmp['trace widths'] / 2, color='orange')

    if centroids['order 2']:
        tmp = centroids['order 2']
        plt.plot(tmp['X centroid'], tmp['Y centroid'], color='black', label='Order 2')
        plt.plot(tmp['X centroid'], tmp['Y centroid'] - tmp['trace widths'] / 2, color='black')
        plt.plot(tmp['X centroid'], tmp['Y centroid'] + tmp['trace widths'] / 2, color='black')

    if centroids['order 3']:
        tmp = centroids['order 3']
        plt.plot(tmp['X centroid'], tmp['Y centroid'], color='red', label='Order 3')
        plt.plot(tmp['X centroid'], tmp['Y centroid'] - tmp['trace widths'] / 2, color='red')
        plt.plot(tmp['X centroid'], tmp['Y centroid'] + tmp['trace widths'] / 2, color='red')

    plt.legend()
    plt.show()

    return


def main():
    """Placeholder for potential multiprocessing."""

    return


if __name__ == '__main__':
    main()
