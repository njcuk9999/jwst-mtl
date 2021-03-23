#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits

from SOSS.extract import soss_read_refs
from SOSS.dms import soss_centroids as cen

from matplotlib import colors
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

    :param subarray: the subarray for which to build a mask.
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
    :param subarray: the subarray for which to build a mask.
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

    # Create a coordinate grid.
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
        mask_above = (ygrid - ytrace) >= 0
        mask_trace = mask_trace | mask_above

    return mask_trace


def build_mask_vertical(subarray='SUBSTRIP256', xlims=None,
                        mask_blue=True, mask_between=True):
    """Mask along the vertical(s) given by xlims.
    If xlims contains 1 element masks pixels blue-wards or red-wards according
    to the value of mask_blue (and mask_between is ignored).
    If xlims contains 2 elements masks pixels between or outside these values
    according to the value of mask_between (and mask_blue is ignored).

    :param subarray: the subarray for which to build a mask.
    :param xlims: the column indices to use as the limits of the masked area.
    :param mask_blue: if True mask pixels blue-wards of xlims, otherwise mask red-wards.
    :param mask_between: if True mask pixels between xlims, otherwise mask outside.

    :type subarray: str
    :type xlims: list[float]
    :type mask_blue: bool
    :type mask_between: bool

    :returns: mask - A mask the removes a vertical region according to xlims.
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

    # TODO better checking of xlims input.
    if xlims is None:
        xlims = [1700]
    else:
        # TODO check if integers.
        pass

    # Prepare the mask array.
    mask = np.zeros((dimy, dimx), dtype='bool')

    if np.size(xlims) == 1:

        # Mask blue-wards or red-wards of a single value.
        if mask_blue:
            mask[:, xlims[0]:] = True
        else:
            mask[:, :xlims[0]] = True

    elif np.size(xlims) == 2:

        # Mask between or exterior to two values.
        if mask_between:
            mask[:, xlims[0]:xlims[1]] = True
        else:
            mask[:, :xlims[0]] = True
            mask[:, xlims[1]:] = True

    else:
        msg = 'xlims must be a list or array of up to 2 indices.'
        raise ValueError(msg)

    return mask


def build_mask_sloped(subarray='SUBSTRIP256', point1=None, point2=None,
                      mask_above=True, verbose=False):
    """Mask pixels above or below the boundary line defined by point1 and point2.

    :param subarray: the subarray for which to build a mask.
    :param point1: the first x, y pair defining the boundary line.
    :param point2: the second x, y pair defining the boundary line.
    :param mask_above: if True mask pixels above the boundary line, else mask below.
    :param verbose: if True be verbose.

    :type subarray: str
    :type point1: list[float]
    :type point2: list[float]
    :type mask_above: bool
    :type verbose: bool

    :returns: mask - A mask the removes a diagonal region along the slope defined by point1 and point2.
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

    # If no points are given use a horizontal line along the bottom of the subarray. TODO better checking of point input.
    if point1 is None:
        point1 = [0, 0]

    if point2 is None:
        point2 = [2048, 0]

    # Obtain the parameters of the line by fitting the point.
    xvals = np.array([point1[0], point2[0]])
    yvals = np.array([point1[1], point2[1]])
    param = np.polyfit(xvals, yvals, 1)

    # Compute the position of the line at every x position.
    xline = np.arange(dimx)
    yline = np.polyval(param, xline)

    if verbose is True:
        print('line fit param:', param)

    # Create a coordinate grid.
    x = np.arange(dimx)
    y = np.arange(dimy)
    _, ygrid = np.meshgrid(x, y)

    # Mask pixels above or below the boundary line.
    if mask_above:
        mask = (ygrid - yline) >= 0
    else:
        mask = (ygrid - yline) < 0

    return mask


def build_mask_order2_contaminated(ytrace_o1, ytrace_o3, subarray='SUBSTRIP256',
                                   halfwidth_o1=25, halfwidth_o3=15, xlim=150):
    """Build a mask that isolates the contaminated part of the order 2 trace.
    This is done by masking the order 1 trace and averything below, the order
    2 trace and everything above and all pixels blue-ward (to the right) of xlim.

    :param ytrace_o1: y position of the order 1 trace at every column.
    :param ytrace_o3: y position of the order 3 trace at every column.
    :param subarray: the subarray for which to build a mask.
    :param halfwidth_o1: the size of the window to mask around the order 1 trace.
    :param halfwidth_o3: the size of the window to mask around the order 3 trace.
    :param xlim: the boundary for masking pixels blue-ward (to the right).

    :type ytrace_o1: array[float]
    :type ytrace_o3: array[float]
    :type subarray: str
    :type halfwidth_o1: float
    :type halfwidth_o3: float
    :type xlim: float

    :returns: mask - A mask that removes everything but the contaminated part of the
              order 2 trace.
    :rtype: array[bool]
    """

    # Mask the order 1 trace and everything below.
    mask_trace_o1 = build_mask_trace(ytrace_o1, subarray=subarray,
                                     halfwidth=halfwidth_o1,
                                     extend_below=True)

    # Mask the order 3 trace and everything above.
    mask_trace_o3 = build_mask_trace(ytrace_o3, subarray=subarray,
                                     halfwidth=halfwidth_o3,
                                     extend_above=True)

    # Mask all pixels blue-ward of xlim.
    mask_blue = build_mask_vertical(subarray=subarray, xlims=[xlim],
                                    mask_blue=True)

    # Combine the masks.
    mask = mask_trace_o1 | mask_trace_o3 | mask_blue

    return mask


def build_mask_order2_uncontaminated(ytrace_o1, ytrace_o3, subarray='SUBSTRIP256',
                                     halfwidth_o1=25, halfwidth_o3=15,
                                     xlims=None,
                                     point1=None, point2=None):
    """Build a mask that isolates the uncontaminated part of the order 2 trace.
    This is done by masking the order 1 trace and averything below, the order
    2 trace and everything above, all pixels outside of the range defined by xlims
    and all pixels below the line defined by point 1 and point 2.

    :param ytrace_o1: y position of the order 1 trace at every column.
    :param ytrace_o3: y position of the order 3 trace at every column.
    :param subarray: the subarray for which to build a mask.
    :param halfwidth_o1: the size of the window to mask around the order 1 trace.
    :param halfwidth_o3: the size of the window to mask around the order 3 trace.
    :param xlims:
    :param point1: the first x, y pair defining the boundary line.
    :param point2: the second x, y pair defining the boundary line.

    :type ytrace_o1: array[float]
    :type ytrace_o3: array[float]
    :type subarray: str
    :type halfwidth_o1: float
    :type halfwidth_o3: float
    :type xlims: list[float]
    :type point1: list[float]
    :type point2: list[float]

    :returns: mask - A mask that removes everything but the uncontaminated part of the
              order 2 trace.
    :rtype: array[bool]
    """

    if xlims is None:
        xlims = [700, 1800]

    if point1 is None:
        point1 = [1249, 31]

    if point2 is None:
        point2 = [1911, 253]

    # Apply a y offset to the points determining the slope to handle
    # different subarray cases TODO This is confusingly only done here and not in build_mask_sloped
    if subarray == 'FULL':
        point1[1] = point1[1] + 1792
        point2[1] = point2[1] + 1792

    if subarray == 'SUBSTRIP96':
        point1[1] = point1[1] - 150
        point2[1] = point2[1] - 150

    # Mask the order 1 trace and everything below.
    mask_trace_o1 = build_mask_trace(ytrace_o1, subarray=subarray,
                                     halfwidth=halfwidth_o1,
                                     extend_below=True)

    # Mask the order 3 trace and everything above.
    mask_trace_o3 = build_mask_trace(ytrace_o3, subarray=subarray,
                                     halfwidth=halfwidth_o3,
                                     extend_above=True)

    # Mask what is on the left side where orders 1 and 2 are well blended
    mask_vertical = build_mask_vertical(subarray=subarray, xlims=xlims,
                                        mask_between=False)

    # Mask the corner below the order 2 trace to remove the wings of the order 1 trace.
    mask_sloped = build_mask_sloped(subarray=subarray, point1=point1, point2=point2,
                                    mask_above=False)

    # Combine the masks.
    mask = (mask_trace_o1 | mask_trace_o3 | mask_vertical | mask_sloped)

    return mask


def build_mask_order3(subarray='SUBSTRIP256', xlim=700, point1=None, point2=None, apex_order1=None):
    """Builds a mask that isolates the order 3 trace.
    This done by masking all pixels blue-ward (to the right) of xlim where the order 3
    transmission goes to zero, and all pixels below the line defined by point1 and point2
    (the order1 trace and order 2 trace).

    :param subarray: the subarray for which to build a mask.
    :param xlim: the boundary for masking pixels blue-ward (to the right).
    :param point1: the first x, y pair defining the boundary line.
    :param point2: the seconf x, y pair defining the boundary line.
    :param apex_order1: The y-position of the order1 apex at 1.3 microns, in the given subarray.

    :type subarray: str
    :type xlim: float
    :type point1: list[float]
    :type point2: list[float]
    :type apex_order1: float

    :returns: mask - A mask that removes everything but the order 3 trace.
    :rtype: array[bool]
    """

    dimx = 2048
    apex_default = 40

    if point1 is None:  # TODO what if is not None? Assume it's in SUBSTRIP256 coordinates? Same for apex.
        point1 = [0, 132]  # Assuming SUBSTRIP256.

    if point2 is None:
        point2 = [1000, 163]  # Assuming SUBSTRIP256.

    if subarray == 'SUBSTRIP96':

        dimy = 96

        # Create an empty mask.
        mask = np.zeros((dimy, dimx), dtype='bool')

        # Nothing to be done because order 3 can not be present.
        print('warning. No mask produced for order 3 when subarray=SUBSTRIP96')

        return mask

    elif subarray == 'SUBSTRIP256':

        dimy = 256

    elif subarray == 'FULL':

        dimy = 2048

        point1[1] += 1792
        point2[1] += 1792

    else:
        msg = 'Unknown subarray: {}'
        raise ValueError(msg.format(subarray))

    if apex_order1 is not None:

        # Can handle different Order 1 apex but with limited freedom
        offset = apex_order1 - apex_default
        point1[1] += offset
        point2[1] += offset

        # Check how close the boundary line is to the top of the subarray.
        if point1[1] > (dimy - 25 - 10):
            msg = 'Warning: masking for order 3 with apex_order1={} leaves too little of order 3 to fit position.'
            print(msg.format(apex_order1))

    # Mask everything beyond where the order 3 transmission approaches zero.
    mask_vertical = build_mask_vertical(subarray=subarray, xlims=[xlim], mask_blue=True)

    # Mask everything below order 3.
    mask_sloped = build_mask_sloped(subarray=subarray, point1=point1, point2=point2, mask_above=False)

    # Combine the masks.
    mask = mask_vertical | mask_sloped

    return mask


def calibrate_widths(width_o1, width_o2_uncont, width_o3, subarray='SUBSTRIP256', debug=False):
    """Fit an exponential function to the wavelength-width relation, for use obtaining the
    contaminated order 2 trace positions.
    # TODO make order 2 and 3 widths optional use those provided?
    :param width_o1: The order 1 trace width at each column, must have shape = (2048,).
    :param width_o2_uncont: The order 2 trace width at each column, must have shape = (2048,).
    :param width_o3: The order 3 trace width at each column, must have shape = (2048,). # TODO Order 3 basically unused.
    :param subarray: The subarray for which to build a mask. TODO basically has no effect?
    :param debug: If set True some diagnostic plots will be made.

    :type width_o1: array[float]
    :type width_o2_uncont: array[float]
    :type width_o3: array[float]
    :type subarray: str
    :type debug: bool

    :returns: pars_width - a list containing the best-fit parameters for the wavelength-width relation.
    :rtype list[float]
    """

    x = np.arange(2048)

    # Convert pixel positions to wavelengths for each order.
    lba_o1 = wavelength_calibration(x, order=1, subarray=subarray)
    lba_o2_uncont = wavelength_calibration(x, order=2, subarray=subarray)
    lba_o3 = wavelength_calibration(x, order=3, subarray=subarray)

    # Join data from orders 1 and 2.
    lba_all = np.concatenate((lba_o1, lba_o2_uncont), axis=None)
    width_all = np.concatenate((width_o1, width_o2_uncont), axis=None)

    # Fit the wavelength vs width of order 1 and 2 using an exponential model.
    mask = np.isfinite(width_all) & np.isfinite(lba_all)
    pars_width = cen.robust_polyfit(np.log(lba_all[mask]), np.log(width_all[mask]), 1)

    # Make a figure of the trace width versus the wavelength
    if debug:

        # Evalaute the best-fit model.
        lba_fit = np.linspace(np.nanmin(lba_all), np.nanmax(lba_all), 101)
        w0, m = np.exp(pars_width[1]), pars_width[0]  # w = w0 * lba^m
        width_fit = w0 * lba_fit ** m

        # Make the figure.
        plt.figure(figsize=(8, 5))

        plt.scatter(lba_o1, width_o1, marker=',', s=1, color='red',
                    label='Order 1')
        plt.scatter(lba_o2_uncont, width_o2_uncont + 0.05, marker=',', s=1,
                    color='orange', label='Order 2 - Uncontaminated')
        plt.scatter(lba_o3, width_o3 + 0.10, marker=',', s=1, color='navy',
                    label='Order 3')

        plt.plot(lba_fit, width_fit, color='black', linewidth=5,
                 label='Order 1 and 2 - Fit:\n width = {:6.2F} $\lambda**({:6.4F})$'.format(w0, m))

        plt.xlabel('Wavelength (microns)', fontsize=12)
        plt.ylabel('Trace Width (pixels)', fontsize=12)
        plt.legend(fontsize=12)

        plt.tight_layout()

        plt.show()
        plt.close()

    return pars_width


def get_soss_centroids(image, mask=None, subarray='SUBSTRIP256', apex_order1=None,
                       calibrate=False, verbose=False, debug=False):  # TODO hardcoded parameters.
    """Determine the traces positions on a real image (native size) with as few
    assumptions as possible using the 'edge trigger' method.

    The algorithm assumes:
    1) The brightest order is order 1 and the target order 1 is the brightest
        of all order 1 traces present.
    2) Order 2 has a minimum in transmission between ~1.0 and ~1.2 microns.
    3) Order 2 widths are the same as order 1 width for the same wavelengths.

    :param image: A 2D image of the detector.
    :param mask: A boolean array of the same shape as image. Pixels corresponding to True values will be masked.
    :param subarray: the subarray for which to build a mask.
    :param apex_order1: The y-position of the order1 apex at 1.3 microns, in the given subarray.
        A rough estimate is sufficient as it is only used to mask rows when subarray='FULL' to
        ensure that the target of interest is detected instead of a field target.
    :param calibrate: If True model the wavelength trace width relation. Default is False. TODO default to True?
    :param verbose: TODO remove? Use verbose of debug in rest of centroids code?
    :param debug: If set True some diagnostic plots will be made.

    :returns: trace_dict - A dictionary containing the trace x, y, width and polynomial fit parameters for each order.
    :rtype: dict
    """

    # Initialize output dictionary.
    trace_dict = dict()

    # Build a mask that restricts the analysis to a SUBSTRIP256-like region centered on the target trace.
    mask_256 = build_mask_256(subarray=subarray, apex_order1=apex_order1)

    # Combine the subsection mask with the user specified mask.
    if mask is not None:
        mask_256 = mask_256 | mask

    if debug:
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_256, np.nan, image)
        hdu.writeto('mask_256.fits', overwrite=True)

    # Get the order 1 trace position.
    result = cen.get_uncontam_centroids_edgetrig(
            image, mask=mask_256, poly_order=11, halfwidth=2,
            mode='combined', verbose=verbose)

    x_o1, y_o1, w_o1, par_o1 = result

    # Add parameters to output dictionary.
    o1_dict = dict()
    o1_dict['X centroid'] = x_o1
    o1_dict['Y centroid'] = y_o1
    o1_dict['trace widths'] = w_o1
    o1_dict['poly coefs'] = par_o1
    trace_dict['order 1'] = o1_dict

    # For SUBSTRIP96 only the order 1 can be measured.
    if subarray == 'SUBSTRIP96':
        return trace_dict

    # Now, one can re-evaluate what the apex of order 1 truly is
    apex_order1_measured = np.round(np.min(y_o1))

    # Make a mask to isolate the order 3 trace and combine it with the user-specified mask.
    mask_o3 = build_mask_order3(apex_order1=apex_order1_measured, subarray=subarray)

    if mask is not None:
        mask_o3 = mask_o3 | mask

    if debug:
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_o3, np.nan, image)
        hdu.writeto('mask_o3.fits', overwrite=True)

    # Get the order 3 trace position.
    result = cen.get_uncontam_centroids_edgetrig(image, mask=mask_o3,
                                                 poly_order=3, halfwidth=2,
                                                 mode='combined', verbose=verbose)
    x_o3, y_o3, w_o3, par_o3 = result

    # Add parameters to output dictionary.
    o3_dict = dict()
    o3_dict['X centroid'] = x_o3
    o3_dict['Y centroid'] = y_o3
    o3_dict['trace widths'] = w_o3
    o3_dict['poly coefs'] = par_o3
    trace_dict['order 3'] = o3_dict

    # Make masks for the second order trace - split in two segments:
    # A) Uncontaminated region 700 < x < 1800 - fit both edges combined (default).
    # B) Contaminated region (x = 0-200) - fit only the top edge.

    # Make a mask to isolate the uncontaminated order 2 trace and combine it with the user-specified mask.
    mask_o2_uncont = build_mask_order2_uncontaminated(y_o1, y_o3,
                                                      subarray=subarray)

    if mask is not None:
        mask_o2_uncont = mask_o2_uncont | mask

    if debug:
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_o2_uncont, np.nan, image)
        hdu.writeto('mask_o2_uncont.fits', overwrite=True)

    # Get the raw trace positions for the uncontaminated part of the order 2 trace.
    result = cen.get_uncontam_centroids_edgetrig(image,
                                                 mask=mask_o2_uncont,
                                                 poly_order=None, halfwidth=2,
                                                 mode='combined', verbose=verbose)

    x_o2_uncont, y_o2_uncont, w_o2_uncont, par_o2_uncont = result

    if calibrate:

        pars_width = calibrate_widths(w_o1, w_o2_uncont, w_o3, subarray=subarray, debug=debug)

    else:
        # Adopt the already computed width relation. The best fit parameters
        # were obtained on the CV3 stack, using halfwidth=2 in the call to
        # get_uncontam_centroids_edgetrig. One should revisit the fit if using
        # a different halfwidth, or different data set.
        pars_width = [-0.20711659, 3.16387517]

    w0, m = np.exp(pars_width[1]), pars_width[0]  # w = w0 * lba^m

    # Make a mask to isolate the contaminated order 2 trace and combine it with the user-specified mask.
    mask_o2_cont = build_mask_order2_contaminated(y_o1, y_o3,
                                                  subarray=subarray)

    if mask is not None:
        mask_o2_cont = mask_o2_cont | mask

    if debug:
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_o2_cont, np.nan, image)
        hdu.writeto('mask_o2_cont.fits', overwrite=True)

    # Get the raw top-edge poistions of the contaminated order 2 trace. TODO rename minedge etc. in get_uncontam_centroids_edgetrig?
    result = cen.get_uncontam_centroids_edgetrig(image,
                                                 mask=mask_o2_cont,
                                                 poly_order=None, halfwidth=2,
                                                 mode='minedge', verbose=verbose)

    x_o2_top, y_o2_top, w_o2_top, par_o2_top = result

    # Convert pixel positions to wavelengths for order 2.
    lba_o2_top = wavelength_calibration(x_o2_top, order=2, subarray=subarray)

    # Use the wavelength width relation to obtain the order 2 trace width.
    w_o2_cont = np.where(np.isfinite(w_o2_top), w0 * lba_o2_top**m, np.nan)

    # Finally combine the top-edge positions and the width to get an estimate of the trace center.
    x_o2_cont = np.copy(x_o2_top)
    y_o2_cont = y_o2_top - w_o2_cont/2.

    # Combine the trace positions from the uncontaminated and contaminated sections.
    mask_comb = np.isfinite(y_o2_uncont)
    x_o2 = np.where(mask_comb, x_o2_uncont, x_o2_cont)
    y_o2 = np.where(mask_comb, y_o2_uncont, y_o2_cont)
    w_o2 = np.where(mask_comb, w_o2_uncont, w_o2_cont)

    # Fit the combined order 2 trace position with a polynomial.
    mask_fit = np.isfinite(x_o2) & np.isfinite(y_o2)
    par_o2 = cen.robust_polyfit(x_o2[mask_fit], y_o2[mask_fit], 5)
    y_o2 = np.polyval(par_o2, x_o2)

    # Add parameters to output dictionary.
    o2_dict = dict()
    o2_dict['X centroid'] = x_o2
    o2_dict['Y centroid'] = y_o2
    o2_dict['trace widths'] = w_o2
    o2_dict['poly coefs'] = par_o2
    trace_dict['order 2'] = o2_dict

    if debug:

        nrows, ncols = image.shape

        if subarray == 'FULL':
            aspect = 1
            figsize = ncols/64, nrows/64
        else:
            aspect = 2
            figsize = ncols/64, nrows/32

        plt.figure(figsize=figsize)

        plt.title('Order 2 Trace Positions')

        tmp = np.ma.masked_array(image, mask=mask)
        plt.imshow(tmp, origin='lower', cmap='inferno', norm=colors.LogNorm(), aspect=aspect)

        plt.plot(x_o2_cont, y_o2_cont, color='red', label='Contaminated')
        plt.plot(x_o2_uncont, y_o2_uncont, color='navy', label='Uncontaminated')
        plt.plot(x_o2, y_o2, color='black', label='Polynomial Fit')

        plt.xlabel('Spectral Pixel', fontsize=14)
        plt.ylabel('Spatial Pixel', fontsize=14)
        plt.legend(fontsize=12)

        plt.xlim(-0.5, ncols - 0.5)
        plt.ylim(-0.5, nrows - 0.5)

        plt.tight_layout()

        plt.show()
        plt.close()

    return trace_dict


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
