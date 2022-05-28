#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:46 2021

@author: MCR

Definitions of the main functions for the APPLESOSS (A Producer of ProfiLEs for
SOSS) module. This class will be initialized and called by the user to create
models of the spatial profiles for the first, second and third order SOSS
traces, for use as the spatprofile reference file required by the ATOCA
algorithm.
"""


from astropy.io import fits
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import warnings

from SOSS.dms.soss_centroids import get_soss_centroids
from SOSS.APPLESOSS import _calibrations
from SOSS.APPLESOSS import interpolation_model
from SOSS.APPLESOSS import plotting
from SOSS.APPLESOSS import utils
from SOSS.dms import soss_ref_files

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


class EmpiricalProfile:
    """Class wrapper around the empirical spatial profile construction module.

    Attributes
    ----------
    clear : np.array
        SOSS CLEAR exposure data frame.
    f277w : np.array
        SOSS exposure data frame using the F277W filter.
    subarray : str
        NIRISS SOSS subarray identifier. One of 'SUBSTRIP256', or 'FULL'.
    pad : int
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions.
    oversample : int
        Oversampling factor. Oversampling will be equal in the spectral and
        spatial directions.
    verbose : int
        Level of verbosity: either 3, 2, 1, or 0.
         3 - show all of progress prints, progress bars, and diagnostic plots.
         2 - show progress prints and bars.
         1 - show only progress prints.
         0 - show nothing.
    order1 : np.array
        First order spatial profile.
    order2 : np.array
        Second order spatial profile.
    order3 : np.array
        Third order spatial profile.

    Methods
    -------
    build_empirical_profile
        Construct the empirical spatial profiles.
    write_specprofile_reference
        Save spatial profile models to reference file
    """

    def __init__(self, clear, f277w=None, pad=0, oversample=1, verbose=0):
        """Initializer for EmpiricalProfile.
        """

        # Initialize input attributes.
        self.clear = clear
        self.f277w = f277w
        self.pad = pad
        self.oversample = oversample
        self.verbose = verbose

        # Validate the parameters and determine the correct subarray.
        self.subarray = self.validate_inputs()
        self.order1 = None
        self.order2 = None
        self.order3 = None

    def build_empirical_profile(self, lazy=True):
        """Run the empirical spatial profile construction module.

        Parameters
        ----------
        lazy : bool
            If True, activate lazy mode (this might actually be better...).
        """

        # Ensure F277W exposure is provided if user does not want lazy method,
        # But warn that its probably better.
        if lazy is False:
            if self.f277w is None:
                msg = 'An F277W exposure must be provided for this method.'
                raise ValueError(msg)
            else:
                msg = 'This method is now depreciated.'
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
        # Run the empirical spatial profile construction.
        o1, o2, o3 = build_empirical_profile(self.clear, self.f277w,
                                             self.subarray,
                                             self.pad, self.oversample,
                                             self.verbose, lazy)
        # Set any niggling negatives to zero (mostly for the bluest end of the
        # second order where things get skrewy).
        for o in [o1, o2, o3]:
            ii = np.where(o < 0)
            o[ii] = 0
        # Store the spatial profiles as attributes.
        self.order1, self.order2, self.order3 = o1, o2, o3

    def write_specprofile_reference(self, subarray, filename=None):
        """Write the spatial profiles to a reference file to be injested by
        ATOCA.

        Parameters
        ----------
        subarray : str
            SOSS subarray, either FULL, SUBSTRIP256 or SUBSTRIP96
        filename : str
            Name of reference file.
        """

        # Create stacked array with all orders.
        stack_full = np.zeros(((2048+2*self.pad)*self.oversample,
                               (2048+2*self.pad)*self.oversample, 3))
        stack_full[(-256-2*self.pad)*self.oversample:, :, 0] = np.copy(self.order1)
        stack_full[(-256-2*self.pad)*self.oversample:, :, 1] = np.copy(self.order2)
        stack_full[(-256-2*self.pad)*self.oversample:, :, 2] = np.copy(self.order3)
        # Pass to reference file creation.
        hdulist = soss_ref_files.init_spec_profile(stack_full, self.oversample,
                                                   self.pad, subarray,
                                                   filename)
        hdu = fits.HDUList(hdulist)
        if filename is None:
            filepattern = 'SOSS_ref_2D_profile_{}.fits'
            filename = filepattern.format(subarray)
        print('Saving to file '+filename)
        hdu.writeto(filename, overwrite=True)

    def validate_inputs(self):
        """Validate the input parameters.
        """
        return utils.validate_inputs(self)


def build_empirical_profile(clear, f277w, subarray, pad,
                            oversample, verbose, lazy=False):
    """Main procedural function for the empirical spatial profile construction
    module. Calling this function will initialize and run all the required
    subroutines to produce a spatial profile for the first, second and third
    orders.= The spatial profiles generated can include oversampling as well
    as padding in both the spatial and spectral directions.

    Parameters
    ----------
    clear : np.array
        SOSS CLEAR exposure data frame.
    f277w : np.array, None
        SOSS exposure data frame using the F277W filter.
    subarray : str
        NIRISS SOSS subarray identifier. One of SUBSTRIP256', or 'FULL'.
    pad : int
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions.
    oversample : int
        Oversampling factor. Oversampling will be equal in the spectral and
        spatial directions.
    verbose : int
        Level of verbosity: either 3, 2, 1, or 0.
         3 - show all of progress prints, progress bars, and diagnostic plots.
         2 - show progress prints and bars.
         1 - show only progress prints.
         0 - show nothing.
    lazy : bool
        If True, activate lazy mode (might actually be better....).

    Returns
    -------
    o1_uncontam : np.array
        Uncontaminated spatial profile for the first order.
    o2_uncontam : np.array
        Uncontaminated spatial profile for the second order.
    o3_uncontam : np.array
        Uncontaminated spatial profile for the third order.

    Raises
    ------
    ValueError
        When the clear dimensions do not match a known subarray.
        If the bad pixel mask is not the same shape as the clear frame.
    """

    if verbose != 0:
        print('Starting the APPLESOSS module.\n')

    # ========= INITIAL SETUP =========
    # Determine correct subarray dimensions.
    dimy, dimx = np.shape(clear)
    # If subarray is FULL - trim down to SUBSTRIP256 and work with that.
    if subarray == 'FULL':
        clear = clear[-256:, :]
        if f277w is not None:
            f277w = f277w[-256:, :]
        # Reset all variable to appropriate SUBSTRIP256 values.
        subarray = 'SUBSTRIP256'
        dimy, dimx = np.shape(clear)

    # Add a floor level such that all pixel values are positive and interpolate
    # bad pixels
    if verbose != 0:
        print(' Initial processing...')
        print('  Interpolating bad pixels...', flush=True)
    floor = np.nanpercentile(clear, 0.1)
    clear -= floor
    clear = utils.replace_badpix(clear, verbose=verbose)
    if f277w is not None:
        floor_f277w = np.nanpercentile(f277w, 0.1)
        f277w -= floor_f277w
        f277w = utils.replace_badpix(f277w, verbose=verbose)

    # Get the centroid positions for both orders from the data using the
    # edgetrig method.
    if verbose != 0:
        print('  Getting trace centroids...')
    centroids = get_soss_centroids(clear, subarray=subarray)
    if verbose == 3:
        plotting.plot_centroid(clear, centroids)

    # The four columns of pixels on the left and right edge of the SOSS
    # detector are reference pixels. Trim them off and replace them with
    # interpolations of the edge-most profiles.
    clear = pad_spectral_axis(clear[:, 5:-5],
                              centroids['order 1']['X centroid'],
                              centroids['order 1']['Y centroid'],
                              pad=5)
    if f277w is not None:
        f277w = pad_spectral_axis(f277w[:, 5:-5],
                                  centroids['order 1']['X centroid'],
                                  centroids['order 1']['Y centroid'],
                                  pad=5)

    # ========= CONSTRUCT SPATIAL PROFILE MODELS =========
    # Build a first estimate of the first and second order spatial profiles.
    # Construct the first order profile.
    if verbose != 0:
        print('  Starting the first order model...', flush=True)
    # Lazy method: no interpolation, no F277W. Just fit the wing of the first
    # order for each wavelength. Relies on some fine tuning and the second
    # order to be low level when it physically overlaps the first order.
    if lazy is True:
        if verbose != 0:
            print('  Lazy method selected...', flush=True)
        o1_native = np.zeros((dimy + 2*pad, dimx))
        first_time = True
        vbs = verbose
        disable = utils.verbose_to_bool(verbose)
        for i in tqdm(range(dimx), disable=disable):
            profile = np.copy(clear[:, i])
            cens = [centroids['order 1']['Y centroid'][i],
                    centroids['order 2']['Y centroid'][i],
                    centroids['order 3']['Y centroid'][i]]
            if first_time is False:
                vbs = 0
            newprof = reconstruct_wings256(profile, ycens=cens,
                                           contamination=[2, 3],
                                           pad=pad, verbose=vbs,
                                           smooth=True)
            first_time = False
            o1_native[:, i] = newprof

    # The original method. Get anchor profiles from the bluest clean order 1
    # profile in the CLEAR exposure, and the reddest in the F277W. Use these
    # anchors to interpolate the contaminated region. Stitch together the
    # F277W, interpolation, and CLEAR for the final model.
    else:
        o1_rough = interpolation_model.construct_order1(clear, f277w,
                                                        centroids,
                                                        subarray=subarray,
                                                        pad=pad,
                                                        verbose=verbose)

        # Rescale the first order profile to the native flux level.
        if verbose != 0:
            print('   Rescaling first order to native flux level...',
                  flush=True)
        o1_native = rescale_model(clear, o1_rough, centroids,
                                  pad=pad, verbose=verbose)
    # Add back the floor.
    o1_uncontam = o1_native + floor

    # Construct the second order profile.
    if verbose != 0:
        print('  Starting the second order trace...')
    o2_out = construct_order23(clear - o1_native[pad:dimy + pad, :],
                               centroids, order='2', verbose=verbose)
    o2_uncontam, o2_native = o2_out[0], o2_out[1]
    # Add padding to the second order if necessary
    if pad != 0:
        o2_uncontam = pad_order23(o2_uncontam, centroids, pad, order='2')

    # Construct the third order profile.
    if verbose != 0:
        print('  Starting the third order trace...')
    o3_out = construct_order23(clear - o1_native[pad:dimy + pad, :] - o2_native,
                               centroids, order='3', pivot=850,
                               verbose=verbose)
    o3_uncontam = o3_out[0]
    # Add padding to the third order if necessary
    if pad != 0:
        o3_uncontam = pad_order23(o3_uncontam, centroids, pad, order='3')

    # ========= FINAL TUNING =========
    # Pad the spectral axis.
    if pad != 0:
        if verbose != 0:
            print(' Adding padding to the spectral axis...')
        o1_uncontam = pad_spectral_axis(o1_uncontam,
                                        centroids['order 1']['X centroid'],
                                        centroids['order 1']['Y centroid'],
                                        pad=pad)
        o2_uncontam = pad_spectral_axis(o2_uncontam,
                                        centroids['order 2']['X centroid'],
                                        centroids['order 2']['Y centroid'],
                                        pad=pad)
        o3_uncontam = pad_spectral_axis(o3_uncontam,
                                        centroids['order 3']['X centroid'],
                                        centroids['order 3']['Y centroid'],
                                        pad=pad)

    # Column normalize. Only want the original detector to sum to 1, not the
    # additional padding + oversampling.
    o1_uncontam /= np.nansum(o1_uncontam, axis=0)
    o2_uncontam /= np.nansum(o2_uncontam, axis=0)
    o3_uncontam /= np.nansum(o3_uncontam, axis=0)
    # Replace NaNs resulting from all zero columns with zeros
    for o in [o2_uncontam, o3_uncontam]:
        ii = np.where(~np.isfinite(o))
        o[ii] = 0

    # Add oversampling.
    if oversample != 1:
        if verbose != 0:
            print(' Oversampling...')
        o1_uncontam = oversample_frame(o1_uncontam, oversample=oversample)
        o2_uncontam = oversample_frame(o2_uncontam, oversample=oversample)
        o3_uncontam = oversample_frame(o3_uncontam, oversample=oversample)

    if verbose != 0:
        print('\nDone.')

    return o1_uncontam, o2_uncontam, o3_uncontam


def chromescale(profile, wave_start, wave_end, ycen, poly_coef):
    """Correct chromatic variations in the PSF along the spatial direction
    using the polynomial relationship derived in _fit_trace_widths.

    Parameters
    ----------
    profile : np.array
        1D PSF profile to be rescaled.
    wave_start : float
        Starting wavelength.
    wave_end : float
        Wavelength after rescaling.
    ycen : float
        Y-pixel position of the order 1 trace centroid.
    poly_coef : tuple
        1st order polynomial coefficients describing the variation of the trace
        width with wavelength, e.g. as output by _fit_trace_widths.

    Returns
    -------
    prof_rescale : np.array
        Rescaled 1D PSF profile.
    """

    # Get the starting and ending trace widths.
    w_start = np.polyval(poly_coef, wave_start)
    w_end = np.polyval(poly_coef, wave_end)

    # Create a rescaled spatial axis. Shift the Y-centroid to zero so
    # that it does not move during the rescaling.
    xax = np.arange(len(profile)) - ycen
    xax_rescale = xax * (w_end / w_start)
    # Rescale the PSF by interpolating onto the new axis.
    prof_rescale = np.interp(xax, xax_rescale, profile)

    # Ensure the total flux remains the same
    # Integrate the profile with a Trapezoidal method.
    flux_i = np.sum(profile[1:-1]) + 0.5 * (profile[0] + profile[-1])
    flux_f = np.sum(prof_rescale[1:-1]) + 0.5 * (
            prof_rescale[0] + prof_rescale[-1])
    # Rescale the new profile so the total encompassed flux remains the same.
    prof_rescale /= (flux_f / flux_i)

    return prof_rescale


def construct_order23(residual, cen, order, pivot=750, halfwidth=12,
                      verbose=0):
    """Reconstruct the wings of the second or third orders after the first
    order spatial profile has been modeled and subtracted off.

    Parameters
    ----------
    residual : np.array
        NIRISS/SOSS data frame residual map - either with o1, or both o1 and
        o2 removed.
    cen : dict
        Centroids dictionary.
    order : str
        The order to reconstruct.
    pivot : int
        For order 2, minimum spectral pixel value for which a wing
        reconstruction will be attempted. For order 3, the maximum pixel value.
        For spectral pixels < or >pivot respectively, the profile at pivot will
        be used.
    halfwidth : int
        Half width in pixels of the spatial profile core.
    verbose : int
        level of verbosity.

    Returns
    -------
    new_frame : np.array
        Model of the second order spatial profile with wings reconstructed.
    new_frame_native : np.array
        New frame, but at the native counts level.
    """

    # Initalize new data frame and get subarray dimensions.
    dimy, dimx = np.shape(residual)
    new_frame = np.zeros_like(residual)
    new_frame_native = np.zeros_like(residual)

    # Get wavelength calibration.
    wavecal_x, wavecal_w = utils.get_wave_solution(order=2)
    # Get width polynomial coefficients.
    width_polys = utils.read_width_coefs(verbose=verbose)

    first_time = True
    if order == '3':
        maxi = pivot
    else:
        maxi = dimx
    # Hard stop for profile reuse - will handle these via padding.
    stop = np.where(cen['order ' + order]['Y centroid'] >= dimy)[0][0] + halfwidth
    for i in range(dimx):
        wave = wavecal_w[i]
        # Skip over columns where the throughput is too low to get a good core
        # and/or the order is buried within another.
        if order == '2' and i < pivot:
            continue
        if order == '3' and i > pivot:
            continue
        # If the centroid is too close to the detector edge, make note of
        # the column and deal with it later
        cen_o = int(round(cen['order ' + order]['Y centroid'][i], 0))
        if cen_o + halfwidth > dimy:
            if i < maxi:
                maxi = i
            continue

        # Get a copy of the spatial profile, and normalize it by its max value.
        working_prof = np.copy(residual[:, i])
        max_val = np.nanmax(working_prof)
        working_prof /= max_val

        # Simulate the wings.
        if first_time is False:
            verbose = 0
        wing, wing2 = simulate_wings(wave, width_polys, verbose=verbose)
        first_time = False
        # Concatenate the wings onto the profile core.
        end = int(round((cen_o + 1 * halfwidth), 0))
        start = int(round((cen_o - 1 * halfwidth), 0))
        stitch = np.concatenate([wing2, working_prof[start:end], wing])
        # Rescale to native flux level.
        stitch_native = stitch * max_val
        # Shift the profile back to its correct centroid position
        stitch = np.interp(np.arange(dimy), np.arange(dimy) - dimy//2 + cen_o,
                           stitch)
        stitch_native = np.interp(np.arange(dimy),
                                  np.arange(dimy) - dimy // 2 + cen_o,
                                  stitch_native)
        new_frame[:, i] = stitch
        new_frame_native[:, i] = stitch_native

    # For columns where the order 2 core is not distinguishable (due to the
    # throughput dropping near 0, or it being buried in order 1) reuse the
    # reddest reconstructed profile.
    if order == '2':
        for i in range(pivot):
            anchor_prof = new_frame[:, pivot]
            anchor_prof_native = new_frame_native[:, pivot]
            sc = cen['order '+order]['Y centroid'][pivot]
            ec = cen['order '+order]['Y centroid'][i]
            working_prof = np.interp(np.arange(dimy), np.arange(dimy) - sc + ec,
                                     anchor_prof)
            new_frame[:, i] = working_prof
            working_prof_native = np.interp(np.arange(dimy),
                                            np.arange(dimy) - sc + ec,
                                            anchor_prof_native)
            new_frame_native[:, i] = working_prof_native

    # For columns where the centroid is off the detector, reuse the bluest
    # reconstructed profile.
    for i in range(maxi, stop):
        anchor_prof = new_frame[:, maxi - 1]
        anchor_prof_native = new_frame_native[:, maxi - 1]
        sc = cen['order '+order]['Y centroid'][maxi - 1]
        ec = cen['order '+order]['Y centroid'][i]
        working_prof = np.interp(np.arange(dimy), np.arange(dimy) - sc + ec,
                                 anchor_prof)
        new_frame[:, i] = working_prof
        working_prof_native = np.interp(np.arange(dimy),
                                        np.arange(dimy) - sc + ec,
                                        anchor_prof_native)
        new_frame_native[:, i] = working_prof_native

        # Handle all rows after hard cut.
        new_frame = pad_order23(new_frame[halfwidth:-halfwidth], cen,
                                pad=halfwidth, order=order)

    return new_frame, new_frame_native


def oversample_frame(frame, oversample=1):
    """Oversample a dataframe by a specified amount. Oversampling is currently
    implemented separately in each dimension, that is, the spatial axis is
    oversampled first, followed by the spectral direction.

    Parameters
    ----------
    frame : np.array
        Dataframe to be oversampled.
    oversample : int
        Oversampling factor to apply to each axis.

    Returns
    -------
    osframe : np.array
        Input dataframe with each axis oversampled by the desired amount.
    """

    # Get dataframe dimensions.
    dimy, dimx = frame.shape
    newdimy, newdimx = dimy * oversample, dimx * oversample

    # Oversample spatial direction.
    osframe1 = np.zeros((newdimy, dimx))
    for i in range(dimx):
        yax_os = np.linspace(0, dimy, newdimy)
        prof_os = np.interp(yax_os, np.arange(dimy), frame[:, i])
        osframe1[:, i] = prof_os

    # Oversample spectral direction.
    osframe = np.zeros((newdimy, newdimx))
    for j in range(newdimy):
        xax_os = np.linspace(0, dimx, newdimx)
        prof_os = np.interp(xax_os, np.arange(dimx), osframe1[j, :])
        osframe[j, :] = prof_os

    return osframe


def pad_order23(dataframe, cen, pad, order):
    """Add padding to the spatial axis of an order 2 or 3 dataframe. Since
    these orders curve almost vertically at short wavelengths, we must take
    special care to properly extend the spatial profile

    Parameters
    ----------
    dataframe : np.array
        A dataframe of order 2 or 3.
    cen : dict
        Centroids dictionary.
    pad : int
        Amount of padding to add to the spatial axis.
    order : str
        Order to pad.

    Returns
    -------
    frame_padded : np.array
        The dataframe with the appropriate amount of padding added to
        the spatial axis.
    """

    # Initalize padded array.
    dimy, dimx = np.shape(dataframe)
    frame_padded = np.zeros((dimy + pad, dimx))
    frame_padded[:-pad] = dataframe

    # Use the shortest wavelength slice along the spatial axis as a reference
    # profile.
    anchor_prof = dataframe[-2, :]
    xcen_anchor = np.where(cen['order '+order]['Y centroid'] >= dimy - 2)[0][0]

    # To pad the upper edge of the spatial axis, shift the reference profile
    # according to extrapolated centroids.
    for yval in range(dimy-2, dimy-1+pad):
        xval = np.where(cen['order '+order]['Y centroid'] >= yval)[0][0]
        shift = xcen_anchor - xval
        working_prof = np.interp(np.arange(dimx), np.arange(dimx) - shift,
                                 anchor_prof)
        frame_padded[yval] = working_prof

    # Pad the lower edge with zeros.
    frame_padded = np.pad(frame_padded, ((pad, 0), (0, 0)), mode='edge')

    return frame_padded


def pad_spectral_axis(frame, xcens, ycens, pad=0, ref_cols=None):
    """Add padding to the spectral axis by interpolating the corresponding
    edge profile onto a set of extrapolated centroids.

    Parameters
    ----------
    frame : np.array
        Data frame.
    xcens : list
        X-coordinates of the trace centroids.
    ycens : list
        Y-coordinates of the trace centroids.
    pad : int
        Amount of padding to add along either end of the spectral axis (in
        pixels).
    ref_cols : list, np.array, None
        Which columns to use as the reference profiles for the padding.

    Returns
    -------
    newframe : np.array
        Data frame with padding on the spectral axis.
    """

    # Set default reference columns.
    if ref_cols is None:
        ref_cols = [6, -6]

    dimy, dimx = np.shape(frame)
    pp = np.polyfit(xcens, ycens, 5)
    xax_pad = np.arange(dimx + 2*pad) - pad
    ycens_pad = np.polyval(pp, xax_pad)

    newframe = np.zeros((dimy, dimx + 2*pad))
    newframe[:, pad:(dimx + pad)] = frame

    for col in range(pad):
        yax = np.arange(dimy)
        newframe[:, col] = np.interp(yax,
                                     yax - ycens[ref_cols[0]] + ycens_pad[col],
                                     frame[:, ref_cols[0]])

    for col in range(dimx + ref_cols[1] + pad, dimx + 2*pad):
        yax = np.arange(dimy)
        newframe[:, col] = np.interp(yax,
                                     yax - ycens[ref_cols[1]] + ycens_pad[col],
                                     frame[:, ref_cols[1]])

    return newframe


def reconstruct_wings256(profile, ycens=None, contamination=[2, 3], pad=0,
                         halfwidth=13, verbose=0, smooth=True, **kwargs):
    """Masks the second and third diffraction orders and reconstructs the
     underlying wing structure of the first order. Also adds padding in the
     spatial direction if required.

    Parameters
    ----------
    profile : np.array
        Spectral trace spatial profile.
    ycens : list, np.array
        Y-coordinates of the trace centroids. Must include all three
        diffraction orders if contamination is True, or only the first order if
        False.
    contamination : list, np.array, None
        List of contaminating orders present on the detector (not including the
        first order). For an uncontaminated detector, pass None.
    pad : int
        Amount to pad each end of the spatial axis (in pixels).
    halfwidth : int
        Half width of the profile core in pixels.
    verbose : int
        Level of verbosity.
    smooth : bool
        If True, smooths over highly deviant pixels in the spatial profile.

    Returns
    -------
    newprof : np.array
        Input spatial profile with reconstructed wings and padding.

    Raises
    ------
    ValueError
        If centroids are not provided for all three orders when contamination
        is not None.
    """

    dimy = len(profile)
    # Convert Y-centroid positions to indices
    ycens = np.atleast_1d(ycens)
    ycens = np.round(ycens, 0).astype(int)
    if contamination is not None and ycens.size != 3:
        errmsg = 'Centroids must be provided for first three orders ' \
                 'if there is contamination.'
        raise ValueError(errmsg)

    # mask negative and zero values.
    profile[profile <= 0] = np.nan

    # ======= RECONSTRUCT CONTAMINATED WING =======
    # Mask the cores of the first three diffraction orders and fit a straight
    # line to the remaining pixels. Additionally mask any outlier pixels that
    # are >3-sigma deviant from that line. Fit a 7th order polynomial to
    # remaining pixels.
    # Get the right wing of the trace profile in log space and spatial axis.
    working_profile, axis = np.copy(profile), np.arange(dimy)
    working_profile = np.log10(working_profile)

    # === Outlier masking ===
    # Mask the cores of each order.
    for order, ycen in enumerate(ycens):
        order += 1
        if order == 1:
            start = 0
            end = int(ycen + 2.5*halfwidth)
        else:
            start = np.min([ycen - int(1.5*halfwidth), dimy-2])
            end = np.min([ycen + int(1.5*halfwidth), dimy-1])
        # Set core of each order to NaN.
        working_profile[start:end] = np.nan
    # Fit the unmasked part of the wing to determine the mean linear trend.
    inds = np.where(np.isfinite(working_profile))[0]
    pp = utils.robust_polyfit(axis[inds], working_profile[inds], (0, 0))
    wing_mean = pp[1] + pp[0] * axis
    # Calculate the standard dev of unmasked points from the mean trend.
    stddev_m = np.sqrt(np.median((working_profile[inds] - wing_mean[inds])**2))

    # === Wing fit ===
    # Get fresh contaminated wing profile.
    working_profile = np.copy(profile)
    working_profile = np.log10(working_profile)
    # Mask first order core.
    working_profile[:int(ycens[0] + halfwidth)] = np.nan
    # Mask second and third orders.
    if contamination is not None:
        for order, ycen in enumerate(ycens):
            order += 1
            if order in contamination:
                if order == 3:
                    halfwidth -= 2
                if order == 2 and ycens[0] - ycens[1] < 5:
                    if (ycens[1]-0.5*halfwidth)-(ycens[0]+1.5*halfwidth) < 5:
                        start = int(np.max([ycen, 0]))
                    else:
                        start = np.max([ycen - halfwidth, 0]).astype(int)
                else:
                    start = np.max([ycen - 1.5*halfwidth, 0]).astype(int)
                end = np.max([ycen + 1.5*halfwidth, 1]).astype(int)
                # Set core of each order to NaN.
                working_profile[start:end] = np.nan
    halfwidth += 2
    # Find all outliers that are >3-sigma deviant from the mean.
    start = int(ycens[0] + 2.5*halfwidth)
    inds2 = np.where(
        np.abs(working_profile[start:] - wing_mean[start:]) > 3*stddev_m)
    # Mask outliers
    working_profile[start:][inds2] = np.nan
    # Mask edge of the detector.
    working_profile[-3:] = np.nan
    # Indices of all unmasked points in the contaminated wing.
    inds3 = np.isfinite(working_profile)

    # Fit with a 9th order polynomial.
    # To ensure that the polynomial does not start turning up in the padded
    # region, extend the linear fit to the edge of the pad to force the fit
    # to continue decreasing.
    ext_ax = np.arange(25) + np.max(axis[inds3]) + np.max([pad, 25])
    ext_prof = pp[1] + pp[0] * ext_ax
    # Concatenate right-hand profile with the extended linear trend.
    fit_ax = np.concatenate([axis[inds3], ext_ax])
    fit_prof = np.concatenate([working_profile[inds3], ext_prof])
    # Use np.polyfit for a first estimate of the coefficients.
    pp_r0 = np.polyfit(fit_ax, fit_prof, 9)
    # Robust fit using the polyfit results as a starting point.
    pp_r = utils.robust_polyfit(fit_ax, fit_prof, pp_r0)

    # === Stitching ===
    newprof = np.copy(profile)
    # Interpolate contaminated regions.
    if contamination is not None:
        for order in contamination:
            # Interpolate around the trace centroid.
            start = np.max([ycens[order - 1]-1.5*halfwidth, ycens[0]+halfwidth]).astype(int)
            if start >= dimy-1:
                # If order is off of the detector.
                continue
            end = np.min([ycens[order - 1]+1.5*halfwidth, dimy-1]).astype(int)
            # Join interpolations to the data.
            newprof = np.concatenate([newprof[:start],
                                      10**np.polyval(pp_r, axis)[start:end],
                                      newprof[end:]])
    # Interpolate nans and negatives with median of surrounding pixels.
    for pixel in np.where(np.isnan(newprof))[0]:
        minp = np.max([pixel-5, 0])
        maxp = np.min([pixel+5, dimy])
        newprof[pixel] = np.nanmedian(newprof[minp:maxp])

    if smooth is True:
        # Replace highly deviant pixels throughout the wings.
        wing_fit = np.polyval(pp_r, axis[int(ycens[0] + 2.5*halfwidth):])
        # Calculate the standard dev of unmasked points from the wing fit.
        stddev_f = np.sqrt(np.nanmedian((np.log10(newprof[int(ycens[0] + 2.5*halfwidth):])-wing_fit)**2))
        # Find all outliers that are >3-sigma deviant from the mean.
        inds4 = np.where(np.abs(np.log10(newprof[int(ycens[0] + 2.5*halfwidth):])-wing_fit) > 3*stddev_f)
        newprof[int(ycens[0] + 2.5*halfwidth):][inds4] = 10**wing_fit[inds4]

    # Add padding - padding values are constant at the median of edge pixels.
    padval_r = np.median(newprof[-8:-3])
    padval_l = np.median(newprof[3:8])
    newprof = np.concatenate([np.tile(padval_l, pad+4), newprof[4:-4],
                              np.tile(padval_r, pad+4)])

    # Do diagnostic plot if requested.
    if verbose == 3:
        plotting.plot_wing_reconstruction(profile, ycens, axis[inds3],
                                          working_profile[inds3], pp_r,
                                          newprof, pad, **kwargs)

    return newprof


def rescale_model(data, model, centroids, pad=0, verbose=0):
    """Rescale a column normalized trace model to the flux level of an actual
    observation. A multiplicative coefficient is determined via Chi^2
    minimization independently for each column, such that the rescaled model
    best matches the data.

    Parameters
    ----------
    data : np.array
        Observed dataframe.
    model : np.array
        Column normalized trace model.
    centroids : dict
        Centroids dictionary.
    pad : int
        Amount of padding on the spatial axis.
    verbose : int
        Level of verbosity.

    Returns
    -------
    model_rescale : np.array
        Trace model after rescaling.
    """

    # Determine first guess coefficients.
    k0 = np.nanmax(data, axis=0)
    ks = []
    # Loop over all columns - surely there is a more vectorized way to do this
    # whenever I try to minimize all of k at once I get nonsensical results?
    disable = utils.verbose_to_bool(verbose)
    for i in tqdm(range(data.shape[1]), disable=disable):
        # Get region around centroid
        ycen = int(round(centroids['order 1']['Y centroid'][i], 0)) + pad
        start = ycen - 13
        end = ycen + 14
        # Minimize the Chi^2.
        lik_args = (data[(start - pad):(end - pad), i], model[start:end, i])
        k = minimize(utils.lik, k0[i], lik_args)
        ks.append(k.x[0])

    # Rescale the column normalized model.
    ks = np.array(ks)
    model_rescale = ks * model

    return model_rescale


def simulate_wings(wavelength, width_coefs, halfwidth=12, verbose=0):
    """Extract the spatial profile wings from a simulated SOSS PSF to
    reconstruct the wings of the second order.

    Parameters
    ----------
    wavelength : float
        Wavelength for which to simulate wings.
    width_coefs : np.array
        Trace width polynomial coefficients.
    halfwidth : int
        Half width of a spatial profile along the spatial axis.
    verbose : int
        Level of verbosty.

    Returns
    -------
    wing : np.array
        Right wing fit.
    wing2 : np.array
        Left wing fit.
    """

    # Open a simulated PSF from which to steal some wings.
    ref_profile = 'Ref_files/SOSS_PSFs/SOSS_os1_256x256_1.000000_0.fits'
    try:
        psf = fits.getdata(ref_profile, 0)
    except FileNotFoundError:
        # If the profile doesn't exist, create it and save it to disk.
        _calibrations.loicpsf([1.0 * 1e-6], save_to_disk=True, oversampling=1,
                              pixel=256, verbose=False, wfe_real=0)
        psf = fits.getdata(ref_profile, 0)
    stand = np.sum(psf, axis=0)
    # Normalize the profile by its maximum.
    max_val = np.nanmax(stand)
    stand /= max_val
    # Scale the profile width to match the current wavelength.
    stand = chromescale(stand, 1, wavelength, 128, width_coefs)

    # Define the edges of the profile 'core'.
    ax = np.arange(256)
    ystart = int(round(256 // 2 - halfwidth, 0))
    yend = int(round(256 // 2 + halfwidth, 0))
    # Get and fit the 'right' wing.
    wing = stand[yend:]
    pp = np.polyfit(ax[yend:], np.log10(wing), 7)
    wing = 10**(np.polyval(pp, ax[yend:]))
    # Get and fit the 'left' wing.
    wing2 = stand[:ystart]
    pp = np.polyfit(ax[:ystart], np.log10(wing2), 7)
    wing2 = 10**(np.polyval(pp, ax[:ystart]))

    # Do diagnostic plot if necessary.
    if verbose == 3:
        plotting.plot_wing_simulation(stand, halfwidth, wing, wing2, ax,
                                      ystart, yend)

    return wing, wing2


if __name__ == '__main__':
    clear = fits.getdata('Ref_files/simulated_data.fits', 0)

    spat_prof = EmpiricalProfile(clear, verbose=1)
    spat_prof.build_empirical_profile()
