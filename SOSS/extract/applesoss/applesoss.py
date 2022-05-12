#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:46 2021

@author: MCR

Definitions of the main functions for the APPLESOSS (A Producer of ProfiLEs for
SOSS) module. This class will be initialized and called by the user to create
models of the spatial profiles for both the first and second order SOSS traces,
for use as the spatprofile reference file required by the ATOCA algorithm.
"""


import os
from astropy.io import fits
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import warnings

from SOSS.dms.soss_centroids import get_soss_centroids
from SOSS.extract.applesoss import plotting
from SOSS.extract.applesoss import _calibrations
from SOSS.extract.applesoss import utils

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
    badpix_mask : np.array
        Bad pixel mask, values of True represent bad pixels. Must be the same
        shape as the CLEAR dataframe.
    subarray : str
        NIRISS SOSS subarray identifier. One of 'SUBSTRIP256', or 'FULL'.
    pad : tuple
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions respectively.
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

    Methods
    -------
    build_empirical_profile
        Construct the empirical spatial profiles.
    save_to_file
        Save spatial profile models to file
    """

    def __init__(self, clear, f277w, badpix_mask, pad=(0, 0), oversample=1,
                 verbose=0):
        """Initializer for EmpiricalProfile.
        """

        # Initialize input attributes.
        self.clear = clear
        self.f277w = f277w
        self.badpix_mask = badpix_mask
        self.pad = pad
        self.oversample = oversample
        self.verbose = verbose

        # Validate the parameters and determine the correct subarray.
        self.subarray = self.validate_inputs()
        self.order1 = None
        self.order2 = None

    def build_empirical_profile(self, normalize=True, lazy=False):
        """Run the empirical spatial profile construction module.

        Parameters
        ----------
        normalize : bool
            if True, column normalize the final spatial profiles such that the
            flux in each column sums to one.
        lazy : bool
            If True, activate lazy mode (this might actually be better...).
        """

        # Force lazy mode if no F277W exposure is provided.
        if self.f277w is None:
            lazy = True
        # Run the empirical spatial profile construction.
        o1, o2 = build_empirical_profile(self.clear, self.f277w,
                                         self.badpix_mask, self.subarray,
                                         self.pad, self.oversample, normalize,
                                         self.verbose, lazy)
        # Store the spatial profiles as attributes.
        self.order1, self.order2 = o1, o2

    def save_to_file(self, filename=None):
        """Write the uncontaminated 2D trace profiles to a fits file.

        Parameters
        ----------
        filename : str (optional)
            Path to file to which to save the spatial profiles. Defaults to
            'SOSS_2D_profile_{subarray}.fits'.
        """

        # Get default filename if none provided.
        if filename is None:
            pad = self.pad[0]
            ovsmp = self.oversample
            sub = self.subarray
            filename = 'SOSS_2D_profile_{0}_os={1}_pad={2}.fits'.format(sub, ovsmp, pad)
        if self.verbose != 0:
            print('Saving trace profiles to file {}...'.format(filename))

        # Print overwrite warning if output file already exists.
        if os.path.exists(filename):
            msg = 'Output file {} already exists.'\
                  ' It will be overwritten'.format(filename)
            warnings.warn(msg)

        # Write trace profiles to disk.
        utils.write_to_file(self.order1, self.order2, self.subarray, filename,
                            self.pad, self.oversample)

    def validate_inputs(self):
        """Validate the input parameters.
        """
        return utils.validate_inputs(self)


def build_empirical_profile(clear, f277w, badpix_mask, subarray, pad,
                            oversample, normalize, verbose, lazy=False):
    """Main procedural function for the empirical spatial profile construction
    module. Calling this function will initialize and run all the required
    subroutines to produce a spatial profile for the first and second orders.
    The spatial profiles generated can include oversampling as well as padding
    in both the spatial and spectral directions.

    Parameters
    ----------
    clear : np.array
        SOSS CLEAR exposure data frame.
    f277w : np.array, None
        SOSS exposure data frame using the F277W filter.
    badpix_mask : np.array
        Bad pixel mask, values of True represent bad pixels. Must be the same
        shape as the CLEAR dataframe.
    subarray : str
        NIRISS SOSS subarray identifier. One of SUBSTRIP256', or 'FULL'.
    pad : tuple
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions, respectively.
    oversample : int
        Oversampling factor. Oversampling will be equal in the spectral and
        spatial directions.
    normalize : bool
        if True, column normalize the final spatial profiles such that the
        flux in each column sums to one.
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
    # Initialize trim variable to False unless the subarray is FULL.
    trim = False
    if subarray == 'FULL':
        # If subarray is FULL - trim down to SUBSTRIP256 and work with that.
        # The rest if the frame is zeros anyways.
        clear = clear[-256:, :]
        if f277w is not None:
            f277w = f277w[-256:, :]
        badpix_mask = badpix_mask[-256:, :]
        # Reset all variable to appropriate SUBSTRIP256 values.
        subarray = 'SUBSTRIP256'
        dimy, dimx = np.shape(clear)
        # Note that the detector was trimmed.
        trim = True

    # Add a floor level such that all pixel values are positive
    floor = np.nanpercentile(clear, 0.1)
    clear_floorsub = clear - floor
    if f277w is not None:
        floor_f277w = np.nanpercentile(f277w, 0.1)
        f277w_floorsub = f277w - floor_f277w

    # Get the centroid positions for both orders from the data using the
    # edgetrig method.
    if verbose != 0:
        print(' Initial processing...')
        print('  Getting trace centroids...')
    centroids = get_soss_centroids(clear_floorsub, mask=badpix_mask,
                                   subarray=subarray)
    if verbose == 3:
        plotting.plot_centroid(clear_floorsub, centroids)

    # The four columns of pixels on the left and right edge of the SOSS
    # detector are reference pixels. Trim them off and replace them with
    # interpolations of the edge-most profiles.
    clear_floorsub = pad_spectral_axis(clear_floorsub[:, 4:-4],
                                       centroids['order 1']['X centroid'],
                                       centroids['order 1']['Y centroid'],
                                       pad=4)
    if f277w is not None:
        f277w_floorsub = pad_spectral_axis(f277w_floorsub[:, 4:-4],
                                           centroids['order 1']['X centroid'],
                                           centroids['order 1']['Y centroid'],
                                           pad=4)

    # ========= CONSTRUCT SPATIAL PROFILE MODELS =========
    # Build a first estimate of the first and second order spatial profiles.
    # Construct the first order profile.
    if verbose != 0:
        print('  Starting the first order model...')
    # Lazy method: no interpolation, no F277W. Just fit the wing of the first
    # order for each wavelength. Relies on some fine tuning and the second
    # order to be very low level when it physically overlaps the first order.
    if lazy is True:
        if verbose != 0:
            print('  Lazy method selected...')
        o1_native = np.zeros((dimy + 2 * pad[0], dimx))
        first_time = True
        for i in tqdm(range(dimx)):
            profile = np.copy(clear_floorsub[:, i])
            cens = [centroids['order 1']['Y centroid'][i],
                    centroids['order 2']['Y centroid'][i],
                    centroids['order 3']['Y centroid'][i]]
            if first_time is False:
                newprof = reconstruct_wings256(profile, ycens=cens,
                                               contamination=[2, 3],
                                               pad=pad[0], verbose=0,
                                               smooth=True)
            else:
                newprof = reconstruct_wings256(profile, ycens=cens,
                                               contamination=[2, 3],
                                               pad=pad[0],
                                               verbose=verbose, smooth=True)
                first_time = False
            o1_native[:, i] = newprof

    # The original method. Get anchor profiles from the bluest clean order 1
    # profile in the CLEAR exposure, and the reddest in the F277W. Use these
    # anchors to interpolate the contaminated region. Stitch together the
    # F277W, interpolation, and CLEAR for the final model.
    else:
        o1_rough = construct_order1(clear_floorsub, f277w_floorsub, centroids,
                                    subarray=subarray, pad=pad[0],
                                    verbose=verbose)

        # Rescale the first order profile to the native flux level.
        if verbose != 0:
            print('   Rescaling first order to native flux level...',
                  flush=True)
        o1_native = rescale_model(clear_floorsub, o1_rough, centroids,
                                  pad=pad[0], verbose=verbose)
    # Add back the floor.
    o1_uncontam = o1_native + floor

    # Construct the second order profile.
    if verbose != 0:
        print('  Starting the second order trace model...')
    o2_uncontam = construct_order2(clear - o1_native[pad[0]:dimy + pad[0], :],
                                   centroids, verbose=verbose)
    # Add padding to the second order if necessary
    if pad[0] != 0:
        o2_uncontam = pad_order2(o2_uncontam, centroids, pad[0])

    # ========= FINAL TUNING =========
    # Pad the spectral axis.
    if pad[1] != 0:
        if verbose != 0:
            print(' Adding padding to the spectral axis...')
        o1_uncontam = pad_spectral_axis(o1_uncontam,
                                        centroids['order 1']['X centroid'],
                                        centroids['order 1']['Y centroid'],
                                        pad=pad[1])
        o2_uncontam = pad_spectral_axis(o2_uncontam,
                                        centroids['order 2']['X centroid'],
                                        centroids['order 2']['Y centroid'],
                                        pad=pad[1])

    # Column normalize. Only want the original detector to sum to 1, not the
    # additional padding + oversampling.
    if normalize is True:
        o1_uncontam /= np.nansum(o1_uncontam[pad[0]:dimy + pad[0]], axis=0)
        o2_uncontam /= np.nansum(o2_uncontam[pad[0]:dimy + pad[0]], axis=0)

    # Add oversampling.
    if oversample != 1:
        if verbose != 0:
            print(' Oversampling...')
        o1_uncontam = oversample_frame(o1_uncontam, oversample=oversample)
        o2_uncontam = oversample_frame(o2_uncontam, oversample=oversample)

    # If the original subarray was FULL - add back the rest of the frame
    if trim is True:
        # Create the FULL frame including oversampling and padding.
        o1_full = np.zeros(((2048 + 2 * pad[0]) * oversample,
                            (2048 + 2 * pad[1]) * oversample))
        o2_full = np.zeros(((2048 + 2 * pad[0]) * oversample,
                            (2048 + 2 * pad[1]) * oversample))
        # Put the uncontaminated SUBSTRIP256 frames on the FULL detector.
        dimy = np.shape(o1_uncontam)[0]
        o1_full[-dimy:, :] = o1_uncontam
        o2_full[-dimy:, :] = o2_uncontam
        o1_uncontam = o1_full
        o2_uncontam = o2_full

    if verbose != 0:
        print('\nDone.')

    return o1_uncontam, o2_uncontam


def _chromescale(profile, wave_start, wave_end, ycen, poly_coef):
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


def construct_order1(clear, f277w, ycens, subarray, pad=0, verbose=0):
    """This creates the full order 1 trace profile model. The region
    contaminated by the second order is interpolated from the CLEAR and F277W
    exposures.
    The steps are as follows:
        1. Determine the red and blue anchor profiles for the interpolation.
        2. Construct the interpolated profile at each wavelength in the
           overlap region.
        3. Stitch together the original CLEAR exposure and the interpolated
           trace model (as well as the F277W exposure if provided).
        4. Mask the contamination from the second and third orders, and
           reconstruct the underlying wing structure of the first order -
           including any padding in the spatial direction.

    Parameters
    ----------
    clear : np.array
        NIRISS SOSS CLEAR exposure dataframe.
    f277w : np.array
        NIRISS SOSS F277W filter exposure dataframe.
    ycens : dict
        Dictionary of Y-coordinates for the trace centroids of the first three
        diffraction orders, ie. as returned by get_soss_centroids.
    subarray : str
        Subarray identifier. "SUBSTRIP256", or "FULL".
    pad : int
        Number of pixels of padding to add on both ends of the spatial axis.
    verbose : int
        Level of verbosity.

    Returns
    -------
    o1frame : np.array
        Interpolated order 1 trace model with padding.
    """

    # Get subarray dimensions
    dimx = 2048
    if subarray == 'SUBSTRIP256':
        dimy = 256
    elif subarray == 'FULL':
        dimy = 2048

    # ========= INITIAL SETUP =========
    # Get wavelengh calibration.
    wavecal_x, wavecal_w = utils.get_wave_solution(order=1)
    # Determine how the trace width changes with wavelength.
    if verbose != 0:
        print('   Calibrating trace widths...')
    width_polys = utils.read_width_coefs(verbose=verbose)

    # ========= GET ANCHOR PROFILES =========
    if verbose != 0:
        print('   Getting anchor profiles...')
    # Determine the anchor profiles - blue anchor.
    # Find the pixel position of 2.1µm.
    i_b = np.where(wavecal_w >= 2.1)[0][-1]
    xdb = int(wavecal_x[i_b])
    ydb = ycens['order 1']['Y centroid'][xdb]
    # Extract the 2.1µm anchor profile from the data - take median profile of
    # neighbouring five columns to mitigate effects of outliers.
    bl_anch = np.median(clear[:, (xdb - 2):(xdb + 2)], axis=1)

    # Mask second and third order, reconstruct wing structure and pad.
    cens = [ycens['order 1']['Y centroid'][xdb],
            ycens['order 2']['Y centroid'][xdb],
            ycens['order 3']['Y centroid'][xdb]]
    bl_anch = reconstruct_wings256(bl_anch, ycens=cens,
                                   contamination=[2, 3], pad=pad,
                                   verbose=verbose, smooth=True,
                                   **{'text': 'Blue anchor'})
    # Remove the chromatic scaling.
    bl_anch = _chromescale(bl_anch, 2.1, 2.5, ydb + pad, width_polys)
    # Normalize
    bl_anch /= np.nansum(bl_anch)

    # Determine the anchor profiles - red anchor.
    # If an F277W exposure is provided, only interpolate out to 2.45µm.
    # Red-wards of 2.45µm we have perfect knowledge of the order 1 trace.
    # Find the pixel position of 2.45µm.
    i_r = np.where(wavecal_w >= 2.45)[0][-1]
    xdr = int(wavecal_x[i_r])
    ydr = ycens['order 1']['Y centroid'][xdr]

    # Extract and rescale the 2.45µm profile - take median of neighbouring
    # five columns to mitigate effects of outliers.
    rd_anch = np.median(f277w[:, (xdr - 2):(xdr + 2)], axis=1)
    # Reconstruct wing structure and pad.
    cens = [ycens['order 1']['Y centroid'][xdr]]
    rd_anch = reconstruct_wings256(rd_anch, ycens=cens,
                                   contamination=None, pad=pad,
                                   verbose=verbose, smooth=True,
                                   **{'text': 'Red anchor'})
    rd_anch = _chromescale(rd_anch, 2.45, 2.5, ydr + pad, width_polys)
    # Normalize
    rd_anch /= np.nansum(rd_anch)

    # Interpolation polynomial coefs, calculated via calc_interp_coefs
    coef_b, coef_r = utils.read_interp_coefs(verbose=verbose)

    # Since there will lkely be a different number of integrations for the
    # F277W exposure vs the CLEAR, the flux levels of the two anchors will
    # likely be different. Rescale the F277W anchor to match the CLEAR level.
    rd_anch = rescale_f277(rd_anch, clear[:, xdr], ycen=ydr, pad=pad,
                           verbose=verbose)

    # ========= INTERPOLATE THE CONTAMINATED REGION =========
    # Create the interpolated order 1 PSF.
    map2d = np.zeros((dimy + 2 * pad, dimx)) * np.nan
    # Get centroid pixel coordinates and wavelengths for interpolation region.
    cenx_d = np.arange(xdb - xdr).astype(int) + xdr
    ceny_d = ycens['order 1']['Y centroid'][cenx_d]
    lmbda = wavecal_w[cenx_d]

    # Create an interpolated 1D PSF at each required position.
    if verbose != 0:
        print('   Interpolating trace...', flush=True)
    disable = utils.verbose_to_bool(verbose)
    for i, vals in tqdm(enumerate(zip(cenx_d, ceny_d, lmbda)),
                        total=len(lmbda), disable=disable):
        cenx, ceny, lbd = int(round(vals[0], 0)), vals[1], vals[2]
        # Evaluate the interpolation polynomials at the current wavelength.
        wb_i = np.polyval(coef_b, lbd)
        wr_i = np.polyval(coef_r, lbd)
        # Recenter the profile of both anchors on the correct Y-centroid.
        bax = np.arange(dimy + 2 * pad) - ydb + ceny
        bl_anch_i = np.interp(np.arange(dimy + 2 * pad), bax, bl_anch)
        rax = np.arange(len(rd_anch)) - ydr + ceny
        rd_anch_i = np.interp(np.arange(dimy + 2 * pad), rax, rd_anch)
        # Construct the interpolated profile.
        prof_int = (wb_i * bl_anch_i + wr_i * rd_anch_i)
        # Re-add the lambda/D scaling.
        prof_int_cs = _chromescale(prof_int, 2.5, lbd, ceny + pad, width_polys)
        # Put the interpolated profile on the detector.
        map2d[:, cenx] = prof_int_cs

        # Note detector coordinates of the edges of the interpolated region.
        bl_end = cenx
        if i == 0:
            # 2.9µm (i=0) limit will be off the end of the detector.
            rd_end = cenx

    # ========= RECONSTRUCT THE FIRST ORDER WINGS =========
    if verbose != 0:
        print('   Stitching data and reconstructing wings...', flush=True)
    # Stitch together the interpolation and data.
    o1frame = np.zeros((dimy + 2 * pad, dimx))
    # Insert interpolated data.
    o1frame[:, rd_end:bl_end] = map2d[:, rd_end:bl_end]
    # Bluer region is known from the CLEAR exposure.
    disable = utils.verbose_to_bool(verbose)
    for col in tqdm(range(bl_end, dimx), disable=disable):
        cens = [ycens['order 1']['Y centroid'][col],
                ycens['order 2']['Y centroid'][col],
                ycens['order 3']['Y centroid'][col]]
        # Mask contamination from second and third orders, reconstruct
        # wings and add padding.
        o1frame[:, col] = reconstruct_wings256(clear[:, col], ycens=cens,
                                               pad=pad, smooth=True)
    # Add on the F277W frame to the red of the model.
    disable = utils.verbose_to_bool(verbose)
    for col in tqdm(range(rd_end), disable=disable):
        cens = [ycens['order 1']['Y centroid'][col]]
        o1frame[:, col] = reconstruct_wings256(f277w[:, col],
                                               ycens=cens,
                                               contamination=None,
                                               pad=pad, smooth=True)

        o1frame[:, col] = rescale_f277(o1frame[:, col], clear[:, col],
                                       ycen=cens[0], verbose=0)

    # Column normalize - necessary for uniformity as anchor profiles are
    # normalized whereas stitched data is not.
    o1frame /= np.nansum(o1frame, axis=0)

    return o1frame


def construct_order2(o1sub, cen, mini=750, halfwidth=12, verbose=0):
    """Reconstruct the wings of the second order after the first order spatial
    profile has been modelled and subtracted off.

    Parameters
    ----------
    o1sub : np.array
        NIRISS/SOSS data frame with the first order subtracted off.
    cen : dict
        Centroids dictionary.
    mini : int
        Minimum spectral pixel value for which a wing reconstruction will be
        attempted. For Spectral pixels <mini, the profile at mini will be used.
    halfwidth : int
        Half width in pixels of the spatial profile core.
    verbose : int
        level of verbosity.

    Returns
    -------
    new_2 : np.array
        Model of the second order spatial profile with wings reconstructed.
    """

    # Initalize new data frame and get subarray dimensions.
    dimy, dimx = np.shape(o1sub)
    new_2 = np.zeros_like(o1sub)

    # Get wavelength calibration.
    wavecal_x, wavecal_w = utils.get_wave_solution(order=2)
    # Get width polynomial coefficients.
    width_polys = utils.read_width_coefs(verbose=verbose)

    first_time = True
    maxi = dimx
    for i in range(dimx):
        wave = wavecal_w[i]
        # Skip over columns where the second order is buried in the first.
        if i < mini:
            continue
            # If the centroid is too close to the detector edge, make note of
            # the column and deal with it later
        cen2 = int(round(cen['order 2']['Y centroid'][i], 0))
        if cen2 + halfwidth > dimy:
            if i < maxi:
                maxi = i
            continue

        # Get a copy of the spatial profile, and normalize it by its max value.
        working_prof = np.copy(o1sub[:, i])
        max_val = np.nanmax(working_prof)
        working_prof /= max_val

        # Simulate the wings.
        if first_time is False:
            verbose = 0
        wing, wing2 = simulate_wings(wave, width_polys, verbose=verbose)
        first_time = False
        # Concatenate the wings onto the profile core.
        end = int(round((cen2 + 1 * halfwidth), 0))
        start = int(round((cen2 - 1 * halfwidth), 0))
        stitch = np.concatenate([wing2, working_prof[start:end], wing])
        # Rescale to native flux level.
        stitch *= max_val
        # Shift the profile back to its correct centroid position
        stitch = np.interp(np.arange(dimy), np.arange(dimy) - dimy//2 + cen2,
                           stitch)
        new_2[:, i] = stitch

    # For columns where the order 2 core is not distinguishable (due to the
    # throughput dropping near 0, or it being buried in order 1) reuse the
    # reddest reconstructed profile.
    for i in range(mini):
        anchor_prof = new_2[:, mini]
        sc = cen['order 2']['Y centroid'][mini]
        ec = cen['order 2']['Y centroid'][i]
        working_prof = np.interp(np.arange(dimy), np.arange(dimy) - sc + ec,
                                 anchor_prof)
        new_2[:, i] = working_prof

    # For columns where the centroid is off the detector, reuse the bluest
    # reconstructed profile.
    for i in range(maxi, dimx):
        anchor_prof = new_2[:, maxi - 1]
        sc = cen['order 2']['Y centroid'][maxi - 1]
        ec = cen['order 2']['Y centroid'][i]
        working_prof = np.interp(np.arange(dimy), np.arange(dimy) - sc + ec,
                                 anchor_prof)
        new_2[:, i] = working_prof

    return new_2


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


def pad_order2(order2, cen, pad):
    """Add padding to the spatial axis of an order 2 dataframe. Since order 2
    curves almost vertically at short wavelengths, we must take special care to
    properly extend the spatial profile

    Parameters
    ----------
    order2 : np.array
        A dataframe of order 2.
    cen : dict
        Centroids dictionary.
    pad : int
        Amount of padding to add to the spatial axis.

    Returns
    -------
    o2_padded : np.array
        The order 2 dataframe with the appropriate amount of padding added to
        the spatial axis.
    """

    # Initalize padded array.
    dimy, dimx = np.shape(order2)
    o2_padded = np.zeros((dimy + pad, dimx))
    o2_padded[:-pad] = order2

    # Use the shortest wavelength slice along the spatial axis as a reference
    # profile.
    anchor_prof = order2[-1]
    ii = np.where(cen['order 2']['Y centroid'] >= dimy - 1)[0][0]
    xcen_anchor = cen['order 2']['X centroid'][ii]

    # To pad the upper edge of the spatial axis, shift the reference profile
    # according to extrapolated centroids.
    for i in range(pad):
        i += 1
        shift = cen['order 2']['X centroid'][ii + i] - xcen_anchor
        working_prof = np.interp(np.arange(dimx),
                                 np.arange(dimx) + shift, anchor_prof)
        o2_padded[dimy + i - 1] = working_prof

    # Pad the lower edge with zeros.
    o2_padded = np.pad(o2_padded, ((pad, 0), (0, 0)), mode='edge')

    return o2_padded


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
        ref_cols = [5, -5]

    dimy, dimx = np.shape(frame)
    pp = np.polyfit(xcens, ycens, 5)
    xax_pad = np.arange(dimx + 2 * pad) - pad
    ycens_pad = np.polyval(pp, xax_pad)

    newframe = np.zeros((dimy, dimx + 2 * pad))
    newframe[:, pad:(dimx + pad)] = frame

    for col in range(pad):
        yax = np.arange(dimy)
        newframe[:, col] = np.interp(yax + ycens[ref_cols[0]] - ycens_pad[col],
                                     yax, frame[:, ref_cols[0]])

    for col in range(dimx + ref_cols[1] + pad, dimx + 2 * pad):
        yax = np.arange(dimy)
        newframe[:, col] = np.interp(yax + ycens[ref_cols[1]] - ycens_pad[col],
                                     yax, frame[:, ref_cols[1]])

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


def rescale_f277(f277_prof, clear_prof, ycen=71, width=50, max_iter=10, pad=0,
                 verbose=0):
    """As the F277W and CLEAR exposures will likely have different flux levels
    due to different throughputs as well as possibly a different number of
    integrations. Rescale the F277W profile to the flux level of the CLEAR
    profile such that both profiles can be used as end points for interplation.

    Parameters
    ----------
    f277_prof : np.array
        Spatial profile in the F277W fiter.
    clear_prof : np.array
        Spatial profile in the CLEAR filter.
    ycen : float
        Y-Centroid position of the F277W profile.
    width : int
        Pixel width on either end of the centroid to consider.
    max_iter : int
        Maximum number of iterations.
    pad : int
        Amount of padding included in the spatial profiles.
    verbose : int
        Level of verbosity

    Returns
    -------
    f277_rescale : np.array
        F277W profile rescaled to the flux level of the CLEAR profile.
    """

    dimy = len(f277_prof)
    start, end = int(ycen - width), int(ycen + width)
    # Iterate over different rescalings + vertical shifts to minimize the
    # Chi^2 between the F277W and CLEAR exposures.
    # Calculate starting Chi^2 value.
    chi2 = np.nansum(((f277_prof[pad:(dimy + pad)] /
                       np.nansum(f277_prof[pad:(dimy + pad)]))[start:end] -
                      (clear_prof / np.nansum(clear_prof))[start:end]) ** 2)
    # Append Chi^2 and normalize F277W profile to arrays.
    chi2_arr = [chi2]
    anchor_arr = [f277_prof / np.nansum(f277_prof)]

    niter = 0
    while niter < max_iter:
        # Subtract an offset so the floors of the two profilels match.
        offset = np.nanpercentile(clear_prof / np.nansum(clear_prof), 1) - \
                 np.nanpercentile(f277_prof / np.nansum(f277_prof), 1)

        # Normalize offset F277W.
        f277_prof = f277_prof / np.nansum(f277_prof) + offset

        # Calculate new Chi^2.
        chi2 = np.nansum(((f277_prof[pad:(dimy + pad)] /
                           np.nansum(f277_prof[pad:(dimy + pad)]))[start:end] -
                          (clear_prof / np.nansum(clear_prof))[
                          start:end]) ** 2)
        chi2_arr.append(chi2)
        anchor_arr.append(f277_prof / np.nansum(f277_prof))

        niter += 1

    # Keep the best fitting F277W profile.
    min_chi2 = np.argmin(chi2_arr)
    f277_rescale = anchor_arr[min_chi2]

    if verbose == 3:
        plotting.plot_f277_rescale(anchor_arr[0][pad:(dimy + pad)],
                                   f277_rescale[pad:(dimy + pad)],
                                   clear_prof)

    return f277_rescale


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
    stand = _chromescale(stand, 1, wavelength, 128, width_coefs)

    # Define the edges of the profile 'core'.
    ax = np.arange(256)
    ystart = int(round(256 // 2 - halfwidth, 0))
    yend = int(round(256 // 2 + halfwidth, 0))
    # Get and fit the 'right' wing.
    wing = stand[yend:]
    pp = np.polyfit(ax[yend:], np.log10(wing), 7)
    wing = 10 ** (np.polyval(pp, ax[yend:]))
    # Get and fit the 'left' wing.
    wing2 = stand[:ystart]
    pp = np.polyfit(ax[:ystart], np.log10(wing2), 7)
    wing2 = 10 ** (np.polyval(pp, ax[:ystart]))

    # Do diagnostic plot if necessary.
    if verbose == 3:
        plotting.plot_wing_simulation(stand, halfwidth, wing, wing2, ax,
                                      ystart, yend)

    return wing, wing2


if __name__ == '__main__':
    filepath = '/Users/michaelradica/transfer/IDTSOSS_clear_noisy_1_flatfieldstep.fits'
    clear_sim = fits.getdata(filepath, 1)
    error = fits.getdata(filepath, 2)
    clear_sim = np.nansum(clear_sim, axis=0)
    filepath = '/Users/michaelradica/transfer/IDTSOSS_f277_noisy_1_flatfieldstep.fits'
    f277_sim = fits.open(filepath)[1].data
    f277_sim = np.nansum(f277_sim, axis=0)

    # Add a floor level such that all pixels are positive
    floor_clear = np.nanpercentile(clear_sim, 0.1)
    clear_sim -= floor_clear
    floor_f277 = np.nanpercentile(f277_sim, 0.1)
    f277_sim -= floor_f277

    # Replace bad pixels.
    clear_sim = utils.replace_badpix(clear_sim, np.isnan(np.log10(clear_sim)))
    f277_sim = utils.replace_badpix(f277_sim, np.isnan(np.log10(f277_sim)))

    # Add back the floor level
    clear_sim += floor_clear
    f277_sim += floor_f277

    bad_pix = np.isnan(clear_sim)
    spat_prof = EmpiricalProfile(clear_sim, f277_sim, bad_pix, verbose=3)
    spat_prof.build_empirical_profile(normalize=False)
