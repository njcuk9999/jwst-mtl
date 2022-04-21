#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 9:35 2020

@author: MCR

File containing the main functions for the creation of 2D trace profiles for
the first and second diffraction orders of NIRISS/SOSS observations.
"""

from astropy.io import fits
import numpy as np
import numpy.ma as ma
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
from SOSS.dms.soss_solver import chi_squared, transform_coords
from SOSS.dms.soss_centroids import get_soss_centroids
from SOSS.extract import soss_read_refs
from SOSS.extract.empirical_trace import plotting
from SOSS.extract.empirical_trace import _calc_interp_coefs
from SOSS.extract.empirical_trace import utils

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# TODO : Get rid of local paths
# Local path to reference files.
path = '/Users/michaelradica/Documents/School/Ph.D./Research/SOSS/Extraction/Input_Files/'


def build_empirical_trace(clear, f277w, badpix_mask, subarray, pad, oversample,
                          normalize, max_iter, verbose):
    """Main procedural function for the empirical trace construction module.
    Calling this function will initialize and run all the required subroutines
    to produce an uncontaminated spatial profile for the first and second
    orders. The spatial profiles generated can include oversampling as well as
    padding in both the spatial and spectral directions.
    It is advisable to include an F277W exposure in addition to the standard
    CLEAR to improve the accuracy of both orders in the overlap region.

    Parameters
    ----------
    clear : np.array
        SOSS CLEAR exposure data frame.
    f277w : np.array, None
        SOSS exposure data frame using the F277W filter. Pass None if no F277W
        exposure is available.
    badpix_mask : np.array
        Bad pixel mask, values of True represent bad pixels. Must be the same
        shape as the CLEAR dataframe.
    subarray : str
        NIRISS SOSS subarray identifier. One of 'SUBSTRIP96', 'SUBSTRIP256', or
        'FULL'.
    pad : tuple
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions, respectively.
    oversample : int
        Oversampling factor. Oversampling will be equal in the spectral and
        spatial directions.
    normalize : bool
        if True, column normalize the final spatial profiles such that the
        flux in each column sums to one.
    max_iter : int
        Number of refinement iterations to complete.
    verbose : int
        Level of verbosity: either 3, 2, 1, or 0.
         3 - show all of progress prints, progress bars, and diagnostic plots.
         2 - show progress prints and bars.
         1 - show only progress prints.
         0 - show nothing.

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
        print('Starting the Empirical Trace Construction module.\n')

    # ========= INITIAL SETUP =========
    # TODO : Get from file header
    # Determine correct subarray dimensions.
    dimy, dimx = np.shape(clear)
    # Initialize trim variable to False unless the subarray is FULL.
    trim = False
    if subarray == 'FULL':
        # If subarray is FULL - trim down to SUBSTRIP256 and work with that.
        # The rest if the frame is zeros anyways.
        clear = clear[-256:, :]
        f277w = f277w[-256:, :]
        badpix_mask = badpix_mask[-256:, :]
        # Reset all variable to appropriate SUBSTRIP256 values.
        subarray = 'SUBSTRIP256'
        dimy, dimx = np.shape(clear)
        # Note that the detector was trimmed.
        trim = True

    # TODO : maybe find a better way to do this? Use DQ flags to find bad pix?
    # Replace bad pixels using the median of pixels in the surrounding 5x5 box.
    if verbose != 0:
        print(' Initial processing...')
        print('  Replacing bad pixels...', flush=True)
    clear = replace_badpix(clear, badpix_mask, verbose=verbose)
    if f277w is not None:
        f277w = replace_badpix(f277w, badpix_mask, verbose=verbose)

    # Get the centroid positions for both orders from the data using the
    # edgetrig method.
    if verbose != 0:
        print('  Getting trace centroids...')
    centroids = get_soss_centroids(clear, mask=badpix_mask, subarray=subarray)
    if verbose == 3:
        plotting.plot_centroid(clear, centroids)

    # ========= CONSTRUCT FIRST PASS MODELS =========
    # Build a first estimate of the first and second order spatial profiles
    # through an interpolation model in the contaminated region, and
    # wing reconstruction for each order.
    # Construct the first order profile.
    if verbose != 0:
        print(' \nConstructing first pass trace models...')
        print('  Starting the first order trace model...')
    # TODO : iterations
    # Pad the trace at this point if no further iterations are to be performed.
    if max_iter == 0:
        pad_i = pad[0]
    else:
        pad_i = 0
    o1_rough = construct_order1(clear, f277w, centroids, subarray=subarray,
                                pad=pad_i, verbose=verbose)

    # For the other subarrays, construct a second order profile.
    # TODO : I think just throughput should be enough here?
    # Rescale the first order profile to the native flux level.
    if verbose != 0:
        print('   Rescaling first order to native flux level...',
              flush=True)
    o1_rescale = rescale_model(clear, o1_rough, centroids, pad=pad_i,
                               verbose=verbose)

    # # Construct the second order profile.
    #
    # # ========= FINAL TUNING =========
    # # Pad the spectral axis.
    # if pad != 0:
    #     if verbose != 0:
    #         print(' Adding padding to the spectral axis...')
    #     o1_uncontam = pad_spectral_axis(o1_uncontam,
    #                                     centroids['order 1']['X centroid'],
    #                                     centroids['order 1']['Y centroid'],
    #                                     pad=pad[1])
    # # Even if padding is not requested, fill in the zero valued area of the
    # # frame where the order 2 trace is off of the detector.
    # edge = np.where(np.nanmedian(o2_uncontam, axis=0) == 0)[0][0] - 5
    # o2_uncontam = pad_spectral_axis(o2_uncontam,
    #                                 centroids['order 2']['X centroid'],
    #                                 centroids['order 2']['Y centroid'],
    #                                 pad=pad[1],
    #                                 ref_cols=(5, edge - dimx))
    #
    # # Plot a map of the residuals.
    # if verbose == 3:
    #     o1_unpad = o1_uncontam[pad[0]:(dimy + pad[0]),
    #                            pad[1]:(dimx+pad[1])]
    #     o2_unpad = o2_uncontam[pad[0]:(dimy + pad[0]),
    #                            pad[1]:(dimx+pad[1])]
    #     plotting.plot_trace_residuals(clear, o1_unpad, o2_unpad)
    #
    # # Add oversampling.
    # if oversample != 1:
    #     if verbose != 0:
    #         print(' Oversampling...')
    #     o1_uncontam = oversample_frame(o1_uncontam, oversample=oversample)
    #     o2_uncontam = oversample_frame(o2_uncontam, oversample=oversample)
    #
    # # Column normalize.
    # if normalize is True:
    #     o1_uncontam /= np.nansum(o1_uncontam, axis=0)
    #     o2_uncontam /= np.nansum(o2_uncontam, axis=0)
    #
    # # If the original subarray was FULL - add back the rest of the frame
    # if trim is True:
    #     # Create the FULL frame including oversampling and padding.
    #     o1_full = np.zeros(((2048 + 2 * pad[0]) * oversample,
    #                         (2048+2*pad[1]) * oversample))
    #     o2_full = np.zeros(((2048 + 2 * pad[0]) * oversample,
    #                         (2048+2*pad[1]) * oversample))
    #     # Put the uncontaminated SUBSTRIP256 frames on the FULL detector.
    #     dimy = np.shape(o1_uncontam)[0]
    #     o1_full[-dimy:, :] = o1_uncontam
    #     o2_full[-dimy:, :] = o2_uncontam
    #     o1_uncontam = o1_full
    #     o2_uncontam = o2_full
    #
    # if verbose != 0:
    #     print('\nDone.')

    #return o1_uncontam, o2_uncontam
    return o1_rescale


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

    xrange = len(profile)
    # Get the starting and ending trace widths.
    if wave_start < 2.1:
        poly_coef_s = poly_coef[1]
    else:
        poly_coef_s = poly_coef[0]
    if wave_end < 2.1:
        poly_coef_e = poly_coef[1]
    else:
        poly_coef_e = poly_coef[0]
    w_start = np.polyval(poly_coef_s, wave_start)
    w_end = np.polyval(poly_coef_e, wave_end)
    # Create a rescaled spatial axis.
    xax = np.linspace(0, round(xrange*(w_end/w_start), 0) - 1, xrange)
    # Find required offset to ensure the centroid remains at the same location.
    offset = xax[int(round(ycen, 0))] - ycen
    # Rescale the PSF by interpolating onto the new axis.
    prof_rescale = np.interp(np.arange(xrange), xax - offset, profile)
    # Ensure the total flux remains the same
    # Integrate the profile with a Trapezoidal method.
    flux_i = np.sum(profile[1:-1]) + 0.5*(profile[0] + profile[-1])
    flux_f = np.sum(prof_rescale[1:-1]) + 0.5*(prof_rescale[0] + prof_rescale[-1])
    # Rescale the new profile so the total encompassed flux remains the same.
    prof_rescale /= (flux_f/flux_i)

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
    # Open wavelength calibration file.
    # TODO : remove local paths
    wavecal = fits.getdata(path+'jwst_niriss_soss-256-ord1_trace.fits', 1)
    # Get wavelength and detector pixel calibration info.
    pp_w = np.polyfit(wavecal['Detector_Pixels'],
                      wavecal['WAVELENGTH'][::-1], 1)
    wavecal_x = np.arange(dimx)
    wavecal_w = np.polyval(pp_w, wavecal_x)
    # Determine how the trace width changes with wavelength.
    if verbose != 0:
        print('   Calibrating trace widths...')
    # TODO : replace with ref file?
    wave_polys = _fit_trace_widths(clear, pp_w, verbose=verbose)

    # ========= GET ANCHOR PROFILES =========
    if verbose != 0:
        print('   Getting anchor profiles...')
    # TODO : remove hardcoding
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
    # Remove the lambda/D scaling.
    # TODO : do this relative
    bl_anch = _chromescale(bl_anch, 2.1, 2.5, ydb + pad, wave_polys)
    # Normalize
    bl_anch /= np.nansum(bl_anch)

    # Determine the anchor profiles - red anchor.
    # If an F277W exposure is provided, only interpolate out to 2.45µm.
    # Red-wards of 2.45µm we have perfect knowledge of the order 1 trace.
    # Find the pixel position of 2.45µm.
    # TODO : this 2.45 is arbitrary, figure out a better red anchor
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
    rd_anch = _chromescale(rd_anch, 2.45, 2.5, ydr + pad, wave_polys)
    # Normalize
    rd_anch /= np.nansum(rd_anch)

    # Interpolation polynomial coefs, calculated via calc_interp_coefs
    coef_b, coef_r = utils.read_interp_coefs(verbose=verbose)

    # Since there will lkely be a different number of integrations for the
    # F277W exposure vs the CLEAR, the flux levels of the two anchors will
    # likely be different. Rescale the F277W anchor to match the CLEAR level.
    rd_anch = rescale_f277(rd_anch, clear[:, xdr], ycen=ydr, verbose=verbose)

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
        bax = np.arange(dimy + 2*pad) - ydb + ceny
        bl_anch_i = np.interp(np.arange(dimy + 2*pad), bax, bl_anch)
        rax = np.arange(len(rd_anch)) - ydr + ceny
        rd_anch_i = np.interp(np.arange(dimy + 2*pad), rax, rd_anch)
        # Construct the interpolated profile.
        prof_int = (wb_i*bl_anch_i + wr_i*rd_anch_i)
        # Re-add the lambda/D scaling.
        prof_int_cs = _chromescale(prof_int, 2.5, lbd, ceny+pad, wave_polys)
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
    # TODO : double check that stitching is working as expected
    o1frame = np.zeros((dimy + 2*pad, dimx))
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
        # TODO : wing reconstruct probably not necessary here
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


# TODO : Consider going with relative widths
def _fit_trace_widths(clear, wave_coefs, verbose=0):
    """Due to the defocusing of the SOSS PSF, the width in the spatial
    direction does not behave as if it is diffraction limited. Calculate the
    width of the spatial trace profile as a function of wavelength, and fit
    with a linear relation to allow corrections of chromatic variations.

    Parameters
    ----------
    clear : np.array
        CLEAR exposure dataframe
    wave_coefs : tuple
        Polynomial coefficients for the wavelength to spectral pixel
        calibration.
    verbose : int
        Level of verbosity.

    Returns
    -------
    wfit_b : tuple
        Polynomial coefficients for a first order fit to the uncontaminated
        trace widths.
    wfit_r : tuple
        Polynomial coefficients for a first order fit to the contaminated trace
        widths.
    """

    # Get subarray dimensions and wavelength to spectral pixel transformation.
    yax, xax = np.shape(clear)
    wax = np.polyval(wave_coefs, np.arange(xax))

    # Determine the width of the trace profile by counting the number of pixels
    # with flux values greater than half of the maximum value in the column.
    trace_widths = []
    for i in range(xax):
        # Oversample by 4 times to get better sub-pixel scale info.
        prof = np.interp(np.linspace(0, yax-1, yax*4), np.arange(yax),
                         clear[:, i])
        # Sort the flux values in the profile.
        prof_sort = np.argsort(prof)
        # To mitigate the effects of any outliers, use the median of the 5
        # highest flux values as the maximum.
        inds = prof_sort[-5:]
        maxx = np.nanmedian(prof[inds])
        # Count how many pixels have flux greater than half this value.
        above_av = np.where(prof >= maxx/2)[0]
        trace_widths.append(len(above_av)/4)

    # TODO : hardcodng of blue anchor wavelength
    # Only fit the trace widths up to the blue anchor (2.1µm) where
    # contamination from the second order begins to be a problem.
    end = np.where(wax < 2.1)[0][0]
    fit_waves_b = np.array(wax[end:])
    fit_widths_b = np.array(trace_widths[end:])
    # Rough sigma clip of huge outliers.
    fit_waves_b, fit_widths_b = utils.sigma_clip(fit_waves_b, fit_widths_b)
    # Robustly fit a straight line.
    pp_b = np.polyfit(fit_waves_b, fit_widths_b, 1)
    wfit_b = utils.robust_polyfit(fit_waves_b, fit_widths_b, pp_b)

    # TODO : treat these regions differently, or just use uncontam region?
    # Separate fit to contaminated region.
    fit_waves_r = np.array(wax[:(end+10)])
    fit_widths_r = np.array(trace_widths[:(end+10)])
    # Rough sigma clip of huge outliers.
    fit_waves_r, fit_widths_r = utils.sigma_clip(fit_waves_r, fit_widths_r)
    # Robustly fit a straight line.
    pp_r = np.polyfit(fit_waves_r, fit_widths_r, 1)
    wfit_r = utils.robust_polyfit(fit_waves_r, fit_widths_r, pp_r)

    # Plot the width calibration fit if required.
    if verbose == 3:
        plotting.plot_width_cal((fit_widths_b, fit_widths_r),
                                (fit_waves_b, fit_waves_r), (wfit_b, wfit_r))

    return wfit_r, wfit_b


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
    newdimy, newdimx = dimy*oversample, dimx*oversample

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
    xax_pad = np.arange(dimx+2*pad)-pad
    ycens_pad = np.polyval(pp, xax_pad)

    newframe = np.zeros((dimy, dimx+2*pad))
    newframe[:, pad:(dimx+pad)] = frame

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
                         verbose=0, smooth=True, **kwargs):
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
        errmsg = 'Centroids must be provided for first three orders '\
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
    prof_r, axis_r = np.log10(profile), np.arange(dimy)

    # === Outlier masking ===
    # Mask the cores of each order.
    for order, ycen in enumerate(ycens):
        if order == 0:
            start = 0
            # TODO : remove hardcoding
            end = ycen+30
        else:
            # TODO : remove hardcoding
            start = np.min([ycen-18, dimy-2])
            end = np.min([ycen+18, dimy-1])
        # Set core of each order to NaN.
        prof_r[start:end] = np.nan
    # Fit the unmasked part of the wing to determine the mean linear trend.
    inds = np.where(np.isfinite(prof_r))[0]
    pp = utils.robust_polyfit(axis_r[inds], prof_r[inds], (0, 0))
    wing_mean = pp[1]+pp[0]*axis_r
    # Calculate the standard dev of unmasked points from the mean trend.
    stddev_m = np.sqrt(np.median((prof_r[inds] - wing_mean[inds])**2))

    # === Wing fit ===
    # Get fresh contaminated wing profile.
    prof_r2 = np.log10(profile)
    # Mask first order core.
    # TODO : remove hardcoding
    prof_r2[:(ycens[0]+13)] = np.nan
    # Mask second and third orders.
    if contamination is not None:
        for order, ycen in enumerate(ycens):
            if order + 1 in contamination:
                # TODO : remove hardcoding
                start = np.max([ycen-18, 0])
                end = np.max([ycen+18, 1])
                # Set core of each order to NaN.
                prof_r2[start:end] = np.nan
    # Find all outliers that are >3-sigma deviant from the mean.
    # TODO : remove hardcoding
    start = ycens[0]+25
    inds2 = np.where(np.abs(prof_r2[start:] - wing_mean[start:]) > 3*stddev_m)
    # Mask outliers
    prof_r2[start:][inds2] = np.nan
    # Mask edge of the detector.
    prof_r2[-3:] = np.nan
    # Indices of all unmasked points in the contaminated wing.
    inds3 = np.isfinite(prof_r2)

    # Fit with a 7th order polynomial.
    # To ensure that the polynomial does not start turning up in the padded
    # region, extend the linear fit to the edge of the pad to force the fit
    # to continue decreasing.
    # TODO : better way to do this?
    # TODO : remove hardcoding
    ext_ax = np.arange(25) + np.max(axis_r[inds3]) + np.max([pad, 25])
    ext_prof = pp[1] + pp[0]*ext_ax
    # Concatenate right-hand profile with the extended linear trend.
    fit_ax = np.concatenate([axis_r[inds3], ext_ax])
    fit_prof = np.concatenate([prof_r2[inds3], ext_prof])
    # Use np.polyfit for a first estimate of the coefficients.
    pp_r0 = np.polyfit(fit_ax, fit_prof, 7)
    # Robust fit using the polyfit results as a starting point.
    pp_r = utils.robust_polyfit(fit_ax, fit_prof, pp_r0)

    # === Stitching ===
    newprof = profile*1
    # Interpolate contaminated regions.
    # TODO : honestly not sure what's going on here - just interpolate the contamination by 2nd + 3rd orders and leave rest alone
    # I think that is what's happening, but need to take a closer look.
    if contamination is not None:
        for order in contamination:
            # Interpolate for +/- 20 pixels around the trace centroid.
            start = np.max([ycens[order-1]-20, ycens[0]+14])
            if start >= dimy-1:
                # If order is off of the detector.
                continue
            end = np.min([ycens[order-1]+20, dimy-1])
            # Join interpolations to the data.
            newprof = np.concatenate([newprof[:start],
                                      10**np.polyval(pp_r, axis_r)[start:end],
                                      newprof[end:]])
    # Interpolate nans and negatives with median of surrounding pixels.
    for pixel in np.where(np.isnan(newprof))[0]:
        minp = np.max([pixel-5, 0])
        maxp = np.min([pixel+5, dimy])
        newprof[pixel] = np.nanmedian(newprof[minp:maxp])

    if smooth is True:
        # TODO : remove hardcoding
        # Replace highly deviant pixels throughout the wings.
        wing_fit = np.polyval(pp_r, axis_r[(ycens[0]+30):])
        # Calculate the standard dev of unmasked points from the wing fit.
        stddev_f = np.sqrt(np.nanmedian((np.log10(newprof[(ycens[0]+30):])
                                         - wing_fit)**2))
        # Find all outliers that are >3-sigma deviant from the mean.
        inds4 = np.where(np.abs(np.log10(newprof[(ycens[0]+30):]) - wing_fit)
                         > 3*stddev_f)
        newprof[(ycens[0]+30):][inds4] = 10**wing_fit[inds4]

    # Add padding - padding values are constant at the median of edge pixels.
    padval_r = np.median(newprof[-8:-3])
    padval_l = np.median(newprof[3:8])
    newprof = np.concatenate([np.tile(padval_l, pad+4), newprof[4:-4],
                              np.tile(padval_r, pad+4)])

    # Do diagnostic plot if requested.
    if verbose == 3:
        plotting.plot_wing_reconstruction(profile, ycens, axis_r[inds3],
                                          prof_r2[inds3], pp_r, newprof,
                                          pad, **kwargs)

    return newprof


def replace_badpix(clear, badpix_mask, fill_negatives=True, verbose=0):
    """Replace all bad pixels with the median of the pixels values of a 5x5 box
    centered on the bad pixel.

    Parameters
    ----------
    clear : np.array
        Dataframe with bad pixels.
    badpix_mask : np.array
        Boolean array with the same dimensions as clear. Values of True
        indicate a bad pixel.
    fill_negatives : bool
        If True, also interpolates all negatives values in the frame.
    verbose : int
        Level of verbosity.

    Returns
    -------
    clear_r : np.array
        Input clear frame with bad pixels interpolated.
    """

    # Get frame dimensions
    dimy, dimx = np.shape(clear)

    # Include all negative and zero pixels in the mask if necessary.
    if fill_negatives is True:
        mask = badpix_mask | (clear <= 0)
    else:
        mask = badpix_mask

    # Loop over all bad pixels.
    clear_r = clear*1
    ys, xs = np.where(mask)

    disable = utils.verbose_to_bool(verbose)
    for y, x in tqdm(zip(ys, xs), total=len(ys), disable=disable):
        # Get coordinates of pixels in the 5x5 box.
        starty = np.max([(y-2), 0])
        endy = np.min([(y+3), dimy])
        startx = np.max([0, (x-2)])
        endx = np.min([dimx, (x+3)])
        # calculate replacement value to be median of surround pixels.
        rep_val = np.nanmedian(clear[starty:endy, startx:endx])
        i = 1
        # if the median value is still bad, widen the surrounding region
        while np.isnan(rep_val) or rep_val <= 0:
            starty = np.max([(y-2-i), 0])
            endy = np.min([(y+3+i), dimy])
            startx = np.max([0, (x-2-i)])
            endx = np.min([dimx, (x+3-i)])
            rep_val = np.nanmedian(clear[starty:endy, startx:endx])
            i += 1
        # Replace bad pixel with the new value.
        clear_r[y, x] = rep_val

    return clear_r


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
        lik_args = (data[(start-pad):(end-pad), i], model[start:end, i])
        k = minimize(utils.lik, k0[i], lik_args)
        ks.append(k.x[0])

    # Rescale the column normalized model.
    ks = np.array(ks)
    model_rescale = ks*model

    return model_rescale


def rescale_f277(f277_prof, clear_prof, ycen=71, width=50, max_iter=10,
                 verbose=0):
    start, end = int(ycen - width), int(ycen + width)
    chi2 = np.nansum(((f277_prof / np.nansum(f277_prof))[start:end] - \
                      (clear_prof / np.nansum(clear_prof))[start:end]) ** 2)
    chi2_arr = [chi2]
    anchor_arr = [f277_prof / np.nansum(f277_prof)]

    niter = 0
    while niter < max_iter:
        offset = np.nanpercentile(clear_prof / np.nansum(clear_prof), 1) - \
                 np.nanpercentile(f277_prof / np.nansum(f277_prof), 1)

        f277_prof = f277_prof / np.nansum(f277_prof) + offset

        chi2 = np.nansum(((f277_prof / np.nansum(f277_prof))[start:end] - \
                          (clear_prof / np.nansum(clear_prof))[
                          start:end]) ** 2)
        chi2_arr.append(chi2)
        anchor_arr.append(f277_prof / np.nansum(f277_prof))

        niter += 1

    min_chi2 = np.argmin(chi2_arr)
    f277_rescale = anchor_arr[min_chi2]

    if verbose == 3:
        plotting.plot_f277_rescale(anchor_arr[0], f277_rescale, clear_prof)

    return f277_rescale
