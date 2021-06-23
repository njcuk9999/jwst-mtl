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
from SOSS.dms.soss_solver import _chi_squared, transform_coords
from SOSS.dms.soss_centroids import get_soss_centroids
from SOSS.extract import soss_read_refs
from SOSS.extract.empirical_trace import plotting
from SOSS.extract.empirical_trace import _calc_interp_coefs
from SOSS.extract.empirical_trace import utils

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Local path to reference files.
path = '/Users/michaelradica/Documents/School/Ph.D./Research/SOSS/Extraction/Input_Files/'


def build_empirical_trace(clear, F277W, badpix_mask, subarray, pad, oversample,
                          normalize, verbose):
    '''Main procedural function for the empirical trace construction module.
    Calling this function will initialize and run all the requred subroutines
    to produce an uncontaminated spatial profile for the first and second
    orders. The spatial profiles generated can include oversampling as well as
    padding in both the spatial and spectral directions.
    It is advisable to include an F277W exposure in addition to the standard
    CLEAR to improve the accuracy of both orders in the overlap region.

    Parameters
    ----------
    clear : np.array of float (2D)
        SOSS CLEAR exposure data frame.
    F277W : np.array of float (2D)
        SOSS exposure data frame using the F277W filter. Pass None if no F277W
        exposure is available.
    badpix_mask : np.ndarray (2D) of bool
        Bad pixel mask, values of True represent bad pixels. Must be the same
        shape as the CLEAR dataframe.
    subarray : str
        NIRISS SOSS subarray identifier. One of 'SUBSTRIP96', 'SUBSTRIP256', or
        'FULL'.
    pad : tuple
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions, repsectively.
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

    Returns
    -------
    order1_uncontam : np.ndarray (2D)
        Uncontaminated spatial profile for the first order.
    order2_uncontam : np.ndarray(2D)
        Uncontaminated spatial profile for the second order.

    Raises
    ------
    ValueError
        When the clear dimensions do not match a known subarray.
        If the bad pixel mask is not the same shape as the clear frame.
    '''

    if verbose != 0:
        print('Starting the Empirical Trace Construction module.\n')

    # ========= INITIAL SETUP =========
    # Determine correct subarray dimensions.
    dimy, dimx = np.shape(clear)
    # Initialize trim variable to False unless the subarray is FULL.
    trim = False
    if subarray == 'FULL':
        # If subarray is FULL - trim down to SUBSTRIP256 and work with that.
        # The rest if the frame is zeros anyways.
        clear = clear[-256:, :]
        F277W = F277W[-256:, :]
        badpix_mask = badpix_mask[-256:, :]
        # Reset all variable to appropriate SUSTRIP256 values.
        subarray = 'SUBSTRIP256'
        dimy, dimx = np.shape(clear)
        # Note that the detector was trimmed.
        trim = True

    # Replace bad pixels using the median of pixels in the surrounding 5x5 box.
    if verbose != 0:
        print(' Initial processing...')
        print('  Replacing bad pixels...', flush=True)
    clear = replace_badpix(clear, badpix_mask, verbose=verbose)
    if F277W is not None:
        F277W = replace_badpix(F277W, badpix_mask, verbose=verbose)

    # Get the centroid positions for both orders from the data using the
    # edgetrig method.
    if verbose != 0:
        print('  Getting trace centroids...')
    centroids = get_soss_centroids(clear, mask=badpix_mask, subarray=subarray)
    # For SUBSTRIP96, the edgetrig method cannot find the second and third
    # order centroids. Use the simple solver method instead.
    if subarray == 'SUBSTRIP96':
        centroids = get_substrip96_centroids(centroids)
    # Overplot the data centroids on the CLEAR exposure if desired.
    if verbose == 3:
        plotting._plot_centroid(clear, centroids)

    # ========= CONSTRUCT FIRST PASS MODELS =========
    # Build a first estimate of the first and second order spatial profiles
    # through an interpolation model in the conatminated region, and
    # wing reconstruction for each order.
    # Construct the first order profile.
    if verbose != 0:
        print(' \nConstructing first pass trace models...')
        print('  Starting the first order trace model...')
    order1_1 = construct_order1(clear, F277W, centroids, pad=0,
                                verbose=verbose, subarray=subarray)
    # If the data is SUBSTRIP96, this is as much as we can do - save the first
    # pass first order trace profile.
    if subarray == 'SUBSTRIP96':
        order1_uncontam = order1_1
        order2_uncontam = None

    # For the other subarrays, construct a second order model and further
    #  improve the first order.
    else:
        # Rescale the first order profile to the native flux level.
        if verbose != 0:
            print('   Rescaling first order to native flux level...',
                  flush=True)
        order1_rescale_1 = rescale_model(clear, order1_1, centroids,
                                         verbose=verbose)

        # Construct the second order profile.
        if verbose != 0:
            print('  Building the second order trace model...')
        order2_1 = construct_order2(clear, order1_rescale_1, centroids,
                                    verbose=verbose)
        if verbose != 0:
            print(' First pass models complete.')
        return order1_rescale_1, order2_1

        # ========= REFINE FIRST PASS MODELS =========
        # Iterate with first estimate solutions to refine cores of each order.
        # Subtract the second order model use the residuals to get a better
        # estimate of the first order core in the contaminated region, and
        # vice-versa for the second order.
        if verbose != 0:
            print(' \nStarting spatial profile refinement...')
            print('  Refining the first order...', flush=True)
        # Refine the first order.
        order1_uncontam = refine_order1(clear, order2_1, centroids, pad[0],
                                        verbose=verbose)

        # Refine the second order.
        if verbose != 0:
            print('  Refining the second order...')
        order2_uncontam = construct_order2(clear,
                                           order1_uncontam[pad[0]:(dimy+pad[0])],
                                           centroids, verbose=verbose,
                                           pad=pad[0])

        # ========= FINAL TUNING =========
        # Pad the spectral axis.
        if pad != 0:
            if verbose != 0:
                print(' Adding padding to the spectral axis...')
            order1_uncontam = pad_spectral_axis(order1_uncontam,
                                                centroids['order 1']['X centroid'],
                                                centroids['order 1']['Y centroid'],
                                                pad=pad[1])
        # Even if padding is not requested, fill in the zero valued area of the
        # frame where the order 2 trace is off of the detector.
        edge = np.where(np.nanmedian(order2_uncontam, axis=0) == 0)[0][0] - 5
        order2_uncontam = pad_spectral_axis(order2_uncontam,
                                            centroids['order 2']['X centroid'],
                                            centroids['order 2']['Y centroid'],
                                            pad=pad[1], ref_col=(5, edge-dimx))

        # Plot a map of the residuals.
        if verbose == 3:
            o1_unpad = order1_uncontam[pad[0]:(dimy+pad[0]),
                                       pad[1]:(dimx+pad[1])]
            o2_unpad = order2_uncontam[pad[0]:(dimy+pad[0]),
                                       pad[1]:(dimx+pad[1])]
            plotting._plot_trace_residuals(clear, o1_unpad, o2_unpad)

    # Add oversampling.
    if oversample != 1:
        if verbose != 0:
            print(' Oversampling...')
        order1_uncontam = oversample_frame(order1_uncontam,
                                           oversample=oversample)
        order2_uncontam = oversample_frame(order2_uncontam,
                                           oversample=oversample)

    # Column normalize.
    if normalize is True:
        order1_uncontam /= np.nansum(order1_uncontam, axis=0)
        if subarray != 'SUBSTRIP96':
            order2_uncontam /= np.nansum(order2_uncontam, axis=0)

    # If the original subarray was FULL - add back the rest of the frame
    if trim is True:
        subarray = 'FULL'
        # Create the FULL frame including oversampling and padding.
        order1_full = np.zeros(((2048+2*pad[0])*oversample,
                                (2048+2*pad[1])*oversample))
        order2_full = np.zeros(((2048+2*pad[0])*oversample,
                                (2048+2*pad[1])*oversample))
        # Put the uncontaminated SUBSTRIP256 frames on the FULL detector.
        dimy = np.shape(order1_uncontam)[0]
        order1_full[-dimy:, :] = order1_uncontam
        order2_full[-dimy:, :] = order2_uncontam
        order1_uncontam = order1_full
        order2_uncontam = order2_full

    if verbose != 0:
        print('\nDone.')

    return order1_uncontam, order2_uncontam


def _chromescale(profile, wave_start, wave_end, ycen, poly_coef):
    '''Correct chromatic variations in the PSF along the spatial direction
    using the polynomial relationshop derived in _fit_trace_widths.

    Parameters
    ----------
    profile : np.array of float
        1D PSF profile to be rescaled.
    wave_start : float
        Starting wavelength.
    wave_end : float
        Wavelength after rescaling.
    ycen : float
        Y-pixel position of the order 1 trace centroid.
    poly_coef : tuple of float
        1st order polynomial coefficients describing the variation of the trace
        width with wavelength, e.g. as output by _fit_trace_widths.

    Returns
    -------
    prof_recsale : np.array of float
        Rescaled 1D PSF profile.
    '''

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


def construct_order1(clear, F277, ycens, subarray, pad=0, verbose=0):
    '''This creates the full order 1 trace profile model. The region
    contaminated by the second order is interpolated from the CLEAR and F277W
    exposures, or just from the CLEAR exposure and a standard red anchor if
    no F277W exposure is available.
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
    clear : np.array of float (2D)
        NIRISS SOSS CLEAR exposure dataframe.
    F277 : np.array of float (2D)
        NIRISS SOSS F277W filter exposure dataframe. If no F277W exposure
        is available, pass None.
    ycens : dict
        Dictionary of Y-coordinates for the trace centroids of the first three
        diffraction orders, ie. as returned by get_contam_centroids.
    subarray : str
        Subarray identifier. One of "SUBSTRIP96", "SUBSTRIP256", or "FULL".
    pad : int
        Number of pixels of padding to add on both ends of the spatial axis.
    verbose : int
        Level of verbosity.

    Returns
    -------
    o1frame : np.array of float (2D)
        Interpolated order 1 trace model with padding.
    '''

    # Get subarray dimensions
    dimx = 2048
    if subarray == 'SUBSTRIP96':
        dimy = 96
        # Get uncontaminated wing for wing reconstruction.
        goodwing = get_goodwing(clear, ycens)
    elif subarray == 'SUBSTRIP256':
        dimy = 256
    elif subarray == 'FULL':
        dimy = 2048

    # ========= INITIAL SETUP =========
    # Open wavelength calibration file.
    wavecal = fits.getdata(path+'jwst_niriss_soss-256-ord1_trace.fits', 1)
    # Get wavelength and detector pixel calibration info.
    pp_w = np.polyfit(wavecal['Detector_Pixels'],
                      wavecal['WAVELENGTH'][::-1], 1)
    wavecal_x = np.arange(dimx)
    wavecal_w = np.polyval(pp_w, wavecal_x)
    # Determine how the trace width changes with wavelength.
    if verbose != 0:
        print('   Calibrating trace widths...')
    wave_polys = _fit_trace_widths(clear, pp_w, verbose=verbose)

    # ========= GET ANCHOR PROFILES =========
    if verbose != 0:
        print('   Getting anchor profiles...')
    # Determine the anchor profiles - blue anchor.
    # Find the pixel position of 2.1µm.
    i_b = np.where(wavecal_w >= 2.1)[0][-1]
    xdb = int(wavecal_x[i_b])
    ydb = ycens['order 1']['Y centroid'][xdb]
    # Extract the 2.1µm anchor profile from the data - take median profile of
    # neighbouring 5 columns to mitigate effects of outliers.
    Banch = np.median(clear[:, (xdb-2):(xdb+2)], axis=1)
    if subarray == 'SUBSTRIP96':
        Banch = reconstruct_wings96(Banch, ydb, goodwing=goodwing,
                                    verbose=verbose, contamination=True,
                                    pad=pad, **{'text': 'Blue anchor'})
    else:
        # Mask second and third order, reconstruct wing structure and pad.
        cens = [ycens['order 1']['Y centroid'][xdb],
                ycens['order 2']['Y centroid'][xdb],
                ycens['order 3']['Y centroid'][xdb]]
        Banch = reconstruct_wings256(Banch, ycens=cens, contamination=[2, 3],
                                     pad=pad, verbose=verbose, smooth=True,
                                     **{'text': 'Blue anchor'})
    # Remove the lambda/D scaling.
    Banch = _chromescale(Banch, 2.1, 2.5, ydb+pad, wave_polys)
    # Normalize
    Banch /= np.nansum(Banch)

    # Determine the anchor profiles - red anchor.
    if F277 is not None:
        # If an F277W exposure is provided, only interpolate out to 2.45µm.
        # Redwards of 2.45µm we have perfect knowledge of the order 1 trace.
        # Find the pixel position of 2.45µm.
        i_r = np.where(wavecal_w >= 2.45)[0][-1]
        xdr = int(wavecal_x[i_r])
        ydr = ycens['order 1']['Y centroid'][xdr]

        # Extract and rescale the 2.45µm profile - take median of neighbouring
        # five columns to mitigate effects of outliers.
        Ranch = np.median(F277[:, (xdr-2):(xdr+2)], axis=1)
        # Reconstruct wing structure and pad.
        if subarray == 'SUBSTRIP96':
            Ranch = reconstruct_wings96(Ranch, ydr, verbose=verbose, pad=pad,
                                        **{'text': 'Red anchor'})
        else:
            cens = [ycens['order 1']['Y centroid'][xdr]]
            Ranch = reconstruct_wings256(Ranch, ycens=cens,
                                         contamination=None, pad=pad,
                                         verbose=verbose, smooth=True,
                                         **{'text': 'Red anchor'})
        Ranch = _chromescale(Ranch, 2.45, 2.5, ydr+pad, wave_polys)
        # Normalize
        Ranch /= np.nansum(Ranch)

        # Interpolation polynomial coeffs, calculated via calc_interp_coefs
        coef_b, coef_r = utils._read_interp_coefs(F277W=True, verbose=verbose)
    else:
        # If no F277W exposure is provided, interpolate out to 2.9µm.
        # Generate a simulated 2.9µm PSF.
        stand = _calc_interp_coefs.loicpsf([2.9*1e-6], save_to_disk=False,
                                           oversampling=1, pixel=256,
                                           verbose=False)[0][0].data
        # Extract the spatial profile.
        Ranch = np.sum(stand[124:132, :], axis=0)
        # Extend y centroids and wavelengths off of the detector.
        pp_w = np.polyfit(wavecal_x, wavecal_w, 1)
        pp_y = np.polyfit(ycens['order 1']['X centroid'],
                          ycens['order 1']['Y centroid'], 9)
        xpix = np.arange(dimx) - 250
        wave_ext = np.polyval(pp_w, xpix)
        ycen_ext = np.polyval(pp_y, xpix)
        # Find position of 2.9µm on extended detector.
        i_r = np.where(wave_ext >= 2.9)[0][-1]
        xdr = int(round(xpix[i_r], 0))
        ydr = ycen_ext[i_r]
        # Interpolate the WebbPSF generated profile to the correct location.
        Ranch = np.interp(np.arange(256), np.arange(256)-128+ydr, Ranch)
        # Reconstruct wing structure and pad.
        Ranch = reconstruct_wings256(Ranch, ycens=[ydr],
                                     contamination=None, pad=pad,
                                     verbose=verbose, smooth=True,
                                     **{'text': 'Red anchor'})
        # Rescale to remove chromatic effects.
        Ranch = _chromescale(Ranch, 2.9, 2.5, ydr+pad, wave_polys)
        # Normalize
        Ranch /= np.nansum(Ranch)
        # Interpolation polynomial coeffs, calculated via calc_interp_coefs
        coef_b, coef_r = utils._read_interp_coefs(F277W=False, verbose=verbose)
        # Pixel coords at which to start the interpolation.
        xdr = 0

    # ========= INTERPOLATE THE CONTAMINATED REGION =========
    # Create the interpolated order 1 PSF.
    map2D = np.zeros((dimy+2*pad, dimx))*np.nan
    # Get centroid pixel coordinates and wavelengths for interpolation region.
    cenx_d = np.arange(xdb - xdr).astype(int) + xdr
    ceny_d = ycens['order 1']['Y centroid'][cenx_d]
    lmbda = wavecal_w[cenx_d]

    # Create an interpolated 1D PSF at each required position.
    if verbose != 0:
        print('   Interpolating trace...', flush=True)
    disable = utils._verbose_to_bool(verbose)
    for i, vals in tqdm(enumerate(zip(cenx_d, ceny_d, lmbda)),
                        total=len(lmbda), disable=disable):
        cenx, ceny, lbd = int(round(vals[0], 0)), vals[1], vals[2]
        # Evaluate the interpolation polynomials at the current wavelength.
        wb_i = np.polyval(coef_b, lbd)
        wr_i = np.polyval(coef_r, lbd)
        # Recenter the profile of both anchors on the correct Y-centroid.
        bax = np.arange(dimy+2*pad)-ydb+ceny
        Banch_i = np.interp(np.arange(dimy+2*pad), bax, Banch)
        rax = np.arange(len(Ranch))-ydr+ceny
        Ranch_i = np.interp(np.arange(dimy+2*pad), rax, Ranch)
        # Construct the interpolated profile.
        prof_int = (wb_i * Banch_i + wr_i * Ranch_i)
        # Re-add the lambda/D scaling.
        prof_int_cs = _chromescale(prof_int, 2.5, lbd, ceny+pad, wave_polys)
        # Put the interpolated profile on the detector.
        map2D[:, cenx] = prof_int_cs

        # Note detector coordinates of the edges of the interpolated region.
        bend = cenx
        if i == 0:
            # 2.9µm (i=0) limit will be off the end of the detector.
            rend = cenx

    # ========= RECONSTRUCT THE FIRST ORDER WINGS =========
    if verbose != 0:
        print('   Stitching data and reconstructing wings...', flush=True)
    # Stitch together the interpolation and data.
    o1frame = np.zeros((dimy+2*pad, dimx))
    # Insert interpolated data.
    o1frame[:, rend:bend] = map2D[:, rend:bend]
    # Bluer region is known from the CLEAR exposure.
    disable = utils._verbose_to_bool(verbose)
    for col in tqdm(range(bend, dimx), disable=disable):
        if subarray == 'SUBSTRIP96':
            cen = ycens['order 1']['Y centroid'][col]
            try:
                if ycens['order 2']['Y centroid'][col] > 110:
                    goodwing_loc = None
                    contamination = False
                else:
                    goodwing_loc = goodwing
                    contamination = True
            except IndexError:
                goodwing_loc = None
                contamination = False
            o1frame[:, col] = reconstruct_wings96(clear[:, col], ycen=cen,
                                                  contamination=contamination,
                                                  goodwing=goodwing_loc,
                                                  pad=pad)

        else:
            cens = [ycens['order 1']['Y centroid'][col],
                    ycens['order 2']['Y centroid'][col],
                    ycens['order 3']['Y centroid'][col]]
            # Mask contamination from second and third orders, reconstruct
            # wings and add padding.
            o1frame[:, col] = reconstruct_wings256(clear[:, col], ycens=cens,
                                                   pad=pad, smooth=True)
    if F277 is not None:
        # Add on the F277W frame to the red of the model.
        disable = utils._verbose_to_bool(verbose)
        for col in tqdm(range(rend), disable=disable):
            # Reconstruct wing structure and pad.
            if subarray == 'SUBSTRIP96':
                cen = ycens['order 1']['Y centroid'][col]
                o1frame[:, col] = reconstruct_wings96(F277[:, col],
                                                      ycen=cen, pad=pad)
            else:
                cens = [ycens['order 1']['Y centroid'][col]]
                o1frame[:, col] = reconstruct_wings256(F277[:, col],
                                                       ycens=cens,
                                                       contamination=None,
                                                       pad=pad, smooth=True)

    # Column normalize - necessary for uniformity as anchor profiles are
    # normalized whereas stitched data is not.
    o1frame /= np.nansum(o1frame, axis=0)

    return o1frame


def construct_order2(clear, order1_rescale, ycens, pad=0, verbose=0):
    '''This creates the full order 2 trace profile model. For wavelengths of
    overlap between orders one and two, use the first order wings to correct
    the oversubtracted second order wings. For wavelengths where there is no
    overlap between the orders, use the wings of the closest neighbouring first
    order wavelength. In cases where the second order core is also
    oversubtracted, use the first order core to correct this as well. This
    method implicitly assumes that the spatial profile is determined solely by
    the optics, and is the same for all orders at a given wavelength.

    Parameters
    ----------
    clear : np.array of float (2D)
        NIRISS SOSS CLEAR exposure dataframe.
    order1_rescale : np.array of float (2D)
        Uncontaminated order 1 trace profile.
    ycens : dict
        Dictionary of Y-coordinates for the trace centroids of the first three
        diffraction orders, ie. as returned by get_contam_centroids.
    pad : int
        Number of pixels of padding to add on both ends of the spatial axis.
    verbose : int
        Level of verbosity.

    Returns
    -------
    o2frame : np.ndarray (2D)
        Uncontaminated second order trace profile at the native flux level.
    '''

    # ========= INITIAL SETUP =========
    # Get wavelength and detector pixel calibration info.
    wavecal_o1 = fits.getdata(path+'jwst_niriss_soss-256-ord1_trace.fits', 1)
    wavecal_o2 = fits.getdata(path+'jwst_niriss_soss-256-ord2_trace.fits', 1)
    pp_p = np.polyfit(wavecal_o1['WAVELENGTH'][::-1],
                      wavecal_o1['Detector_Pixels'], 1)
    pp_w2 = np.polyfit(wavecal_o2['Detector_Pixels'],
                       wavecal_o2['WAVELENGTH'][::-1], 1)
    # Get throughput information for each order.
    ttab_file = soss_read_refs.RefTraceTable()
    thpt_o1 = ttab_file('THROUGHPUT', subarray='SUBSTRIP256', order=1)
    thpt_o2 = ttab_file('THROUGHPUT', subarray='SUBSTRIP256', order=2)

    # Useful storage arrays
    ks, pixs = [], []
    notdone = []
    # First estimate of second order is the residuals from subtraction of
    # first order model from original data.
    sub = clear - order1_rescale
    dimy, dimx = np.shape(order1_rescale)
    o2frame = np.zeros((dimy, dimx))

    # ========= RECONSTRUCT SECOND ORDER WINGS =========
    # Determine a coefficient to rescale the first order profile to the flux
    # level of the second order at the same wavelength. Since the
    # wavelengths are the same, we can assume that the spatial profiles are
    # also the same. Thus, use the first order wings to reconstruct the
    # second order wings.
    if verbose != 0:
        print('   Reconstructing oversubtracted wings...', flush=True)
    disable = utils._verbose_to_bool(verbose)
    for o2pix in tqdm(range(dimx), disable=disable):
        # Get the wavelength for each column in order 2.
        o2wave = np.polyval(pp_w2, o2pix)
        # Find the corresponding column in order 1.
        o1pix = int(round(np.polyval(pp_p, o2wave), 0))

        # === FIND CORRECT COLUMNS ===
        if o1pix < dimx - 5:
            # Region where wavelengths in order 1 and order 2 overlap.
            # Get throughput for order 1.
            thpt_1i = np.where(thpt_o1[0] >= o2wave)[0][0]
            ycen_o1i = int(round(ycens['order 1']['Y centroid'][o1pix], 0))
            # Save current order 1 info for special cases below.
            # Since increasing pixel position moves from red to blue, always
            # save the bluest order 1 profile to reuse when lambda<0.8µm.
            o1pix_r = o1pix
            thpt_1i_r = thpt_1i
            marker = True
        else:
            # For region where lambda<0.8µm, there is no profile in order 1.
            # We will reuse the 'bluest' order 1 profile from the above for
            # this region, assuming that the spatial profile does not vary too
            # drastically with wavelength.
            thpt_1i = thpt_1i_r
            o1pix = o1pix_r
            ycen_o1i = int(round(ycens['order 1']['Y centroid'][o1pix_r], 0))
            marker = False
        # Get trace centroid and throughput for the second order.
        thpt_2i = np.where(thpt_o2[0] >= o2wave)[0][0]
        ycen_o2i = int(round(ycens['order 2']['Y centroid'][o2pix], 0))
        # If the centroid is off of the detector, skip the column.
        if ycen_o2i > dimy:
            continue

        # === DETERMINE O1->O2 SCALING ===
        # Find coefficient to scale order 1 to the flux level of order 2.
        # Initial guess for scaling coefficient is ratio of throughputs.
        # Won't be exactly correct due to different spectral resolutions.
        k0 = thpt_2i / thpt_1i
        # Use region +/- 13 pixels around centroid to calculate scaling.
        max1, max2 = np.min([ycen_o1i+14, dimy]), np.min([ycen_o2i+14, dimy])
        min1, min2 = np.max([ycen_o1i-13, 0]), np.max([ycen_o2i-13, 0])
        # Ensure that the order 1 and order 2 arrays are the same size
        if max1 - ycen_o1i != max2 - ycen_o2i:
            mindif = np.min([max1 - ycen_o1i, max2 - ycen_o2i])
            max1, max2 = ycen_o1i + mindif, ycen_o2i + mindif
        if ycen_o1i - min1 != ycen_o2i - min2:
            mindif = np.min([ycen_o1i - min1, ycen_o2i - min2])
            min1, min2 = ycen_o1i - mindif, ycen_o2i - mindif
        # Determine optimal scaling coefficient.
        k = minimize(utils._lik, k0, (sub[min2:max2, o2pix],
                     order1_rescale[min1:max1, o1pix])).x
        if k <= 0:
            notdone.append(o2pix)
            continue
        # Rescale the first order profile to the flux level of order 2.
        o1prof = order1_rescale[:, o1pix]*k
        if marker is True:
            ks.append(k[0])
            pixs.append(o2pix)

        # === RECONSTRUCT SPATIAL PROFILE ===
        # Replace any oversubtracted pixels in o2 with o1.
        inds = np.isnan(np.log10(sub[min2:max2, o2pix]))
        sub[min2:max2, o2pix][inds] = o1prof[min1:max1][inds]
        # Stitch together o2 core with o1 wings.
        newprof = np.concatenate([o1prof[max1:][::-1],
                                  sub[min2:max2, o2pix], o1prof[max1:]])

        # Put the spatial profile on the detector. The order 2 profile is
        # already at the detector scale.
        o2_ycen = ycens['order 2']['Y centroid'][o2pix]
        oldax = np.arange(len(newprof)) - len(newprof)/2 + o2_ycen
        # Find position where 'right' wing is off the detector.
        end = np.where(oldax < dimy)[0][-1]
        ext = 0
        # Find position where 'left' wing is off the detector.
        try:
            start = np.where(oldax < 0)[0][-1]
        except IndexError:
            # If it is never off, repeat the end value to make up space.
            start = 0
            ext = dimy - (end - start)
            val = np.nanmedian(newprof[:5])
            newprof = np.concatenate([np.tile(val, ext), newprof])
        # Put the profile on the detector.
        o2frame[:, o2pix] = newprof[start:(end+ext)]

    # ========= CORRECT OVERSUBTRACTED COLUMNS =========
    # Deal with notdone columns: columns who's scaling coef was <= 0 due to
    # oversubtraction of the first order.
    pixs, ks = np.array(pixs), np.array(ks)
    # Rough sigma clip of huge outliers.
    pixs, ks = utils._sigma_clip(pixs, ks)
    # Fit a polynomial to all positive scaling coefficients, and use the fit to
    # interpolate the correct scaling for notdone columns.
    pp_k = np.polyfit(pixs, ks, 6)
    pp_k = utils._robust_polyfit(pixs, ks, pp_k)
    # Plot the results if necessary.
    if verbose == 3:
        plotting._plot_scaling_coefs(pixs, ks, pp_k)

    if verbose != 0 and len(notdone) != 0:
        print('   Dealing with oversubtracted cores...')
    for o2pix in notdone:
        # Get the order 2 wavelength and corresponding order 1 column.
        o2wave = np.polyval(pp_w2, o2pix)
        o1pix = int(round(np.polyval(pp_p, o2wave), 0))
        # If order 1 pixel is off of the detector, use trailing blue profile.
        if o1pix >= 2048:
            o1pix = o1pix_r
        # Interpolate the appropriate scaling coefficient.
        k = np.polyval(pp_k, o2pix)
        # Rescale the order 1 profile to the flux level of order 2.
        o1prof = order1_rescale[:, o1pix]*k
        # Get bounds of the order 1 trace core.
        o1_ycen = ycens['order 1']['Y centroid'][o1pix]
        min1 = np.max([int(round(o1_ycen, 0)) - 13, 0])
        max1 = np.min([int(round(o1_ycen, 0)) + 14, dimy])
        # Use the first order core as well as the wings to reconstruct a first
        # guess of the second order profile.
        newprof = np.concatenate([o1prof[max1:][::-1], o1prof[min1:]])

        # Put the spatial profile on the detector as before.
        o2_ycen = ycens['order 2']['Y centroid'][o2pix]
        oldax = np.arange(len(newprof)) - len(newprof)/2 + o2_ycen
        end = np.where(oldax < dimy)[0][-1]
        ext = 0
        try:
            start = np.where(oldax < 0)[0][-1]
        except IndexError:
            start = 0
            ext = dimy - (end - start)
            val = np.nanmedian(newprof[:5])
            newprof = np.concatenate([np.tile(val, ext), newprof])
        o2frame[:, o2pix] = newprof[start:(end+ext)]

    # ========= SMOOTH DISCONTINUITIES =========
    # Smooth over discontinuities (streaks and oversubtractions) in both the
    # spatial and spectral directions.
    if verbose != 0:
        print('   Smoothing...', flush=True)
    o2frame = smooth_spec_discont(o2frame, verbose=verbose)
    o2frame = smooth_spat_discont(o2frame, ycens)

    # ========= ADD PADDING =========
    # Add padding to the spatial axis by repeating the median of the 5 edge
    # pixels for each column.
    if pad != 0:
        o2frame_pad = np.zeros((dimy+2*pad, dimx))
        padded = np.repeat(np.nanmedian(o2frame[-5:, :, np.newaxis], axis=0),
                           pad, axis=1).T
        o2frame_pad[-pad:] = padded
        padded = np.repeat(np.nanmedian(o2frame[:5, :, np.newaxis], axis=0),
                           pad, axis=1).T
        o2frame_pad[:pad] = padded
        o2frame_pad[pad:-pad] = o2frame
        o2frame = o2frame_pad

    return o2frame


def _fit_trace_widths(clear, wave_coefs, verbose=0):
    '''Due to the defocusing of the SOSS PSF, the width in the spatial
    direction does not behave as if it is diffraction limited. Calculate the
    width of the spatial trace profile as a function of wavelength, and fit
    with a linear relation to allow corrections of chromatic variations.

    Parameters
    ----------
    clear : np.ndarray
        CLEAR exposure dataframe
    wave_coefs : tuple of float
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
    '''

    # Get subarray diemsnions and wavelength to spectral pixel transformation.
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
        # Count how many pixels have flux greater than twice this value.
        above_av = np.where(prof >= maxx/2)[0]
        trace_widths.append(len(above_av)/4)

    # Only fit the trace widths up to the blue anchor (2.1µm) where
    # contamination from the second order begins to be a problem.
    end = np.where(wax < 2.1)[0][0]
    fit_waves_b = np.array(wax[end:])
    fit_widths_b = np.array(trace_widths[end:])
    # Reject clear outliers (>5sigma)
    pp_b = np.polyfit(fit_waves_b, fit_widths_b, 1)
    mod_widths = np.polyval(pp_b, fit_waves_b)
    stddev = np.median(np.sqrt((fit_widths_b - mod_widths)**2))
    inds = np.where(np.abs(fit_widths_b - mod_widths) < 5*stddev)
    # Robustly fit a straight line.
    pp_b = np.polyfit(fit_waves_b[inds], fit_widths_b[inds], 1)
    wfit_b = utils._robust_polyfit(fit_waves_b[inds], fit_widths_b[inds], pp_b)

    # Seperate fit to contaminated region.
    fit_waves_r = np.array(wax[:(end+10)])
    fit_widths_r = np.array(trace_widths[:(end+10)])
    # Reject clear outliers (>5sigma)
    pp_r = np.polyfit(fit_waves_r, fit_widths_r, 1)
    mod_widths = np.polyval(pp_r, fit_waves_r)
    stddev = np.median(np.sqrt((fit_widths_r - mod_widths)**2))
    inds = np.where(np.abs(fit_widths_r - mod_widths) < 5*stddev)
    # Robustly fit a straight line.
    pp_r = np.polyfit(fit_waves_r[inds], fit_widths_r[inds], 1)
    wfit_r = utils._robust_polyfit(fit_waves_r[inds], fit_widths_r[inds], pp_r)

    # Plot the width calibration fit if required.
    if verbose == 3:
        plotting._plot_width_cal(wax, trace_widths, (fit_waves_b, fit_waves_r),
                                 (wfit_b, wfit_r))

    return wfit_r, wfit_b


def get_goodwing(clear96, centroids):
    '''Obtain an uncontaminated wing of the first order spatial profile to use
    as a reference wing for SUBSTRIP96 wing reconstruction.

    Parameters
    ----------
    clear96 : np.ndarray (2D)
        SUBSTRIP96 2D trace profile dataframe.
    centroids : dict
        Centroids dictionary.

    Returns
    -------
    goodwing : np.ndarray
        Uncontaminated first order wing.
    '''

    # Find the column where the first order is lowest on the detector. This
    # ensures that we will never have to extend the goodwing.
    ymin = np.min(centroids['order 1']['Y centroid'])
    ind = np.where(centroids['order 1']['Y centroid'] == ymin)[0][0]
    # Ensure that the second order core is off the detector so there is no
    # contamination.
    while centroids['order 2']['Y centroid'][ind] < 100:
        ind += 1

    # Extract the right wing of the chosen column.
    ycen = int(round(centroids['order 1']['Y centroid'][ind], 0))
    goodwing = np.nanmedian(clear96[(ycen+13):, (ind-2):(ind+2)], axis=1)

    return goodwing


def get_substrip96_centroids(centroids):
    '''For the SUBSTRIP96 subarray, the edgetrig centroiding method cannot
    locate the centroids for orders 2 and 3 as they are not on the detector.
    This function estimates the centroid positions for these orders by
    comparing the edgetrig first order centroids to the centroids in the trace
    table reference file. Analagous to the simple solver, the necessary
    rotation and x/y offsets are calculated to transform the reference first
    order centroids to match the data. All orders are assumed to transform
    rigidly, such that applying the transformation to the reference second, or
    third order centroids gives an estimate of the data centroids.

    Parameters
    ----------
    centroids : dict
        Centroids dictionary, as returned by get_soss_centroids, containing
        only information for the first order.

    Returns
    -------
    centroids : dict
        Centroids dictionary, with second and third order centroids appended.
    '''

    # Get first order centroids from the data.
    xcen_dat = centroids['order 1']['X centroid']
    ycen_dat = centroids['order 1']['Y centroid']
    # Get first order centroids in the trace table reference file.
    ttab_file = soss_read_refs.RefTraceTable()
    xcen_ref = ttab_file('X', subarray='SUBSTRIP96')[1]
    # Extend centroids beyond edges of the subarray for more accurate fitting.
    inds = np.where((xcen_ref >= -50) & (xcen_ref < 2098))
    xcen_ref = xcen_ref[inds]
    ycen_ref = ttab_file('Y', subarray='SUBSTRIP96')[1][inds]

    # Fit reference file centroids to the data to determine necessary rotation
    # and offsets.
    guess_params = (0.15, 1, 1)
    lik_args = (xcen_ref, ycen_ref, xcen_dat, ycen_dat)
    fit = minimize(_chi_squared, guess_params, lik_args).x
    rot_ang, x_shift, y_shift = fit

    # Transform centroids to detector frame for orders 2 and 3.
    for order in [2, 3]:
        # Get centroids from the reference file.
        xcen_ref = ttab_file('X', subarray='SUBSTRIP96', order=order)[1]
        # Extend centroids beyond edges of the subarray for more accurate
        # fitting.
        inds = np.where((xcen_ref >= -50) & (xcen_ref < 2098))
        xcen_ref = xcen_ref[inds]
        ycen_ref = ttab_file('Y', subarray='SUBSTRIP96', order=2)[1][inds]
        # Transform reference centroids to the data frame.
        rot_x, rot_y = transform_coords(rot_ang, x_shift, y_shift, xcen_ref,
                                        ycen_ref)
        # Ensure there is a y-centroid for every x-pixel (even if they extend
        # off of the detector).
        rot_y = np.interp(np.arange(2048), rot_x[::-1], rot_y[::-1])
        # Add the transformed centroids to the centroid dict.
        tmp = {}
        tmp['X centroid'], tmp['Y centroid'] = np.arange(2048), rot_y
        centroids['order {}'.format(order)] = tmp

    return centroids


def oversample_frame(frame, oversample=1):
    '''Oversample a dataframe by a specified amount. Oversampling is currently
    implemented seperately in each dimension, that is, the spatial axis is
    oversampled first, followed by the spectral direction.

    Parameters
    ----------
    frame : np.array (2D)
        Dataframe to be oversampled.
    oversample : int
        Oversampling factor to apply to each axis.

    Returns
    -------
    osframe : np.array (2D)
        Input dataframe with each axis oversampled by the desired amount.
    '''

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


def pad_spectral_axis(frame, xcens, ycens, pad=0, ref_col=(5, -5)):
    '''Add padding to the spectral axis by interpolating the corresponding
    edge profile onto a set of extrapolated centroids.

    Parameters
    ----------
    frame : np.array (2D)
        Data frame.
    xcens : list
        X-coordinates of the trace centroids.
    ycens : list
        Y-coordinates of the trace centroids.
    pad : int
        Amount of padding to add along either end of the spectral axis (in
        pixels).

    Returns
    -------
    newframe : np.array (2D)
        Data frame with padding on the spectral axis.
    '''

    dimy, dimx = np.shape(frame)
    pp = np.polyfit(xcens, ycens, 5)
    xax_pad = np.arange(dimx+2*pad)-pad
    ycens_pad = np.polyval(pp, xax_pad)

    newframe = np.zeros((dimy, dimx+2*pad))
    newframe[:, (pad):(dimx+pad)] = frame

    for col in range(pad):
        yax = np.arange(dimy)
        newframe[:, col] = np.interp(yax+ycens[ref_col[0]]-ycens_pad[col], yax,
                                     frame[:, ref_col[0]])

    for col in range(dimx+ref_col[1]+pad, dimx+2*pad):
        yax = np.arange(dimy)
        newframe[:, col] = np.interp(yax+ycens[ref_col[1]]-ycens_pad[col], yax,
                                     frame[:, ref_col[1]])

    return newframe


def reconstruct_wings256(profile, ycens=None, contamination=[2, 3], pad=0,
                         verbose=0, smooth=True, **kwargs):
    '''Masks the second and third diffraction orders and reconstructs the
     underlying wing structure of the first order. Also adds padding in the
     spatial direction if required.

    Parameters
    ----------
    profile : np.array
        Spectral trace spatial profile.
    ycens : list
        Y-coordinates of the trace centroids. Must include all three
        diffraction orders if contamination is True, or only the first order if
        False.
    contamination : list
        List of contaminating orders present on the detector (not including the
        first order). For an uncontaminated detector, pass None.
    pad : int
        Amount to pad each end of the spartial axis (in pixels).
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
    '''

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

    # ======= RECONSTRUCT RIGHT WING =======
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
            end = ycen+30
        else:
            start = np.min([ycen-17, dimy-2])
            end = np.min([ycen+17, dimy-1])
        # Set core of each order to NaN.
        prof_r[start:end] = np.nan
    # Fit the unmasked part of the wing to determine the mean linear trend.
    inds = np.where(np.isfinite(prof_r))[0]
    pp = utils._robust_polyfit(axis_r[inds], prof_r[inds], (0, 0))
    wing_mean = pp[1]+pp[0]*axis_r
    # Calculate the standard dev of unmasked points from the mean trend.
    stddev_m = np.sqrt(np.median((prof_r[inds] - wing_mean[inds])**2))

    # === Wing fit ===
    # Get fresh right wing profile.
    prof_r2 = np.log10(profile)
    # Mask first order core.
    prof_r2[:(ycens[0]+13)] = np.nan
    # Mask second and third orders.
    if contamination is not None:
        for order, ycen in enumerate(ycens):
            if order + 1 in contamination:
                start = np.max([ycen-17, 0])
                end = np.max([ycen+17, 1])
                # Set core of each order to NaN.
                prof_r2[start:end] = np.nan
    # Find all outliers that are >3-sigma deviant from the mean.
    start = ycens[0]+25
    inds2 = np.where(np.abs(prof_r2[start:] - wing_mean[start:]) > 3*stddev_m)
    # Mask outliers
    prof_r2[start:][inds2] = np.nan
    # Mask edge of the detector.
    prof_r2[-3:] = np.nan
    # Indices of all unmasked points in the left wing.
    inds3 = np.isfinite(prof_r2)

    # Fit with a 7th order polynomial.
    # To ensure that the polynomial does not start turning up in the padded
    # region, extend the linear fit to the edge of the pad to force the fit
    # to continue decreasing.
    ext_ax = np.arange(25) + np.max(axis_r[inds3]) + np.max([pad, 25])
    ext_prof = pp[1] + pp[0]*ext_ax
    # Concatenate right-hand profile with the extended linear trend.
    fit_ax = np.concatenate([axis_r[inds3], ext_ax])
    fit_prof = np.concatenate([prof_r2[inds3], ext_prof])
    # Use np.polyfit for a first estimate of the coefficients.
    pp_r0 = np.polyfit(fit_ax, fit_prof, 7)
    # Robust fit using the polyfit results as a starting point.
    pp_r = utils._robust_polyfit(fit_ax, fit_prof, pp_r0)

    # === Stitching ===
    newprof = profile*1
    # Interpolate contaminated regions.
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
        plotting._plot_wing_reconstruction(profile, ycens, axis_r[inds3],
                                           prof_r2[inds3], pp_r, newprof, pad,
                                           **kwargs)

    return newprof


def reconstruct_wings96(profile, ycen, goodwing=None, contamination=False,
                        pad=0, verbose=0, **kwargs):
    '''Wing reconstruction for the SUBSTRIP96 subarray. As not enough of the
    first order wings remain on the detector to perform the full wing
    reconstruction, a standard uncontaminated wing profile is used to correct
    second order contamination.

    Parameters
    ----------
    profile : np.array
        Spectral trace spatial profile.
    ycens : list
        Y-coordinates of the trace centroids. Must include all three
        diffraction orders if contamination is True, or only the first order if
        False.
    goodwing : np.ndarray or None
        Uncontaminated wing profile.
    contamination : bool
        If True, profile is contaminated by the second order.
    pad : int
        Amount to pad each end of the spartial axis (in pixels).
    verbose : int
        Level of verbosity.

    Returns
    -------
    newprof : np.array
        Input spatial profile with reconstructed wings and padding.

    Raises
    ------
    ValueError
        If contamination is True and goodwing is None.
    '''

    # Get rid of second order contamination by stitching an uncontminated wing
    # to the uncontaminated first order core.
    if contamination is True:
        # Stitch together the contaminated profile, and uncontaminated wing.
        ycen = int(round(ycen, 0))
        if goodwing is None:
            errmsg = 'Uncontamined wing must not be None if profile is \
                      contaminated.'
            raise ValueError(errmsg)
        newprof = np.concatenate([profile[:(ycen+13)], goodwing])
        # Smooth over the joint with a polynomial fit.
        fitprof = newprof[(ycen+10):(ycen+20)]
        fitprof = ma.masked_array(fitprof, mask=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        fitax = ma.masked_array(np.arange(ycen+20 - (ycen+10)) + ycen+10,
                                mask=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        # Fit the pixels on either side of the joint.
        pp = ma.polyfit(fitax, fitprof, 5)
        # Evaluate the above polynominal in the joint region.
        newprof[(ycen+12):(ycen+16)] = np.polyval(pp, np.arange(4)+ycen+12)
        # Trim to the size of SUBSTRIP96
        newprof = newprof[:96]
    else:
        newprof = profile

    # Add padding - padding values are constant at the median of edge pixels.
    padval_r = np.median(newprof[-8:-3])
    padval_l = np.median(newprof[3:8])
    newprof = np.concatenate([np.tile(padval_l, pad+4), newprof[4:-4],
                              np.tile(padval_r, pad+4)])

    if verbose == 3:
        plotting._plot_wing_reconstruction96(profile, newprof, goodwing,
                                             **kwargs)

    return newprof


def refine_order1(clear, o2frame, centroids, pad, verbose=0):
    '''Refine the first order trace model using a clear exposure with the
    second order subtracted.

    Parameters
    ----------
    clear : np.ndarray (2D)
        CLEAR data frame.
    o2frame : np.ndarray (2D)
        Model of the second order spatial profile.
    centroids : dict
        Centroid dictionary.
    pad : int
        padding to add to the spatial axis.
    verbose : int
        Verbose level.

    Returns
    -------
    order1_uncontam : np.ndarray (2D)
        First order trace model.
    '''

    dimy, dimx = np.shape(clear)
    # Create an uncontaminated frame by subtracting the second order model from
    # the CLEAR exposure.
    order1_uncontam_unref = clear - o2frame
    order1_uncontam = np.zeros((dimy+2*pad, dimx))

    # Reconstruct first order wings.
    disable = utils._verbose_to_bool(verbose)
    for i in tqdm(range(dimx), disable=disable):
        ycens = [centroids['order 1']['Y centroid'][i],
                 centroids['order 2']['Y centroid'][i],
                 centroids['order 3']['Y centroid'][i]]
        prof_refine = reconstruct_wings256(order1_uncontam_unref[:, i],
                                           ycens, contamination=[3], pad=pad)
        order1_uncontam[:, i] = prof_refine

    return order1_uncontam


def replace_badpix(clear, badpix_mask, fill_negatives=True, verbose=0):
    '''Replace all bad pixels with the median of the pixels values of a 5x5 box
    centered on the bad pixel.

    Parameters
    ----------
    clear : np.ndarray (2D)
        Dataframe with bad pixels.
    badpix_mask : np.ndarray (2D)
        Boolean array with the same dimensions as clear. Values of True
        indicate a bad pixel.
    fill_negatives : bool
        If True, also interpolates all negatives values in the frame.
    verbose : int
        Level of verbosity.

    Returns
    -------
    clear_r : np.ndarray (2D)
        Input clear frame with bad pixels interpolated.
    '''

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

    disable = utils._verbose_to_bool(verbose)
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


def rescale_model(data, model, centroids, verbose=0):
    '''Rescale a column normalized trace model to the flux level of an actual
    observation. A multiplicative coefficient is determined via Chi^2
    minimization independantly for each column, such that the rescaled model
    best matches the data.

    Parameters
    ----------
    data : np.ndarray (2D)
        Observed dataframe.
    model : np.ndarray (2D)
        Column normalized trace model.
    verbose : int
        Level of verbosity.

    Returns
    -------
    model_rescale : np.ndarray (2D)
        Trace model after rescaling.
    '''

    # Determine first guess coefficients.
    k0 = np.nanmax(data, axis=0)
    ks = []
    # Loop over all columns - surely there is a more vectorized way to do this
    # whenever I try to minimize all of k at once I get nonsensical results?
    disable = utils._verbose_to_bool(verbose)
    for i in tqdm(range(data.shape[1]), disable=disable):
        # Get region around centroid
        ycen = int(round(centroids['order 1']['Y centroid'][i], 0))
        start = ycen - 13
        end = ycen + 14
        # Minimize the Chi^2.
        lik_args = (data[start:end, i], model[start:end, i])
        k = minimize(utils._lik, k0[i], lik_args)
        ks.append(k.x[0])

    # Rescale the column normalized model.
    ks = np.array(ks)
    model_rescale = ks*model

    return model_rescale


def smooth_spat_discont(o2frame, ycens):
    '''Smooth oversubtracted pixels in the spatial direction. If the flux in a
    pixel is >3sigma deviant from the mean value of the trace core in its
    column, it is replaced by a median of flux values over its nieghbours in
    the spectral direction.

    Parameters
    ----------
    o2frame : np.ndarray (2D)
        Uncontaminated second order trace profile.
    ycens : dict
        Centroids dictionary.

    Returns
    -------
    o2frame : np.ndarray (2D)
        Uncontaminated trace profile with oversubtracted pixels interpolated.
    '''

    for col in range(o2frame.shape[1]):
        # Get lower and upper bounds of trace core
        start = int(round(ycens['order 2']['Y centroid'][col], 0)) - 11
        end = int(round(ycens['order 2']['Y centroid'][col], 0)) + 11
        # Calculate mean and standard deviation of core.
        mn = np.mean(np.log10(o2frame[start:end, col]))
        sd = np.std(np.log10(o2frame[start:end, col]))
        diff = np.abs(np.log10(o2frame[start:end, col]) - mn)
        # Find where flux values are >3sigma deviant from mean
        inds = np.where(diff > 3*sd)[0]
        for i in inds:
            # Pixel location in spatial profile.
            loc = start+i
            # Replace bad pixel with median of rows.
            start2 = np.max([col-25, 0])
            end2 = np.min([col+25, o2frame.shape[1]])
            rep_val = np.median(o2frame[loc, start2:end2])
            o2frame[loc, col] = rep_val

    return o2frame


def smooth_spec_discont(o2frame, verbose):
    '''Smooth over streaks (especially in the contaminated region). If the mean
    flux value of a column is >10% deviant from that of the surrounding
    columns, replace it with the median of its neighbours.

    Parameters
    ----------
    o2frame : np.ndarray (2D)
        Uncontaminated second order trace profile.
    verbose : int
        Level of verbosity.

    Returns
    -------
    o2frame : np.ndarray (2D)
        Uncontaminated trace profile with column-to-column discontinuities
        smoothed.
    '''

    # Get mean flux values for each column.
    col_mean = np.nanmean(o2frame, axis=0)
    # Find where order 2 ends.
    end = np.where(col_mean == 0)[0][0] - 5
    # For each column, find the local mean of the surrounding 6 columns.
    loc_mean = utils._local_mean(col_mean, step=3)
    # Calculate the deviation of each column from the local mean.
    dev_0 = np.abs((col_mean[3:end] - loc_mean[3:end])/loc_mean[3:end])
    # Replace all columns whose mean value is >10% deviant from the local mean
    # with the median of its neighbours.
    dev, iter = dev_0, 0
    # Iterate until no columns are >10% deviant, or 10 iterations have run.
    while np.any(dev > 0.1) and iter < 10:
        # Get all >10% deviant columns.
        inds = np.where(dev >= 0.1)[0]
        for i in inds:
            i += 3
            # For each column, calculate the median of the 'local region'.
            # Expand the region by one pixel each iteration.
            local = np.concatenate([o2frame[:, (i-2-iter):i],
                                    o2frame[:, (i+1):(i+3+iter)]], axis=1)
            o2frame[:, i] = np.nanmedian(local, axis=1)
        # Recalculate the flux deviations as before.
        col_mean = np.nanmean(o2frame, axis=0)
        loc_mean = utils._local_mean(col_mean, step=3)
        dev = np.abs((col_mean[3:end] - loc_mean[3:end])/loc_mean[3:end])
        # Increment iteration.
        iter += 1

    # Plot the change in flux deviations after all iterations are complete.
    if verbose == 3:
        plotting._plot_flux_deviations(dev_0, dev, iter)

    return o2frame
