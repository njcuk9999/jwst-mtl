#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:37 2022

@author: MCR

Functions for the now depreciated interpolation model which makes use of an
F277W filter exposure to interpolate over the contaminated region of order 1.
"""

import numpy as np
from tqdm import tqdm

from SOSS.APPLESOSS import applesoss
from SOSS.APPLESOSS import plotting
from SOSS.APPLESOSS import utils


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
    bl_anch = applesoss.reconstruct_wings256(bl_anch, ycens=cens,
                                             contamination=[2, 3], pad=pad,
                                             verbose=verbose, smooth=True,
                                             **{'text': 'Blue anchor'})
    # Remove the chromatic scaling.
    bl_anch = applesoss.chromescale(bl_anch, 2.1, 2.5, ydb + pad, width_polys)
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
    rd_anch = applesoss.reconstruct_wings256(rd_anch, ycens=cens,
                                             contamination=None, pad=pad,
                                             verbose=verbose, smooth=True,
                                             **{'text': 'Red anchor'})
    rd_anch = applesoss.chromescale(rd_anch, 2.45, 2.5, ydr + pad, width_polys)
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
        prof_int_cs = applesoss.chromescale(prof_int, 2.5, lbd, ceny + pad,
                                            width_polys)
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
        o1frame[:, col] = applesoss.reconstruct_wings256(clear[:, col],
                                                         ycens=cens, pad=pad,
                                                         smooth=True)
    # Add on the F277W frame to the red of the model.
    disable = utils.verbose_to_bool(verbose)
    for col in tqdm(range(rd_end), disable=disable):
        cens = [ycens['order 1']['Y centroid'][col]]
        o1frame[:, col] = applesoss.reconstruct_wings256(f277w[:, col],
                                                         ycens=cens,
                                                         contamination=None,
                                                         pad=pad, smooth=True)

        o1frame[:, col] = rescale_f277(o1frame[:, col], clear[:, col],
                                       ycen=cens[0], verbose=0)

    # Column normalize - necessary for uniformity as anchor profiles are
    # normalized whereas stitched data is not.
    o1frame /= np.nansum(o1frame, axis=0)

    return o1frame


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
