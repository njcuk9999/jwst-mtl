#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 9:35 2020

@author: MCR

File containing the necessary functions to create an empirical interpolated
trace model in the overlap region for SOSS order 1.
"""

import os
import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from tqdm import tqdm
import webbpsf
from SOSS.extract import soss_read_refs
from SOSS.extract.empirical_trace import centroid as ctd
from SOSS.extract.empirical_trace import plotting as plotting
from SOSS.extract.overlap import TrpzOverlap, TrpzBox
from SOSS.extract.throughput import ThroughputSOSS
from SOSS.extract.convolution import WebbKer

# Local path to reference files.
path = '/Users/michaelradica/Documents/School/Ph.D./Research/SOSS/Extraction/Input_Files/'


def build_empirical_trace(clear, F277W, badpix_mask,
                          filename='spatial_profile.fits', pad=(0, 0),
                          oversample=1, verbose=False):
    ''' Procedural function to wrap around construct orders 1 and 2.
    Will do centroiding and call the functions to construct the models.

    ***Will eventually want clear and F277W to be the full fits with headers
    to get parameters from***

    Parameters
    ----------
    clear : np.array of float (2D) - eventually path
    F277W : np.array of float (2D) - eventually path
    filename : str
    pad : tuple
    oversample : int
    verbose : bool
    '''

    # Print overwrite warning if output file already exists.
    if os.path.exists(filename):
        print('Output file {} already exists. It will be overwritten'.format(filename))

    # Replace bad pixels.
    if verbose is True:
        print('Replacing bad pixels...', flush=True)
    clear = replace_badpix(clear, badpix_mask, verbose=verbose)
    if F277W is not None:
        F277W = replace_badpix(F277W, badpix_mask, verbose=verbose)

    # Get the centroid positions for both orders from the data.
    if verbose is True:
        print('Getting trace centroids...')
    centroids, rot_pars = ctd.get_contam_centroids(clear, bound=False,
                                                   verbose=verbose)

    # Overplot the data centroids on the CLEAR exposure if desired.
    if verbose is True:
        plotting._plot_centroid(clear, centroids)

    # Construct the first order profile.
    if verbose is True:
        print('Building the first order trace model...')
    o1frame = construct_order1(clear, F277W, rot_pars, centroids, pad=pad[0],
                               verbose=verbose, subarray='SUBSTRIP256')
    # Pad the spectral axis.
    if pad[1] != 0:
        if verbose is True:
            print('  Adding padding to first order spectral axis...')
        o1frame = pad_spectral_axis(o1frame, centroids['order 1'][0],
                                    centroids['order 1'][1], pad=pad[1])

    # Add oversampling
    if oversample != 1:
        if verbose is True:
            print('  Oversampling...')
        o1frame = oversample_frame(o1frame, oversample=oversample)

    # Get the extraction parameters
    #extract_params = get_extract_params()
    #ref_file_args = get_ref_file_args(o1frame)
    # Construct the second order profile.
    #o2frame_contam = construct_order2(clear, ref_file_args, extract_params)
    # Create a mask to remove residuals from the first order.
    #o2frame = mask_order(o2frame_contam, x2, y2)
    # Set any spurious negative values to zero.
    #o2frame[o2frame < 0] = 0
    # Normalize the profile in each column.
    #o2frame /= np.nansum(o2frame, axis=0)

    # Write the trace model to disk.
    #hdu = fits.PrimaryHDU()
    #hdu.data = np.dstack((o1frame, o2frame))
    #hdu.writeto(filename, overwrite=True)

    if verbose is True:
        print('Done.')

    return o1frame#, o2frame


def _chromescale(wave_start, profile, ycen, wave_end=2.5, invert=False):
    '''Remove the lambda/D chromatic PSF scaling by re-interpolating a
    monochromatic PSF function onto a rescaled axis.

    Parameters
    ----------
    wave_start : float
        Wavelength corresponding to the input 1D PSF profile.
    profile : np.array of float
        1D PSF profile to be rescaled.
    ycen : float
        Y-pixel position of the order 1 trace centroid.
    wave_end : float
        Wavelength after rescaling.
    invert : bool
        If True, add back the lambda/D scaling instead of removing it.

    Returns
    -------
    prof_recsale : np.array of float
        Rescaled 1D PSF profile.
    '''

    xrange = len(profile)
    # Assume that the axis corresponding to wave_end can be constructed with
    # np.arange - this is the 'standard' axis.
    # Construct the input profile axis by rescaling the standard axis by the
    # ratio of wave_end/wave_start.
    xax = np.linspace(0, round(xrange*(wave_end/wave_start), 0) - 1, xrange)
    # Calculate the offset of the centroid y-position in the standard and
    # rescaled axis, to ensure it remains at the same pixel position.
    offset = xax[int(round(ycen, 0))] - ycen

    # Interpolate the profile onto the standard axis.
    if invert is False:
        prof_rescale = np.interp(np.arange(xrange), xax - offset, profile)
    # Or interpolate the profile from the standard axis to re-add
    # the lambda/D scaling.
    else:
        prof_rescale = np.interp(xax - offset, np.arange(xrange), profile)

    return prof_rescale


def construct_order1(clear, F277, rot_params, ycens, subarray, pad=0,
                     verbose=False):
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
    rot_params : list of float
        List containing the rotation angle, X and Y anchor points, and X
        and Y offset required to transform OM coordinates to the detector
        frame.
    ycens : dict
        Dictionary of Y-coordinates for the trace centroids of the first three
        diffraction orders, ie. as returned by get_contam_centroids.
    subarray : str
        Subarray identifier. One of "SUBSTRIP96", "SUBSTRIP256", or "FULL".
    pad : int
        Number of pixels of padding to add on both ends of the spatial axis.
    verbose : bool
        if True, do diagnostic plotting and progress prints.

    Returns
    -------
    newmap : np.array of float (2D)
        Interpolated order 1 trace model with padding.
    '''

    # Get subarray dimensions
    dimx = 2048
    if subarray == 'SUBSTRIP96':
        dimy = 96
    elif subarray == 'SUBSTRIP256':
        dimy = 256
    else:
        dimy = 2048

    # Open wavelength calibration file.
    wavecal = fits.getdata(path+'jwst_niriss_soss-256-ord1_trace.fits', 1)
    # Get wavelength and detector pixel calibration info.
    pp_w = np.polyfit(wavecal['Detector_Pixels'], wavecal['WAVELENGTH'][::-1], 1)
    wavecal_x = np.arange(dimx)
    wavecal_w = np.polyval(pp_w, wavecal_x)

    if verbose is True:
        print('  Getting anchor profiles...')

    # Determine the anchor profiles - blue anchor.
    # Find the pixel position of 2.1µm.
    i_b = np.where(wavecal_w >= 2.1)[0][-1]
    xdb = int(wavecal_x[i_b])
    ydb = ycens['order 1'][1][xdb]

    # Extract the 2.1µm anchor profile from the data - take median profile of
    # neighbouring 5 columns to mitigate effects of outliers.
    Banch = np.median(clear[:, (xdb-2):(xdb+2)], axis=1)
    # Mask second and third order, reconstruct wing structure and pad.
    cens = [ycens['order 1'][1][xdb], ycens['order 2'][1][xdb],
            ycens['order 3'][1][xdb]]
    Banch = reconstruct_wings(Banch, ycens=cens, contamination=True, pad=pad,
                              verbose=verbose, **{'text': 'Blue anchor'})
    # Remove the lambda/D scaling.
    Banch = _chromescale(2.1, Banch, ydb+pad)
    # Normalize
    Banch /= np.nansum(Banch)

    # Determine the anchor profiles - red anchor.
    if F277 is not None:
        # If an F277W exposure is provided, only interpolate out to 2.45µm.
        # Redwards of 2.45µm we have perfect knowledge of the order 1 trace.
        # Find the pixel position of 2.45µm.
        i_r = np.where(wavecal_w >= 2.45)[0][-1]
        xdr = int(wavecal_x[i_r])
        ydr = ycens['order 1'][1][xdr]

        # Extract and rescale the 2.45µm profile - take median of neighbouring
        # five columns to mitigate effects of outliers.
        Ranch = np.median(F277[:, (xdr-2):(xdr+2)], axis=1)
        # Reconstruct wing structure and pad.
        cens = [ycens['order 1'][1][xdr]]
        Ranch = reconstruct_wings(Ranch, ycens=cens, contamination=False,
                                  pad=pad, verbose=verbose,
                                  **{'text': 'Red anchor'})
        Ranch = _chromescale(2.45, Ranch, ydr+pad)
        # Normalize
        Ranch /= np.nansum(Ranch)

        # Interpolation polynomial coeffs, calculated via calc_interp_coefs
        coef_b = [1.51850915, -9.76581613, 14.80720191]
        coef_r = [-1.51850915,  9.76581613, -13.80720191]
    else:
        # If no F277W exposure is provided, interpolate out to 2.9µm.
        # Generate a simulated 2.9µm PSF.
        stand = loicpsf([2.9*1e-6], save_to_disk=False, oversampling=1,
                        pixel=256, verbose=False)[0][0].data
        # Extract the spatial profile.
        Ranch = np.sum(stand[124:132, :], axis=0)

        # Extend y centroids and wavelengths off of the detector.
        pp_w = np.polyfit(wavecal_x, wavecal_w, 1)
        pp_y = np.polyfit(ycens['order 1'][0], ycens['order 1'][1], 9)
        xpix = np.arange(dimx) - 250
        wave_ext = np.polyval(pp_w, xpix)
        ycen_ext = np.polyval(pp_y, xpix)

        # Find position of 2.9µm on extended detector.
        i_r = np.where(wave_ext >= 2.9)[0][-1]
        xdr = int(round(xpix[i_r], 0))
        ydr = ycen_ext[i_r]

        # Interpolate the WebbPSF generated profile to the correct location.
        Ranch = np.interp(np.arange(dimy), np.arange(dimy)-dimy/2+ydr, Ranch)
        # Reconstruct wing structure and pad.
        Ranch = reconstruct_wings(Ranch, ycens=[ydr], contamination=False,
                                  pad=pad, verbose=verbose,
                                  **{'text': 'Red anchor'})
        # Rescale to remove chromatic effects.
        Ranch = _chromescale(2.9, Ranch, ydr+pad)
        # Normalize
        Ranch /= np.nansum(Ranch)

        # Interpolation polynomial coeffs, calculated via calc_interp_coefs
        coef_b = [0.80175603, -5.27434345, 8.54474316]
        coef_r = [-0.80175603, 5.27434345, -7.54474316]
        # Pixel coords at which to start the interpolation.
        xdr = 0

    # Create the interpolated order 1 PSF.
    map2D = np.zeros((dimy+2*pad, dimx))*np.nan
    # Get centroid pixel coordinates and wavelengths for interpolation region.
    cenx_d = np.arange(xdb - xdr).astype(int) + xdr
    ceny_d = ycens['order 1'][1][cenx_d]
    lmbda = wavecal_w[cenx_d]

    # Create an interpolated 1D PSF at each required position.
    if verbose is True:
        print('  Interpolating trace...', flush=True)
    for i, vals in tqdm(enumerate(zip(cenx_d, ceny_d, lmbda)),
                        total=len(lmbda), disable=not verbose):
        cenx, ceny, lbd = vals[0], vals[1], vals[2]
        # Evaluate the interpolation polynomials at the current wavelength.
        wb_i = np.polyval(coef_b, lbd)
        wr_i = np.polyval(coef_r, lbd)
        # Recenter the profile of both anchors on the correct Y-centroid.
        bax = np.arange(dimy+2*pad)-ydb+ceny
        Banch_i = np.interp(np.arange(dimy+2*pad), bax, Banch)
        rax = np.arange(dimy+2*pad)-ydr+ceny
        Ranch_i = np.interp(np.arange(dimy+2*pad), rax, Ranch)
        # Construct the interpolated profile.
        prof_int = (wb_i * Banch_i + wr_i * Ranch_i)
        # Re-add the lambda/D scaling.
        prof_int_cs = _chromescale(lbd, prof_int, ceny+pad, invert=True)
        # Put the interpolated profile on the detector.
        map2D[:, int(round(cenx, 0))] = prof_int_cs

        # Note detector coordinates of the edges of the interpolated region.
        bend = int(round(cenx, 0))
        if i == 0:
            # 2.9µm (i=0) limit may be off the end of the detector.
            rend = int(round(cenx, 0))

    if verbose is True:
        print('  Stitching models and reconstructing wings...', flush=True)
    # Stitch together the interpolation and data.
    newmap = np.zeros((dimy+2*pad, dimx))
    # Insert interpolated data.
    newmap[:, rend:bend] = map2D[:, rend:bend]
    # Bluer region is known from the CLEAR exposure.
    for col in tqdm(range(bend, dimx), disable=not verbose):
        cens = [ycens['order 1'][1][col], ycens['order 2'][1][col],
                ycens['order 3'][1][col]]
        # Mask contamination from second and third orders, reconstruct wings
        # and add padding.
        newmap[:, col] = reconstruct_wings(clear[:, col], ycens=cens, pad=pad)
    if F277 is not None:
        # Add on the F277W frame to the red of the model.
        for col in tqdm(range(rend), disable=not verbose):
            cens = [ycens['order 1'][1][col]]
            # Reconstruct wing structure and pad.
            newmap[:, col] = reconstruct_wings(F277[:, col], ycens=cens,
                                               contamination=False, pad=pad)

    # Column normalize.
    newmap /= np.nansum(newmap, axis=0)
    # Add noise floor to prevent arbitrarily low values in padded wings.
    floor = np.nanpercentile(newmap[pad:(-1-pad), :], 2)
    newmap += floor
    # Column renormalize.
    newmap /= np.nansum(newmap, axis=0)

    return newmap


def construct_order2(clear, ref_file_args, extract_params):
    '''Preforms an extraction and reconstructs the detector with only the
    first order trace profile. The second order profile is then isolated
    through the difference of the original and reconstructed detector.

    Parameters
    ----------
    clear : np.array of float (2D)
        CLEAR data frame.
    ref_file_args : list
        List of parameters of the reference files.
    extract_params : dict
        Dictionary of arguments required by the extraction algorithm.

    Returns
    -------
    residual : np.array of float (2D)
        Detector with only the order 2 trace profile.
    '''

    # Set up the extraction.
    extra = TrpzOverlap(*ref_file_args, **extract_params, orders=[1])
    # Preform the extraction with only the first order.
    f_k = extra.extract(data=clear)
    # Rebuild the detector.
    rebuilt = extra.rebuild(f_k)
    rebuilt[np.isnan(rebuilt)] = 0
    # Isolate the second order by subtracting the reconstructed first
    # order from the data
    residual = clear - rebuilt

    return rebuilt


def get_extract_params():
    '''
    '''
    params = {}
    # Map of expected noise (sig)
    bkgd_noise = 10.
    # Oversampling
    params["n_os"] = 1
    # Threshold on the spatial profile
    params["thresh"] = 1e-6

    return params


def get_ref_file_args(o1frame):
    '''
    '''

    path = '/Users/michaelradica/Documents/School/Ph.D./Research/SOSS/Extraction/Input_Files/SOSS_Ref_Files/'
    # List of orders to consider in the extraction
    order_list = [1]

    # Wavelength solution
    wave_maps = []
    wavemap_file = soss_read_refs.Ref2dWave()
    wave_maps.append(wavemap_file(order=1, subarray='SUBSTRIP256', native=True))

    # Spatial profiles
    spat_pros = []
    spat_pros.append(o1frame)

    # Convert data from fits files to float (fits precision is 1e-8)
    wave_maps = [wv.astype('float64') for wv in wave_maps]
    spat_pros = [p_ord.astype('float64') for p_ord in spat_pros]

    # Throughputs
    thrpt_list = [ThroughputSOSS(order) for order in order_list]

    # Convolution kernels
    ker_list = [WebbKer(wv_map) for wv_map in wave_maps]

    # Put all inputs from reference files in a list
    ref_file_args = [spat_pros, wave_maps, thrpt_list, ker_list]

    return ref_file_args


def loicpsf(wavelist=None, wfe_real=None, filepath='', save_to_disk=True,
            oversampling=10, pixel=128, verbose=True):
    '''Calls the WebbPSF package to create monochromatic PSFs for NIRISS
    SOSS observations and save them to disk.

    Parameters
    ----------
    wavelist : list
        List of wavelengths (in meters) for which to generate PSFs.
    wfe_real : int
        Index of wavefront realization to use for the PSF (if non-default
        WFE realization is desired).
    filepath : str
        Path to the directory to which the PSFs will be written.
        Defaults to the current directory.
    save_to_disk : bool
        Whether to save PSFs to disk, or return them from the function.
    oversampling : int
        Oversampling pixels scale for the PSF.
    pixel : int
        Width of the PSF in native pixels.
    verbose : bool
        Whether to print explanatory comments.

    Returns
    -------
    psf_list : list
        If save_to_disk is False, a list of the generated PSFs.
    '''

    psf_list = []

    if wavelist is None:
        # List of wavelengths to generate PSFs for
        wavelist = np.linspace(0.5, 5.2, 95) * 1e-6
    # Dimension of the PSF in native pixels
    pixel = pixel

    # Select the NIRISS instrument
    niriss = webbpsf.NIRISS()

    # Override the default minimum wavelength of 0.6 microns
    niriss.SHORT_WAVELENGTH_MIN = 0.5e-6
    # Set correct filter and pupil wheel components
    niriss.filter = 'CLEAR'
    niriss.pupil_mask = 'GR700XD'

    # Change the WFE realization if desired
    if wfe_real is not None:
        niriss.pupilopd = ('OPD_RevW_ote_for_NIRISS_predicted.fits.gz',
                           wfe_real)

    # Loop through all wavelengths to generate PSFs
    for wave in wavelist:
        if verbose is True:
            print('Calculating PSF at wavelength ',
                  round(wave/1e-6, 2), ' microns')
        psf = niriss.calc_psf(monochromatic=wave, fov_pixels=pixel,
                              oversample=oversampling, display=False)
        psf_list.append(psf)

        if save_to_disk is True:
            # Save psf realization to disk
            text = '{0:5f}'.format(wave*1e+6)
            psf.writeto(str(filepath)+'SOSS_os'+str(oversampling)+'_'+str(pixel)
                        + 'x'+str(pixel)+'_'+text+'_'+str(wfe_real)+'.fits',
                        overwrite=True)

        if save_to_disk is False:
            return psf_list


# depreciated
def mask_order(frame, xpix, ypix):
    '''
    Create a pixel mask to isolate only the detector pixels belonging to
    a specific diffraction order.

    Parameters
    ----------
    frame : np.array of float (2D)
        Science data frame.
    xpix : np.array
        Data x centroids for the desired order
    ypix : np.array
        Data y centroids for the desired order

    Returns
    -------
    O1frame : np.array of float (2D)
        The input data frame, with all pixels other than those within
        +/- 20 pixels of ypix masked.
    '''

    mask = np.zeros([256, 2048])
    xx = np.round(xpix, 0).astype(int)
    yy = np.round(ypix, 0).astype(int)
    xr = np.linspace(np.min(xx), np.max(xx), np.max(xx)+1).astype(int)

    # Set all pixels within the extent of the spatial profile to 1
    for xxx, yyy in zip(xr, yy):
        # Ensure that we stay on the detector
        p_max = np.min([yyy+20, 255])
        p_min = np.max([yyy-21, 0])
        mask[p_min:p_max, xxx] = 1

    # Mask pixels not in the order we want
    O1frame = (mask * frame)

    return O1frame


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


def pad_spectral_axis(frame, xcens, ycens, pad=0):
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

    ylen = int(frame.shape[0])
    xlen = int(frame.shape[1])
    pp = np.polyfit(xcens, ycens, 5)
    xax_pad = np.arange(xlen+2*pad)-pad
    ycens_pad = np.polyval(pp, xax_pad)

    newframe = np.zeros((ylen, xlen+2*pad))
    newframe[:, (pad):(xlen+pad)] = frame

    for col in range(pad):
        yax = np.arange(ylen)
        newframe[:, col] = np.interp(yax+ycens[5]-ycens_pad[col], yax,
                                     frame[:, 5])

    for col in range(xlen+pad, xlen+2*pad):
        yax = np.arange(ylen)
        newframe[:, col] = np.interp(yax+ycens[-10]-ycens_pad[col], yax,
                                     frame[:, -10])

    return newframe


def reconstruct_wings_v2(profile, ycens=None, contamination=True, pad=0,
                      verbose=False, **kwargs):
    '''Masks the second and third diffraction orders and reconstructs the
     underlying wing structure of the first order. Also adds padding in the
     spatial direction if required.
     Note: the algorithm struggles in the region where the first and second
     orders completely overlap (pixels ~0-450). However, this region is either
     interpolated or from an F277W exposure in the empirical trace model,
     circumventing this issue.

    Parameters
    ----------
    profile : np.array
        Spectral trace spatial profile.
    ycens : list
        Y-coordinates of the trace centroids. Must include all three
        diffraction orders if contamination is True, or only the first order if
        False.
    contamination : bool
        If True, profile has contamination from the second and third
        diffraction orders.
    pad : int
        Amount to pad each end of the spartial axis (in pixels).
    verbose : bool
        If True, does diagnostic plotting.

    Returns
    -------
    newprof : np.array
        Input spatial profile with reconstructed wings and padding.

    Raises
    ------
    ValueError
        If centroids are not provided for all three orders when contamination
        is set to True.
    '''

    dimy = len(profile)
    # Convert Y-centroid positions to indices
    ycens = np.atleast_1d(ycens)
    ycens = np.round(ycens, 0).astype(int)
    if contamination is True and ycens.size != 3:
        raise ValueError('Centroids must be provided for first three orders if there is contamination.')

    # mask negative and zero values.
    profile[profile <= 0] = np.nan

    # ====== Reconstruct right wing ======
    # Mask the cores of the first three diffraction orders and fit a straight
    # line to the remaining pixels. Additionally mask any outlier pixels that
    # are >3-sigma deviant from the mean. Fit a 7th order polynomial to
    # remaining pixels.
    # Get the right wing of the trace profile in log space.
    prof_r = np.log10(profile)
    # and corresponding axis.
    axis_r = np.arange(dimy)

    # === Outlier masking ===
    # Mask the cores of each order.
    for order, ycen in enumerate(ycens):
        if order == 0:
            start = 0
            end = ycen+25
        elif order == 1:
            start = np.min([ycen-17, dimy-2])
            end = np.min([ycen+17, dimy-1])
        else:
            start = np.min([ycen-17, dimy-2])
            end = np.min([ycen+17, dimy-1])
        # Set core of each order to NaN.
        prof_r[start:end] = np.nan

    # Fit the unmasked part of the wing to determine the mean trend.
    inds = np.where(np.isfinite(prof_r))[0]
    pp = _robust_polyfit(axis_r[inds], prof_r[inds], (0, 0))
    wing_mean = pp[1]+pp[0]*axis_r[inds]

    # Calculate the standard dev of unmasked points from the mean trend.
    stddev = np.sqrt(np.median((prof_r[inds] - wing_mean)**2))
    # Find all outliers that are >3-sigma deviant from the mean.
    inds2 = np.where(np.abs(prof_r[inds] - wing_mean) > 3*stddev)

    # === Wing fit ===
    # Get fresh right wing profile.
    prof_r2 = np.log10(profile)
    # Mask first order core.
    prof_r2[:(ycens[0]+12)] = np.nan
    # Mask edge of the detector.
    prof_r2[-4:] = np.nan
    # Mask second and third orders.
    if contamination is True:
        for order, ycen in enumerate(ycens):
            if order == 1:
                start = np.max([ycen-17, 0])
                end = np.max([ycen+17, 1])
            elif order == 2:
                start = np.max([ycen-17, 0])
                end = np.max([ycen+17, 1])
            # Set core of each order to NaN.
            prof_r2[start:end] = np.nan
    # Mask outliers
    prof_r2[inds[inds2]] = np.nan

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
    pp_r0 = np.polyfit(fit_ax, fit_prof, 11)
    # Robust fit using the polyfit results as a starting point.
    pp_r = _robust_polyfit(fit_ax, fit_prof, pp_r0)

    # === Stitching ===
    # Find pixel to stitch right wing fit.
    jjr = ycens[0]+14
    # Pad the right axis.
    axis_r_pad = np.arange(len(axis_r[ycens[0]+14:])+pad) + axis_r[ycens[0]+14]
    iir = np.where(axis_r_pad == jjr)[0][0]
    # Join right wing to old trace profile.
    newprof = np.concatenate([profile[:jjr], 10**np.polyval(pp_r, axis_r_pad)[iir:]])

    # Interpolate between wing and core profiles for continuity.
    # Get pixels and values around the joint.
    xs = np.concatenate([jjr-1-np.arange(3), np.arange(3)+jjr+1])
    ys = newprof[xs]
    # Interpolation over the above pixels.
    interp = np.polyfit(xs, ys, 5)
    xnew = np.arange(2)+(jjr-1)
    # Insert interpolation.
    newprof[(jjr-1):(jjr+1)] = np.polyval(interp, xnew)

    # ====== Reconstruct left wing ======
    # Mask the core of the first order, and fit a third order polynomial to all
    # remaining pixels to capture the wing behaviour near the core. Reuse the
    # 7th order solution from the right wing for the extended behaviour.
    # Get the profile for the left wing in log space.
    prof_l = np.log10(profile[:(ycens[0]-12)])
    # and corresponding axis.
    axis_l = np.arange(dimy)[:(ycens[0]-12)]
    # Fit with third order polynomial - exclude detector edge.
    # Use np.polyfit for a first estimate of the coefficients.
    pp_l0 = np.polyfit(axis_l[3:], prof_l[3:], 5)
    # Robust fit using the polyfit results as a starting point.
    pp_l = _robust_polyfit(axis_l[3:], prof_l[3:], pp_l0)

    # === Stitching ===
    # Find pixels to stitch left wing fit.
    jjl = ycens[0]-14  # Join fit to profile.
    jjl2 = ycens[0]-30  # Join core and extended fits.
    # Pad the left axis.
    axis_l_pad = np.linspace(axis_l[0]-pad, axis_l[-1], len(axis_l)+pad)
    #  Mirror of axis_l_pad to the right.
    axis_l_pad2 = np.arange(len(axis_l_pad)) + axis_r[ycens[0]+14]
    iil = np.where(axis_l_pad == jjl)[0][0]
    iil2 = np.where(axis_l_pad == jjl2)[0][0]
    # Join left wing to old trace profile.
    newprof = np.concatenate([10**np.polyval(pp_r, axis_l_pad2)[::-1][:iil2],
                              10**np.polyval(pp_l, axis_l_pad)[iil2:iil],
                              newprof[jjl:]])

    # Interpolate between wing and core profiles for continuity.
    # Get pixels and values around the joint.
    xs = np.concatenate([jjl-1-np.arange(3), np.arange(3)+jjl+1])
    ys = newprof[xs]
    # Interpolation over the above pixels.
    interp = np.polyfit(xs, ys, 5)
    xnew = np.arange(2)+(jjl-1)
    # Insert interpolation.
    newprof[(jjl-1):(jjl+1)] = np.polyval(interp, xnew)
    # Interpolate between wing and extended solutions for continuity.
    # Get pixels and values around the joint.
    xs = np.concatenate([np.arange(4)+np.max([0, iil2-9]),
                         np.min([iil2+9, dimy])-np.arange(4)])
    ys = newprof[xs]
    # Interpolation over the above pixels.
    interp = np.polyfit(xs, ys, 5)
    xnew = np.arange(18)+(iil2-9)
    # Insert interpolation.
    newprof[np.max([0, iil2-9]):np.min([iil2+9, dimy])] = np.polyval(interp, xnew)

    # Set any negatives to zero
    newprof[newprof < 0] = 0

    # Do diagnostic plot if requested.
    if verbose is True:
        plotting._plot_wing_reconstruction(profile, ycens, axis_r[inds3],
                                           prof_r2[inds3], axis_l_pad,
                                           axis_r_pad, pp_r, newprof, **kwargs)

    return newprof


def reconstruct_wings(profile, ycens=None, contamination=True, pad=0,
                      verbose=False, **kwargs):
    '''Masks the second and third diffraction orders and reconstructs the
     underlying wing structure of the first order. Also adds padding in the
     spatial direction if required.
     Note: the algorithm struggles in the region where the first and second
     orders completely overlap (pixels ~0-450). However, this region is either
     interpolated or from an F277W exposure in the empirical trace model,
     circumventing this issue.

    Parameters
    ----------
    profile : np.array
        Spectral trace spatial profile.
    ycens : list
        Y-coordinates of the trace centroids. Must include all three
        diffraction orders if contamination is True, or only the first order if
        False.
    contamination : bool
        If True, profile has contamination from the second and third
        diffraction orders.
    pad : int
        Amount to pad each end of the spartial axis (in pixels).
    verbose : bool
        If True, does diagnostic plotting.

    Returns
    -------
    newprof : np.array
        Input spatial profile with reconstructed wings and padding.

    Raises
    ------
    ValueError
        If centroids are not provided for all three orders when contamination
        is set to True.
    '''

    dimy = len(profile)
    # Convert Y-centroid positions to indices
    ycens = np.atleast_1d(ycens)
    ycens = np.round(ycens, 0).astype(int)
    if contamination is True and ycens.size != 3:
        raise ValueError('Centroids must be provided for first three orders if there is contamination.')

    # mask negative and zero values.
    profile[profile <= 0] = np.nan

    # ====== Reconstruct right wing ======
    # Mask the cores of the first three diffraction orders and fit a straight
    # line to the remaining pixels. Additionally mask any outlier pixels that
    # are >3-sigma deviant from that mean. Fit a 9th order polynomial to
    # remaining pixels.
    # Get the right wing of the trace profile in log space.
    prof_r = np.log10(profile)
    # and corresponding axis.
    axis_r = np.arange(dimy)

    # === Outlier masking ===
    # Mask the cores of each order.
    for order, ycen in enumerate(ycens):
        if order == 0:
            start = 0
            end = ycen+25
        elif order == 1:
            start = np.min([ycen-18, dimy-2])
            end = np.min([ycen+18, dimy-1])
        else:
            start = np.min([ycen-17, dimy-2])
            end = np.min([ycen+17, dimy-1])
        # Set core of each order to NaN.
        prof_r[start:end] = np.nan

    # Fit the unmasked part of the wing to determine the mean trend.
    inds = np.where(np.isfinite(prof_r))[0]
    pp = _robust_polyfit(axis_r[inds], prof_r[inds], (0, 0))
    wing_mean = pp[1]+pp[0]*axis_r

    # Calculate the standard dev of unmasked points from the mean trend.
    stddev = np.sqrt(np.median((prof_r[inds] - wing_mean[inds])**2))
    # Find all outliers that are >3-sigma deviant from the mean.
    inds2 = np.where(np.abs(prof_r[inds] - wing_mean[inds]) > 3*stddev)

    # === Wing fit ===
    # Get fresh right wing profile.
    prof_r2 = np.log10(profile)
    # Mask first order core.
    prof_r2[:(ycens[0]+18)] = np.nan
    # Mask second and third orders.
    if contamination is True:
        for order, ycen in enumerate(ycens):
            if order == 1:
                start = np.max([ycen-18, 0])
                end = np.max([ycen+18, 1])
            elif order == 2:
                start = np.max([ycen-17, 0])
                end = np.max([ycen+17, 1])
            # Set core of each order to NaN.
            prof_r2[start:end] = np.nan
    # Mask outliers
    prof_r2[inds[inds2]] = np.nan
    # Mask edge of the detector.
    prof_r2[-4:] = np.nan
    # Indices of all unmasked points in the left wing.
    inds3 = np.isfinite(prof_r2)

    # Fit with a 9th order polynomial.
    fit_ax = axis_r[inds3]
    fit_prof = prof_r2[inds3]
    # Use np.polyfit for a first estimate of the coefficients.
    pp_r0 = np.polyfit(fit_ax, fit_prof, 9)
    # Robust fit using the polyfit results as a starting point.
    pp_r = _robust_polyfit(fit_ax, fit_prof, pp_r0)

    # === Stitching ===
    # Find pixel to stitch right wing fit - try to retain as much of the
    # original data as possible.
    if contamination is True:
        # Option 3: If completely contaminated by O2.
        # Option 2: Some contamination from O2.
        # Option 1: In uncontaminated case, smooth over background noise.
        stitch_pix = np.min([ycens[0]+50, np.max([ycens[1]-20, ycens[0]+14])])
    else:
        stitch_pix = ycens[0]+50
    jjr = stitch_pix
    # Pad the right axis.
    axis_r_pad = np.arange(len(axis_r[stitch_pix:])+pad) + axis_r[stitch_pix]
    iir = np.where(axis_r_pad == jjr)[0][0]
    # Join right wing to old trace profile.
    newprof = np.concatenate([profile[:jjr], 10**np.polyval(pp_r, axis_r_pad)[iir:]])

    # Interpolate between wing and core profiles for continuity.
    # Get pixels and values around the joint.
    xs = np.concatenate([jjr-1-np.arange(3), np.arange(3)+jjr+1])
    ys = newprof[xs]
    # Interpolation over the above pixels.
    interp = np.polyfit(xs, ys, 5)
    xnew = np.arange(2)+(jjr-1)
    # Insert interpolation.
    newprof[(jjr-1):(jjr+1)] = np.polyval(interp, xnew)

    # ====== Reconstruct left wing ======
    # Mask the core of the first order, and fit a third order polynomial to all
    # remaining pixels to capture the wing behaviour near the core. Reuse the
    # 7th order solution from the right wing for the extended behaviour.
    # Get the profile for the left wing in log space.
    prof_l = np.log10(profile[:(ycens[0]-12)])
    # and corresponding axis.
    axis_l = np.arange(dimy)[:(ycens[0]-12)]
    # Fit with third order polynomial - exclude detector edge.
    # Use np.polyfit for a first estimate of the coefficients.
    pp_l0 = np.polyfit(axis_l[3:], prof_l[3:], 5)
    # Robust fit using the polyfit results as a starting point.
    pp_l = _robust_polyfit(axis_l[3:], prof_l[3:], pp_l0)

    # === Stitching ===
    # Find pixels to stitch left wing fit.
    jjl2 = np.max([4, ycens[0]-50])  # Join core and extended fits.
    # Pad the left axis.
    axis_l_pad = np.linspace(axis_l[0]-pad, axis_l[-1], len(axis_l)+pad)
    #  Mirror of axis_l_pad to the right.
    axis_l_pad2 = np.arange(len(axis_l_pad)) + axis_r[ycens[0]+14]
    iil2 = np.where(axis_l_pad == jjl2)[0][0]
    # Join left wing to old trace profile.
    newprof = np.concatenate([10**np.polyval(pp_r, axis_l_pad2)[::-1][:iil2],
                              #10**np.polyval(pp_l, axis_l_pad)[iil2:iil],
                              newprof[jjl2:]])

    # Interpolate between wing and core profiles for continuity.
    # Get pixels and values around the joint.
    xs = np.concatenate([jjl2-1-np.arange(3), np.arange(3)+jjl2+1])
    ys = newprof[xs]
    # Interpolation over the above pixels.
    interp = np.polyfit(xs, ys, 5)
    xnew = np.arange(2)+(jjl2-1)
    # Insert interpolation.
    newprof[(jjl2-1):(jjl2+1)] = np.polyval(interp, xnew)

    # Set any negatives to zero
    newprof[newprof < 0] = 0

    # Do diagnostic plot if requested.
    if verbose is True:
        plotting._plot_wing_reconstruction(profile, ycens, axis_r[inds3],
                                           prof_r2[inds3], axis_l_pad,
                                           axis_r_pad, pp_r, newprof, **kwargs)

    return newprof


def reconstruct_wings2(frame, ycen, pad_factor=1):
    '''
    Depreciated!
    Takes a reconstructed trace profile which has been truncated about the
    centroid and reconstructs the extended wing structure using an exponential
    profile. Also adds padding in the spatial direction by extending the
    exponetial fit.

    Parameters
    ----------
    frame : np.ndarray of float (2D)
        Empirical trace model.
    ycen : np.array of float
        Y-pixel centroid positions.
    pad_factor : int
        Multiplicative padding factor on the spatial axis. Defaults to 1 (no
        padding).

    Returns
    -------
    newframe : np.ndarray of float (2D)
        Trace model with reconstructed extended wing structure, and required
        padding.
    '''

    # Create new detector array and spatial axis taking into account padding.
    newframe = np.zeros(((pad_factor)*frame.shape[0], frame.shape[1]))
    fullax = np.arange(frame.shape[0])
    fullax_pad = np.arange((pad_factor)*frame.shape[0]) - (frame.shape[0]/2)*(pad_factor-1)

    # Loop over each column on the detector.
    for col in range(frame.shape[1]):

        # Temporary hack for NaN columns
        if np.any(np.isnan(frame[:, col])):
            continue

        # Get the centroid Y-position.
        cen = int(round(ycen[col], 0))

        # Extract the left wing
        start = np.max([0, cen-75])
        ax_l = np.arange(start, cen-9)
        lwing = np.log10(frame[start:(cen-9), col])
        # Find where the log is finite
        lwing_noi = lwing[np.isfinite(lwing)]
        ax_l_noi = ax_l[np.isfinite(lwing)]
        # Fit a first order polynomial to the finite value of the log wing.
        # Equivalent to fitting an exponential to the wing.
        pp_l = np.polyfit(ax_l_noi, lwing_noi, 1)

        # Locate pixels for stitching - where the left wing goes to zero.
        ii_l = np.where(np.isinf(lwing))[0][-1]
        # Location in full axis.
        jj_l = np.where(fullax == ax_l[ii_l])[0][0]
        # Location in padded axis.
        kk_l = np.where(fullax_pad == ax_l[ii_l])[0][0]

        # Extract the right wing
        end = np.min([cen+50, 255])
        ax_r = np.arange(cen+9, end)
        rwing = np.log10(frame[(cen+9):end, col])
        # Find where the log is finite
        rwing_noi = rwing[np.isfinite(rwing)]
        ax_r_noi = ax_r[np.isfinite(rwing)]
        # Fit a first order polynomial to the finite value of the log wing.
        # Equivalent to fitting an exponential to the wing.
        pp_r = np.polyfit(ax_r_noi, rwing_noi, 1)

        # Locate pixels for stitching - where the right wing goes to zero.
        ii_r = np.where(np.isinf(rwing))[0][0]
        jj_r = np.where(fullax == ax_r[ii_r])[0][0]
        kk_r = np.where(fullax_pad == ax_r[ii_r])[0][0]

        # Stitch the wings to the original profile, add padding if necessary.
        newcol = np.concatenate([10**(np.polyval(pp_l, fullax_pad[:kk_l])),
                                 frame[jj_l:jj_r, col],
                                 10**(np.polyval(pp_r, fullax_pad[kk_r:]))])

        try:
            # Find any remaining pixels where the profile is zero.
            inds = np.where(newcol == 0)[0][0]
            # If there are remainining zeros, replace with mean of neighbours.
            newcol[inds] = np.mean([newcol[inds-1], newcol[inds+1]])
        except IndexError:
            pass

        # Renormalize the column.
        newframe[:, col] = newcol / np.nansum(newcol)

    return newframe


def replace_badpix(clear, badpix_mask, fill_negatives=True, verbose=False):
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
    for y, x in tqdm(zip(ys, xs), total=len(ys), disable=not verbose):
        # Get coordinates of pixels in the 5x5 box.
        starty = np.max([(y-2), 0])
        endy = np.min([(y+2), dimy])
        startx = np.max([0, (x-2)])
        endx = np.min([dimx, (x+2)])
        # calculate replacement value to be median of surround pixels.
        rep_val = np.nanmedian(clear[starty:endy, startx:endx])
        i = 1
        # if the median value is still bad, widen the surrounding region
        while np.isnan(rep_val) or rep_val <= 0:
            starty = np.max([(y-2-i), 0])
            endy = np.min([(y+2+i), dimy])
            startx = np.max([0, (x-2-i)])
            endx = np.min([dimx, (x+2-i)])
            rep_val = np.nanmedian(clear[starty:endy, startx:endx])
            i += 1
        # Replace bad pixel with the new value.
        clear_r[y, x] = rep_val

    return clear_r


def _robust_polyfit(x, y, p0):
    '''Wrapper around scipy's least_squares fitting routine implementing the
     Huber loss function - to be more resistant to outliers.

    Parameters
    ----------
    x : list
        Data describing dependant variable.
    y : list
        Data describing independant variable.
    p0 : tuple
        Initial guess straight line parameters. The length of p0 determines the
        polynomial order to be fit - i.e. a length 2 tuple will fit a 1st order
        polynomial, etc.

    Returns
    -------
    res.x : list
        Best fitting parameters of the desired polynomial order.
    '''

    def poly_res(p, x, y):
        '''Residuals from a polynomial'''
        return np.polyval(p, x) - y

    # Preform outlier resistant fitting.
    res = least_squares(poly_res, p0, loss='huber', f_scale=0.1, args=(x, y))
    return res.x
