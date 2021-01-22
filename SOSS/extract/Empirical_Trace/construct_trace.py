#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 9:35 2020

@author: MCR

File containing the necessary functions to create an empirical
interpolated trace model in the overlap region for SOSS order 1.
"""

import os
import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares
import webbpsf
from SOSS.trace import tracepol as tp
from SOSS.extract.empirical_trace import centroid as ctd
from SOSS.extract.empirical_trace import plotting as plotting
from SOSS.extract.overlap import TrpzOverlap, TrpzBox
from SOSS.extract.throughput import ThroughputSOSS
from SOSS.extract.convolution import WebbKer


def build_empirical_trace(clear, F277W, filename='spatial_profile.fits',
                          pad=(0, 0), doplot=False, verbose=False):
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
    doplot : bool
    verbose : bool
    '''

    # Print overwrite warning if output file already exists.
    if os.path.exists(filename):
        print('Output file {} already exists. It will be overwritten'.format(filename))
    # Get the centroid positions for both orders from the data.
    if verbose is True:
        print('Getting trace centroids...')
    centroids, rot_pars = ctd.get_contam_centroids(clear, doplot=doplot,
                                                   bound=False,
                                                   showprogress=verbose)

    # Overplot the data centroids on the CLEAR exposure if desired.
    if doplot is True:
        plotting._plot_centroid(clear, centroids)

    # Construct the first order profile.
    if verbose is True:
        print('Interpolating the first order trace...')
    o1frame = construct_order1(clear, F277W, rot_pars, centroids, pad=pad[0],
                               doplot=doplot)
    # Pad the spectral axis.
    if pad[1] != 0:
        if verbose is True:
            print('Adding padding to first order spectral axis...')
        o1frame = pad_spectral_axis(o1frame, centroids['order 1'][0],
                                    centroids['order 1'][1], pad=pad[1])

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


def calc_interp_coefs(make_psfs=False, doplot=True, F277W=True, filepath=''):
    '''Function to calculate the interpolation coefficients necessary to
    construct a monochromatic PSF profile at any wavelength between
    the two 1D PSF anchor profiles. Linear combinations of the blue and red
    anchor are iteratively fit to each intermediate wavelength to find the
    best fitting combination. The mean linear coefficients across the 10 WFE
    error realizations are returned for each wavelengths.
    When called, 2D moncochromatic PSF profiles will be generated and saved
    to disk if the user does not already have them available.
    This should not need to be called by the end user except in rare cases.

    Parameters
    ----------
    make_psfs : bool
        Whether or not WebbPSF will have to generate the monochromatic
        PSFs used for the fitting.
    doplot : bool
        Whether to show the diagnostic plots for the model derivation.
    F277W : bool
        Set to False if no F277W exposure is available for the observation.
        Finds coefficients for the entire 2.2 - 2.8µm region in this case.
    filepath : str
        Path to directory containing the WebbPSF monochromatic PSF fits
        files, or the directory to which they will be stored when made.
        Defaults to the current directory.

    Returns
    -------
    pb : np.array of float
        Polynomial coefficients of the interpolation index fits for the
        blue anchor.
    pr : np.array of float
        Polynomial coefficients of the interpolation index fits for the
        red anchor.
    '''

    # Red anchor is 2.8µm without an F277W exposure.
    if F277W is False:
        wave_range = np.linspace(2.1, 2.85, 7)
    # Red anchor is 2.5µm with F277W exposure.
    else:
        wave_range = np.linspace(2.1, 2.45, 7)

    # Read in monochromatic PSFs generated by WebbPSF.
    PSFs = []
    # Loop over all 10 available WFE realizations.
    for i in range(10):
        psf_run = []
        # Create the PSF if user has indicated to.
        if make_psfs is True:
            loicpsf(wavelist=wave_range*1e-6, wfe_real=i)
        # If user already has PSFs generated.
        for w in wave_range:
            try:
                psf_run.append(fits.open('{0:s}SOSS_os10_128x128_{1:.6f}_{2:.0f}.fits'
                                         .format(filepath, w, i))[0].data)
            # Generate missing PSFs if necessary.
            except FileNotFoundError:
                print('No monochromatic PSF found for {0:.2f}µm and WFE realization {1:.0f}.'
                      .format(w, i))
                loicpsf(wavelist=[w*1e-6], wfe_real=i, filepath=filepath)
                psf_run.append(fits.open('{0:s}SOSS_os10_128x128_{1:.6f}_{2:.0f}.fits'
                                         .format(filepath, w, i))[0].data)
        PSFs.append(psf_run)

    # Determine specific interpolation coefficients for all WFEs
    wb, wr = [], []

    for E in range(10):
        # Generate the blue wavelength anchor.
        # The width of the 1D PSF has lambda/D dependence, so rescale all
        # profiles to a common wavelength to remove these chromatic effects.
        rngeb = np.linspace(0, round(1280*(2.5/2.1), 0) - 1, 1280)
        offsetb = rngeb[640] - 640
        newb = np.interp(np.arange(1280), rngeb - offsetb,
                         np.sum(PSFs[E][0][600:700, :], axis=0))

        # Generate the red wavelength anchor.
        if F277W is False:
            # Use 2.85µm for CLEAR.
            rnger = np.linspace(0, round(1280*(2.5/2.85), 0) - 1, 1280)
        else:
            # Or 2.42µm for F277W.
            rnger = np.linspace(0, round(1280*(2.5/2.45), 0) - 1, 1280)
        offsetr = rnger[640] - 640
        # Remove lambda/D scaling.
        newr = np.interp(np.arange(1280), rnger - offsetr,
                         np.sum(PSFs[E][6][600:700, :], axis=0))

        # Loop over all monochrmatic PSFs to determine interpolation coefs.
        for f, wave in enumerate(wave_range):
            # Lists for running counts of indicies and model residuals.
            resid, ind_i, ind_j = [], [], []

            # Rescale monochrmotic PSF to remove lambda/D.
            newrnge = np.linspace(0, round(1280*(2.5/wave), 0) - 1, 1280)
            newoffset = newrnge[640] - 640
            newpsf = np.interp(np.arange(1280), newrnge - newoffset,
                               np.sum(PSFs[E][f][600:700, :], axis=0))

            # Brute force the coefficient determination.
            for i in range(1, 100):
                for j in range(1, 100):
                    i /= 10
                    j /= 10
                    # Create a test profile which is a mix of the two anchors.
                    mix = (i*newb + j*newr) / (i + j)

                    # Save current iteration in running counts.
                    resid.append(np.sum(np.abs(newpsf - mix)[450:820]))
                    ind_i.append(i)
                    ind_j.append(j)

            # Determine which combination of indices minimizes model residuals.
            ind = np.where(resid == np.min(resid))[0][0]
            # Save the normalized indices for this wavelength.
            wb.append(ind_i[ind] / (ind_i[ind] + ind_j[ind]))
            wr.append(ind_j[ind] / (ind_i[ind] + ind_j[ind]))

    wb = np.reshape(wb, (10, 7))
    wr = np.reshape(wr, (10, 7))

    # Fit a second order polynomial to the mean of the interpolation indices.
    pb = np.polyfit(wave_range, np.mean(wb, axis=0), 2)
    pr = np.polyfit(wave_range, np.mean(wr, axis=0), 2)

    # Show the diagnostic plot if necessary.
    if doplot is True:
        plotting._plot_interpmodel(wave_range, wb, wr, pb, pr)

    return pb, pr, wb, wr


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


def construct_order1(clear, F277, rot_params, ycens, pad=0, doplot=False):
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
    pad : int
        Number of pixels of padding to add on both ends of the spatial axis.
    doplot : bool
        if True, do diagnostic plotting.

    Returns
    -------
    newmap : np.array of float (2D)
        Interpolated order 1 trace model with padding.
    '''

    # Determine the anchor profiles - blue anchor.
    # Note: Y-values returned from OM are inverted relative to the UdeM
    # coordinate system.
    ###### TODO Switch this to use wavecal ######
    xOM, yOM, tp2 = ctd.get_om_centroids()
    # Get OM centroid pixel coords at 2.1µm.
    xom22 = tp.wavelength_to_pix(2.1, tp2, 1)[0]
    yom22 = 256 - tp.wavelength_to_pix(2.1, tp2, 1)[1]
    # Use rot params to find location of the 2.1µm centroids in the data frame.
    xd22, yd22 = ctd.rot_centroids(*rot_params, xom22, yom22)
    xd22 = np.round(xd22, 0).astype(int)[0]
    yd22 = np.round(yd22, 0).astype(int)[0]
    # Extract the 2.1µm anchor profile from the data.
    Banch = clear[:, xd22]
    # Mask second and third order, reconstruct wing structure and pad.
    cens = [ycens['order 1'][1][xd22], ycens['order 2'][1][xd22],
            ycens['order 3'][1][xd22]]
    Banch = reconstruct_wings(Banch, ycens=cens, contamination=True, pad=pad,
                              doplot=doplot)
    # Remove the lambda/D scaling.
    Banch = _chromescale(2.1, Banch, yd22)
    # Normalize
    Banch /= np.nansum(Banch)

    # Determine the anchor profiles - red anchor.
    if F277 is None:
        # If no F277W exposure is provided, interpolate out to 2.85µm.
        # Generate a simulated 2.85µm PSF.
        stand = loicpsf([2.85*1e-6], save_to_disk=False, oversampling=1,
                        verbose=False)[0][0].data
        # Extract and rescale the spatial profile.
        Ranch = np.sum(stand[60:70, (64-24):(64+25)], axis=0)
        Ranch = _chromescale(2.85, Ranch)
        # Normalize
        Ranch /= np.nansum(Ranch)

        # Interpolation polynomial coeffs, calculated via calc_interp_coefs
        coef_b = [0.86772837, -5.6445105, 9.03072711]
        coef_r = [-0.86772837, 5.6445105, -8.03072711]
        # Pixel coords at which to start and end interpolation in OM frame.
        end = int(round(tp.wavelength_to_pix(2.1, tp2, 1)[0], 0))
        # Determine the OM coords of first pixel on detector
        start = int(round(tp.wavelength_to_pix(2.85, tp2, 1)[0], 0))
        rlen = end - start

    else:
        # If an F277W exposure is provided, only interpolate out to 2.45µm.
        # Redwards of 2.45µm we have perfect knowledge of the order 1 trace.
        # Get OM centroid pixel coords at 2.45µm
        ###### TODO Switch this to use wavecal ######
        xom25 = tp.wavelength_to_pix(2.45, tp2, 1)[0]
        yom25 = 256 - tp.wavelength_to_pix(2.45, tp2, 1)[1]
        # Transform into the data frame.
        xd25, yd25 = ctd.rot_centroids(*rot_params, xom25, yom25)
        xd25 = np.round(xd25, 0).astype(int)[0]
        yd25 = np.round(yd25, 0).astype(int)[0]
        # Extract and rescale the 2.5µm profile.
        Ranch = F277[:, xd25-1]
        # Reconstruct wing structure and pad.
        cens = [ycens['order 1'][1][xd25-1]]
        Ranch = reconstruct_wings(Ranch, ycens=cens, contamination=False,
                                  pad=pad)
        Ranch = _chromescale(2.45, Ranch, yd25)
        # Normalize
        Ranch /= np.nansum(Ranch)

        # Interpolation polynomial coeffs, calculated via calc_interp_coefs
        coef_b = [1.51850915, -9.76581613, 14.80720191]
        coef_r = [-1.51850915,  9.76581613, -13.80720191]
        # Pixel coords at which to start and end interpolation in OM frame.
        end = int(round(tp.wavelength_to_pix(2.1, tp2, 1)[0], 0))
        start = int(round(tp.wavelength_to_pix(2.45, tp2, 1)[0], 0))
        rlen = end - start

    # Create the interpolated order 1 PSF.
    map2D = np.zeros((256+2*pad, 2048))*np.nan
    # Get OM X-pixel values for the region to be interpolated.
    cenx_om = np.arange(rlen) + start
    # Find the wavelength at each X centroid
    lmbda = tp.specpix_to_wavelength(cenx_om, tp2, 1)[0]
    # Y centroid at each wavelength
    ceny_om = 256 - tp.wavelength_to_pix(lmbda, tp2, 1)[1]
    # Transform the OM centroids onto the detector.
    cenx_d, ceny_d = ctd.rot_centroids(*rot_params, cenx_om, ceny_om,
                                       bound=False)

    # Create an interpolated 1D PSF at each required position.
    for i, vals in enumerate(zip(cenx_d, ceny_d, lmbda)):
        cenx, ceny, lbd = vals[0], vals[1], vals[2]
        # Evaluate the interpolation polynomials at the current wavelength.
        wb_i = np.polyval(coef_b, lbd)
        wr_i = np.polyval(coef_r, lbd)
        # Recenter the profile of both anchors on the correct Y-centroid.
        bax = np.arange(256+2*pad)-yd22+ceny
        Banch_i = np.interp(np.arange(256+2*pad), bax, Banch)
        rax = np.arange(256+2*pad)-yd25+ceny
        Ranch_i = np.interp(np.arange(256+2*pad), rax, Ranch)
        # Construct the interpolated profile.
        prof_int = (wb_i * Banch_i + wr_i * Ranch_i)
        # Re-add the lambda/D scaling.
        prof_int_cs = _chromescale(lbd, prof_int, ceny, invert=True)
        # Put the interpolated profile on the detector.
        map2D[:, int(round(cenx, 0))] = prof_int_cs

        # Note detector coordinates of the edges of the interpolated region.
        bend = int(round(cenx, 0))
        if i == 0:
            # 2.85µm (i=0) limit may be off the end of the detector.
            rend = np.max([int(round(cenx, 0)), 0])

    # Stitch together the interpolation and data.
    newmap = np.zeros((256+2*pad, 2048))
    # Insert interpolated data
    newmap[:, rend:bend] = map2D[:, rend:bend]
    # Bluer region is known from the CLEAR exposure.
    # Mask contamination from second and third orders and reconstruct wings.
    for col in range(bend, 2048):
        cens = [ycens['order 1'][1][col], ycens['order 2'][1][col],
                ycens['order 3'][1][col]]
        newmap[:, col] = reconstruct_wings(clear[:, col], ycens=cens, pad=pad)
    if F277 is not None:
        # Add on the F277W frame to the red of the model.
        # Reconstruct wing structure and pad.
        for col in range(rend):
            cens = [ycens['order 1'][1][col]]
            newmap[:, col] = reconstruct_wings(F277[:, col], ycens=cens,
                                               contamination=False, pad=pad)
    # Insert interpolated data to the red of the data.
    else:
        newmap[:, 0:rend] = map2D[:, 0:rend]

    # Column normalize.
    newmap /= np.nansum(newmap, axis=0)
    # Add noise floor to prevent arbitrarily low values in padded wings.
    floor = np.nanpercentile(newmap[pad:(-1-pad), :], 2)
    newmap += floor

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
    extra = TrpzOverlap(*ref_file_args, **extract_params)
    # Preform the extraction with only the first order.
    f_k = extra.extract(data=clear)
    # Rebuild the detector.
    rebuilt = extra.rebuild(f_k)
    rebuilt[np.isnan(rebuilt)] = 0
    # Isolate the second order by subtracting the reconstructed first
    # order from the data
    residual = clear - rebuilt

    return residual


def get_extract_params():
    '''
    '''
    params = {}
    # Map of expected noise (sig)
    bkgd_noise = 20.
    # Oversampling
    params["n_os"] = 1
    # Threshold on the spatial profile
    params["thresh"] = 1e-6

    return params


def get_ref_file_args(o1frame):
    '''
    '''
    path = '/Users/michaelradica/Documents/GitHub/jwst-mtl/SOSS/'
    # List of orders to consider in the extraction
    order_list = [1]

    # Wavelength solution
    wave_maps = []
    wave_maps.append(fits.getdata(path+"extract/Ref_files/wavelengths_m1.fits"))

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
            oversampling=10, verbose=True):
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
    pixel = 128

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


def mask_order(frame, xpix, ypix):
    '''
    Depreciated!
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


def reconstruct_wings(profile, ycens=None, contamination=True, pad=0,
                      doplot=False):
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
    contamination : bool
        If True, profile has contamination from the second and third
        diffraction orders.
    pad : int
        Amount to pad each end of the spartial axis (in pixels).
    doplot : bool
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

    # Convert Y-centroid positions to indices
    ycens = np.atleast_1d(ycens)
    ycens = np.round(ycens, 0).astype(int)
    if contamination is True and ycens.size != 3:
        raise ValueError('Centroids must be provided for first three orders if there is contamination.')

    # ====== Reconstruct left wing ======
    # Get the left wing of the trace profile in log space.
    prof_l = np.log10(profile[0:(ycens[0]-12)])
    # and corresponding axis.
    axis_l = np.arange(256)[0:(ycens[0]-12)]

    # === Outlier masking ===
    # Mask the cores of each order.
    for order, ycen in enumerate(ycens):
        if order == 0:
            start = ycen-25
            end = len(profile)
        elif order == 1:
            start = np.max([ycen-15, 0])
            end = np.max([ycen+15, 1])
        else:
            start = np.max([ycen-10, 0])
            end = np.max([ycen+10, 1])
        # Set core of each order to NaN.
        prof_l[start:end] = np.nan

    # Fit the unmasked part of the wing to determine the mean trend.
    inds = np.where(np.isfinite(prof_l))[0]
    pp = _robust_linefit(axis_l[inds], prof_l[inds], (0, 0))
    wing_mean = pp[0]+pp[1]*axis_l[inds]
    # Calculate the standard dev of unmasked points from the mean trend.
    stddev = np.sqrt(np.median((prof_l[inds] - wing_mean)**2))
    # Find all outliers that are >3-sigma deviant from the mean.
    inds2 = np.where(prof_l[inds] - wing_mean > 3*stddev)

    # === Wing fit ===
    # Get fresh left wing profile.
    prof_l2 = np.log10(profile[0:(ycens[0]-12)])
    # Mask second and third orders.
    if contamination is True:
        for order, ycen in enumerate(ycens):
            if order == 1:
                start = np.max([ycen-15, 0])
                end = np.max([ycen+15, 1])
            elif order == 2:
                start = np.max([ycen-10, 0])
                end = np.max([ycen+10, 1])
            # Set core of each order to NaN.
            prof_l2[start:end] = np.nan
    # Mask outliers
    prof_l2[inds[inds2]] = np.nan

    # Indices of all unmasked points in the left wing.
    inds3 = np.isfinite(prof_l2)
    # Fit with a 7th order polynomial.
    pp_l = np.polyfit(axis_l[inds3], prof_l2[inds3], 7)

    # ====== Reconstruct right wing ======
    # Get the profile for the right wing in log space.
    prof_r = np.log10(profile[(ycens[0]+12):])
    # and corresponding axis.
    axis_r = np.arange(256)[(ycens[0]+12):]
    # Fit with third order polynomial.
    pp_r = np.polyfit(axis_r, prof_r, 3)

    # ===== Stitching =====
    # Find pixel to stitch right wing fit.
    jjr = ycens[0]+15
    iir = np.where(axis_r == jjr)[0][0]
    # Pad the right axis.
    axis_r_pad = np.linspace(axis_r[0], axis_r[-1]+pad, len(axis_r)+pad)
    # Join right wing to old trace profile.
    newprof = np.concatenate([profile[:jjr], 10**np.polyval(pp_r, axis_r_pad)[iir:]])

    # Find pixel to stitch left wing fit.
    jjl = ycens[0]-15
    # Pad the left axis.
    axis_l_pad = np.linspace(axis_l[0]-pad, axis_l[-1], len(axis_l)+pad)
    iil = np.where(axis_l_pad == jjl)[0][0]
    # Join left wing to old trace profile.
    newprof = np.concatenate([10**np.polyval(pp_l, axis_l_pad)[:iil], newprof[jjl:]])

    if doplot is True:
        plotting._plot_wing_reconstruction(profile, ycens, axis_l, axis_l_pad,
                                           axis_r_pad, pp_l, pp_r, prof_l2,
                                           newprof)

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


def _robust_linefit(x, y, p0):
    '''Wrapper around scipy's least_squares line fitting routine implementing
    the Huber loss function - to be more resistant to outliers.

    Parameters
    ----------
    x : list
        Data describing dependant variable.
    y : list
        Data describing independant variable.
    p0 : tuple
        Initial guess straight line parameters.

    Returns
    -------
    res.x : list
        Best fitting parameters of a straight line.
    '''

    def line_res(p, x, y):
        '''Residuals from a straight line'''
        return p[0]+p[1]*x-y

    # Preform outlier resistant fitting.
    res = least_squares(line_res, p0, loss='huber', f_scale=0.1, args=(x, y))

    return res.x
