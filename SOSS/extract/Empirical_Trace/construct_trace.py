#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 9:35 2020

@author: MCR

File containing the necessary functions to create an empirical
interpolated trace model in the overlap region for SOSS order 1.
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import webbpsf
import sys
tppath = '/Users/michaelradica/Documents/GitHub/jwst-mtl/SOSS/trace'
sys.path.insert(1, tppath)
import tracepol as tp
sspath = '/Users/michaelradica/Documents/GitHub/jwst-mtl/SOSS/extract/simple_solver/'
sys.path.insert(0, sspath)
import simple_solver as ss


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
        wave_range = np.linspace(2.2, 2.8, 7)
    # Red anchor is 2.5µm with F277W exposure.
    else:
        wave_range = np.linspace(2.2, 2.5, 7)

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
                print('No monochromatic PSF found for {0:.1f}µm and WFE realization {1:.0f}.'
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
        rnge2 = np.linspace(0, round(1280*(2.5/2.2), 0) - 1, 1280)
        offset2 = rnge2[640] - 640
        new2 = np.interp(np.arange(1280), rnge2 - offset2,
                         np.sum(PSFs[E][0][600:700, :], axis=0))

        # Generate the red wavelength anchor.
        if F277W is False:
            # Use 2.8µm for CLEAR.
            rnger = np.linspace(0, round(1280*(2.5/2.8), 0) - 1, 1280)
        else:
            # Or 2.5µm for F277W.
            rnger = np.linspace(0, round(1280*(2.5/2.5), 0) - 1, 1280)
        offsetr = rnger[640] - 640
        # Remove lambda/D scaling.
        newr = np.interp(np.arange(1280), rnger - offsetr,
                         np.sum(PSFs[E][6][600:700, :], axis=0))

        # Loop over all monochrmatic PSFs to determine interpolation coefs.
        for f, wave in enumerate(wave_range):
            # Lists for running counts of indicies and model residuals.
            resid, II, JJ = [], [], []

            # Rescale monochrmotic PSF to remove lambda/D.
            newrnge = np.linspace(0, round(1280*(2.5/wave), 0) - 1, 1280)
            newoffset = newrnge[640] - 640
            newpsf = np.interp(np.arange(1280), newrnge - newoffset,
                               np.sum(PSFs[E][f][600:700, :], axis=0))

            # Brute force the coefficient determination.
            # It should only take a minute or so.
            for i in range(1, 100):
                for j in range(1, 100):
                    I = i / 10
                    J = j / 10
                    # Create a test profile which is a mix of the two anchors.
                    mix = (I*new2 + J*newr) / (I + J)

                    # Save current iteration in running counts.
                    resid.append(np.sum(np.abs(newpsf - mix)[450:820]))
                    II.append(I)
                    JJ.append(J)

            # Determine which combination of indices minimizes model residuals.
            ind = np.where(resid == np.min(resid))[0][0]
            # Save the normalized indices for this wavelength.
            wb.append(II[ind] / (II[ind] + JJ[ind]))
            wr.append(JJ[ind] / (II[ind] + JJ[ind]))

    wb = np.reshape(wb, (10, 7))
    wr = np.reshape(wr, (10, 7))

    # Fit a second order polynomial to the mean of the interpolation indices.
    pb = np.polyfit(wave_range, np.mean(wb, axis=0), 2)
    pr = np.polyfit(wave_range, np.mean(wr, axis=0), 2)

    # Show the diagnostic plot if necessary.
    if doplot is True:
        _plot_interpmodel(wave_range, wb, wr, pb, pr)

    return pb, pr


def _chromescale(wave, profile, invert=False):
    '''Utility function to remove the lambda/D chromatic PSF scaling by
    interpolating a monochromatic PSF function onto a standard axis.

    Parameters
    ----------
    wave : float
        Wavelength corresponding to the input 1D PSF profile.
    profile : np.array of float
        1D PSF profile to be rescaled.
    invert : bool
        If True, add back the lambda/D scaling instead of removing it.

    Returns
    -------
    new : np,.array of float
        Rescaled 1D PSF profile.
    '''

    # Create the standard axis
    rnge = np.linspace(0, round(49*(2.5/wave), 0) - 1, 49)
    offset = rnge[24] - 24

    # Interpolate the profile onto the standard axis.
    if invert is False:
        new = np.interp(np.arange(49), rnge - offset, profile)
    # Or interpolate the profile from the standard axis to re-add
    # the lambda/D scaling.
    else:
        new = np.interp(rnge - offset, np.arange(49), profile)

    return new


def construct_order1(clear, F277, do_plots=False, filename=None):
    '''This creates the full order 1 trace profile model. The region
    contaminated by the second order is interpolated from the CLEAR and F277W
    exposures, or just from the CLEAR exposure and a standard red anchor if
    no F277W exposure is available.
    The steps are as follows:
        1. Fit the optics model to the data centroids to determine the correct
           rotation and offset parameters.
        2. Determine the red and blue anchor profiles for the interpolation.
        3. Construct the interpolated profile at each wavelength in the
           overlap region.
        4. Stitch together the original CLEAR exposure and the interpolated
           trace model (as well as the F277W exposure if provided).
    This is the main function that the end user will call.

    Parameters
    ----------
    clear : np.array of float (2D)
        NIRISS SOSS CLEAR exposure dataframe.
    F277 : np.array of float (2D)
        NIRISS SOSS F277W filter exposure dataframe. If no F277W exposure
        is available, pass None for this parameter.
    do_plots : bool
        Whether to show the diagnostic plots.
    filename : str
        Name of file to which to write the trace model. If a filename is
        provided, the trace model will be written to disk instead of returned
        by the function.

    Returns
    -------
    O1frame : np.array of float (2D)
        Complete interpolated order 1 trace model, if not written to disk.
    '''

    # Get the centroid positions from the optics model and the data.
    pixels = np.linspace(0, 2047, 2048)+0.5
    xOM, yOM, tp2 = ss.get_om_centroids()  # OM
    xdat, ydat = ss.get_uncontam_centroids(clear, atthesex=pixels)  # data
    # Overplot the data centroids on the CLEAR exposure if necessary
    # to verify accuracy.
    if do_plots is True:
        _plot_centroid(clear, xdat, ydat)

    # Use MCMC to brute force find the best fitting angle, rotation center,
    # and offset parameters necessary to fit the OM to the data centroids.
    ang_samp = ss._do_emcee(xOM, yOM, xdat, ydat)
    # Show the MCMC results in a corner plot if necessary.
    if do_plots is True:
        ss._plot_corner(ang_samp)
    # The MCMC results have been well behaved in all test cases.
    flat_samples = ang_samp.get_chain(discard=500, thin=15, flat=True)
    ang = np.percentile(flat_samples[:, 0], 50)
    xanch = np.percentile(flat_samples[:, 1], 50)
    yanch = np.percentile(flat_samples[:, 2], 50)
    xshift = np.percentile(flat_samples[:, 3], 50)
    yshift = np.percentile(flat_samples[:, 4], 50)

    # Determine the anchor profiles - blue anchor.
    # Note: Y-values returned from OM are inverted relative to the UdeM
    # coordinate system.
    # Get OM centroid pixel coords at 2.2µm.
    xom22 = tp.wavelength_to_pix(2.2, tp2, 1)[0]
    yom22 = 256 - tp.wavelength_to_pix(2.2, tp2, 1)[1]
    # Use rot params to find location of the 2.2µm centroids in the data frame.
    xd22, yd22 = ss.rot_centroids(ang, xanch, yanch, xshift, yshift, xom22,
                                  yom22, fill_det=False)
    xd22, yd22 = int(round(xd22, 0)), int(round(yd22, 0))
    # Extract the 2.2µm anchor profile from the data.
    Banch = clear[(yd22-24):(yd22+25), xd22]
    # Remove the lambda/D scaling.
    Banch = _chromescale(2.2, Banch)

    # Determine the anchor profiles - red anchor.
    if F277 is None:
        # ********TODO1 - Needs to be updated!!***********
        # Use a simulated F277W exposure as the 2.8µm anchor if no F277W
        # exposure is provided to ensure that there is no contamination.
        stand = fits.open('/Users/michaelradica/Documents/School/Ph.D./Research/SOSS/Extraction/simu_F277_CLEAR/f277.fits')[0].data[::-1, :]
        # Find the appropriate X and Y centroids.
        # The simulation is in the OM coordinate system.
        # Get OM centroid pixel coords at 2.8µm.
        xom28 = int(round(tp.wavelength_to_pix(2.8, tp2, 1)[0], 0))
        yom28 = int(round(256 - tp.wavelength_to_pix(2.8, tp2, 1)[1], 0))
        # Extract and rescale the 2.8µm profile.
        Ranch = stand[(yom28-24):(yom28+25), xom28]
        Ranch = _chromescale(2.8, Ranch)

    else:
        # If an F277W exposure is provided, only interpolate out to 2.5µm.
        # From 2.5 - 2.8µm we have perfect knowledge of the order 1 trace.
        # Get OM centroid pixel coords at 2.5µm
        xom25 = tp.wavelength_to_pix(2.5, tp2, 1)[0]
        yom25 = 256 - tp.wavelength_to_pix(2.5, tp2, 1)[1]
        # Transform into the data frame.
        xd25, yd25 = ss.rot_centroids(ang, xanch, yanch, xshift, yshift, xom25,
                                      yom25, fill_det=False)
        xd25, yd25 = int(round(xd25, 0)), int(round(yd25, 0))
        # Extract and rescale the 2.5µm profile.
        Ranch = F277[(yd25-24):(yd25+25), xd25]
        Ranch = _chromescale(2.5, Ranch)

    # The interpolation polynomial coefficients, calculated via
    # calc_interp_coefs. These have been robust in all tests, and the hope is
    # that they will be robust for all future observations.
    if F277 is None:
        coef_b = [1.0311255, -6.81906843, 11.01065922]
        coef_r = [-1.0311255, 6.81906843, -10.01065922]
    else:
        coef_b = [2.04327661, -12.90780135, 19.50319999]
        coef_r = [-2.04327661, 12.90780135, -18.50319999]

    # Create the interpolated order 1 PSF.
    map2D = np.zeros((256, 2048))*np.nan
    # Pixel coordinate at which to start and end the interpolation in OM frame.
    # ********TODO2 - Needs to be updated!!***********
    if F277 is None:
        start = 4.5
        rlen = 606
    else:
        start = 307.5
        rlen = 303

    # Transform OM centroids onto the detector.
    # Get OM X-pixel values for the region to be interpolated.
    cenx_om = np.arange(rlen) + start
    # Find the wavelength at each X centroid
    lmbda = tp.specpix_to_wavelength(cenx_om, tp2, 1)[0]
    # Y centroid at each wavelength
    ceny_om = 256 - tp.wavelength_to_pix(lmbda, tp2, 1)[1]
    # Transform the OM centroids onto the detector.
    cenx_d, ceny_d = ss.rot_centroids(ang, xanch, yanch, xshift, yshift,
                                      cenx_om, ceny_om, fill_det=False)

    # Create an interpolated 1D PSF at each required position.
    for i, vals in enumerate(zip(cenx_d, ceny_d, lmbda)):
        cenx, ceny, lbd = vals[0], vals[1], vals[2]
        # Evaluate the interpolation polynomials at the current wavelength.
        wb_i = np.polyval(coef_b, lbd)
        wr_i = np.polyval(coef_r, lbd)
        # Construct the interpolated profile.
        prof_int = (wb_i * Banch + wr_i * Ranch)
        # Re-add the lambda/D scaling.
        prof_int_cs = _chromescale(lbd, prof_int, invert=True)

        # Put the interpolated profile on the detector.
        axis = np.linspace(-24, 24, 49) + ceny
        inds = np.where((axis < 256) & (axis >= 0))[0]
        profile = np.interp(np.arange(256), axis[inds], prof_int_cs[inds])

        # Subtract the noisy wing edges.
        map2D[:, int(cenx)] = profile - np.nanpercentile(profile, 2.5)

        # Note detector coordinates of the edges of the interpolated region.
        if i == rlen - 1:
            bend = int(cenx)
        elif i == 0:
            rend = int(cenx)

    # Stitch together the interpolation and data.
    newmap = np.zeros((256, 2048))
    # Insert interpolated data
    newmap[:, rend:bend] = map2D[:, rend:bend]
    # Bluer region is known from the CLEAR exposure.
    newmap[:, bend:2048] = clear[:, bend:2048]
    if F277 is None:
        # Insert interpolated data to the red as well if no F277W.
        newmap[:, 0:rend] = map2D[:, 0:rend]
    # Or add on the F277W frame to the red if available.
    else:
        newmap[:, 0:rend] = F277[:, 0:rend]
    # Normalize the profile in each column.
    newmap = newmap / np.nanmax(newmap, axis=0)
    # Create a mask to remove the second order from the CLEAR data.
    O1frame = mask_order(newmap, xdat, ydat)

    # Write the trace model to disk if requested.
    if filename is not None:
        hdu = fits.PrimaryHDU()
        hdu.data = O1frame
        hdu.writeto('%s.fits' % filename, overwrite=True)
    # Or return the frame itself.
    else:
        return O1frame


def loicpsf(wavelist=None, wfe_real=None, filepath=''):
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
    '''

    if wavelist is None:
        # List of wavelengths to generate PSFs for
        wavelist = np.linspace(0.5, 5.2, 95) * 1e-6
    # Dimension of the PSF in native pixels
    pixel = 128
    # Pixel oversampling factor
    oversampling = 10

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
        print('Calculating PSF at wavelength ',
              round(wave/1e-6, 2), ' microns')
        psf = niriss.calc_psf(monochromatic=wave, fov_pixels=pixel,
                              oversample=oversampling, display=False)

        # Save psf realization to disk
        text = '{0:5f}'.format(wave*1e+6)
        psf.writeto(str(filepath)+'SOSS_os'+str(oversampling)+'_'+str(pixel)
                    + 'x'+str(pixel)+'_'+text+'_'+str(wfe_real)+'.fits',
                    overwrite=True)


def mask_order(frame, xpix, ypix):
    '''Create a pixel mask to isolate only the detector pixels
    belonging to a specific diffraction order.

    Parameters
    ----------
    frame : np.array of float (2D)
        Science data frame.
    xpix : np.array of int
        Data x centroids for the desired order
    ypix : np.array of int
        Data y centroids for the desired order

    Returns
    -------
    O1frame : np.array of float (2D)
        The input data frame, with all pixels other than those
        within +/- 20 pixels of yCV masked.
    '''

    mask = np.zeros([256, 2048])
    xx = np.round(xpix, 0).astype(int)
    yy = np.round(ypix, 0).astype(int)
    xr = np.linspace(np.min(xx), np.max(xx), np.max(xx)+1).astype(int)

    # Set all pixels within the extent of the order 1 trace to 1 in the mask.
    for xxx, yyy in zip(xr, yy):
        mask[(yyy-21):(yyy+20), xxx] = 1

    O1frame = (mask * frame) / np.nanmax(mask*frame, axis=0)

    return O1frame


def _plot_centroid(clear, xpix, ypix):
    '''Utility function to overplot the trace centroids extracted from
    the data over the data isetfl to verify accuracy. Called by makemod.
    '''
    plt.figure(figsize=(15, 3))
    plt.plot(xpix, ypix, c='black')
    plt.imshow(clear/np.nanmax(clear, axis=0), origin='lower', cmap='jet')

    return None


def _plot_interpmodel(waves, nw1, nw2, p1, p2):
    '''Plot the diagnostic results of the derive_model function. Four plots
    are generated, showing the normalized interpolation coefficients for the
    blue and red anchors for each WFE realization, as well as the mean trend
    across WFE for each anchor profile, and the resulting polynomial fit to
    the mean trends.

    Parameters
    ----------
    waves : np.array of float
        Wavelengths at which WebbPSF monochromatic PSFs were created.
    nw1 : np.array of float
        Normalized interpolation coefficient for the blue anchor
        for each PSF profile.
    nw2 : np.array of float
        Normalized interpolation coefficient for the red anchor
        for each PSF profile.
    p1 : np.array of float
        Polynomial coefficients of the fit to the mean interpolation
        coefficients for the blue anchor.
    p2 : np.array of float
        Polynomial coefficients of the fit to the mean interpolation
        coefficients for the red anchor.
    '''

    f, ax = plt.subplots(2, 2, figsize=(14, 6))
    for i in range(10):
        ax[0, 0].plot(waves, nw1[i])
        ax[1, 0].plot(waves, nw2[i])

    ax[0, 1].plot(waves, np.mean(nw1, axis=0))
    ax[0, 1].plot(waves, np.mean(nw2, axis=0))

    ax[-1, 0].set_xlabel('Wavelength [µm]', fontsize=14)
    ax[-1, 1].set_xlabel('Wavelength [µm]', fontsize=14)

    y1 = np.polyval(p1, waves)
    y2 = np.polyval(p2, waves)

    ax[1, 1].plot(waves, y1, c='r', ls=':')
    ax[1, 1].plot(waves, y2, c='b', ls=':')
    ax[1, 1].plot(waves, np.mean(nw1, axis=0), c='b', label='Blue Anchor')
    ax[1, 1].plot(waves, np.mean(nw2, axis=0), c='r', label='Red Anchor')
    ax[1, 1].set_xlim(np.min(waves), np.max(waves))
    ax[1, 1].legend(loc=1, fontsize=12)

    f.tight_layout()
