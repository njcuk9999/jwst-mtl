#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 11:31 2021

@author: MCR

Functions to calculate the interpolation coefficients for the empirical trace
construction.
"""

from astropy.io import fits
import numpy as np
import os
import pandas as pd
import webbpsf
from SOSS.extract.empirical_trace import plotting

# Local path to reference files.
path = '/Users/michaelradica/Documents/GitHub/jwst-mtl/SOSS/extract/empirical_trace/Ref_files/'


def calc_interp_coefs(F277W=True, verbose=0):
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
    F277W : bool
        Set to False if no F277W exposure is available for the observation.
        Finds coefficients for the entire 2.1 - 2.9µm region in this case.
    verbose : int
        Level of verbosity.

    Returns
    -------
    pb : np.array of float
        Polynomial coefficients of the interpolation index fits for the
        blue anchor.
    pr : np.array of float
        Polynomial coefficients of the interpolation index fits for the
        red anchor.
    '''

    if verbose != 0:
        print('Calculating interpolation coefficients.')
    # Red anchor is 2.9µm without an F277W exposure.
    if F277W is False:
        wave_range = np.linspace(2.1, 2.9, 7)
    # Red anchor is 2.45µm with F277W exposure.
    else:
        wave_range = np.linspace(2.1, 2.45, 7)

    # Read in monochromatic PSFs generated by WebbPSF.
    PSFs = []
    # Loop over all 10 available WFE realizations.
    for i in range(10):
        psf_run = []
        # Import the PSFs,
        for w in wave_range:
            # If the user already has the PSFs generated, import them.
            try:
                infile = path+'{0:s}SOSS_os10_128x128_{1:.6f}_{2:.0f}.fits'\
                         .format('SOSS_PSFs/', w, i)
                psf_run.append(fits.open(infile)[0].data)
            # Generate missing PSFs if necessary.
            except FileNotFoundError:
                errmsg = ' No monochromatic PSF found for {0:.2f}µm and WFE '\
                         'realization {1:.0f}. Creating it now.'.format(w, i)
                print(errmsg)
                loicpsf(wavelist=[w*1e-6], wfe_real=i, verbose=False)
                psf_run.append(fits.open(infile)[0].data)
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
            rnger = np.linspace(0, round(1280*(2.5/2.9), 0) - 1, 1280)
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
    if verbose == 3:
        plotting._plot_interpmodel(wave_range, wb, wr, pb, pr)

    # Save the coefficients to disk so that they can be accessed by the
    # empirical trace construction module.
    try:
        df = pd.read_csv(path+'interpolation_coefficients.csv')
    except FileNotFoundError:
        # If the interpolation coefficients file does not already exist, create
        # a new dictionary.
        df = {}
    # Replace the data for F277W or no F277W depending on which was run.
    if F277W is True:
        df['F_red'] = pr
        df['F_blue'] = pb
    else:
        df['NF_red'] = pr
        df['NF_blue'] = pb
    # Write to file.
    df = pd.DataFrame(data=df)
    df.to_csv(path+'interpolation_coefficients.csv', index=False)

    return pb, pr


def loicpsf(wavelist=None, wfe_real=None, save_to_disk=True, oversampling=10,
            pixel=128, verbose=True):
    '''Calls the WebbPSF package to create monochromatic PSFs for NIRISS
    SOSS observations and save them to disk.

    Parameters
    ----------
    wavelist : list
        List of wavelengths (in meters) for which to generate PSFs.
    wfe_real : int
        Index of wavefront realization to use for the PSF (if non-default
        WFE realization is desired).
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

    # Create PSF storage array.
    psf_list = []
    # PSFs will be saved to a SOSS_PSFs directory. If it does not already
    # exist, create it.
    if save_to_disk is True:
        filepath = path + 'SOSS_PSFs/'
        if os.path.exists(filepath):
            pass
        else:
            os.mkdir(filepath)

    if wavelist is None:
        # List of wavelengths to generate PSFs for
        wavelist = np.linspace(0.5, 5.2, 95) * 1e-6

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
            filepars = [oversampling, pixel, text, wfe_real]
            outfile = filepath+'SOSS_os{0}_{1}x{1}_{2}_{3}.fits'.format(*filepars)
            psf.writeto(outfile, overwrite=True)

    if save_to_disk is False:
        return psf_list
