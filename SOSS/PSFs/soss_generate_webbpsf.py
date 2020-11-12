#!/usr/bin/env python
# coding: utf-8

""" soss_generate_webbpsf.py: This calls WebbPSF to generate SOSS PSFs
                              used as the basis of Jason Rowe's simulations"""
__author__ = "Loic Albert"


import webbpsf
import numpy as np
from astropy.io import fits


def loicpsf(wavelist=None, wfe_real=None, save_to_disk=True, savepath=None):
    '''Utility function which calls the WebbPSF package to create monochromatic
    PSFs for NIRISS SOSS mode obserations.

    Parameters
    ----------
    wavelist : list
        List of wavelengths (in meters) for which to generate PSFs.
    wfe_real : int
        Index of wavefront realization to use for the PSF (if non-default
        WFE realization is desired).
    save_to_disk  : bool
        Whether to save PSFs to disk.

    Returns
    -------
    None : NoneType
        If PSFs are written to disk.
    psf-list : list
        List of np.ndarrays with the PSF data.
    '''

    if savepath != None:
        PSF_DIR = savepath
    else:
        PSF_DIR = '/genesis/jwst/jwst-ref-soss/monochromatic_PSFs/'

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
    if save_to_disk is False:
        psf_list = []  # Create running list of PSF realizations
    for wave in wavelist:
        print('Calculating PSF at wavelength ', wave/1e-6, ' microns')
        psf = niriss.calc_psf(monochromatic=wave, fov_pixels=pixel,
                              oversample=oversampling, display=False)

        # Save psf realization to disk if desired
        if save_to_disk is True:
            text = '{0:5f}'.format(wave*1e+6)
            psf.writeto(PSF_DIR + 'SOSS_os'+str(oversampling)+'_'+str(pixel)
                        + 'x'+str(pixel)+'_'+text+'_test.fits', overwrite=True)
        else:
            psf_list.append(psf[0].data)

    if save_to_disk is False:
        return psf_list
    else:
        return None

loicpsf(wavelist=None, wfe_real=None, save_to_disk=True,
        savepath='/genesis/jwst/userland-soss/loic_review/')