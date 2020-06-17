#!/usr/bin/env python
# coding: utf-8

""" soss_generate_webbpsf.py: This calls WebbPSF to generate SOSS PSFs
                              used as the basis of Jason Rowe's simulations"""
__author__ = "Loic Albert"


import webbpsf
import numpy as np
from astropy.io import fits

# List of wavelengths to generate PSFs for
wavelist = np.linspace(0.5, 5.2, 95) * 1e-6
# Size of the PSF files in native pixels
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

# Loop through all wavelengths to generate PSFs and write them to disk
for wave in wavelist:
    print('Calculate PSF at wavelength ', wave/1e-6, ' microns')
    text = '{0:5f}'.format(wave*1e+6)
    psf = niriss.calc_psf(monochromatic=wave, fov_pixels=pixel,
                          oversample=oversampling, display=False)
    psf.writeto('SOSS_os'+str(oversampling)+'_'+str(pixel)+'x'+str(pixel)+'_'
                +text+'.fits', overwrite=True)
