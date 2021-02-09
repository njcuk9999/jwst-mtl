#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np

from astropy.io import fits


def main():
    """"""

    # Fixed parameters for the spectral kernels reference file.
    specos = 10
    halfwidth = 7
    nwave = 95
    wavemin = 0.5
    wavemax = 5.2
    soss_ref_spectral_kernel = 'SOSS_ref_spectral_kernel.fits'  # Output SOSS reference file.

    # TODO Read file provided by Loïc, replace with function that generates this info in the future.
    kernels = fits.getdata('spectral_kernel_matrix_os_10_width_15pixels.fits')

    # Build the wavelenth array.
    wavelengths = np.linspace(wavemin, wavemax, nwave)
    wavelengths = np.ones_like(kernels) * wavelengths

    # Start building the output fits file.
    hdul = list()
    hdu = fits.PrimaryHDU()
    hdu.header['DATE'] = (datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'Date this file was created (UTC)')
    hdu.header['ORIGIN'] = ('SOSS Team MTL', 'Orginazation responsible for creating file')
    hdu.header['TELESCOP'] = ('JWST', 'Telescope used to acquire the data')
    hdu.header['INSTRUME'] = ('NIRISS', 'Instrument used to acquire the data')
    hdu.header['FILENAME'] = (soss_ref_spectral_kernel, 'Name of the file')
    hdu.header['REFTYPE'] = ('SPECKERNEL', 'Reference file type')
    hdu.header['PEDIGREE'] = ('GROUND', 'The pedigree of the refernce file')
    hdu.header['DESCRIP'] = ('Convolution kernels', 'Desription of the reference file')
    hdu.header['AUTHOR'] = ('Loic Albert', 'Author of the reference file')
    hdu.header['USEAFTER'] = ('2000-01-01T00:00:00', 'Use after date of the reference file')
    hdu.header['EXP_TYPE'] = ('NIS_SOSS', 'Type of data in the exposure')
    hdul.append(hdu)

    # Create the wavelength extension.
    hdu = fits.ImageHDU(wavelengths.astype('float32'))
    hdu.header['EXTVER'] = 1
    hdu.header['EXTNAME'] = 'WAVELENGTHS'
    hdu.header['NWAVE'] = (nwave, 'The number of wavelengths used.')
    hdu.header['WAVEMIN'] = (wavemin, 'Wavelength of the first kernel (in microns).')
    hdu.header['WAVEMAX'] = (wavemax, 'Wavelength of the last kernel (in microns).')
    hdul.append(hdu)

    # Create the kernel extension.
    hdu = fits.ImageHDU(kernels.astype('float32'))
    hdu.header['EXTVER'] = 1
    hdu.header['EXTNAME'] = 'KERNELS'
    hdu.header['SPECOS'] = (specos, 'The spectral oversampling of the native pixels.')
    hdu.header['HALFWIDT'] = (halfwidth, 'The half width of the kernel in native pixels.')
    hdu.header['INDCENTR'] = (specos*halfwidth + 1, 'Index of the kernel centroid.')
    hdul.append(hdu)

    # Write the file to disk.
    hdul = fits.HDUList(hdul)
    hdul.writeto(soss_ref_spectral_kernel, overwrite=True)
    hdul.writeto(soss_ref_spectral_kernel + '.gz', overwrite=True)

    return


if __name__ == '__main__':
    main()
