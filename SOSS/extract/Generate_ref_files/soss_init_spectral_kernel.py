#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

    # Add the kernels to the fits file.
    hdu = fits.PrimaryHDU(kernels)

    # Build the header.
    hdu.header['CREATOR'] = 'Loic Albert'
    hdu.header['SPECOS'] = (specos, 'The spectral oversampling of the native pixels.')
    hdu.header['HALFWIDT'] = (halfwidth, 'The half width of the kernel in native pixels.')
    hdu.header['INDCENTR'] = (specos*halfwidth + 1, 'Index of the kernel centroid.')
    hdu.header['NWAVE'] = (nwave, 'The number of wavelengths used.')
    hdu.header['WAVE0'] = (wavemin, 'Wavelength of the first kernel (in microns).')
    hdu.header['INDWAVE0'] = (1, 'Index of the first kernel')
    hdu.header['WAVEN'] = (wavemax, 'Wavelength of the last kernel (in microns).')
    hdu.header['INDWAVEN'] = (nwave, 'Index of the last kernel')

    hdu.writeto(soss_ref_spectral_kernel, overwrite=True)
    hdu.writeto(soss_ref_spectral_kernel + '.gz', overwrite=True)

    return


if __name__ == '__main__':
    main()
