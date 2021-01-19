#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from astropy.io import fits


def main():
    """"""

    # Fixed parameters for the 2D wavelength map reference file.
    padding = 10
    oversample = 2
    orders = [1, 2, 3]
    soss_ref_2d_profile = 'SOSS_ref_2D_profile.fits'  # Output SOSS reference file.

    # Start building the output fits file.
    hdul = list()
    hdu = fits.PrimaryHDU()
    hdu.header['CREATOR'] = 'Loic Albert'
    hdul.append(hdu)

    for m in orders:

        # TODO Read file provided by Loïc, replace with function that generates this info in the future.
        profile_2d = fits.getdata('2DTrace.fits', ext=0)
        profile_2d = profile_2d[m-1]

        nrows, ncols = profile_2d.shape
        dimy = oversample*(2048 + 2*padding)
        dimx = oversample*(2048 + 2*padding)

        tmp = np.full((dimy, dimx), fill_value=np.nan)
        tmp[-nrows:] = profile_2d
        profile_2d = tmp

        # Add the 2D wavelength map to the fits file.
        hdu = fits.ImageHDU(profile_2d)
        hdu.header['ORDER'] = (m, 'Spectral order.')
        hdu.header['OVERSAMP'] = (oversample, 'Pixel oversampling.')
        hdu.header['PADDING'] = (padding, 'Native pixel-size padding around the image.')

        lrow = oversample*(2048 - 246)
        urow = oversample*(2048 - 150 + 2*padding)
        lcol = 1
        ucol = oversample*(2048 + 2*padding)
        index96 = '[{}:{},{}:{}]'.format(lrow, urow, lcol, ucol)
        hdu.header['INDEX96'] = (index96, 'SUBSTRIP96, including padding.')

        lrow = oversample*(2048 - 256)
        urow = oversample*(2048 + 2*padding)
        lcol = 1
        ucol = oversample*(2048 + 2*padding)
        index256 = '[{}:{},{}:{}]'.format(lrow, urow, lcol, ucol)
        hdu.header['INDEX256'] = (index256, 'SUBSTRIP256, including padding.')

        hdul.append(hdu)

    hdul = fits.HDUList(hdul)
    hdul.writeto(soss_ref_2d_profile, overwrite=True)
    hdul.writeto(soss_ref_2d_profile + '.gz', overwrite=True)

    return


if __name__ == '__main__':
    main()
