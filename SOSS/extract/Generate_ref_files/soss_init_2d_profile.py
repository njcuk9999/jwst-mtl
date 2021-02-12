#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np

from astropy.io import fits


def main():
    """"""

    # Fixed parameters for the 2D wavelength map reference file.
    padding = 10
    oversample = 2
    orders = [1, 2, 3]
    subarrays = ['FULL', 'SUBSTRIP96', 'SUBSTRIP256']
    filepattern = 'SOSS_ref_2D_profile_{}.fits'  # Output SOSS reference file.

    for subarray in subarrays:

        if subarray == 'FULL':
            lrow = 0
            urow = oversample * (2048 + 2 * padding)
            lcol = 0
            ucol = oversample * (2048 + 2 * padding)
            filename = filepattern.format(subarray)
        elif subarray == 'SUBSTRIP96':
            lrow = oversample * (2048 - 246)
            urow = oversample * (2048 - 150 + 2 * padding)
            lcol = 0
            ucol = oversample * (2048 + 2 * padding)
            filename = filepattern.format(subarray)
        elif subarray == 'SUBSTRIP256':
            lrow = oversample * (2048 - 256)
            urow = oversample * (2048 + 2 * padding)
            lcol = 0
            ucol = oversample * (2048 + 2 * padding)
            filename = filepattern.format(subarray)
        else:
            raise ValueError('Unknown subarray: {}'.format(subarray))

        # Start building the output fits file.
        hdul = list()
        hdu = fits.PrimaryHDU()
        hdu.header['DATE'] = (datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'Date this file was created (UTC)')
        hdu.header['ORIGIN'] = ('SOSS Team MTL', 'Orginazation responsible for creating file')
        hdu.header['TELESCOP'] = ('JWST', 'Telescope used to acquire the data')
        hdu.header['INSTRUME'] = ('NIRISS', 'Instrument used to acquire the data')
        hdu.header['SUBARRAY'] = (subarray, 'Subarray used')
        hdu.header['FILENAME'] = (filename, 'Name of the file')
        hdu.header['REFTYPE'] = ('SPECPROFILE', 'Reference file type')
        hdu.header['PEDIGREE'] = ('GROUND', 'The pedigree of the refernce file')
        hdu.header['DESCRIP'] = ('2D trace profile', 'Desription of the reference file')
        hdu.header['AUTHOR'] = ('Loic Albert', 'Author of the reference file')
        hdu.header['USEAFTER'] = ('2000-01-01T00:00:00', 'Use after date of the reference file')
        hdu.header['EXP_TYPE'] = ('NIS_SOSS', 'Type of data in the exposure')
        hdul.append(hdu)

        for m in orders:

            # TODO Read file provided by Loïc, replace with function that generates this info in the future.
            profile_2d = fits.getdata('2DTrace.fits', ext=0)
            profile_2d = profile_2d[m - 1]

            nrows, ncols = profile_2d.shape
            dimy = oversample*(2048 + 2*padding)
            dimx = oversample*(2048 + 2*padding)

            tmp = np.full((dimy, dimx), fill_value=np.nan)
            tmp[-nrows:] = profile_2d
            profile_2d = tmp

            # Add the 2D wavelength map to the fits file.
            hdu = fits.ImageHDU(profile_2d[lrow:urow, lcol:ucol].astype('float32'))
            hdu.header['ORDER'] = (m, 'Spectral order.')
            hdu.header['OVERSAMP'] = (oversample, 'Pixel oversampling.')
            hdu.header['PADDING'] = (padding, 'Native pixel-size padding around the image.')
            hdu.header['EXTNAME'] = 'ORDER'
            hdu.header['EXTVER'] = m
            hdul.append(hdu)

        hdul = fits.HDUList(hdul)
        hdul.writeto(filename, overwrite=True)
        hdul.writeto(filename + '.gz', overwrite=True)

    return


if __name__ == '__main__':
    main()
