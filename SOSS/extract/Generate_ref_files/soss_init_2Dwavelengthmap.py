#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:02:42 2020

@author: albert
"""

import sys
sys.path.insert(0, "../../trace/")
from astropy.io import fits
import tracepol


def main():
    """Generate the NIRISS SOSS 2D wavelength reference file."""

    # Fixed parameters for the 2D wavelength map reference file.
    padding = 10
    oversample = 3
    orders = [1, 2, 3]
    trace_ref = 'SOSS_trace_ref.fits'
    wave2d_ref = 'SOSS_wave2d_ref.fits'

    # Start building the output fits file.
    hdul = list()
    hdu = fits.PrimaryHDU()
    hdu.header['CREATOR'] = 'Geert Jan Talens'
    hdul.append(hdu)

    for m in orders:

        # Read the 1D trace reference file.
        data = fits.getdata(trace_ref, ext=m)

        # Unpack the 1D trace info.
        wavegrid = data['WAVELENGTH']
        x_dms = data['X_SUBSTRIP256']
        y_dms = data['Y_SUBSTRIP256']
        tilt = data['TILT']

        # Convert from dms to native coordinates.
        y_trace = tracepol.specpix_frame_to_ref(x_dms, frame='dms')
        x_trace = tracepol.spatpix_frame_to_ref(y_dms, frame='dms', subarray='SUBSTRIP256')

        # Compute the 2D wavelength map.
        wave_map_2d = tracepol._get_wave_map(wavegrid, x_trace, y_trace, tilt, subarray='FULL', oversample=oversample,
                                             padding=padding)

        # Add the 2D wavelength map to the fits file.
        hdu = fits.ImageHDU(wave_map_2d)
        hdu.header['ORDER'] = (m, 'Spectral order.')
        hdu.header['OVERSAMP'] = (oversample, 'Pixel oversampling.')
        hdu.header['PADDING'] = (padding, 'Native pixel-size padding around the image.')
        # hdu.header['LBSUB96'] = oversample*(2048 - 246)
        # hdu.header['UBSUB96'] = oversample*(2048 - 150 + 2*padding)
        # hdu.header['LBSUB256'] = oversample*(2048 - 256)
        # hdu.header['UBSUB256'] = oversample*(2048 + 2*padding)
        hdul.append(hdu)

    hdul = fits.HDUList(hdul)
    hdul.writeto(wave2d_ref, overwrite=True)

    return


if __name__ == '__main__':
    main()
