#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:02:42 2020

@author: albert
"""

import sys
sys.path.insert(0, "../../trace/")
import numpy as np
import tracepol as tp
from astropy.io import fits
from astropy.table import Table


def get_tiltpars(filename=None):
    """Read a file containing the tilt angle for orders 1, 2 and 3.

    :param filename: The file containing the tilt data.

    :type filename: str

    :returns: wavegrid, tilt_o1, tilt_o2, tilt_o3 - A grid of wavelengths and corresponding tilt angles for each order.
    :rtype: Tuple(array[float], array[float], array[float], array[float])
    """

    if filename is None:
        filename = 'SOSS_wavelength_dependent_tilt.ecsv'  # TODO Switch to pkg_resources in the future.

    tab = Table.read(filename)
    wavegrid = tab['Wavelength']
    tilt_o1 = tab['order 1']
    tilt_o2 = tab['order 2']
    tilt_o3 = tab['order 3']

    # TODO make polynomial fit like tp.get_tracepars?

    return wavegrid, tilt_o1, tilt_o2, tilt_o3


def wavelength_to_tilt(wavelength, tiltpars, m=1):
    """Interpolate the tilt values to the input wavelength grid.

    :param wavelength: wavelength values in microns.
    :param tiltpars: the tilt parameters returned by get_tiltpars.
    :param m: the spectral order.

    :type wavelength: array[float]
    :type tiltpars: Tuple(array[float], array[float], array[float], array[float])
    :type m: int

    :returns: tilt - The tilt angles corresponding to wavelength for order m.
    :rtype: array[float]
    """

    wavegrid, tilt_o1, tilt_o2, tilt_o3 = tiltpars

    if m == 1:
        tilt = np.interp(wavelength, wavegrid, tilt_o1)
    elif m == 2:
        tilt = np.interp(wavelength, wavegrid, tilt_o2)
    elif m == 3:
        tilt = np.interp(wavelength, wavegrid, tilt_o3)
    else:
        raise ValueError('Order m must be 1, 2, or 3.')

    return tilt

     
def image_nat_to_dms(image):
    """This function converts images from native (ds9) to DMS coordinates.

    :param image: The input image data in native coordinates.

    :type image: array[float]

    :returns: out - the image in DMS coordinates.
    :rtype: array[float]
    """

    ndim = image.ndim

    if ndim == 2:
        out = np.flip(np.rot90(image, axes=(0, 1)), axis=-1)
    elif ndim == 3:
        out = np.flip(np.rot90(image, axes=(1, 2)), axis=-1)
    elif ndim == 4:
        out = np.flip(np.rot90(image, axes=(2, 3)), axis=-1)
    else:
        raise ValueError('Image with {} dimensions are not supported.'.format(ndim))

    return out


def make_2d_wavemap(m=1, subarray='SUBSTRIP256', frame='dms', tilt_angle=None, oversample=1, padding=10, maxiter=5,
                    dtol=1e-2):
    """Compute the 2D wavelength map for NIRISS SOSS.

    :param m: the spectral order.
    :param subarray: the subarray of the output coordinates (SUBARRAY256 or SUBARRAY96).
    :param frame: the coordinate frame of the input coordinates (nat, dms or sim).
    :param tilt_angle: a constant tilt angle to use.
    :param oversample: the oversampling factor of the output array.
    :param padding: the padding of the output array in pixels.
    :param maxiter: the maximum number of iterations to use when solving the wavelength map.
    :param dtol: the tolerance in pixels at which to end the iterations.

    :type m: int
    :type subarray: str
    :type frame: str
    :type tilt_angle: float
    :type oversample: int
    :type padding: int
    :type maxiter: int
    :type dtol: float

    :returns: wave_map_2d - A 2D array of wavelength values across the detector.
    :rtype: array[float]
    """

    # Assuming that tracepol is oriented in native (ds9) coordinates,
    # i.e. the spectral axis is along Y, the spatial axis is along X, with the red
    # wavelengths going up and blue curving left.
    if subarray == 'FULL':
        # Assume a spatial dimension of 300 and at the end pad with NaNs.
        dimy = 2048  # spectral axis
        dimx = 300  # spatial axis
    elif subarray == 'SUBSTRIP96':
        # Use the same dimensions as SUBSTRIP256 and at the end shrink it down.
        dimy = 2048  # spectral axis
        dimx = 256  # spatial axis
    elif subarray == 'SUBSTRIP256':
        dimy = 2048  # spectral axis
        dimx = 256  # spatial axis
    else:
        raise ValueError('subarray must be one of SUBSTRIP256, SUBSTRIP96 or FULL')

    # Padding so that an oversampled array may be safely shifted.
    xpad = np.copy(padding)
    ypad = np.copy(padding)

    # The oversampling is an integer number that will scale the output 2D map.
    os = np.copy(oversample)

    # Get the tilt parameters.
    tiltpars = get_tiltpars()

    # Get the trace parameters.
    trace_file = '../../trace/NIRISS_GR700_trace_extended.csv'
    tracepars = tp.get_tracepars(trace_file)

    # Compute the trace x, y positions for order m.
    wave = np.linspace(0.5, 3.0, 2501)
    y_trace, x_trace, mask = tp.wavelength_to_pix(wave, tracepars, m=m, frame='nat')

    # Generate the oversampled grid of pixel coordinates.
    x_vec = np.arange((dimx + 2*xpad)*os)/os - (os - 1)/(2*os) - xpad
    y_vec = np.arange((dimy + 2*ypad)*os)/os - (os - 1)/(2*os) - ypad
    x_pixel, y_pixel = np.meshgrid(x_vec, y_vec)

    # The gain is for the iterative approach to finding the wavelength.
    gain = -1.0
    delta_y = 0.0

    # Iteratively compute the wavelength at each pixel.
    for niter in range(maxiter):

        # Assume all x have same wavelength.
        wave_iterated = np.interp(y_pixel + gain*delta_y, y_trace, wave)  # TODO why not use tp.specpix_to_wavelength?

        # Compute the tilt angle at the wavelengths.
        if tilt_angle is not None:
            tilt_tmp = np.copy(tilt_angle)
        else:
            tilt_tmp = wavelength_to_tilt(wave_iterated, tiltpars, m=m)

        # Compute the trace position at the wavelengths.
        x_estimate = np.interp(wave_iterated, wave, x_trace)  # TODO why not use tp.wavelength_to_pix?
        y_estimate = np.interp(wave_iterated, wave, y_trace)

        # Project that back to pixel coordinates.
        y_iterated = y_estimate + (x_pixel - x_estimate)*np.tan(np.deg2rad(tilt_tmp))

        # Measure error between requested and iterated position.
        delta_y = delta_y + (y_iterated - y_pixel)

        # If the desired precision has been reached end iterations.
        if np.all(np.abs(y_iterated - y_pixel) < dtol):
            break

    wave_map_2d = wave_iterated

    # Crop or expand to the appropriate size for the subarray.
    if subarray == 'FULL':
        # We pad the FULL subarray with NaNs now.
        tmp = np.full((os*(dimy + 2*ypad), os*(dimy + 2*xpad)), fill_value=np.nan)
        lx = 0
        ux = os*(dimx + 2*xpad)
        tmp[:, lx:ux] = wave_map_2d
        wave_map_2d = tmp
    elif subarray == 'SUBSTRIP96':
        # The SUBSTRIP96 subarray is offset relative to the SUBSTRIP256 by N pixels.
        offset = 11
        lx = os*(dimx - offset - 96)
        ux = os*(dimx - offset + 2*xpad)
        wave_map_2d = wave_map_2d[:, lx:ux]
    elif subarray == 'SUBSTRIP256':
        lx = 0
        ux = os*(256 + 2*xpad)
        wave_map_2d = wave_map_2d[:, lx:ux]
    else:
        raise ValueError('subarray must be one of SUBSTRIP256, SUBSTRIP96 or FULL')

    # Transform to the correct coordinate frame.
    if frame == 'nat':
        pass
    elif frame == 'dms':
        wave_map_2d = image_nat_to_dms(wave_map_2d)
    elif frame == 'sim':
        wave_map_2d = image_nat_to_dms(wave_map_2d)
        wave_map_2d = np.flip(wave_map_2d, axis=1)
    else:
        raise ValueError('Unknown coordinate frame: {}'.format(frame))

    return wave_map_2d


def main():
    """Generate the NIRISS SOSS 2D wavelength reference file."""

    padding = 10
    oversample = 3
    filename = 'SOSS_wave2D_os{}_pad{}.fits'.format(oversample, padding)

    # Generate the 2D wavelength map for each order.
    wave_map_2d_order1 = make_2d_wavemap(m=1, oversample=oversample, padding=padding)
    wave_map_2d_order2 = make_2d_wavemap(m=2, oversample=oversample, padding=padding)
    wave_map_2d_order3 = make_2d_wavemap(m=3, oversample=oversample, padding=padding)

    # Create the reference file.
    hdu0 = fits.PrimaryHDU()
    hdu0.header['CREATOR'] = 'Geert Jan Talens'

    # Order 1 wavelength map.
    hdu1 = fits.ImageHDU(wave_map_2d_order1)
    hdu1.header['ORDER'] = (1, 'Spectral order.')
    hdu1.header['OVERSAMP'] = (oversample, 'Pixel oversampling.')
    hdu1.header['PADDING'] = (padding, 'Native pixel-size padding around the image.')

    # Order 2 wavelength map.
    hdu2 = fits.ImageHDU(wave_map_2d_order2)
    hdu2.header['ORDER'] = (2, 'Spectral order.')
    hdu2.header['OVERSAMP'] = (oversample, 'Pixel oversampling.')
    hdu2.header['PADDING'] = (padding, 'Native pixel-size padding around the image.')

    # Order 3 wavelength map.
    hdu3 = fits.ImageHDU(wave_map_2d_order3)
    hdu3.header['ORDER'] = (3, 'Spectral order.')
    hdu3.header['OVERSAMP'] = (oversample, 'Pixel oversampling.')
    hdu3.header['PADDING'] = (padding, 'Native pixel-size padding around the image.')

    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])
    hdul.writeto(filename, overwrite=True)

    return


if __name__ == '__main__':
    main()
