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


def read_tilt_file(
        filename='./SOSS_wavelength_dependent_tilt.txt'):
    """
    Read a file containing the tilt angle for orders 1, 2 and 3.
    """

    # Use the best guess estimate for what the tilt is from CV3
    # Initialize arrays read from reference file.
    w = []
    o1, o2, o3 = [], [], []

    # Read in the reference tilt file
    f = open(filename, 'r')
    for line in f:

        # Ignore comments (lines starting with #
        if line[0] != '#':
            columns = line.split()
            w.append(float(columns[0]))
            o1.append(float(columns[1]))
            o2.append(float(columns[2]))
            o3.append(float(columns[3]))

    # Make sure to convert from lists to numpy arrays
    w = np.array(w)
    o1 = np.array(o1)
    o2 = np.array(o2)
    o3 = np.array(o3)

    return w, o1, o2, o3


def tilt_solution(wave_queried, tilt_columns, m=1):
    """
    Interpolate the tilt values to the input wavelength grid.
    """

    w, o1, o2, o3 = tilt_columns

    if m == 1:
        tilt_queried = np.interp(wave_queried, w, o1)
    elif m == 2:
        tilt_queried = np.interp(wave_queried, w, o2)
    elif m == 3:
        tilt_queried = np.interp(wave_queried, w, o3)
    else:
        raise ValueError('order m must be 1, 2, or 3')

    return tilt_queried

     
def image_ds9_to_dms(image):
    """
    This function converts images from ds9 (native) to DMS coordinates.
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


def make_2d_wavemap(m=1, subarray='SUBSTRIP256', frame='dms',
                    tilt_angle=None, oversampling=1, padding=10):
    """This script generates and writes on disk the reference file that
    describes the wavelength at the center of each pixel in the subarray.
    The map will have two 'slices', i.e. a cube of 2 layers, one for each
    spectral order of SOSS.

    subarray_name: SUBSTRIP96, SUBSTRIP256, FF
    coordinate_system: DS9 (native) or DMS or Jason's
    tilt_table: the name of the 4-column table describing the monochromatic
       tilt as a function of wavelength (in microns). Interpolation will be
       made from that table. Col 1 = microns, col 2 = first order, col 3 =
       second order. No longer used here. Put in tilt_vs_spectralpixel instead.
    tilt_constant: if that is set then its value is the monochromatic tilt
       in degrees whose value is constant for all wavelengths. It then
       bypasses the tilt described in the tilt_table.
    The convention for the tilt sign is described in the
    tilt_vs_spectralpixel() function above.
    padding: the number of native pixels to add as padding on each sides
    and top/bottom of the generated map. That will allow the extract 1D Solver
    to simply apply rotation+offsets without incurring border artifacts.
    """

    # Assuming that tracepol is oriented in the ds9 (native detector) coordinates,
    # i.e. the spectral axis is along Y, the spatial axis is along X, with the red
    # wavelengths going up and blue curving left.
    if subarray == 'SUBSTRIP96':
        # Seed the larger subarray then we will shrink it later to 96
        dimy = 2048  # spectral axis
        dimx = 256  # spatial axis
    elif subarray == 'FF':
        # Assume a spatial dimension of 300 and at the end pad with NaNs.
        dimy = 2048  # spectral axis
        dimx = 300  # spatial axis
    elif subarray == 'SUBSTRIP256':
        dimy = 2048  # spectral axis
        dimx = 256  # spatial axis
    else:
        raise ValueError('subarray must be one of SUBSTRIP256, SUBSTRIP96 or FF')

    # Padding so that an oversampled array may be safely shifted.
    xpad = np.copy(padding)
    ypad = np.copy(padding)

    # The oversampling is an integer number that will scale the output 2D map
    os = np.copy(oversampling)
    wave_map_2d = np.zeros(((dimy + 2*ypad)*os, (dimx + 2*xpad)*os))

    # Inititalize the tilt solution
    tilt_columns = read_tilt_file()

    # The gain is for the iterative approach to finding the wavelength
    gain = -1.0

    # First, query the x,y for order m+1 (so order 1 or 2 or 3)
    # Get the trace parameters, function found in tracepol imported above
    trace_file = '../../trace/NIRISS_GR700_trace_extended.csv'
    tracepars = tp.get_tracepars(trace_file)

    # Get wavelength (in um) of first and last pixel of the Order m trace
    wave = np.linspace(0.5, 3.0, 2501)
    y, x, mask = tp.wavelength_to_pix(wave, tracepars, m=m,
                                      frame='nat',
                                      subarray=subarray,
                                      oversample=oversampling)

    # Loop over spectral order (m), spatial axis (x) and spectral axis (y)
    # For each pixel, project back to trace center iteratively to
    # recover the wavelength.
    for i in range((dimx + 2*xpad)*os):  # the spatial axis

        for j in range((dimy + 2*ypad)*os):  # the spectral axis

            x_queried = np.float(i)/os - xpad
            y_queried = np.float(j)/os - ypad

            delta_y = 0.0
            for niter in range(5):

                # Assume all x have same lambda
                wave_queried = np.interp(y_queried + gain*delta_y, y/os, wave)

                # Monochromatic tilt at that wavelenength is:
                if tilt_angle is not None:
                    tilt_tmp = np.copy(tilt_angle)
                else:
                    tilt_tmp = tilt_solution(wave_queried, tilt_columns, m=m)

                # Plug the lambda to spit out the x, y.
                x_estimate = np.interp(wave_queried, wave, x)
                y_estimate = np.interp(wave_queried, wave, y)

                # Project that back to requested x assuming a tilt of tilt_degree
                # x_iterated = np.copy(x_queried) not used?
                y_iterated = y_estimate + \
                    (x_queried-x_estimate) * \
                    np.tan(np.deg2rad(tilt_tmp))

                # Measure error between requested and iterated position.
                delta_y = delta_y + (y_iterated - y_queried)

            wave_map_2d[j, i] = wave_queried

    # Crop or expand to the appropriate size for the subarray.
    if subarray == 'SUBSTRIP96':
        # The SUBSTRIP96 subarray is offset relative to the SUBSTRIP256 by
        # nnn pixels
        offset = 11
        wave_map_2d = wave_map_2d[:, :, os*(2*xpad+256-96-offset):os*(xpad+256-offset)]
    elif subarray == 'FF':
        tmp = np.zeros(((dimy+2*ypad)*os, (dimy+2*xpad)*os)) * np.nan
        tmp[:, os*(xpad+0):os*(2*xpad+dimx)] = wave_map_2d
        wave_map_2d = tmp
    else:
        pass

    # Transform to the correct coordinate frame.
    if frame == 'nat':
        pass
    elif frame == 'dms':
        wave_map_2d = image_ds9_to_dms(wave_map_2d)
    elif frame == 'sim':
        wave_map_2d = image_ds9_to_dms(wave_map_2d)
        wave_map_2d = np.flip(wave_map_2d, axis=1)
    else:
        raise ValueError('Unknown coordinate frame: {}'.format(frame))

    return wave_map_2d


def main():

    # Generate the 2D wavelength map for order 1.
    wave_map_2d = make_2d_wavemap()

    # Save the 2D wavelength map.
    fitsmap_name = './SOSS_wave2D_ref.fits'

    hdu = fits.PrimaryHDU()
    hdu.data = wave_map_2d
    hdu.header['oversamp'] = (1, 'Pixel oversampling')
    hdu.header['padding'] = (10, 'Native pixel-size padding around the image.')
    hdu.writeto(fitsmap_name, overwrite=True)

    return


if __name__ == '__main__':
    main()
