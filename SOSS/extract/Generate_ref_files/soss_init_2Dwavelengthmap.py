#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:02:42 2020

@author: albert
"""

# Initialy written in the NIRISS/simuSOSS/simu2.pro IDL package.
# Translated to python

import sys
sys.path.insert(0, "/genesis/jwst/github/jwst-mtl/SOSS/trace/")
import numpy as np
import tracepol as tp
# import matplotlib.pyplot as plt
from astropy.io import fits


def read_tilt_file():
    """For spectral axis pixels (1 to 2048), return the monochromatic tilt
    in degrees, for a given spectral order (1 or 2) as a function of spectral
    pixel. This is the function where a varying tilt can be defined.
    Based on CV3 data, there seems to be some variations of that tilt angle
    for order 1. It is however difficult to quantify well.

    The tilt sign is defined the following way:
    Given the spatial axis going from the blue to the red is positive;
    Given that the spectral axis is positive from the blue to the red;
    Then the tilt is the angle away from the positive spatial axis towards
    the positive spectral axis. In that direction, the tilt is positive.

               red (+)
                  |
         tilt (+) | s
            \     | p
             \    | a
              \   | t
               \  | i
                \ | a
                 \| l
     red(+)-------- blue
          spectral
    """

    filename = '/genesis/jwst/github/jwst-mtl/SOSS/extract/Generate_ref_files/SOSS_wavelength_dependent_tilt.txt'

    if True:
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

    # The tilt relation is
    tilt_columns = w, o1, o2, o3

    return tilt_columns


def tilt_vs_spectralpixel(wavelength_micron, order):
    """For spectral axis pixels (1 to 2048), return the monochromatic tilt
    in degrees, for a given spectral order (1 or 2) as a function of spectral
    pixel. This is the function where a varying tilt can be defined.
    Based on CV3 data, there seems to be some variations of that tilt angle
    for order 1. It is however difficult to quantify well.
    
    The tilt sign is defined the following way:
    Given the spatial axis going from the blue to the red is positive;
    Given that the spectral axis is positive from the blue to the red;
    Then the tilt is the angle away from the positive spatial axis towards
    the positive spectral axis. In that direction, the tilt is positive.

               red (+)
                  |
         tilt (+) | s
            \     | p
             \    | a
              \   | t
               \  | i
                \ | a
                 \| l
     red(+)-------- blue
          spectral
    """

    if False:

        # Used this sinusoidaly changing tilt for debugging.
        if order == 1:
            tilt_degrees = 1.0 + 10.0*np.sin(wavelength_micron/50.)
        elif order == 2:
            tilt_degrees = 1.0 + 10.0*np.sin(wavelength_micron/100.)    
        else:
            print('Call tilt_vs_spectralpixel with order=1 or order=2.')
            sys.exit()

    if True:

        # Use the best guess estimate for what the tilt is from CV3
        # Initialize arrays read from reference file.
        w = []
        o1, o2, o3 = [], [], []

        # Read in the reference tilt file
        filename = '/genesis/jwst/github/jwst-mtl/SOSS/extract/Generate_ref_files/SOSS_wavelength_dependent_tilt.txt'
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

        # Branch according to the spectral order
        if order == 1:
            tilt_degrees = np.interp(wavelength_micron, w, o1)
        elif order == 2:
            tilt_degrees = np.interp(wavelength_micron, w, o2)
        elif order == 3:
            tilt_degrees = np.interp(wavelength_micron, w, o3)
        else:
            print('Call tilt_vs_spectralpixel with order=1 or order=2 or order=3')
            sys.exit()

    return tilt_degrees


def tilt_solution(wave_queried, tilt_columns, order=1):
    """"""

    w, o1, o2, o3 = tilt_columns
    if order == 1:
        tilt_queried = np.interp(wave_queried, w, o1)
    elif order == 2:
        tilt_queried = np.interp(wave_queried, w, o2)
    elif order == 3:
        tilt_queried = np.interp(wave_queried, w, o3)
    else:
        print('in tilt_solution, use order 1, 2 or 3.')
        sys.exit()

    return tilt_queried

     
def image_native_to_dms(image):
    """ This function converts from ds9 (native) to DMS coordinates.
    x_dms = y_native, y_dms = 2048-x_native
    See: https://outerspace.stsci.edu/pages/viewpage.action?
    spaceKey=NC&title=NIRISS+Commissioning&preview=/13498420/13499105/NIRISS_subarrays.pdf
    """

    size = np.shape(image)
    # a single 2D image
    ndim = len(size)
    if ndim == 2:        
        out = np.flip(np.rot90(image, axes=(0, 1)), axis=-1)
    # a cube of images, or more than a cube (assume that last 2 are x and y)
    if ndim == 3:
        out = np.flip(np.rot90(image, axes=(1, 2)), axis=-1)
    if ndim == 4:
        out = np.flip(np.rot90(image, axes=(2, 3)), axis=-1)

    return out


def make_2d_wavemap(fitsmap_name, subarray_name='SUBSTRIP256', coordinate_system='DMS',
                    tilt_constant=None, oversampling=1, padding=10):
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
    if subarray_name == 'SUBSTRIP96':
        # Seed the larger subarray then we will shrink it later to 96
        dimy = 2048  # spectral axis
        dimx = 256  # spatial axis
    elif subarray_name == 'FF':
        # Assume a spatial dimension of 300 and at the end pad with NaNs.
        dimy = 2048  # spectral axis
        dimx = 300  # spatial axis
    elif subarray_name == 'SUBSTRIP256':
        dimy = 2048  # spectral axis
        dimx = 256  # spatial axis
    else:
        # Assume SUBSTRIP256 by default
        print('No subarray_name was passed to make_2D_wavemap, assuming SUBSTRIP256.')
        dimy = 2048  # spectral axis
        dimx = 256  # spatial axis

    # number of spectral orders (1 to 3)
    norder = 3
        
    # xpad and ypad are not currently necessary but may prove to be in the
    # the future. Keep for now. They are given in native pixel size.
    xpad = np.copy(padding)  # not required
    ypad = np.copy(padding)  # not required
    # The oversampling is an integer number that will scale the output 2D map
    os = np.copy(oversampling)
    lambda_map = np.zeros((norder, (dimy+2*ypad)*os, (dimx+2*xpad)*os))
    
    # The gain is for the iterative approach to finding the wavelength
    gain = -1.0

    # Inititalize the tilt solution
    tilt_columns = read_tilt_file()
    
    for m in range(norder):  # repeat for each order
    
        order = m+1

        # First, query the x,y for order m+1 (so order 1 or 2 or 3)
        # Get the trace parameters, function found in tracepol imported above
        trace_file = '/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv'
        tracepars = tp.get_tracepars(trace_file)

        # Get wavelength (in um) of first and last pixel of the Order m trace
        lba = np.linspace(0.5, 3.0, 2501)
        y, x, mask = tp.wavelength_to_pix(lba, tracepars, m=order,
                                          # frame = str.lower(coordinate_system),
                                          frame='nat',
                                          subarray=subarray_name,
                                          oversample=oversampling)

        # Loop over spectral order (m), spatial axis (x) and spectral axis (y)
        # For each pixel, project back to trace center iteratively to
        # recover the wavelength.
        for i in range((dimx+2*xpad)*os):  # the spatial axis

            print('Order {:} Spatial pixel {:}'.format(order, i))

            for j in range((dimy+2*ypad)*os):  # the spectral axis

                x_queried = np.float(i)/os-xpad
                y_queried = np.float(j)/os-ypad
                # Debugging for os > 1
                # x_queried = (np.float(i)-xpad)/os
                # y_queried = (np.float(j)-ypad)/os

                delta_y = 0.0
                for niter in range(5):

                    # Assume all x have same lambda
                    lba_queried = np.interp(y_queried+gain*delta_y, y/os, lba)
                    # print(x_queried, y_queried, lba_queried)

                    # Monochromatic tilt at that wavelenength is:
                    if tilt_constant is not None:
                        tilt_tmp = np.copy(tilt_constant)
                    else:
                        tilt_tmp = tilt_solution(lba_queried, tilt_columns, order=order)

                    # Plug the lambda to spit out the x,y
                    x_estimate = np.interp(lba_queried, lba, x)
                    y_estimate = np.interp(lba_queried, lba, y)

                    # Project that back to requested x assuming a tilt of tilt_degree
                    # x_iterated = np.copy(x_queried) not used?
                    y_iterated = y_estimate + \
                        (x_queried-x_estimate) * \
                        np.tan(np.deg2rad(tilt_tmp))

                    # Measure error between requested and iterated position.
                    delta_y = delta_y + (y_iterated-y_queried)
                    # print(i,j,iter, x_queried, y_queried, tilt_tmp,
                    #      lba_queried, x_estimate, y_estimate, delta_y)

                lambda_map[m, j, i] = lba_queried

    # Crop or expand to the appropriate size for the subarray.
    if subarray_name == 'SUBSTRIP96':
        # The SUBSTRIP96 subarray is offset relative to the SUBSTRIP256 by
        # nnn pixels
        offset = 11
        lambda_map = lambda_map[:, :, os*(2*xpad+256-96-offset):os*(xpad+256-offset)]
    elif subarray_name == 'FF':
        tmp = np.zeros((norder, (dimy+2*ypad)*os, (dimy+2*xpad)*os)) * np.nan
        tmp[:, :, os*(xpad+0):os*(2*xpad+dimx)] = lambda_map
        lambda_map = tmp
    else:
        # Nothing to do for other sizes.
        pass

    # Prepare the output map
    print('Saving the 2D wavelength map as {:}'.format(fitsmap_name))

    if (coordinate_system == 'DMS') | (coordinate_system == 'dms'):
        print('in the DMS Coordinate System.')
        lambda_map = image_native_to_dms(lambda_map)
    elif (coordinate_system == 'JASON') | (coordinate_system == 'jason'):
        # Do like DMS but invert Y axis afterward
        print('in Jason s original Coordinate System (DMS with Y flipped.')
        lambda_map = image_native_to_dms(lambda_map)
        lambda_map = np.flip(lambda_map, axis=1)
    else:
        # DS9 must be. Do nothing.
        print('in ds9 or Native Coordinate System')
        lambda_map = lambda_map * 1.0
    
    hdu = fits.PrimaryHDU()
    hdu.data = lambda_map
    hdu.header['oversamp'] = (int(os), 'Pixel oversampling')
    hdu.header['padding'] = (int(padding), 'Native pixel-size padding around the image.')
    hdu.writeto(fitsmap_name, overwrite=True)
    
    return lambda_map


make_2d_wavemap('/genesis/jwst/userland-soss/loic_review/wave2Dmap_os1.fits',
                subarray_name='SUBSTRIP256', oversampling=1)
