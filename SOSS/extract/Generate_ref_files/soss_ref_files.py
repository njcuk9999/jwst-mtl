#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:35:32 2020

@author: albert
"""

from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.table import Table

from SOSS.trace import tracepol  # TODO in the future this should not depend on tracepol.


# ==============================================================================
# Code for generating the spectral trace reference file.
# ==============================================================================


def init_spec_trace(throughput_file=None, tilt_file=None):
    """Generate the NIRISS SOSS 1D trace reference file.
    TODO This code will need to be updated once in-flight measurements are available.
    TODO CAR NIS-018-GR700XD Wavelength Calibration
    TODO CAR NIS-017-GR700XD Flux Calibration
    """

    if throughput_file is None:
        throughput_file = 'files/NIRISS_Throughput_STScI.fits'

    if tilt_file is None:
        tilt_file = 'files/SOSS_wavelength_dependent_tilt.ecsv'

    # Fixed parameters for the 2D wavelength map reference file.
    subarrays = ['FULL', 'SUBSTRIP96', 'SUBSTRIP256']
    filepattern = 'SOSS_ref_trace_table_{}.fits'  # Output SOSS reference file.

    # Reference wavelength grid in micron.
    wave_grid = np.linspace(0.5, 5.5, 5001)

    # Read the SOSS total throughput as a function of wavelength.
    tab, hdr = fits.getdata(throughput_file, ext=1, header=True)

    throughput_wave = tab[0]['LAMBDA']/1e3
    throughput_order1 = tab[0]['SOSS_ORDER1']
    throughput_order2 = tab[0]['SOSS_ORDER2']
    throughput_order3 = tab[0]['SOSS_ORDER3']

    # Interpolate to the reference wavelength grid.
    throughput_order1 = np.interp(wave_grid, throughput_wave, throughput_order1)
    throughput_order2 = np.interp(wave_grid, throughput_wave, throughput_order2)
    throughput_order3 = np.interp(wave_grid, throughput_wave, throughput_order3)

    # Fix small negative throughput values.
    throughput_order1 = np.where(throughput_order1 < 0, 0, throughput_order1)
    throughput_order2 = np.where(throughput_order2 < 0, 0, throughput_order2)
    throughput_order3 = np.where(throughput_order3 < 0, 0, throughput_order3)

    # Read the tilt as a function of wavelength.
    tab = Table.read(tilt_file)

    tilt_wave = tab['Wavelength']
    tilt_order1 = tab['order 1']
    tilt_order2 = tab['order 2']
    tilt_order3 = tab['order 3']

    # Interpolate the tilt to the same wavelengths as the throughput.
    # Default bounds handling (constant boundary) is fine.
    tilt_order1 = np.interp(wave_grid, tilt_wave, tilt_order1)
    tilt_order2 = np.interp(wave_grid, tilt_wave, tilt_order2)
    tilt_order3 = np.interp(wave_grid, tilt_wave, tilt_order3)

    # Get the trace parameters, function found in tracepol imported above.
    # TODO once in-flight data are available tracepol should not be used. Use interpolation instead.
    tracepars = tracepol.get_tracepars('../../trace/NIRISS_GR700_trace_extended.csv')

    for subarray in subarrays:

        filename = filepattern.format(subarray)

        # Create the reference file.
        hdul = list()
        hdu = fits.PrimaryHDU()
        hdu.header['DATE'] = (datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'Date this file was created (UTC)')
        hdu.header['ORIGIN'] = ('SOSS Team MTL', 'Orginazation responsible for creating file')
        hdu.header['TELESCOP'] = ('JWST', 'Telescope used to acquire the data')
        hdu.header['INSTRUME'] = ('NIRISS', 'Instrument used to acquire the data')
        hdu.header['SUBARRAY'] = (subarray, 'Subarray used')
        hdu.header['FILENAME'] = (filename, 'Name of the file')
        hdu.header['REFTYPE'] = ('SPECTRACE', 'Reference file type')
        hdu.header['PEDIGREE'] = ('GROUND', 'The pedigree of the refernce file')
        hdu.header['DESCRIP'] = ('1D trace decscription', 'Desription of the reference file')
        hdu.header['AUTHOR'] = ('Geert Jan Talens', 'Author of the reference file')
        hdu.header['USEAFTER'] = ('2000-01-01T00:00:00', 'Use after date of the reference file')
        hdu.header['EXP_TYPE'] = ('NIS_SOSS', 'Type of data in the exposure')
        hdul.append(hdu)

        # Using tracepol, compute trace positions for each order and mask out-of-bound values.
        xtrace, ytrace, mask = tracepol.wavelength_to_pix(wave_grid, tracepars, m=1, frame='dms', subarray=subarray)
        xtrace = np.where(mask, xtrace, np.nan)
        ytrace = np.where(mask, ytrace, np.nan)

        # Order 1 table.
        col1 = fits.Column(name='WAVELENGTH', format='F', array=wave_grid)
        col2 = fits.Column(name='X', format='F', array=xtrace)
        col3 = fits.Column(name='Y', format='F', array=ytrace)
        col4 = fits.Column(name='THROUGHPUT', format='F', array=throughput_order1)
        col5 = fits.Column(name='TILT', format='F', array=tilt_order1)
        cols = fits.ColDefs([col1, col2, col3, col4, col5])

        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header['ORDER'] = (1, 'Spectral order.')
        hdu.header['EXTNAME'] = 'ORDER'
        hdu.header['EXTVER'] = 1
        hdul.append(hdu)

        # Using tracepol, compute trace positions for each order and mask out-of-bound values.
        xtrace, ytrace, mask = tracepol.wavelength_to_pix(wave_grid, tracepars, m=2, frame='dms', subarray=subarray)
        xtrace = np.where(mask, xtrace, np.nan)
        ytrace = np.where(mask, ytrace, np.nan)

        # Order 2 table.
        col1 = fits.Column(name='WAVELENGTH', format='F', array=wave_grid)
        col2 = fits.Column(name='X', format='F', array=xtrace)
        col3 = fits.Column(name='Y', format='F', array=ytrace)
        col4 = fits.Column(name='THROUGHPUT', format='F', array=throughput_order2)
        col5 = fits.Column(name='TILT', format='F', array=tilt_order2)
        cols = fits.ColDefs([col1, col2, col3, col4, col5])

        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header['ORDER'] = (2, 'Spectral order.')
        hdu.header['EXTNAME'] = 'ORDER'
        hdu.header['EXTVER'] = 2
        hdul.append(hdu)

        # Using tracepol, compute trace positions for each order and mask out-of-bound values.
        xtrace, ytrace, mask = tracepol.wavelength_to_pix(wave_grid, tracepars, m=3, frame='dms', subarray=subarray)
        xtrace = np.where(mask, xtrace, np.nan)
        ytrace = np.where(mask, ytrace, np.nan)

        # Order 3 table.
        col1 = fits.Column(name='WAVELENGTH', format='F', array=wave_grid)
        col2 = fits.Column(name='X', format='F', array=xtrace)
        col3 = fits.Column(name='Y', format='F', array=ytrace)
        col4 = fits.Column(name='THROUGHPUT', format='F', array=throughput_order3)
        col5 = fits.Column(name='TILT', format='F', array=tilt_order3)
        cols = fits.ColDefs([col1, col2, col3, col4, col5])

        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header['ORDER'] = (3, 'Spectral order.')
        hdu.header['EXTNAME'] = 'ORDER'
        hdu.header['EXTVER'] = 3
        hdul.append(hdu)

        hdul = fits.HDUList(hdul)
        hdul.writeto(filename, overwrite=True)
        hdul.writeto(filename + '.gz', overwrite=True)

    return


# ==============================================================================
# Code for generating the wave map reference file.
# ==============================================================================


def calc_2d_wave_map(wave_grid, x_dms, y_dms, tilt, oversample=2, padding=10, maxiter=5, dtol=1e-2):
    """Compute the 2D wavelength map on the detector.

    :param wave_grid: The wavelength corresponding to the x_dms, y_dms, and tilt values.
    :param x_dms: the trace x position on the detector in DMS coordinates.
    :param y_dms: the trace y position on the detector in DMS coordinates.
    :param tilt: the trace tilt angle in degrees.
    :param oversample: the oversampling factor of the input coordinates.
    :param padding: the native pixel padding around the edge of the detector.
    :param maxiter: the maximum number of iterations used when solving for the wavelength at each pixel.
    :param dtol: the tolerance of the iterative solution in pixels.

    :type wave_grid: array[float]
    :type x_dms: array[float]
    :type y_dms: array[float]
    :type tilt: array[float]
    :type oversample: int
    :type padding: int
    :type maxiter: int
    :type dtol: float

    :returns: wave_map_2d - an array containing the wavelength at each pixel on the detector.
    :rtype: array[float]
    """

    os = np.copy(oversample)
    xpad = np.copy(padding)
    ypad = np.copy(padding)

    # No need to compute wavelengths across the entire detector, slightly larger than SUBSTRIP256 will do.
    dimx, dimy = 2048, 300
    y_dms = y_dms + (dimy - 2048)  # Adjust y-coordinate to area of interest.

    # Generate the oversampled grid of pixel coordinates.
    x_vec = np.arange((dimx + 2*xpad)*os)/os - (os - 1)/(2*os) - xpad
    y_vec = np.arange((dimy + 2*ypad)*os)/os - (os - 1)/(2*os) - ypad
    x_grid, y_grid = np.meshgrid(x_vec, y_vec)

    # Iteratively compute the wavelength at each pixel.
    delta_x = 0.0  # A shift in x represents a shift in wavelength.
    for niter in range(maxiter):

        # Assume all y have same wavelength.
        wave_iterated = np.interp(x_grid - delta_x, x_dms[::-1], wave_grid[::-1])  # Invert arrays to get increasing x.

        # Compute the tilt angle at the wavelengths.
        tilt_tmp = np.interp(wave_iterated, wave_grid, tilt)

        # Compute the trace position at the wavelengths.
        x_estimate = np.interp(wave_iterated, wave_grid, x_dms)
        y_estimate = np.interp(wave_iterated, wave_grid, y_dms)

        # Project that back to pixel coordinates.
        x_iterated = x_estimate + (y_grid - y_estimate)*np.tan(np.deg2rad(tilt_tmp))

        # Measure error between requested and iterated position.
        delta_x = delta_x + (x_iterated - x_grid)

        # If the desired precision has been reached end iterations.
        if np.all(np.abs(x_iterated - x_grid) < dtol):
            break

    # Evaluate the final wavelength map, this time setting out-of-bounds values to NaN.
    wave_map_2d = np.interp(x_grid - delta_x, x_dms[::-1], wave_grid[::-1], left=np.nan, right=np.nan)

    # Extend to full detector size.
    tmp = np.full((os*(dimx + 2*xpad), os*(dimx + 2*xpad)), fill_value=np.nan)
    tmp[-os*(dimy + 2*ypad):] = wave_map_2d
    wave_map_2d = tmp

    return wave_map_2d


def init_wave_map():
    """Generate the NIRISS SOSS 2D wavelength reference file."""

    # Fixed parameters for the 2D wavelength map reference file.
    padding = 10
    oversample = 2
    orders = [1, 2, 3]
    subarrays = ['FULL', 'SUBSTRIP96', 'SUBSTRIP256']
    soss_ref_trace_table = 'SOSS_ref_trace_table_FULL.fits'  # Input SOSS reference file.
    filepattern = 'SOSS_ref_2D_wave_{}.fits'  # Output SOSS reference file.

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
        hdu.header['REFTYPE'] = ('WAVEMAP', 'Reference file type')
        hdu.header['PEDIGREE'] = ('GROUND', 'The pedigree of the refernce file')
        hdu.header['DESCRIP'] = ('2D wavelength map', 'Desription of the reference file')
        hdu.header['AUTHOR'] = ('Geert Jan Talens', 'Author of the reference file')
        hdu.header['USEAFTER'] = ('2000-01-01T00:00:00', 'Use after date of the reference file')
        hdu.header['EXP_TYPE'] = ('NIS_SOSS', 'Type of data in the exposure')
        hdul.append(hdu)

        for m in orders:

            # Read the 1D trace reference file.
            data = fits.getdata(soss_ref_trace_table, ext=m)

            # Unpack the 1D trace info.
            wave_grid = data['WAVELENGTH']
            x_dms = data['X']
            y_dms = data['Y']
            tilt = data['TILT']

            # Compute the 2D wavelength map.
            wave_map_2d = calc_2d_wave_map(wave_grid, x_dms, y_dms, tilt, oversample=oversample, padding=padding)

            # Add the 2D wavelength map to the fits file.
            hdu = fits.ImageHDU(wave_map_2d[lrow:urow, lcol:ucol].astype('float32'))
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


# ==============================================================================
# Code for generating the spectral profile reference file.
# ==============================================================================


def init_spec_profile(profile_file=None):
    """"""

    if profile_file is None:
        'files/2DTrace.fits'

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
            profile_2d = fits.getdata(profile_file, ext=0)
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


# ==============================================================================
# Code for generating the spectral kernel reference file.
# ==============================================================================


def init_spec_kernel(kernel_file=None):
    """"""

    if kernel_file is None:
        kernel_file = 'files/spectral_kernel_matrix_os_10_width_15pixels.fits'

    # Fixed parameters for the spectral kernels reference file. # TODO get from kernel_file.
    specos = 10
    halfwidth = 7
    nwave = 95
    wavemin = 0.5
    wavemax = 5.2
    soss_ref_spectral_kernel = 'SOSS_ref_spectral_kernel.fits'  # Output SOSS reference file.

    # TODO Read file provided by Loïc, replace with function that generates this info in the future.
    kernels = fits.getdata(kernel_file)

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


def main():

    init_spec_trace()
    init_wave_map()
    init_spec_profile()
    init_spec_kernel()

    return


if __name__ == '__main__':
    main()
