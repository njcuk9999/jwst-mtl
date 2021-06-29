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


def init_spec_trace(throghput_file='../Ref_files/NIRISS_Throughput_STScI.fits',
                    tilt_file='../../trace/SOSS_wavelength_dependent_tilt.ecsv'):
    """Generate the NIRISS SOSS 1D trace reference file.
    TODO This code will need to be updated once in-flight measurements are available.
    TODO CAR NIS-018-GR700XD Wavelength Calibration
    TODO CAR NIS-017-GR700XD Flux Calibration
    """

    # Fixed parameters for the 2D wavelength map reference file.
    subarrays = ['FULL', 'SUBSTRIP96', 'SUBSTRIP256']
    filepattern = 'SOSS_ref_trace_table_{}.fits'  # Output SOSS reference file.

    # Reference wavelength grid in micron.
    wave_grid = np.linspace(0.5, 5.5, 5001)

    # Read the SOSS total throughput as a function of wavelength.
    tab, hdr = fits.getdata(throghput_file, ext=1, header=True)

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


def main():

    return


if __name__ == '__main__':
    main()
