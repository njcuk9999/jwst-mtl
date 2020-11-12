#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:35:32 2020

@author: albert
"""

import sys
sys.path.insert(0, "../../trace/")

import numpy as np
from astropy.io import fits
from astropy.table import Table
import tracepol


def main():
    """Generate the NIRISS SOSS 1D trace reference file."""

    throughput_ref = 'NIRISS_Throughput_STScI.fits'
    tilt_ref = 'SOSS_wavelength_dependent_tilt.ecsv'
    trace_xy_ref = '../../trace/NIRISS_GR700_trace_extended.csv'
    trace_1d_ref = 'SOSS_trace_ref.fits'

    # Read the SOSS total throughput as a function of wavelength.
    tab, hdr = fits.getdata(throughput_ref, ext=1, header=True)

    master_wave = tab[0]['LAMBDA']/1000.  # Use this wavelength grid as the reference for all columns.
    throughput_order1 = tab[0]['SOSS_ORDER1']
    throughput_order2 = tab[0]['SOSS_ORDER2']
    throughput_order3 = tab[0]['SOSS_ORDER3']

    # Fix small negative throughput values.
    throughput_order1 = np.where(throughput_order1 < 0, 0, throughput_order1)
    throughput_order2 = np.where(throughput_order2 < 0, 0, throughput_order2)
    throughput_order3 = np.where(throughput_order3 < 0, 0, throughput_order3)

    # Read the tilt as a function of wavelength.
    tab = Table.read(tilt_ref)

    tilt_wave = tab['Wavelength']
    tilt_order1 = tab['order 1']
    tilt_order2 = tab['order 2']
    tilt_order3 = tab['order 3']

    # Interpolate the tilt to the same wavelengths as the throughput.
    # Default bounds handling (constant boundary) is fine.
    tilt_order1 = np.interp(master_wave, tilt_wave, tilt_order1)
    tilt_order2 = np.interp(master_wave, tilt_wave, tilt_order2)
    tilt_order3 = np.interp(master_wave, tilt_wave, tilt_order3)

    # Get the trace parameters, function found in tracepol imported above.
    tracepars = tracepol.get_tracepars(trace_xy_ref)

    # Using tracepol, compute trace positions for each order and mask out-of-bound values.
    x_sub256_order1, y_sub256_order1, mask = tracepol.wavelength_to_pix(master_wave, tracepars, m=1, frame='dms',
                                                                        subarray='SUBSTRIP256')
    x_sub256_order1 = np.where(mask, x_sub256_order1, np.nan)
    y_sub256_order1 = np.where(mask, y_sub256_order1, np.nan)

    x_sub256_order2, y_sub256_order2, mask = tracepol.wavelength_to_pix(master_wave, tracepars, m=2, frame='dms',
                                                                        subarray='SUBSTRIP256')
    x_sub256_order2 = np.where(mask, x_sub256_order2, np.nan)
    y_sub256_order2 = np.where(mask, y_sub256_order2, np.nan)

    x_sub256_order3, y_sub256_order3, mask = tracepol.wavelength_to_pix(master_wave, tracepars, m=3, frame='dms',
                                                                        subarray='SUBSTRIP256')
    x_sub256_order3 = np.where(mask, x_sub256_order3, np.nan)
    y_sub256_order3 = np.where(mask, y_sub256_order3, np.nan)

    # Create the reference file.
    hdu0 = fits.PrimaryHDU()
    hdu0.header['CREATOR'] = 'Geert Jan Talens'

    # Order 1 table.
    col1 = fits.Column(name='WAVELENGTH', format='F', array=master_wave)
    col2 = fits.Column(name='THROUGHPUT', format='F', array=throughput_order1)
    col3 = fits.Column(name='TILT', format='F', array=tilt_order1)
    col4 = fits.Column(name='X_SUBSTRIP256', format='F', array=x_sub256_order1)
    col5 = fits.Column(name='Y_SUBSTRIP256', format='F', array=y_sub256_order1)
    cols = fits.ColDefs([col1, col2, col3, col4, col5])

    hdu1 = fits.BinTableHDU.from_columns(cols)
    hdu1.header['ORDER'] = (1, 'Spectral order.')
    hdu1.header['DY_FULL'] = (1792, 'Y offset for FULL frame images.')  # Hardcode the offsets for the other subarrays.
    hdu1.header['DY_SUB96'] = (-10, 'Y offset for SUBSTRIP96 images.')

    # Order 2 table.
    col1 = fits.Column(name='WAVELENGTH', format='F', array=master_wave)
    col2 = fits.Column(name='THROUGHPUT', format='F', array=throughput_order2)
    col3 = fits.Column(name='TILT', format='F', array=tilt_order2)
    col4 = fits.Column(name='X_SUBSTRIP256', format='F', array=x_sub256_order2)
    col5 = fits.Column(name='Y_SUBSTRIP256', format='F', array=y_sub256_order2)
    cols = fits.ColDefs([col1, col2, col3, col4, col5])

    hdu2 = fits.BinTableHDU.from_columns(cols)
    hdu2.header['ORDER'] = (2, 'Spectral order.')
    hdu2.header['DY_FULL'] = (1792, 'Y offset for FULL frame images.')  # Hardcode the offsets for the other subarrays.
    hdu2.header['DY_SUB96'] = (-10, 'Y offset for SUBSTRIP96 images.')

    # Order 3 table.
    col1 = fits.Column(name='WAVELENGTH', format='F', array=master_wave)
    col2 = fits.Column(name='THROUGHPUT', format='F', array=throughput_order3)
    col3 = fits.Column(name='TILT', format='F', array=tilt_order3)
    col4 = fits.Column(name='X_SUBSTRIP256', format='F', array=x_sub256_order3)
    col5 = fits.Column(name='Y_SUBSTRIP256', format='F', array=y_sub256_order3)
    cols = fits.ColDefs([col1, col2, col3, col4, col5])

    hdu3 = fits.BinTableHDU.from_columns(cols)
    hdu3.header['ORDER'] = (3, 'Spectral order.')
    hdu3.header['DY_FULL'] = (1792, 'Y offset for FULL frame images.')  # Hardcode the offsets for the other subarrays.
    hdu3.header['DY_SUB96'] = (-10, 'Y offset for SUBSTRIP96 images.')

    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])
    hdul.writeto(trace_1d_ref, overwrite=True)

    return


if __name__ == '__main__':
    main()
