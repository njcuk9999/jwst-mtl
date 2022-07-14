#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:37 2022
@author: MCR
Routines to convert ATOCA output to match the Eureka! Stage 3 file formats.
Requires the astraeus package for xarray functionality. Installation
insctructions are here: https://kevin218.github.io/Astraeus/
"""

from astraeus import xarrayIO as xrio
from astropy.io import fits
from astropy.time import Time
from jwst import datamodels
import numpy as np


def unpack_spectra(filename, quantities=('WAVELENGTH', 'FLUX', 'FLUX_ERROR')):
    """Unpack the ATOCA extract_1d output into a dictoionary of wavelengths,
    fluxes, and errors.
    Parameters
    ----------
    filename : str
        Path to the ATOCA output extract1dstep file.
    quantities : array-like
        Quantities to extract.
    Returns
    -------
    all_spec : dict
        Dictionary containing arrays of the requested quantities for each
        order.
    """

    multi_spec = datamodels.open(filename)

    all_spec = {sp_ord: {quantity: [] for quantity in quantities}
                for sp_ord in [1, 2, 3]}
    for spec in multi_spec.spec:
        sp_ord = spec.spectral_order
        for quantity in quantities:
            all_spec[sp_ord][quantity].append(spec.spec_table[quantity])

    for sp_ord in all_spec:
        for key in all_spec[sp_ord]:
            all_spec[sp_ord][key] = np.array(all_spec[sp_ord][key])

    return all_spec


def make_temporal_axis(filename):
    """Create the time axis for a TSO based on the information included in the
    file header.
    Parameters
    ----------
    filename : str
        Path to input file.
    Returns
    -------
    t : array-like
        BJD time axis. Time stamps correspond to midpoints of each
        integration.
    """

    header = fits.getheader(filename, 0)
    # Observation start time
    t_start = header['DATE-OBS'] + 'T' + header['TIME-OBS']
    t_start = Time(t_start, format='isot', scale='utc')
    # Group time in s
    tgroup = header['TGROUP'] / 3600 / 24
    # Group time
    ngroup = header['NGROUPS'] + 1  # ngroup+1 to take into account reset
    # Number of integrations
    nint = header['NINTS']

    # Create time axis
    t = np.arange(nint) * tgroup * ngroup + t_start.jd + (tgroup*ngroup)/2

    return t


def convert_atoca_to_eureka(infile, outfile, orders=None):
    """Exactly what it says on the tin.
    **Not all data variables are currently filled in, and many are currently
    just place-holders as the required information is not contained within the
    ATOCA extract1dstep output and will require a more involved appraoch.
    Parameters
    ----------
    infile : str
        Path to ATOCA extract1dstep output.
    outfile : str
        Output file name.
    orders : list[str]
        Orders to consider.
    Returns
    -------
    outdata : xarray
        ATOCA 1d extracted spectra in Eureka! Stage 3 format.
    """

    if orders is None:
        orders = ['1', '2', '3']
    indata = unpack_spectra(infile)
    outdata = xrio.makeDataset()

    tgroup = fits.getheader(infile, 0)['TGROUP'] / 3600 / 24
    ngroup = fits.getheader(infile, 0)['NGROUPS'] - 1
    gain = 1.6
    dn2e = tgroup * ngroup * gain
    dimx = fits.getheader(infile, 0)['SUBSIZE1']
    dimy = fits.getheader(infile, 0)['SUBSIZE2']
    nint = fits.getheader(infile, 0)['nints']

    outdata['medflux'] = (['y', 'x'], np.zeros((dimy, dimx)) * dn2e)
    outdata['medflux'].attrs['flux_units'] = "ELECTRONS"

    for order in orders:
        outdata['atocaspec_o' + order] = (['time', 'x'], indata[int(order)]['FLUX'][:, ::-1] * dn2e)
        outdata['atocaspec_o' + order].attrs['flux_units'] = "ELECTRONS"
        outdata['atocaspec_o' + order].attrs['time_units'] = "BJD_TBD"

        outdata['atocaerr_o' + order] = (['time', 'x'], indata[int(order)]['FLUX_ERROR'][:, ::-1] * dn2e)
        outdata['atocaerr_o' + order].attrs['flux_units'] = "ELECTRONS"
        outdata['atocaerr_o' + order].attrs['time_units'] = "BJD_TBD"

        outdata['src_ypos_exact_o' + order] = (['time', 'x'], np.zeros((nint, dimx)))
        outdata['src_ypos_exact_o' + order].attrs['time_units'] = "BJD_TBD"
        outdata['src_ypos_width_o' + order] = (
        ['time', 'x'], np.zeros((nint, dimx)))
        outdata['src_ypos_width_o' + order].attrs['time_units'] = "BJD_TBD"

        outdata['wave_1d_o' + order] = (['x'], indata[int(order)]['WAVELENGTH'][0][::-1])
        outdata['wave_1d_o' + order].attrs['wave_units'] = "micron"

        outdata['wave_2d_o' + order] = (['y', 'x'], np.zeros((dimy, dimx)))
        outdata['wave_2d_o' + order].attrs['wave_units'] = "micron"

    outdata.coords['x'] = np.arange(dimx)
    outdata.coords['y'] = np.arange(dimy)
    outdata.coords['time'] = make_temporal_axis(infile)

    xrio.writeXR(outfile, outdata)

    return outdata
