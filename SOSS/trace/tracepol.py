#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:31:58 2020

@author: talens-irex
"""

import numpy as np
from astropy.io import ascii

def trace_polynomial(trace, m=1, method=None):
    """ Fit a polynomial to the trace of order m and return a
    dictionary containing the parameters and validity intervals.
    _________________________________________________
    Inputs: trace - csv file with trace data
            m - trace order
            method - wavelength or pixel
    Outputs: Dictionary containing trace parameters
    """

    # Select the data for order m.
    mask = (trace['order'] == m)
    wave = trace['Wavelength'][mask]
    # We assume x is the position on the spectral, and y on the spatial axis.
    ypos = trace['xpos'][mask]  # x and y directions are reversed in csv
    xpos = trace['ypos'][mask]

    # Find the edges of the domain.
    wmin = np.amin(wave)
    wmax = np.amax(wave)

    xmin = np.amin(xpos)
    xmax = np.amax(xpos)

    # Determine trace parameters
    if method is None or method == 'wavelength':
        # Compute pixel positions of trace from wavelength data
        order = 0
        while True:
            xpars = np.polyfit(wave, xpos, order)
            ypars = np.polyfit(wave, ypos, order)

            x = np.polyval(xpars, wave)
            y = np.polyval(ypars, wave)

            # If better than half micron (wavelength step size)
            if np.all(np.abs(xpos - x) < 0.5) & np.all(np.abs(ypos - y) < 0.5):
                pars = dict()  # Initialize the results dictionary
                pars['xpar'] = xpars
                break

            order += 1

    if method == 'pixel':
        # Compute y pixel position of trace from x positions
        order = 0
        while True:
            ypars = np.polyfit(xpos, ypos, order)

            y = np.polyval(ypars, xpos)

            # If better than half micron (wavelength step size)
            if np.all(np.abs(ypos - y) < 0.5):
                pars = dict()  # Initialize the results dictionary
                break

            order += 1

    # Compute the transform back to wavelength.
    wavegrid = wmin + (wmax - wmin)*np.linspace(0., 1., 501)  # ~10x oversampled
    ygrid = np.polyval(ypars, wavegrid)
    wpars = np.polyfit(ygrid, wavegrid, order)

    # Add the common parameters to a dictionary.
    pars['ypar'] = ypars
    pars['xmin'] = xmin
    pars['xmax'] = xmax
    pars['wpar'] = wpars
    pars['wmin'] = wmin
    pars['wmax'] = wmax

    return pars

def get_tracepars(filename=None, method=None):
    """ Read a file containing the trace profile and generate
    polynomial parameters for each order.
    _________________________________________________
    Inputs: filename - path to csv file with trace data
            method - wavelength or pixel
    Outputs: Dictionary containing trace parameters
    """

    if filename is None:
        filename = 'NIRISS_GR700_trace.csv' # pkg_resources?

    # Read the trace.
    trace = ascii.read(filename)
    trace['xpos'] /= 0.018  # convert arcsec to pixels
    trace['ypos'] /= 0.018

    # Compute polynomial parameters for different orders.
    tracepars = dict()
    for m in np.unique(trace['order']):
        pars = trace_polynomial(trace, m=m, method=method)
        tracepars[m] = pars

    return tracepars

def bounds_check(array, lower, upper):
    """ Perform asimple bounds check on an array. """

    mask = (array >= lower) & (array <= upper)

    return mask

def wavelength2x(wavelength, tracepars, m=1):
    """ Convert wavelength to x-position for order m. """

    x = np.polyval(tracepars[m]['xpar'], wavelength)
    mask = bounds_check(wavelength, tracepars[m]['wmin'], tracepars[m]['wmax'])

    return x, mask

def wavelength2y(wavelength, tracepars, m=1):
    """ Convert wavelength to y-position for order m. """

    y = np.polyval(tracepars[m]['ypar'], wavelength)
    mask = bounds_check(wavelength, tracepars[m]['wmin'], tracepars[m]['wmax'])

    return y, mask

def y2wavelength(y, tracepars, m=1):
    """ Convert y-position to wavelength for order m. """

    wavelength = np.polyval(tracepars[m]['wpar'], y)
    mask = bounds_check(y, tracepars[m]['ymin'], tracepars[m]['ymax'])

    return wavelength, mask

def wavelength2xy(wavelength, tracepars, m=1):
    """ Convert wavelength to x,y-position for order m. """

    x, mask = wavelength2x(wavelength, tracepars, m=m)
    y, mask = wavelength2y(wavelength, tracepars, m=m)

    return x, y, mask

def coords_ds9_to_dms(x_ds9, y_ds9):
    """ Transfrom ds9 coordinates to DMS coordinates. """

    x_dms = 2048 - y_ds9
    y_dms = 256 - x_ds9

    return x_dms, y_dms

def coords_dms_to_ds9(x_dms, y_dms):
    """ Transfrom DMS coordinates to ds9 coordinates. """

    x_ds9 = 256 - y_dms
    y_ds9 = 2048 - x_dms

    return x_ds9, y_ds9

def wavelength_map_2D(m, tracepars, use_tilt=False):
    """ Compute the wavelengths of order m across the SUBSTRIP256 subarray. """

    if use_tilt:
        raise ValueError("The 'use_tilt' option has not yet been implemented.")

    # Get the coordinates of the pixels in the subarray.
    y_dms, x_dms = np.indices((256, 2048))
    x_ds9, y_ds9 = coords_dms_to_ds9(x_dms, y_dms)

    # Convert to wavelengths using the trace polynomials.
    wavelength_map, mask = y2wavelength(y_ds9, tracepars, m=m)

    # Set out-of-bounds and reference pixels to zero.
    wavelength_map[~mask] = 0
    wavelength_map[-4:] = 0
    wavelength_map[:,:4] = 0
    wavelength_map[:,-4:] = 0

    return wavelength_map
