#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:31:58 2020

@author: talens-irex
"""

import numpy as np

from astropy.io import ascii

def trace_polynomial(trace, m=1):
    """ Fit a polynomial to the trace of order m and return a 
    dictionary containing the parameters and validity intervals.
    """
    
    # Select the data for order m.
    mask = (trace['order'] == m)
    wave = trace['Wavelength'][mask]
    xpos = trace['xpos'][mask]
    ypos = trace['ypos'][mask]
    
    # Find the edges of the domain.
    wmin = np.amin(wave)
    wmax = np.amax(wave)
    
    ymin = np.amin(ypos)
    ymax = np.amax(ypos)

    # Compute the polynomial parameters for x and y.
    order = 0
    while True:
        xpars = np.polyfit(wave, xpos, order)
        ypars = np.polyfit(wave, ypos, order)
        
        x = np.polyval(xpars, wave)
        y = np.polyval(ypars, wave)
        
        if np.all(np.abs(xpos - x) < 0.5) & np.all(np.abs(ypos - y) < 0.5):
            break
            
        order += 1
    
    # Compute the transform back to wavelength.
    wavegrid = wmin + (wmax - wmin)*np.linspace(0., 1., 501)
    ygrid = np.polyval(ypars, wavegrid)
    wpars = np.polyfit(ygrid, wavegrid, order)

    # Add the parameters to a dictionary.
    pars = dict()
    pars['x'] = xpars
    pars['y'] = ypars
    pars['ymin'] = ymin
    pars['ymax'] = ymax
    pars['w'] = wpars
    pars['wmin'] = wmin
    pars['wmax'] = wmax
    
    return pars

def get_tracepars(filename=None):
    """ Read a file containing the trace profile and generate
    polynomial parameters for each order.
    """
    
    if filename is None:
        filename = 'NIRISS_GR700_trace.csv' # When incorporated into a package pkg_resources should be used for this.
    
    # Read the trace.
    trace = ascii.read(filename)
    trace['xpos'] /= 0.018
    trace['ypos'] /= 0.018

    # Compute polynomial parameters for different orders.
    tracepars = dict()
    for m in np.unique(trace['order']):
        pars = trace_polynomial(trace, m=m)
        tracepars[m] = pars
        
    return tracepars

def bounds_check(array, lower, upper):
    """ Perform asimple bounds check on an array. """
    
    mask = (array >= lower) & (array <= upper)
    
    return mask

def wavelength2x(wavelength, tracepars, m=1):
    """ Convert wavelength to x-position for order m. """

    x = np.polyval(tracepars[m]['x'], wavelength)
    mask = bounds_check(wavelength, tracepars[m]['wmin'], tracepars[m]['wmax'])
    
    return x, mask

def wavelength2y(wavelength, tracepars, m=1):
    """ Convert wavelength to y-position for order m. """
    
    y = np.polyval(tracepars[m]['y'], wavelength)
    mask = bounds_check(wavelength, tracepars[m]['wmin'], tracepars[m]['wmax'])
    
    return y, mask

def y2wavelength(y, tracepars, m=1):
    """ Convert y-position to wavelength for order m. """
    
    wavelength = np.polyval(tracepars[m]['w'], y)
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