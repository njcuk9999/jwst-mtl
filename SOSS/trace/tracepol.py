"""
Created on Mon Feb 17 13:31:58 2020

@author: talens-irex
"""

import numpy as np

from astropy.io import ascii


def trace_polynomial(trace, m=1):
    """Fit a polynomial to the trace of order m and return a
    dictionary containing the parameters and validity intervals.
    """

    # TODO added arbitrary maxorder to deal with poor exrapolatian, revisit when extrapolotion fixed.
    
    # Select the data for order m.
    mask = (trace['order'] == m)
    wave = trace['Wavelength'][mask]
    x_ds9 = trace['xpos'][mask]
    y_ds9 = trace['ypos'][mask]
    
    # Find the edges of the domain.
    wmin = np.amin(wave)
    wmax = np.amax(wave)
    
    ymin = np.amin(y_ds9)
    ymax = np.amax(y_ds9)

    # Compute the polynomial parameters for x and y.
    order = 0
    while order <= maxorder:

        xpars = np.polyfit(wave, x_ds9, order)
        ypars = np.polyfit(wave, y_ds9, order)

        xp_ds9 = np.polyval(xpars, wave)
        yp_ds9 = np.polyval(ypars, wave)

        if np.all(np.abs(x_ds9 - xp_ds9) < 0.5) & np.all(np.abs(y_ds9 - yp_ds9) < 0.5):
            break
            
        order += 1
    
    # Compute the transform back to wavelength.
    wavegrid = wmin + (wmax - wmin)*np.linspace(0., 1., 501)
    ygrid = np.polyval(ypars, wavegrid)
    wpars = np.polyfit(ygrid, wavegrid, order)

    # Add the parameters to a dictionary.
    pars = dict()
    pars['xpars'] = xpars
    pars['ypars'] = ypars
    pars['ymin'] = ymin
    pars['ymax'] = ymax
    pars['wpars'] = wpars
    pars['wmin'] = wmin
    pars['wmax'] = wmax
    
    return pars


def get_tracepars(filename=None):
    """Read a file containing the trace profile and generate
    polynomial parameters for each order.
    """
    
    if filename is None:
        filename = 'NIRISS_GR700_trace_extended.csv'  # TODO Switch to pkg_resources in the future.
    
    # Read the trace.
    trace = ascii.read(filename)  # Read the Code V trace model from file. DS9 coordinates are used.
    trace['xpos'] /= 0.018  # Convert from micron to pixels.
    trace['ypos'] /= 0.018  # Convert from micron to pixels.

    # Compute polynomial parameters for different orders.
    tracepars = dict()
    for m in np.unique(trace['order']):
        pars = trace_polynomial(trace, m=m)
        tracepars[m] = pars
        
    return tracepars


def bounds_check(array, lower, upper):
    """Perform asimple bounds check on an array."""
    
    mask = (array >= lower) & (array <= upper)
    
    return mask


def wavelength_to_xy(wavelength, tracepars, m=1, frame='dms', oversample=1.):
    """Convert wavelength to x,y-position for order m."""

    x_ds9 = np.polyval(tracepars[m]['xpars'], wavelength)
    y_ds9 = np.polyval(tracepars[m]['ypars'], wavelength)
    mask = bounds_check(wavelength, tracepars[m]['wmin'], tracepars[m]['wmax'])

    if frame == 'ds9':
        x, y = x_ds9, y_ds9
    elif frame == 'dms':
        x, y = coords_ds9_to_dms(x_ds9, y_ds9)
    elif frame == 'sim':
        x, y = coords_ds9_to_sim(x_ds9, y_ds9)
    else:
        raise ValueError('Unknown coordinate frame: {}'.format(frame))

    x = x*oversample
    y = y*oversample

    return x, y, mask


def xy_to_wavelength(x, y, tracepars, m=1, frame='dms', oversample=1.):
    """Convert pixel position to wavelength for order m."""

    x = x/oversample
    y = y/oversample

    if frame == 'ds9':
        y_ds9 = y
    elif frame == 'dms':
        _, y_ds9 = coords_dms_to_ds9(x, y)
    elif frame == 'sim':
        _, y_ds9 = coords_sim_to_ds9(x, y)
    else:
        raise ValueError('Unknown coordinate frame: {}'.format(frame))

    wavelength = np.polyval(tracepars[m]['wpars'], y_ds9)
    mask = bounds_check(y_ds9, tracepars[m]['ymin'], tracepars[m]['ymax'])
    
    return wavelength, mask


def coords_ds9_to_dms(x_ds9, y_ds9, oversample=1.):
    """Transfrom ds9 coordinates to DMS coordinates."""

    x_dms = 2048*oversample - y_ds9
    y_dms = 256*oversample - x_ds9
    
    return x_dms, y_dms


def coords_dms_to_ds9(x_dms, y_dms, oversample=1.):
    """Transfrom DMS coordinates to ds9 coordinates."""
    
    x_ds9 = 256*oversample - y_dms
    y_ds9 = 2048*oversample - x_dms
    
    return x_ds9, y_ds9


def coords_ds9_to_sim(x_ds9, y_ds9, oversample=1.):
    """Transform DS9 coordinates to Simulation coordinates."""

    x_sim = 2048*oversample - y_ds9
    y_sim = x_ds9

    return x_sim, y_sim


def coords_sim_to_ds9(x_sim, y_sim, oversample=1.):
    """Transform Simulation coordinates to DS9 coordinates."""

    x_ds9 = y_sim
    y_ds9 = 2048*oversample - x_sim

    return x_ds9, y_ds9


def coords_dms_to_sim(x_dms, y_dms, oversample=1.):
    """Transform DMS coordinates to Simulation coordinates."""

    x_ds9, y_ds9 = coords_dms_to_ds9(x_dms, y_dms, oversample=oversample)
    x_sim, y_sim = coords_ds9_to_sim(x_ds9, y_ds9, oversample=oversample)

    return x_sim, y_sim


def coords_sim_to_dms(x_sim, y_sim, oversample=1.):
    """Transform Simulation coordinates to DMS coordinates."""

    x_ds9, y_ds9 = coords_sim_to_ds9(x_sim, y_sim, oversample=oversample)
    x_dms, y_dms = coords_ds9_to_dms(x_ds9, y_ds9, oversample=oversample)

    return x_dms, y_dms


def wavelength_map_2d(m, tracepars, oversample=1., use_tilt=False):
    """Compute the wavelengths of order m across the SUBSTRIP256 subarray."""
    
    if use_tilt:
        raise ValueError("The 'use_tilt' option has not yet been implemented.")
        
    # Get the coordinates of the pixels in the subarray.
    y_dms, x_dms = np.indices((256*int(oversample), 2048*int(oversample)))

    # Convert to wavelengths using the trace polynomials.
    wavelength_map, mask = xy_to_wavelength(x_dms, y_dms, tracepars, m=m, frame='dms', oversample=oversample)
    
    # Set out-of-bounds and reference pixels to zero.
    wavelength_map[~mask] = 0
    wavelength_map[-4*int(oversample):] = 0
    wavelength_map[:, :4*int(oversample)] = 0
    wavelength_map[:, -4*int(oversample):] = 0
    
    return wavelength_map
