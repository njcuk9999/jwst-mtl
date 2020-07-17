"""
Created on Mon Feb 17 13:31:58 2020

@author: talens-irex
"""

import numpy as np

from astropy.io import ascii


def trace_polynomial(trace, m=1, maxorder=15):
    """Fit a polynomial to the trace of order m and return a
    dictionary containing the parameters and validity intervals.
    """

    # TODO added arbitrary maxorder to deal with poor exrapolatian, revisit when extrapolotion fixed.
    
    # Select the data for order m.
    mask = (trace['order'] == m)
    wave = trace['Wavelength'][mask]
    spatpix_ds9 = trace['xpos'][mask]
    specpix_ds9 = trace['ypos'][mask]
    
    # Find the edges of the domain.
    wavemin = np.amin(wave)
    wavemax = np.amax(wave)
    
    specmin = np.amin(specpix_ds9)
    specmax = np.amax(specpix_ds9)

    # Compute the polynomial parameters for x and y.
    order = 0
    while order <= maxorder:

        spatpars = np.polyfit(wave, spatpix_ds9, order)
        specpars = np.polyfit(wave, specpix_ds9, order)

        spatpixp_ds9 = np.polyval(spatpars, wave)
        specpixp_ds9 = np.polyval(specpars, wave)

        if np.all(np.abs(spatpix_ds9 - spatpixp_ds9) < 0.5) & np.all(np.abs(specpix_ds9 - specpixp_ds9) < 0.5):
            break
            
        order += 1
    
    # Compute the transform back to wavelength.
    wavegrid = wavemin + (wavemax - wavemin)*np.linspace(0., 1., 501)
    specgrid = np.polyval(specpars, wavegrid)
    wavepars = np.polyfit(specgrid, wavegrid, order)

    # Add the parameters to a dictionary.
    pars = dict()
    pars['spatpars'] = spatpars
    pars['specpars'] = specpars
    pars['specmin'] = specmin
    pars['specmax'] = specmax
    pars['wavepars'] = wavepars
    pars['wavemin'] = wavemin
    pars['wavemax'] = wavemax
    
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
    """Perform a simple bounds check on an array."""
    
    mask = (array >= lower) & (array <= upper)
    
    return mask


def specpix_ds9_to_frame(specpix_ds9, frame='dms', oversample=1):
    """Convert specpix from ds9 coordinates to the specified frame."""

    if frame == 'ds9':
        specpix = specpix_ds9
    elif frame == 'dms':
        specpix = 2048 * oversample - specpix_ds9
    elif frame == 'sim':
        specpix = 2048 * oversample - specpix_ds9
    else:
        ValueError('Unknown coordinate frame: {}'.format(frame))

    return specpix


def spatpix_ds9_to_frame(spatpix_ds9, frame='dms', oversample=1):
    """Convert spatpix from ds9 coordinates to the specified frame."""

    if frame == 'ds9':
        spatpix = spatpix_ds9
    elif frame == 'dms':
        spatpix = 256 * oversample - spatpix_ds9
    elif frame == 'sim':
        spatpix = spatpix_ds9
    else:
        ValueError('Unknown coordinate frame: {}'.format(frame))

    return spatpix


def pix_ds9_to_frame(specpix_ds9, spatpix_ds9, frame='dms', oversample=1):
    """Convert from ds9 to coordinates to the specified frame."""

    specpix = specpix_ds9_to_frame(specpix_ds9, frame=frame, oversample=oversample)
    spatpix = spatpix_ds9_to_frame(spatpix_ds9, frame=frame, oversample=oversample)

    return specpix, spatpix


def specpix_frame_to_ds9(specpix, frame='dms', oversample=1):
    """Convert specpix from an arbitrary frame to ds9 coordinates."""

    if frame == 'ds9':
        specpix_ds9 = specpix
    elif frame == 'dms':
        specpix_ds9 = 2048 * oversample - specpix
    elif frame == 'sim':
        specpix_ds9 = 2048 * oversample - specpix
    else:
        ValueError('Unknown coordinate frame: {}'.format(frame))

    return specpix_ds9


def spatpix_frame_to_ds9(spatpix, frame='dms', oversample=1):
    """Convert spatpix from an arbitrary frame to ds9 coordinates."""

    if frame == 'ds9':
        spatpix_ds9 = spatpix
    elif frame == 'dms':
        spatpix_ds9 = 256 * oversample - spatpix
    elif frame == 'sim':
        spatpix_ds9 = spatpix
    else:
        ValueError('Unknown coordinate frame: {}'.format(frame))

    return spatpix_ds9


def pix_frame_to_ds9(specpix, spatpix, frame='dms', oversample=1):
    """Convert from an arbitrary frame to ds9 coordinates."""

    specpix_ds9 = specpix_frame_to_ds9(specpix, frame=frame, oversample=oversample)
    spatpix_ds9 = spatpix_frame_to_ds9(spatpix, frame=frame, oversample=oversample)

    return specpix_ds9, spatpix_ds9


def wavelength_to_pix(wavelength, tracepars, m=1, frame='dms', oversample=1):
    """Convert wavelength to pixel coordinates for order m."""

    # Convert wavelenght to ds9 pixel coordinates.
    specpix_ds9 = np.polyval(tracepars[m]['specpars'], wavelength)
    spatpix_ds9 = np.polyval(tracepars[m]['spatpars'], wavelength)
    mask = bounds_check(wavelength, tracepars[m]['wavemin'], tracepars[m]['wavemax'])

    # Convert coordinates to the requested frame.
    specpix, spatpix = pix_ds9_to_frame(specpix_ds9, spatpix_ds9, frame=frame)

    # Oversample the coordinates.
    specpix = specpix*oversample
    spatpix = spatpix*oversample

    return specpix, spatpix, mask


def specpix_to_wavelength(specpix, tracepars, m=1, frame='dms', oversample=1):
    """Convert the spectral pixel coordinate to wavelength for order m."""

    # Remove any oversampling.
    specpix = specpix/oversample

    # Convert the input coordinates to ds9 coordinates.
    specpix_ds9 = specpix_frame_to_ds9(specpix, frame=frame)

    # Convert the specpix coordinates to wavelength.
    wavelength = np.polyval(tracepars[m]['wavepars'], specpix_ds9)
    mask = bounds_check(specpix_ds9, tracepars[m]['specmin'], tracepars[m]['specmax'])

    return wavelength, mask


def wavelength_map_2d(m, tracepars, oversample=1, use_tilt=False):
    """Compute the wavelengths of order m across the SUBSTRIP256 subarray."""
    
    if use_tilt:
        raise ValueError("The 'use_tilt' option has not yet been implemented.")
        
    # Get the coordinates of the pixels in the subarray.
    spatpix_dms, specpix_dms = np.indices((256*int(oversample), 2048*int(oversample)))

    # Convert to wavelengths using the trace polynomials.
    wavelength_map, mask = specpix_to_wavelength(specpix_dms, tracepars, m=m, frame='dms', oversample=oversample)
    
    # Set out-of-bounds and reference pixels to zero.
    wavelength_map[~mask] = 0
    wavelength_map[-4*int(oversample):] = 0
    wavelength_map[:, :4*int(oversample)] = 0
    wavelength_map[:, -4*int(oversample):] = 0
    
    return wavelength_map
