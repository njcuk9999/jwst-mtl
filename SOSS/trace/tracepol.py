"""
Created on Mon Feb 17 13:31:58 2020

@author: talens-irex
"""

import numpy as np
from numpy.polynomial import Legendre

from astropy.io import ascii

# SUBSTRIP256 keeps solumns 0:255 (0 based) in the nat frame.
# SUBSTRIP96 keeps columns 150:245 (0 based) in the nat frame.


def trace_polynomial(trace, m=1, maxorder=15):
    """Fit a polynomial to the trace of order m and return a
    dictionary containing the parameters and validity intervals.

    :param trace: astropy table containing modelled trace points.
    :param m: spectral order for which to fit a polynomial.
    :param maxorder: maximum polynomial order to use.

    :type trace: astropy.table.Table
    :type m: int
    :type maxorder: int

    :returns: pars - dictionary containg the polynomial solution.
    :rtype: dict
    """

    # TODO added arbitrary maxorder to deal with poor exrapolatian, revisit when extrapolation fixed.
    
    # Select the data for order m.
    mask = (trace['order'] == m)
    wave = trace['Wavelength'][mask]
    spatpix_ref = trace['xpos'][mask]
    specpix_ref = trace['ypos'][mask]
    
    # Find the edges of the domain.
    wavemin = np.amin(wave)
    wavemax = np.amax(wave)
    
    specmin = np.amin(specpix_ref)
    specmax = np.amax(specpix_ref)

    # Compute the polynomial parameters for x and y.
    order = 0
    while order <= maxorder:

        spatpol = Legendre.fit(np.log(wave), spatpix_ref, order)
        specpol = Legendre.fit(np.log(wave), specpix_ref, order)

        spatpixp_nat = spatpol(np.log(wave))
        specpixp_nat = specpol(np.log(wave))

        if np.all(np.abs(spatpix_ref - spatpixp_nat) < 0.5) & np.all(np.abs(specpix_ref - specpixp_nat) < 0.5):
            break
            
        order += 1
    
    # Compute the transform back to wavelength.
    wavegrid = wavemin + (wavemax - wavemin)*np.linspace(0., 1., 501)
    specgrid = specpol(np.log(wavegrid))
    wavepol = Legendre.fit(specgrid, np.log(wavegrid), order)

    # Add the parameters to a dictionary.
    pars = dict()
    pars['spat_coef'] = spatpol.coef
    pars['spat_domain'] = spatpol.domain
    pars['spec_coef'] = specpol.coef
    pars['spec_domain'] = specpol.domain
    pars['wave_coef'] = wavepol.coef
    pars['wave_domain'] = wavepol.domain
    
    return pars


def get_tracepars(filename=None):
    """Read a file containing the trace profile and generate
    polynomial parameters for each order.

    :param filename: file containing modelled trace points.

    :type filename: str

    :returns: tracepars - a dictionary containg the parameters for the polynomial fits.
    :rtype: dict
    """
    
    if filename is None:
        filename = 'NIRISS_GR700_trace_extended.csv'  # TODO Switch to pkg_resources in the future.
    
    # Read the trace.
    trace = ascii.read(filename)  # Read the Code V trace model from file. DS9 coordinates are used.
    trace['xpos'] /= 0.018  # Convert from micron to pixels.
    trace['ypos'] /= 0.018  # Convert from micron to pixels.
    trace['xpos'] -= 0.5  # Set the origin at the center of the lower-left pixel.
    trace['ypos'] -= 0.5  # Set the origin at the center of the lower-left pixel.

    # Compute polynomial parameters for different orders.
    tracepars = dict()
    for m in np.unique(trace['order']):
        pars = trace_polynomial(trace, m=m)
        tracepars[m] = pars
        
    return tracepars


def bounds_check(values, lower, upper):
    """Perform a simple bounds check on an array.

    :param values: an array of values.
    :param lower: the lower bound of the valid range.
    :param upper: the upper bound of the valid range.

    :type values: array[float]
    :type lower: float
    :type upper: float

    :returns: mask - a boolean array that is True when values is between lower and upper.
    :rtype: array[bool]
    """
    
    mask = (values >= lower) & (values <= upper)
    
    return mask


def specpix_ref_to_frame(specpix_ref, frame='dms', oversample=1):
    """Convert specpix from nat coordinates to the specified frame.

    :param specpix_ref: specpix coordinates in the dms frame of SUBSTRIP256.
    :param frame: the output coordinate frame.
    :param oversample: the oversampling factor of the input coordinates.

    :type specpix_ref: array[float]
    :type frame: str
    :type oversample: int

    :returns: specpix - the input coordinates transformed to the requested coordinate frame.
    :rtype: array[float]
    """

    if frame == 'nat':
        specpix = specpix_ref
    elif frame == 'dms':
        specpix = 2047*oversample - specpix_ref
    elif frame == 'sim':
        specpix = 2047*oversample - specpix_ref
    else:
        raise ValueError('Unknown coordinate frame: {}'.format(frame))

    return specpix


def spatpix_ref_to_frame(spatpix_ref, frame='dms', subarray='SUBSTRIP256', oversample=1):
    """Convert spatpix from nat coordinates in SUBSTRIP256 to the specified frame and subarray.

    :param spatpix_ref: spatpix coordinates in the dms frame of SUBSTRIP256.
    :param frame: the output coordinate frame.
    :param subarray: the output coordinate subarray.
    :param oversample: the oversampling factor of the input coordinates.

    :type spatpix_ref: array[float]
    :type frame: str
    :type subarray: str
    :type oversample: int

    :returns: spatpix - the input coordinates transformed to the requested coordinate frame and subarray.
    :rtype: array[float]
    """

    if (frame == 'nat') & (subarray == 'SUBSTRIP256'):
        spatpix = spatpix_ref
    elif (frame == 'dms') & (subarray == 'SUBSTRIP256'):
        spatpix = 255*oversample - spatpix_ref
    elif (frame == 'sim') & (subarray == 'SUBSTRIP256'):
        spatpix = spatpix_ref
    elif (frame == 'nat') & (subarray == 'SUBSTRIP96'):
        spatpix = spatpix_ref - 150*oversample
    elif (frame == 'dms') & (subarray == 'SUBSTRIP96'):
        spatpix = 245*oversample - spatpix_ref
    elif (frame == 'sim') & (subarray == 'SUBSTRIP96'):
        spatpix = spatpix_ref - 150*oversample
    else:
        raise ValueError('Unknown coordinate frame or subarray: {} {}'.format(frame, subarray))

    return spatpix


def pix_ref_to_frame(specpix_ref, spatpix_ref, frame='dms', subarray='SUBSTRIP256', oversample=1):
    """Convert from nat coordinates in SUBSTRIP256 to the specified frame and subarray.

    :param specpix_ref: specpix coordinates in the dms frame of SUBSTRIP256.
    :param spatpix_ref: spatpix coordinates in the dms frame of SUBSTRIP256.
    :param frame: the output coordinate frame.
    :param subarray: the output coordinate subarray.
    :param oversample: the oversampling factor of the input coordinates.

    :type specpix_ref: array[float]
    :type spatpix_ref: array[float]
    :type frame: str
    :type subarray: str
    :type oversample: int

    :returns: spatpix, specpix - the input coordinates transformed to the requested coordinate frame and subarray.

    :rtype: Typle(array[float], array[float])
    """

    specpix = specpix_ref_to_frame(specpix_ref, frame=frame, oversample=oversample)
    spatpix = spatpix_ref_to_frame(spatpix_ref, frame=frame, subarray=subarray, oversample=oversample)

    return specpix, spatpix


def specpix_frame_to_ref(specpix, frame='dms', oversample=1):
    """Convert specpix from an arbitrary frame to nat coordinates.

    :param specpix: specpix coordinates in an arbitrary coordinate frame.
    :param frame: the input coordinate frame.
    :param oversample: the oversampling factor of the input coordinates.

    :type specpix: array[float]
    :type frame: str
    :type oversample: int

    :returns: specpix_ref - the input coordinates transformed to nat coordinate frame.
    :rtype: array[float]
    """

    if frame == 'nat':
        specpix_ref = specpix
    elif frame == 'dms':
        specpix_ref = 2047*oversample - specpix
    elif frame == 'sim':
        specpix_ref = 2047*oversample - specpix
    else:
        raise ValueError('Unknown coordinate frame: {}'.format(frame))

    return specpix_ref


def spatpix_frame_to_ref(spatpix, frame='dms', subarray='SUBSTRIP256', oversample=1):
    """Convert spatpix from an arbitrary frame to nat coordinates in SUBSTRIP256.

    :param spatpix: spatpix coordinates in an arbitrary coordinate frame.
    :param frame: the input coordinate frame.
    :param subarray: the input coordinate subarray.
    :param oversample: the oversampling factor of the input coordinates.

    :type spatpix: array[float]
    :type frame: str
    :type subarray: str
    :type oversample: int

    :returns: spatpix_ref - the input coordinates transformed to nat coordinate frame and SUBSTRIP256 subarray.
    :rtype: array[float]
    """

    if (frame == 'nat') & (subarray == 'SUBSTRIP256'):
        spatpix_ref = spatpix
    elif (frame == 'dms') & (subarray == 'SUBSTRIP256'):
        spatpix_ref = 255*oversample - spatpix
    elif (frame == 'sim') & (subarray == 'SUBSTRIP256'):
        spatpix_ref = spatpix
    elif (frame == 'nat') & (subarray == 'SUBSTRIP96'):
        spatpix_ref = spatpix + 150*oversample
    elif (frame == 'dms') & (subarray == 'SUBSTRIP96'):
        spatpix_ref = 245*oversample - spatpix
    elif (frame == 'sim') & (subarray == 'SUBSTRIP96'):
        spatpix_ref = spatpix + 150*oversample
    else:
        raise ValueError('Unknown coordinate frame or subarray: {} {}'.format(frame, subarray))

    return spatpix_ref


def pix_frame_to_ref(specpix, spatpix, frame='dms', subarray='SUBSTRIP256', oversample=1):
    """Convert from an arbitrary frame and subarray to nat coordinates in SUBSTRIP256.

    :param specpix: specpix coordinates in an arbitrary coordinate frame.
    :param spatpix: spatpix coordinates in an arbitrary coordinate frame.
    :param frame: the input coordinate frame.
    :param subarray: the input coordinate subarray.
    :param oversample: the oversampling factor of the input coordinates.

    :type specpix: array[float]
    :type spatpix: array[float]
    :type frame: str
    :type subarray: str
    :type oversample: int

    :returns: specpix_ref, spatpix_ref - the input coordinates transformed to nat coordinate frame and SUBSTRIP256 subarray.
    :rtype: Tuple(array[float], array[float])
    """

    specpix_ref = specpix_frame_to_ref(specpix, frame=frame, oversample=oversample)
    spatpix_ref = spatpix_frame_to_ref(spatpix, frame=frame, subarray=subarray, oversample=oversample)

    return specpix_ref, spatpix_ref


def wavelength_to_pix(wavelength, tracepars, m=1, frame='dms', subarray='SUBSTRIP256', oversample=1):
    """Convert wavelength to pixel coordinates for order m.

    :param wavelength: wavelength values in microns.
    :param tracepars: the trace polynomial solutions returned by get_tracepars.
    :param m: the spectral order.
    :param frame: the coordinate frame of the output coordinates (nat, dms or sim).
    :param subarray: the subarray of the output coordinates (SUBARRAY256 or SUBARRAY96).
    :param oversample: the oversampling factor of the outpur coordinates.

    :type wavelength: array[float]
    :type tracepars: dict
    :type m: int
    :type frame: str
    :type subarray: str
    :type oversample: int

    :returns: specpix - the spectral pixel coordinates, spatpix - the spatial pixel coordinates,
    mask - an array that is True when the specpix values were within the valid range of the polynomial.
    :rtype: Tuple(array[float], array[float], array[bool])
    """

    # Convert wavelenght to nat pixel coordinates.
    w2spec = Legendre(tracepars[m]['spec_coef'], domain=tracepars[m]['spec_domain'])
    w2spat = Legendre(tracepars[m]['spat_coef'], domain=tracepars[m]['spat_domain'])

    specpix_nat = w2spec(np.log(wavelength))
    spatpix_nat = w2spat(np.log(wavelength))
    mask = bounds_check(np.log(wavelength), tracepars[m]['spec_domain'][0], tracepars[m]['spec_domain'][1])

    # Convert coordinates to the requested frame.
    specpix, spatpix = pix_ref_to_frame(specpix_nat, spatpix_nat, frame=frame, subarray=subarray)

    # Oversample the coordinates.
    specpix = specpix*oversample
    spatpix = spatpix*oversample

    return specpix, spatpix, mask


def specpix_to_wavelength(specpix, tracepars, m=1, frame='dms', oversample=1):
    """Convert the spectral pixel coordinate to wavelength for order m.

    :param specpix: the pixel values.
    :param tracepars: the trace polynomial solutions returned by get_tracepars.
    :param m: the spectral order.
    :param frame: the coordinate frame of the input coordinates (nat, dms or sim).
    :param oversample: the oversampling factor of the input coordinates.

    :type specpix: array[float]
    :type tracepars: dict
    :type m: int
    :type frame: str
    :type oversample: int

    :returns: wavelength - an array containing the wavelengths corresponding to specpix,
    mask - an array that is True when the specpix values were within the valid range of the polynomial.
    :rtype: Tuple(array[float], array[bool])
    """

    # Remove any oversampling.
    specpix = specpix/oversample

    # Convert the input coordinates to nat coordinates.
    specpix_nat = specpix_frame_to_ref(specpix, frame=frame)

    # Convert the specpix coordinates to wavelength.
    spec2w = Legendre(tracepars[m]['wave_coef'], domain=tracepars[m]['wave_domain'])

    with np.errstate(over='ignore'):
        wavelength = np.exp(spec2w(specpix_nat))
    mask = bounds_check(specpix_nat, tracepars[m]['wave_domain'][0], tracepars[m]['wave_domain'][1])

    return wavelength, mask


def wavelength_map_2d(tracepars, m=1, subarray='SUBSTRIP256', oversample=1, use_tilt=False):
    """Compute the wavelengths of order m in dms coordinates for the specified subarray.

    :param tracepars: the trace polynomial solutions returned by get_tracepars.
    :param m: the spectral order.
    :param subarray: the subarray of the output coordinates (SUBARRAY256 or SUBARRAY96).
    :param oversample: the oversampling factor of the input coordinates.
    :param use_tilt: Include the effect of tilt in the output.

    :type tracepars: dict
    :type m: int
    :type subarray: str
    :type oversample: int
    :type use_tilt: bool

    :returns: wavelength_map - A 2D array of wavelength values across the detector.
    :rtype: array[float]
    """
    
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

    if subarray == 'SUBSTRIP256':
        pass
    elif subarray == 'SUBSTRIP96':
        wavelength_map = wavelength_map[10*int(oversample):106*int(oversample)]
    else:
        raise ValueError('Unknown subarray: {}'.format(subarray))

    return wavelength_map
