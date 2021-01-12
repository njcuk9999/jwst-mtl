"""
Created on Mon Feb 17 13:31:58 2020

@author: talens-irex
"""

import numpy as np
from numpy.polynomial import Legendre

from astropy.io import ascii

# SUBSTRIP256 keeps solumns 0:255 (0 based) in the nat frame.
# SUBSTRIP96 keeps columns 150:245 (0 based) in the nat frame.


def apply_rotation(coords, origin=np.array([1514., 456.]), angle=1.489):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.

    :param coords: x, y coordinates to rotate.
    :param origin: point around which to rotate the coordinates.
    :param angle: angle in degrees.

    :type coords: Tuple(array[float], array[float])
    :type origin: array[float]
    :type angle: float

    :returns: x_rot, y_rot - rotated coordinates.
    :rtype: Tuple(array[float], array[float])

    """

    x, y = coords
    origin_x, origin_y = origin
    angle = np.deg2rad(angle)

    dx, dy = x - origin_x, y - origin_y
    x_rot = np.cos(angle)*dx - np.sin(angle)*dy + origin_x
    y_rot = np.sin(angle)*dx + np.cos(angle)*dy + origin_y

    return x_rot, y_rot


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


def get_tracepars(filename=None, origin=np.array([1514, 456]), angle=1.489):
    """Read a file containing the trace profile and generate
    polynomial parameters for each order.

    :param filename: file containing modelled trace points.
    :param origin: point around which to rotate the coordinates.
    :param angle: angle in degrees.

    :type filename: str
    :type origin: array[float]
    :type angle: float

    :returns: tracepars - a dictionary containg the parameters for the polynomial fits.
    :rtype: dict
    """
    
    if filename is None:
        filename = 'NIRISS_GR700_trace_extended.csv'  # TODO Switch to pkg_resources in the future.
    
    # Read the trace.
    trace = ascii.read(filename)  # Read the Code V trace model from file. DS9 coordinates are used.

    # Convert to pixel coordinates.
    trace['xpos'] /= 0.018  # Convert from micron to pixels.
    trace['ypos'] /= 0.018  # Convert from micron to pixels.
    trace['xpos'] -= 0.5  # Set the origin at the center of the lower-left pixel.
    trace['ypos'] -= 0.5  # Set the origin at the center of the lower-left pixel.

    # Apply rotation around point.
    trace['xpos'], trace['ypos'] = apply_rotation((trace['xpos'], trace['ypos']), origin=origin, angle=angle)

    # Compute polynomial parameters for different orders.
    tracepars = dict()
    for m in np.unique(trace['order']):
        pars = trace_polynomial(trace, m=m)
        tracepars[m] = pars
        
    return tracepars


def get_tiltpars(filename=None):
    """Read a file containing the tilt angle for orders 1, 2 and 3.

    :param filename: The file containing the tilt data.

    :type filename: str

    :returns: wavegrid, tilt_o1, tilt_o2, tilt_o3 - A grid of wavelengths and corresponding tilt angles for each order.
    :rtype: Tuple(array[float], array[float], array[float], array[float])
    """

    if filename is None:
        filename = 'SOSS_wavelength_dependent_tilt.ecsv'  # TODO Switch to pkg_resources in the future.

    tab = ascii.read(filename)
    wavegrid = tab['Wavelength']
    tilt_o1 = tab['order 1']
    tilt_o2 = tab['order 2']
    tilt_o3 = tab['order 3']

    # TODO make polynomial fit like tp.get_tracepars?

    return wavegrid, tilt_o1, tilt_o2, tilt_o3


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
    elif (frame == 'nat') & (subarray == 'FULL'):
        spatpix = spatpix_ref
    elif (frame == 'dms') & (subarray == 'FULL'):
        spatpix = 2047*oversample - spatpix_ref
    elif (frame == 'sim') & (subarray == 'FULL'):
        spatpix = 1791*oversample + spatpix_ref
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
    elif (frame == 'nat') & (subarray == 'FULL'):
        spatpix_ref = spatpix
    elif (frame == 'dms') & (subarray == 'FULL'):
        spatpix_ref = 2047*oversample - spatpix
    elif (frame == 'sim') & (subarray == 'FULL'):
        spatpix_ref = spatpix - 1791*oversample
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


def image_nat_to_dms(image):
    """This function converts images from native (ds9) to DMS coordinates.

    :param image: The input image data in native coordinates.

    :type image: array[float]

    :returns: out - the image in DMS coordinates.
    :rtype: array[float]
    """

    ndim = image.ndim

    if ndim == 2:
        out = np.flip(np.rot90(image, axes=(0, 1)), axis=-1)
    elif ndim == 3:
        out = np.flip(np.rot90(image, axes=(1, 2)), axis=-1)
    elif ndim == 4:
        out = np.flip(np.rot90(image, axes=(2, 3)), axis=-1)
    else:
        raise ValueError('Image with {} dimensions are not supported.'.format(ndim))

    return out


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


def wavelength_to_tilt(wavelength, tiltpars, m=1):
    """Interpolate the tilt values to the input wavelength grid.

    :param wavelength: wavelength values in microns.
    :param tiltpars: the tilt parameters returned by get_tiltpars.
    :param m: the spectral order.

    :type wavelength: array[float]
    :type tiltpars: Tuple(array[float], array[float], array[float], array[float])
    :type m: int

    :returns: tilt - The tilt angles corresponding to wavelength for order m.
    :rtype: array[float]
    """

    wavegrid, tilt_o1, tilt_o2, tilt_o3 = tiltpars

    if m == 1:
        tilt = np.interp(wavelength, wavegrid, tilt_o1)
    elif m == 2:
        tilt = np.interp(wavelength, wavegrid, tilt_o2)
    elif m == 3:
        tilt = np.interp(wavelength, wavegrid, tilt_o3)
    else:
        raise ValueError('Order m must be 1, 2, or 3.')

    return tilt


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


def _get_wave_map(wavegrid, x_trace, y_trace, tilt, subarray='SUBSTRIP256', frame='dms', oversample=1, padding=0,
                  maxiter=5, dtol=1e-2):
    """"""

    os = np.copy(oversample)
    xpad = np.copy(padding)
    ypad = np.copy(padding)

    dimx, dimy = 300, 2048

    # Generate the oversampled grid of pixel coordinates.
    x_vec = np.arange((dimx + 2 * xpad) * os) / os - (os - 1) / (2 * os) - xpad
    y_vec = np.arange((dimy + 2 * ypad) * os) / os - (os - 1) / (2 * os) - ypad
    x_pixel, y_pixel = np.meshgrid(x_vec, y_vec)

    # The gain is for the iterative approach to finding the wavelength.
    gain = -1.0
    delta_y = 0.0

    # Iteratively compute the wavelength at each pixel.
    for niter in range(maxiter):

        # Assume all x have same wavelength.
        wave_iterated = np.interp(y_pixel + gain*delta_y, y_trace, wavegrid)

        # Compute the tilt angle at the wavelengths.
        tilt_tmp = np.interp(wave_iterated, wavegrid, tilt)

        # Compute the trace position at the wavelengths.
        x_estimate = np.interp(wave_iterated, wavegrid, x_trace)
        y_estimate = np.interp(wave_iterated, wavegrid, y_trace)

        # Project that back to pixel coordinates.
        y_iterated = y_estimate + (x_pixel - x_estimate) * np.tan(np.deg2rad(tilt_tmp))

        # Measure error between requested and iterated position.
        delta_y = delta_y + (y_iterated - y_pixel)

        # If the desired precision has been reached end iterations.
        if np.all(np.abs(y_iterated - y_pixel) < dtol):
            break

    # Evaluate the final wavelength map, this time setting out-of-bounds values to NaN.
    wave_map_2d = np.interp(y_pixel + gain*delta_y, y_trace, wavegrid, left=np.nan, right=np.nan)

    # Crop or expand to the appropriate size for the subarray.
    if subarray == 'FULL':
        # We pad the FULL subarray with NaNs now.
        tmp = np.full((os * (2048 + 2 * ypad), os * (2048 + 2 * xpad)), fill_value=np.nan)
        lx = 0
        ux = os * (dimx + 2 * xpad)
        tmp[:, lx:ux] = wave_map_2d
        wave_map_2d = tmp
    elif subarray == 'SUBSTRIP96':
        # The SUBSTRIP96 keeps columns 150 to 245.
        lx = os * (150)
        ux = os * (246 + 2 * xpad)
        wave_map_2d = wave_map_2d[:, lx:ux]
    elif subarray == 'SUBSTRIP256':
        # The SUBSTRIP256 keeps columns 0 to 255.
        lx = 0
        ux = os * (256 + 2 * xpad)
        wave_map_2d = wave_map_2d[:, lx:ux]
    else:
        raise ValueError('subarray must be one of SUBSTRIP256, SUBSTRIP96 or FULL')

    # Transform to the correct coordinate frame.
    if frame == 'nat':
        pass
    elif frame == 'dms':
        wave_map_2d = image_nat_to_dms(wave_map_2d)
    elif frame == 'sim':
        wave_map_2d = image_nat_to_dms(wave_map_2d)
        wave_map_2d = np.flip(wave_map_2d, axis=1)
    else:
        raise ValueError('Unknown coordinate frame: {}'.format(frame))

    return wave_map_2d


def make_2d_wavemap(m=1, subarray='SUBSTRIP256', frame='dms', tilt_angle=None, oversample=1, padding=0, maxiter=5,
                    dtol=1e-2):
    """Compute the 2D wavelength map for NIRISS SOSS.

    :param m: the spectral order.
    :param subarray: the subarray of the output coordinates (SUBARRAY256 or SUBARRAY96).
    :param frame: the coordinate frame of the input coordinates (nat, dms or sim).
    :param tilt_angle: a constant tilt angle to use.
    :param oversample: the oversampling factor of the output array.
    :param padding: the padding of the output array in pixels.
    :param maxiter: the maximum number of iterations to use when solving the wavelength map.
    :param dtol: the tolerance in pixels at which to end the iterations.

    :type m: int
    :type subarray: str
    :type frame: str
    :type tilt_angle: float
    :type oversample: int
    :type padding: int
    :type maxiter: int
    :type dtol: float

    :returns: wave_map_2d - A 2D array of wavelength values across the detector.
    :rtype: array[float]
    """

    # Get the trace parameters.
    tracepars = get_tracepars()

    # Get the tilt parameters.
    tiltpars = get_tiltpars()

    # Compute the trace x, y positions and tilt angles for order m.
    wavegrid = np.linspace(0.5, 5.0, 4501)
    y_trace, x_trace, mask = wavelength_to_pix(wavegrid, tracepars, m=m, frame='nat')

    if tilt_angle is None:
        tilt = wavelength_to_tilt(wavegrid, tiltpars, m=m)
    else:
        tilt = np.full_like(wavegrid, fill_value=tilt_angle)

    # Compute the 2D wavelength map in native (ds9) coordinates.
    wave_map_2d = _get_wave_map(wavegrid, x_trace, y_trace, tilt, subarray=subarray, frame=frame, oversample=oversample,
                                padding=padding, maxiter=maxiter, dtol=dtol)

    return wave_map_2d
