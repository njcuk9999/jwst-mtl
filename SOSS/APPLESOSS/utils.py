#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Mar 11 14:35 2020

@author: MCR

Miscellaneous utility functions for APPleSOSS.
"""

from astropy.io import fits
import numpy as np
from tqdm import tqdm
import warnings
import webbpsf

from SOSS.APPLESOSS import plotting


def generate_psfs(wave_increment=0.1, npix=400, verbose=0):
    """Generate 1D SOSS PSFs across the full 0.5 - 2.9µm range of all orders.

    Parameters
    ----------
    wave_increment : float
        Wavelength step (in µm) for PSF simulation.
    npix : int
        Size (in native pixels) of the 1D PSFs.
    verbose : int
        Level of verbosity.
    Returns
    -------
    psfs : array-like
        Array of 1D PSFs at specified wavelength increments.
    """

    # Calculate the number of PSFs to generate based on the SOSS wavelength
    # range and the chosen increment.
    nsteps = int((2.9 - 0.5) / wave_increment)
    # Estimate time to completion assuming ~5s per PSF.
    time_frame = int((nsteps * 5) / 60)
    if verbose != 0:
        print('  Generating {0} PSFs... Expected to take about {1} min(s).'.format(nsteps, time_frame))
    wavelengths = (np.linspace(0.5, 2.9, nsteps) * 1e-6)[::-1]

    # Set up WebbPSF simulation for NIRISS.
    niriss = webbpsf.NIRISS()
    # Override the default minimum wavelength of 0.6 microns.
    niriss.SHORT_WAVELENGTH_MIN = 0.5e-6
    # Set correct filter and pupil wheel components.
    niriss.filter = 'CLEAR'
    niriss.pupil_mask = 'GR700XD'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cube = niriss.calc_datacube(wavelengths=wavelengths, fov_pixels=npix,
                                    oversample=1)
    # Collapse into 1D PSF
    psfs_1d = np.nansum(cube[0].data, axis=1)

    # Turn into record array and attach wavelength info
    psfs = np.recarray((nsteps, npix),
                       dtype=[('Wave', float), ('PSF', float)])
    psfs['Wave'] = wavelengths[:, None]*1e6  # Convert to µm
    psfs['PSF'] = psfs_1d

    return psfs


def get_wave_solution(order):
    """Extract wavelength calibration information from the wavelength solution
    reference file.

    Parameters
    ----------
    order : int
        Diffraction order.

    Returns
    -------
    wavecal_x : np.array
        X pixel coordinate.
    wavecal_w : np.array
        Wavelength value.
    """

    # Get wavelength calibration reference file.
    wave_soln = 'Ref_files/jwst_niriss_wavemap_0013.fits'
    wavemap = fits.getdata(wave_soln, order)
    header = fits.getheader(wave_soln, order)
    ovs = header['OVERSAMP']
    pad = header['PADDING']

    # Bin the map down to native resolution and remove padding.
    nrows, ncols = wavemap.shape
    trans_map = wavemap.reshape((nrows // ovs), ovs, (ncols // ovs), ovs)
    trans_map = trans_map.mean(1).mean(-1)
    trans_map = trans_map[pad:-pad, pad:-pad]
    dimy, dimx = np.shape(trans_map)
    # Collapse over the spatial dimension.
    wavecal_w = np.nanmean(trans_map, axis=0)
    wavecal_x = np.arange(dimx)

    return wavecal_x, wavecal_w


def interpolate_profile(w, w_cen, wavelengths, psfs, psfs_cen, os_factor=10):
    """For efficiency, 1D SOSS PSFs were generated through WebbPSF at
    discrete intervals. This function performs the linear interpolation to
    construct profiles at a specified wavelength.

    Parameters
    ----------
    w : float
        Wavelength at which to return a PSF (in µm).
    w_cen : float
        Centroid position of the profile at w.
    wavelengths : array_like
        Wavelengths (in µm) corresponding to the PSF array.
    psfs : array-like
        WebbPSF simulated 1D PSFs.
    psfs_cen : array_like
        Array of centroid positions for the profiles in psfs.
    os_factor : int
        Oversampling factor for recentroiding.

    Returns
    -------
    profile : np.array
        1D SOSS PSF at wavelength w.
    """

    # Get the simulated PSF anchors for the interpolation.
    low = np.where(wavelengths < w)[0][0]
    up = np.where(wavelengths > w)[0][-1]
    anch_low = wavelengths[low]
    anch_up = wavelengths[up]

    # Shift the anchor profiles to the centroid position of the wavelength of
    # interest.
    len_psf = np.shape(psfs)[1]
    # Oversample
    psf_up = np.interp(np.linspace(0, (os_factor*len_psf - 1)/os_factor,
                                   (os_factor*len_psf - 1) + 1),
                       np.arange(len_psf), psfs[up])
    psf_low = np.interp(np.linspace(0, (os_factor*len_psf - 1)/os_factor,
                                   (os_factor*len_psf - 1) + 1),
                        np.arange(len_psf), psfs[low])
    # Shift the profiles to the correct cenntroid
    psf_up = np.interp(np.arange(len_psf*os_factor),
                       np.arange(len_psf*os_factor) - psfs_cen[up]*os_factor + w_cen*os_factor,
                       psf_up)
    psf_low = np.interp(np.arange(len_psf*os_factor),
                        np.arange(len_psf*os_factor) - psfs_cen[low]*os_factor + w_cen*os_factor,
                        psf_low)
    # Resample to the native pixel sampling.
    psf_up = np.interp(np.arange(len_psf), np.linspace(0, (os_factor*len_psf-1)/os_factor,
                                                       (os_factor*len_psf-1)+1),
                       psf_up)
    psf_low = np.interp(np.arange(len_psf), np.linspace(0, (os_factor*len_psf-1)/os_factor,
                                                        (os_factor*len_psf-1)+1),
                        psf_low)

    # Assume that the PSF varies linearly over the interval.
    # Calculate the weighting coefficients for each anchor.
    diff = np.abs(anch_up - anch_low)
    weight_low = 1 - (w - anch_low) / diff
    weight_up = 1 - (anch_up - w) / diff

    # Linearly interpolate the anchor profiles to the wavelength of interest.
    profile = np.average(np.array([psf_low, psf_up]),
                         weights=np.array([weight_low, weight_up]), axis=0)

    return profile


def replace_badpix(clear, thresh=5, box_size=10, verbose=0):
    """Replace bad pixels with the median of the pixels values of a box
    centered on the bad pixel.

    Parameters
    ----------
    clear : array-like
        Data frame.
    thresh : int
        Threshold in standard deviations to flag a bad pixel.
    box_size : int
        Box side half-length to use.
    verbose : int
        level of verbosity.

    Returns
    -------
    clear_r : np.array
        Data frame with bad pixels interpolated.
    """

    # Initial setup of arrays and variables
    disable = verbose_to_bool(verbose)
    clear_r = clear.copy()
    counts = 0
    dimy, dimx = np.shape(clear)

    def get_interp_box(data, box_size, i, j, dimx, dimy):
        """ Get median and standard deviation of a box centered on a specified
        pixel.
        """

        # Get the box limits.
        low_x = np.max([i - box_size, 0])
        up_x = np.min([i + box_size, dimx - 1])
        low_y = np.max([j - box_size, 0])
        up_y = np.min([j + box_size, dimy - 1])

        # Calculate median and std deviation of box.
        median = np.nanmedian(data[low_y:up_y, low_x:up_x])
        stddev = np.nanstd(data[low_y:up_y, low_x:up_x])

        # Pack into array.
        box_properties = np.array([median, stddev])

        return box_properties

    # Loop over all pixels and interpolate those that deviate by more than the
    # threshold from the surrounding median.
    for i in tqdm(range(dimx), disable=disable):
        for j in range(dimy):
            box_size_i = box_size
            box_prop = get_interp_box(clear_r, box_size_i, i, j, dimx, dimy)

            # Ensure that the median and std dev we extract are good.
            # If not, increase the box size until they are.
            while np.any(np.isnan(box_prop)) or np.any(box_prop < 0):
                box_size_i += 1
                box_prop = get_interp_box(clear_r, box_size_i, i, j, dimx,
                                          dimy)
            med, std = box_prop[0], box_prop[1]

            # If central pixel is too deviant (or nan, or negative) replace it.
            if np.abs(clear_r[j, i] - med) > thresh*std or np.isnan(clear_r[j, i]) or clear_r[j, i] < 0:
                clear_r[j, i] = med
                counts += 1

    # Print statistics and show debugging plot if necessary.
    if verbose != 0:
        print('   {:.2f}% of pixels interpolated.'.format(counts/dimx/dimy*100))
        if verbose == 3:
            plotting.plot_badpix(clear, clear_r)

    return clear_r


def validate_inputs(etrace):
    """Validate the input parameters for the empirical trace construction
    module, and determine the correct subarray for the data.

    Parameters
    ----------
    etrace : EmpiricalTrace instance
        Instance of an EmpiricalTrace object.

    Returns
    -------
    subarray : str
        The correct NIRISS/SOSS subarray identifier corresponding to the CLEAR
        dataframe.
    """

    # Ensure padding and oversampling are integers.
    if type(etrace.pad) != int:
        raise ValueError('Padding must be an integer.')
    if type(etrace.oversample) != int:
        raise ValueError('Oversampling factor must be an integer.')

    # Determine correct subarray dimensions.
    dimy, dimx = np.shape(etrace.clear)
    if dimy == 96:
        subarray = 'SUBSTRIP96'
    elif dimy == 256:
        subarray = 'SUBSTRIP256'
    elif dimy == 2048:
        subarray = 'FULL'
    else:
        raise ValueError('Unrecognized subarray: {}x{}.'.format(dimy, dimx))

    return subarray


def verbose_to_bool(verbose):
    """Convert integer verbose to bool to disable or enable progress bars.
    """

    if verbose in [2, 3]:
        verbose_bool = False
    else:
        verbose_bool = True

    return verbose_bool
