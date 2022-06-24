#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:46 2021

@author: MCR

Definitions of the main functions for the APPLESOSS (A Producer of ProfiLEs for
SOSS) module. This class will be initialized and called by the user to create
models of the spatial profiles for the first, second, and third order SOSS
traces, for use as the specprofile reference file required by the ATOCA
algorithm.
"""


from astropy.io import fits
import numpy as np
from scipy.interpolate import interp2d
import warnings

from SOSS.dms.soss_centroids import get_soss_centroids
from SOSS.APPLESOSS import plotting
from SOSS.APPLESOSS import utils
from SOSS.dms import soss_ref_files

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


class EmpiricalProfile:
    """Class wrapper around the empirical spatial profile construction module.

    Attributes
    ----------
    clear : np.array
        SOSS CLEAR exposure data frame.
    subarray : str
        NIRISS SOSS subarray identifier. One of 'SUBSTRIP256', or 'FULL'.
    pad : int
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions.
    oversample : int
        Oversampling factor. Oversampling will be equal in the spectral and
        spatial directions.
    verbose : int
        Level of verbosity: either 3, 2, 1, or 0.
         3 - show all of progress prints, progress bars, and diagnostic plots.
         2 - show progress prints and bars.
         1 - show only progress prints.
         0 - show nothing.
    order1 : np.array
        First order spatial profile.
    order2 : np.array
        Second order spatial profile.
    order3 : np.array
        Third order spatial profile.

    Methods
    -------
    build_empirical_profile
        Construct the empirical spatial profiles.
    write_specprofile_reference
        Save spatial profile models to reference file.
    """

    def __init__(self, clear, pad=0, oversample=1, verbose=0):
        """Initializer for EmpiricalProfile.
        """

        # Initialize input attributes.
        self.clear = clear
        self.pad = pad
        self.oversample = oversample
        self.verbose = verbose

        # Validate the parameters and determine the correct subarray.
        self.subarray = self.validate_inputs()
        self.order1 = None
        self.order2 = None
        self.order3 = None

    def build_empirical_profile(self):
        """Run the empirical spatial profile construction module.
        """

        # Run the empirical spatial profile construction.
        o1, o2, o3 = build_empirical_profile(self.clear, self.subarray,
                                             self.pad, self.oversample,
                                             self.verbose)
        # Set any niggling negatives to zero (mostly for the bluest end of the
        # second order where things get skrewy).
        for o in [o1, o2, o3]:
            ii = np.where(o < 0)
            o[ii] = 0
        # Store the spatial profiles as attributes.
        self.order1, self.order2, self.order3 = o1, o2, o3

    def write_specprofile_reference(self, subarray, filename=None):
        """Write the spatial profiles to a reference file to be injested by
        ATOCA.

        Parameters
        ----------
        subarray : str
            SOSS subarray, either FULL, SUBSTRIP256 or SUBSTRIP96
        filename : str
            Name of reference file.
        """

        # Just make sure that everything is the same shape
        assert self.order1.shape == self.order2.shape == self.order3.shape
        dimy, dimx = self.order1.shape
        # Create stacked array with all orders.
        stack_full = np.zeros(((2048+2*self.pad)*self.oversample,
                               (2048+2*self.pad)*self.oversample, 3))
        stack_full[-dimy:, :, 0] = np.copy(self.order1)
        stack_full[-dimy:, :, 1] = np.copy(self.order2)
        stack_full[-dimy:, :, 2] = np.copy(self.order3)
        # Pass to reference file creation.
        hdulist = soss_ref_files.init_spec_profile(stack_full, self.oversample,
                                                   self.pad, subarray,
                                                   filename)
        hdu = fits.HDUList(hdulist)
        if filename is None:
            filepattern = 'APPLESOSS_ref_2D_profile_{0}_os{1}_pad{2}.fits'
            filename = filepattern.format(subarray, self.oversample, self.pad)
        print('Saving to file '+filename)
        hdu.writeto(filename, overwrite=True)

    def validate_inputs(self):
        """Validate the input parameters.
        """
        return utils.validate_inputs(self)


def build_empirical_profile(clear, subarray, pad, oversample, verbose):
    """Main procedural function for the empirical spatial profile construction
    module. Calling this function will initialize and run all the required
    subroutines to produce a spatial profile for the first, second and third
    orders.= The spatial profiles generated can include oversampling as well
    as padding in both the spatial and spectral directions.

    Parameters
    ----------
    clear : np.array
        SOSS CLEAR exposure data frame.
    subarray : str
        NIRISS SOSS subarray identifier. One of 'SUBSTRIP256', or 'FULL'.
    pad : int
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions.
    oversample : int
        Oversampling factor. Oversampling will be equal in the spectral and
        spatial directions.
    verbose : int
        Level of verbosity: either 3, 2, 1, or 0.
         3 - show all of progress prints, progress bars, and diagnostic plots.
         2 - show progress prints and bars.
         1 - show only progress prints.
         0 - show nothing.

    Returns
    -------
    o1_uncontam : np.array
        Uncontaminated spatial profile for the first order.
    o2_uncontam : np.array
        Uncontaminated spatial profile for the second order.
    o3_uncontam : np.array
        Uncontaminated spatial profile for the third order.

    Raises
    ------
    ValueError
        When the clear dimensions do not match a known subarray.
    """

    if verbose != 0:
        print('Starting the APPLESOSS module.\n')

    # ========= INITIAL SETUP =========
    # Determine correct subarray dimensions.
    dimy, dimx = np.shape(clear)
    # If subarray is FULL - trim down to SUBSTRIP256 and work with that.
    if subarray == 'FULL':
        clear = clear[-256:, :]
        # Reset all variable to appropriate SUBSTRIP256 values.
        subarray = 'SUBSTRIP256'
        dimy, dimx = np.shape(clear)

    # Add a floor level such that all pixel values are positive and interpolate
    # bad pixels
    if verbose != 0:
        print(' Initial processing...')
        print('  Interpolating bad pixels...', flush=True)
    floor = np.nanpercentile(clear, 0.1)
    clear -= floor
    #clear = utils.replace_badpix(clear, verbose=verbose)

    # Get the centroid positions for both orders from the data using the
    # edgetrig method.
    if verbose != 0:
        print('  Getting trace centroids...')
    centroids = get_soss_centroids(clear, subarray=subarray)
    if verbose == 3:
        plotting.plot_centroid(clear, centroids)
    clear += floor

    # The four columns of pixels on the left and right edge of the SOSS
    # detector are reference pixels. Trim them off and replace them with
    # interpolations of the edge-most profiles.
    clear = pad_spectral_axis(clear[:, 5:-5],
                              centroids['order 1']['X centroid'][5:-5],
                              centroids['order 1']['Y centroid'][5:-5],
                              pad=5, ref_cols=[0, -1], replace=True)

    # ========= CONSTRUCT SPATIAL PROFILE MODELS =========
    # Build a first estimate of the first and second order spatial profiles.
    # Construct the first order profile.
    if verbose != 0:
        print('  Starting the first order model...', flush=True)
    # Lazy method: no interpolation, no F277W. Just fit the wing of the first
    # order for each wavelength. Relies on some fine tuning and the second
    # order to be low level when it physically overlaps the first order.
    psfs = utils.generate_psfs(wave_increment=0.1, verbose=verbose)
    o1_uncontam, o1_native = construct_order23(clear, centroids, 1, psfs)

    # Construct the second order profile.
    if verbose != 0:
        print('  Starting the second order trace...')
    o2_out = construct_order23(clear - o1_native[pad:dimy + pad, :],
                               centroids, 2, psfs, verbose=verbose, o1_prof=o1_native[pad:dimy + pad, :])
    o2_uncontam, o2_native = o2_out[0], o2_out[1]
    # Add padding to the second order if necessary
    if pad != 0:
        o2_uncontam = pad_order23(o2_uncontam, centroids, pad, order='2')

    # Construct the third order profile.
    if verbose != 0:
        print('  Starting the third order trace...')
    o3_out = construct_order23(clear - o1_native[pad:dimy + pad, :] - o2_native,
                               centroids, 3, psfs, pivot=700,
                               verbose=verbose, o2_prof=o2_uncontam)
    o3_uncontam = o3_out[0]
    # Add padding to the third order if necessary
    if pad != 0:
        o3_uncontam = pad_order23(o3_uncontam, centroids, pad, order='3')

    # ========= FINAL TUNING =========
    # Pad the spectral axis.
    if pad != 0:
        if verbose != 0:
            print(' Adding padding to the spectral axis...')
        o1_uncontam = pad_spectral_axis(o1_uncontam,
                                        centroids['order 1']['X centroid'],
                                        centroids['order 1']['Y centroid'],
                                        pad=pad)
        o2_uncontam = pad_spectral_axis(o2_uncontam,
                                        centroids['order 2']['X centroid'],
                                        centroids['order 2']['Y centroid'],
                                        pad=pad)
        o3_uncontam = pad_spectral_axis(o3_uncontam,
                                        centroids['order 3']['X centroid'],
                                        centroids['order 3']['Y centroid'],
                                        pad=pad)

    # Column normalize. Only want the original detector to sum to 1, not the
    # additional padding + oversampling.
    o1_uncontam /= np.nansum(o1_uncontam, axis=0)
    o2_uncontam /= np.nansum(o2_uncontam, axis=0)
    o3_uncontam /= np.nansum(o3_uncontam, axis=0)
    # Replace NaNs resulting from all zero columns with zeros
    for o in [o2_uncontam, o3_uncontam]:
        ii = np.where(~np.isfinite(o))
        o[ii] = 0

    # Add oversampling.
    if oversample != 1:
        if verbose != 0:
            print(' Oversampling...')
        o1_uncontam = oversample_frame(o1_uncontam, os=oversample)
        o2_uncontam = oversample_frame(o2_uncontam, os=oversample)
        o3_uncontam = oversample_frame(o3_uncontam, os=oversample)

    if verbose != 0:
        print('\nDone.')

    return o1_uncontam, o2_uncontam, o3_uncontam


def construct_order23(residual, cen, order, psfs, pivot=750, halfwidth=12,
                      o1_prof=None, o2_prof=None, verbose=0):
    """Reconstruct the wings of the second or third orders after the first
    order spatial profile has been modeled and subtracted off.

    Parameters
    ----------
    residual : np.array
        NIRISS/SOSS data frame residual map - either with o1, or both o1 and
        o2 removed.
    cen : dict
        Centroids dictionary.
    order : str
        The order to reconstruct.
    pivot : int
        For order 2, minimum spectral pixel value for which a wing
        reconstruction will be attempted. For order 3, the maximum pixel value.
        For spectral pixels < or >pivot respectively, the profile at pivot will
        be used.
    halfwidth : int
        Half width in pixels of the spatial profile core.
    verbose : int
        level of verbosity.

    Returns
    -------
    new_frame : np.array
        Model of the second order spatial profile with wings reconstructed.
    new_frame_native : np.array
        New frame, but at the native counts level.
    """

    # Initalize new data frame and get subarray dimensions.
    dimy, dimx = np.shape(residual)
    new_frame = np.zeros_like(residual)
    new_frame_native = np.zeros_like(residual)

    # Get wavelength calibration.
    wavecal_x, wavecal_w = utils.get_wave_solution(order=order)

    first_time = True
    if order == 3:
        maxi = pivot
    else:
        maxi = dimx
    # Hard stop for profile reuse - will handle these via padding.
    if order in [2, 3]:
        stop = np.where(cen['order ' + str(order)]['Y centroid'] >= dimy)[0][0] + halfwidth
    for i in range(dimx):
        wave = wavecal_w[i]
        # Skip over columns where the throughput is too low to get a good core
        # and/or the order is buried within another.
        if order == 2 and i < pivot:
            continue
        if order == 3 and i > pivot:
            continue
        # If the centroid is too close to the detector edge, make note of
        # the column and deal with it later
        cen_o = int(round(cen['order ' + str(order)]['Y centroid'][i], 0))
        if cen_o + halfwidth > dimy:
            if i < maxi:
                maxi = i
            continue

        # Get a copy of the spatial profile, and normalize it by its max value.
        working_prof = np.copy(residual[:, i])
        max_val = np.nanpercentile(working_prof, 99.5)
        working_prof /= max_val

        # Simulate the wings.
        if first_time is False:
            verbose = 0
        wing, wing2 = simulate_wings(wave, psfs, halfwidth=halfwidth,
                                     verbose=verbose)
        first_time = False
        # Concatenate the wings onto the profile core.
        end = int(round((cen_o + 1 * halfwidth), 0))
        start = int(round((cen_o - 1 * halfwidth), 0))
        stitch = np.concatenate([wing2, working_prof[start:end], wing])
        # Rescale to native flux level.
        stitch_native = stitch * max_val
        # Shift the profile back to its correct centroid position
        stitch = np.interp(np.arange(dimy), np.arange(400) - 400//2 + cen_o,
                           stitch)
        stitch_native = np.interp(np.arange(dimy),
                                  np.arange(400) - 400//2 + cen_o,
                                  stitch_native)
        new_frame[:, i] = stitch
        new_frame_native[:, i] = stitch_native

    # For columns where the order 2 core is not distinguishable (due to the
    # throughput dropping near 0, or it being buried in order 1) reuse the
    # reddest reconstructed profile.
    if order == 2:
        wavecal_x_o1, wavecal_w_o1 = utils.get_wave_solution(order=1)
        for i in range(pivot):
            wave_o2 = wavecal_w[i]
            up = np.where(wavecal_w_o1 > wave_o2)[0][-1]
            low = np.where(wavecal_w_o1 < wave_o2)[0][0]
            anch_low = wavecal_w_o1[low]
            anch_up = wavecal_w_o1[up]

            # Assume that the PSF varies linearly over the interval.
            # Calculate the weighting coefficients for each anchor.
            diff = np.abs(anch_up - anch_low)
            weight_low = 1 - (wave_o2 - anch_low) / diff
            weight_up = 1 - (anch_up - wave_o2) / diff

            profile = np.average(np.array([o1_prof[:, low], o1_prof[:, up]]),
                                 weights=np.array([weight_low, weight_up]),
                                 axis=0)

            co2 = cen['order 2']['Y centroid'][i]
            co1_l = cen['order 1']['Y centroid'][low]
            co1_u = cen['order 1']['Y centroid'][up]
            co1 = np.mean([co1_l, co1_u])
            working_prof = np.interp(np.arange(dimy), np.arange(dimy) - co1 + co2,
                                     profile)

            new_frame[:, i] = working_prof

    # O3
    if order == 3:
        wavecal_x_o2, wavecal_w_o2 = utils.get_wave_solution(order=2)
        for i in range(maxi, stop):
            wave_o3 = wavecal_w[i]
            up = np.where(wavecal_w_o2 > wave_o3)[0][-1]
            low = np.where(wavecal_w_o2 < wave_o3)[0][0]
            anch_low = wavecal_w_o2[low]
            anch_up = wavecal_w_o2[up]

            # Assume that the PSF varies linearly over the interval.
            # Calculate the weighting coefficients for each anchor.
            diff = np.abs(anch_up - anch_low)
            weight_low = 1 - (wave_o3 - anch_low) / diff
            weight_up = 1 - (anch_up - wave_o3) / diff

            profile = np.average(
                np.array([o2_prof[:, low], o2_prof[:, up]]),
                weights=np.array([weight_low, weight_up]),
                axis=0)

            co3 = cen['order 3']['Y centroid'][i]
            co2_l = cen['order 2']['Y centroid'][low]
            co2_u = cen['order 2']['Y centroid'][up]
            co2 = np.mean([co2_l, co2_u])
            working_prof = np.interp(np.arange(dimy),
                                     np.arange(dimy) - co2 + co3,
                                     profile)

            new_frame[:, i] = working_prof

    # For columns where the centroid is off the detector, reuse the bluest
    # reconstructed profile.
    if order in [2]:
        for i in range(maxi, stop):
            anchor_prof = new_frame[:, maxi - 1]
            anchor_prof_native = new_frame_native[:, maxi - 1]
            sc = cen['order '+str(order)]['Y centroid'][maxi - 1]
            ec = cen['order '+str(order)]['Y centroid'][i]
            working_prof = np.interp(np.arange(dimy), np.arange(dimy) - sc + ec,
                                     anchor_prof)
            new_frame[:, i] = working_prof
            working_prof_native = np.interp(np.arange(dimy),
                                            np.arange(dimy) - sc + ec,
                                            anchor_prof_native)
            new_frame_native[:, i] = working_prof_native

            # # Handle all rows after hard cut.
            # new_frame = pad_order23(new_frame[halfwidth:-halfwidth], cen,
            #                         pad=halfwidth, order=order)

    return new_frame, new_frame_native


def oversample_frame(dataframe, os=1):
    """Oversample a dataframe by a specified amount.

    Parameters
    ----------
    dataframe : np.array
        Dataframe to be oversampled.
    os : int
        Oversampling factor to apply to each axis.

    Returns
    -------
    data_os : np.array
        Input dataframe with each axis oversampled by the desired amount.
    """

    # Generate native and oversampled axes.
    dimy, dimx = np.shape(dataframe)
    x, x_os = np.arange(dimx), np.arange(dimx * os) / os
    y, y_os = np.arange(dimy), np.arange(dimy * os) / os

    # Interpolate onto the oversampled grid.
    pp = interp2d(x, y, dataframe)
    data_os = pp(x_os, y_os)

    return data_os


def pad_order23(dataframe, cen, pad, order):
    """Add padding to the spatial axis of an order 2 or 3 dataframe. Since
    these orders curve almost vertically at short wavelengths, we must take
    special care to properly extend the spatial profile

    Parameters
    ----------
    dataframe : np.array
        A dataframe of order 2 or 3.
    cen : dict
        Centroids dictionary.
    pad : int
        Amount of padding to add to the spatial axis.
    order : str
        Order to pad.

    Returns
    -------
    frame_padded : np.array
        The dataframe with the appropriate amount of padding added to
        the spatial axis.
    """

    # Initalize padded array.
    dimy, dimx = np.shape(dataframe)
    frame_padded = np.zeros((dimy + pad, dimx))
    frame_padded[:-pad] = dataframe

    # Use the shortest wavelength slice along the spatial axis as a reference
    # profile.
    anchor_prof = dataframe[-2, :]
    xcen_anchor = np.where(cen['order '+str(order)]['Y centroid'] >= dimy - 2)[0][0]

    # To pad the upper edge of the spatial axis, shift the reference profile
    # according to extrapolated centroids.
    for yval in range(dimy-2, dimy-1+pad):
        xval = np.where(cen['order '+str(order)]['Y centroid'] >= yval)[0][0]
        shift = xcen_anchor - xval
        working_prof = np.interp(np.arange(dimx), np.arange(dimx) - shift,
                                 anchor_prof)
        frame_padded[yval] = working_prof

    # Pad the lower edge with zeros.
    frame_padded = np.pad(frame_padded, ((pad, 0), (0, 0)), mode='edge')

    return frame_padded


def pad_spectral_axis(frame, xcens, ycens, pad=0, ref_cols=None,
                      replace=False):
    """Add padding to the spectral axis by interpolating the corresponding
    edge profile onto a set of extrapolated centroids.

    Parameters
    ----------
    frame : np.array
        Data frame.
    xcens : list
        X-coordinates of the trace centroids.
    ycens : list
        Y-coordinates of the trace centroids.
    pad : int
        Amount of padding to add along either end of the spectral axis (in
        pixels).
    ref_cols : list, np.array, None
        Which columns to use as the reference profiles for the padding.
    replace : bool
        Toggle for functionality to replace reference pixel columns.

    Returns
    -------
    newframe : np.array
        Data frame with padding on the spectral axis.
    """

    # Set default reference columns.
    if ref_cols is None:
        ref_cols = [6, -6]

    dimy, dimx = np.shape(frame)
    # Get centroids and extended centroids.
    pp = np.polyfit(xcens, ycens, 5)
    if replace:
        xax_pad = np.arange(dimx + 2 * pad)
    else:
        xax_pad = np.arange(dimx + 2*pad) - pad
    ycens_pad = np.polyval(pp, xax_pad)
    # Construct padded dataframe and paste in orignal data.
    newframe = np.zeros((dimy, dimx + 2*pad))
    newframe[:, pad:(dimx + pad)] = frame

    # Loop over columns to pad and stitch on the shifted reference column.
    for col in range(pad):
        yax = np.arange(dimy)
        newframe[:, col] = np.interp(yax,
                                     yax - ycens[ref_cols[0]] + ycens_pad[col],
                                     frame[:, ref_cols[0]])
    for col in range(dimx + ref_cols[1] + pad+1, dimx + 2*pad):
        yax = np.arange(dimy)
        newframe[:, col] = np.interp(yax,
                                     yax - ycens[ref_cols[1]] + ycens_pad[col],
                                     frame[:, ref_cols[1]])

    return newframe


def simulate_wings(w, psfs, halfwidth=12, verbose=0):
    """Extract the wings from a simulated WebbPSF 1D profile.

    Parameters
    ----------
    w : float
        Wavelength of interest (µm).
    psfs : np.recarray
        Array of simulated SOSS PSFs.
    halfwidth : int
        Half width of the SOSS trace.
    verbose : int
        Level of verbosity.

    Returns
    -------
    wing : np.array
        Extracted right wing.
    wing2 : np.array
        Extracted left wing.
    """

    # Get the simulated profile at the desired wavelength.
    stand = utils.interpolate_profile(w, psfs)
    psf_size = np.shape(psfs['PSF'])[1]
    # Normalize to a max value of one to match the simulated profile.
    max_val = np.nanpercentile(stand, 99.5)
    stand /= max_val

    # Define the edges of the profile 'core'.
    ax = np.arange(psf_size)
    ystart = int(round(psf_size//2 - halfwidth, 0))
    yend = int(round(psf_size//2 + halfwidth, 0))
    # Get and fit the 'right' wing.
    wing = stand[yend:]
    pp = np.polyfit(ax[yend:], np.log10(wing), 9)
    wing = 10**np.polyval(pp, ax[yend:])
    # Get and fit the 'left' wing.
    wing2 = stand[:ystart]
    pp = np.polyfit(ax[:ystart], np.log10(wing2), 9)
    wing2 = 10**np.polyval(pp, ax[:ystart])

    # Do diagnostic plot if necessary.
    if verbose == 3:
        plotting.plot_wing_simulation(stand, halfwidth, wing, wing2, ax,
                                      ystart, yend)

    return wing, wing2


if __name__ == '__main__':
    clear_data = fits.getdata('Ref_files/simulated_data.fits', 0)

    spat_prof = EmpiricalProfile(clear_data, verbose=1)
    spat_prof.build_empirical_profile()
