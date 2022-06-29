#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:46 2021

@author: MCR

Definitions of the main functions for the APPleSOSS (A Producer of ProfiLEs
for SOSS) module. This class will be initialized and called by the user to
create models of the spatial profiles for the first, second, and third order
SOSS traces, for use as the specprofile reference file required by the ATOCA
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
    clear : array-like
        SOSS CLEAR exposure data frame.
    subarray : str
        NIRISS SOSS subarray identifier. One of 'SUBSTRIP256', 'SUBSTRIP96',
        or 'FULL'.
    pad : int
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions.
    oversample : int
        Oversampling factor. Oversampling will be equal in the spectral and
        spatial directions.
    order1 : array-like
        First order spatial profile.
    order2 : array-like
        Second order spatial profile.
    order3 : array-like
        Third order spatial profile.

    Methods
    -------
    build_empirical_profile
        Construct the empirical spatial profiles.
    write_specprofile_reference
        Save spatial profile models to reference file.
    """

    def __init__(self, clear, pad=0, oversample=1):
        """Initializer for EmpiricalProfile.
        """

        # Initialize input attributes.
        self.clear = clear
        self.pad = pad
        self.oversample = oversample

        # Validate the parameters and determine the correct subarray.
        self.subarray = self.validate_inputs()
        self.order1 = None
        self.order2 = None
        self.order3 = None

    def build_empirical_profile(self, wave_increment=0.1, halfwidth=12,
                                verbose=0):
        """Run the empirical spatial profile construction module.

        Parameters
        ----------
        wave_increment : float
            Wavelength step (in µm) for PSF simulations. For accuracy, it is
            advisable not to use steps larger than 0.1µm.
        halfwidth : int
            half-width of the trace in native pixels.
        verbose: int
            Level of verbosity: either 3, 2, 1, or 0.
            3 - show all of progress prints, progress bars, and diagnostic
            plots.
            2 - show progress prints and bars.
            1 - show only progress prints.
            0 - show nothing.
        """

        # Run the empirical spatial profile construction.
        o1, o2, o3 = build_empirical_profile(self.clear, self.subarray,
                                             self.pad, self.oversample,
                                             wave_increment, halfwidth,
                                             verbose)
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
            SOSS subarray, either 'FULL', 'SUBSTRIP256', or 'SUBSTRIP96'
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


def build_empirical_profile(clear, subarray, pad, oversample, wave_increment,
                            halfwidth, verbose):
    """Main procedural function for the empirical spatial profile construction
    module. Calling this function will initialize and run all the required
    subroutines to produce a spatial profile for the first, second, and third
    orders. The spatial profiles generated can include oversampling as well
    as padding in both the spatial and spectral directions.

    Parameters
    ----------
    clear : array-like
        SOSS CLEAR exposure data frame.
    subarray : str
        NIRISS SOSS subarray identifier. One of 'SUBSTRIP256', 'SUBSTRIP96',
        or 'FULL'.
    pad : int
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions.
    oversample : int
        Oversampling factor. Oversampling will be equal in the spectral and
        spatial directions.
    wave_increment : float
        Wavelength step (in µm) for PSF simulations.
    halfwidth : int
        Half-width of the trace in native pixels.
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
    # If subarray is FULL - trim down to SUBSTRIP256 and work with that.
    if subarray == 'FULL':
        clear = clear[-256:, :]
        # Reset all variable to appropriate SUBSTRIP256 values.
        subarray = 'SUBSTRIP256'

    # Add a floor level such that all pixel values are positive and interpolate
    # bad pixels
    if verbose != 0:
        print(' Initial processing.')
        print('  Interpolating bad pixels...', flush=True)
    floor = np.nanpercentile(clear, 0.1)
    clear -= floor
    clear = utils.replace_badpix(clear, verbose=verbose)

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
    # Build a first estimate of the first, second, and third order spatial
    # profiles. The cores can be mostly read off of the clear dataframe - it
    # is just the wings that need reconstruction. For this, we will use
    # WebbPSF.
    # Generate WebbPSF 1D profiles across a range of wavelengths.
    psfs = utils.generate_psfs(wave_increment=wave_increment, verbose=verbose)

    # === First Order ===
    # Construct the first order profile.
    if verbose != 0:
        print(' Building the spatial profile models.')
        print('  Starting the first order model...', flush=True)
    o1_uncontam, o1_native = reconstruct_order(clear, centroids, order=1,
                                               psfs=psfs, halfwidth=halfwidth,
                                               pad=0)
    # Add padding to first order spatial axis if necessary.
    if pad != 0:
        o1_uncontam = np.pad(o1_uncontam, ((pad, pad), (0, 0)), mode='edge')

    # If the subarray is SUBSTRIP96, this is all we can do. However, for
    # SUBSTRIP256 we can reconstruct the second and third orders as well.
    if subarray != 'SUBSTRIP96':
        # === Second Order ===
        # Construct the second order profile.
        if verbose != 0:
            print('  Starting the second order trace...')
        o2_out = reconstruct_order(clear - o1_native, centroids, order=2,
                                   psfs=psfs, halfwidth=halfwidth, pad=pad,
                                   o1_prof=o1_uncontam[pad:, :],
                                   verbose=verbose)
        o2_uncontam, o2_native = o2_out[0], o2_out[1]
        # Add padding to the lower edge of spatial axis if necessary.
        if pad != 0:
            o2_uncontam = np.pad(o2_uncontam, ((pad, 0), (0, 0)), mode='edge')

        # === Third Order ===
        # Construct the third order profile.
        if verbose != 0:
            print('  Starting the third order trace...')
        o3_out = reconstruct_order(clear - o1_native - o2_native, centroids,
                                   order=3, psfs=psfs, pivot=700,
                                   halfwidth=halfwidth, pad=pad,
                                   o2_prof=o2_uncontam[pad:, :],
                                   verbose=verbose)
        o3_uncontam = o3_out[0]
        # Add padding to the lower edge of the spatial axis if necessary.
        if pad != 0:
            o3_uncontam = np.pad(o3_uncontam, ((pad, 0), (0, 0)), mode='edge')
    else:
        msg = 'Only order 1 can be reconstructed for SUBSTRIP96.'
        warnings.warn(msg)
        o2_uncontam = np.ones_like(o1_uncontam)
        o3_uncontam = np.ones_like(o1_uncontam)

    # ========= FINAL TUNING =========
    # Pad the spectral axes.
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


def oversample_frame(dataframe, os):
    """Oversample a dataframe by a specified amount.

    Parameters
    ----------
    dataframe : array-like
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
    pp = interp2d(x, y, dataframe, kind='cubic')
    data_os = pp(x_os, y_os)

    return data_os


def pad_spectral_axis(frame, xcens, ycens, pad=0, ref_cols=None,
                      replace=False):
    """Add padding to the spectral axis by interpolating the corresponding
    edge profile onto a set of extrapolated centroids.

    Parameters
    ----------
    frame : array-like
        Data frame.
    xcens : array-like
        X-coordinates of the trace centroids.
    ycens : array-like
        Y-coordinates of the trace centroids.
    pad : int
        Amount of padding to add along either end of the spectral axis (in
        pixels).
    ref_cols : array-like
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


def reconstruct_order(residual, cen, order, psfs, halfwidth, pad, pivot=750,
                      o1_prof=None, o2_prof=None, os_factor=10, verbose=0):
    """Reconstruct the wings of the the spatial profiles using simulated
    WebbPSF PSFs. Will also add padding to the spatial axes of orders 2 and 3,
    where the trace touches the top edge of the detector.

    Parameters
    ----------
    residual : array-like
        NIRISS/SOSS data frame residual map - either with o1, or both o1 and
        o2 removed.
    cen : dict
        Centroids dictionary.
    order : int
        The order to reconstruct.
    psfs : array-like
        Array of simulated 1D SOSS PSFs.
    halfwidth : int
        Half-width of the trace in native pixels.
    pad : int
        Amount of padding in native pixels to add to the spatial axis.
    pivot : int
        For order 2, minimum spectral pixel value for which a wing
        reconstruction will be attempted. For order 3, the maximum pixel value.
        For spectral pixels < or >pivot respectively, the profile at pivot will
        be used.
    o1_prof : array-like
        Uncontaminated order 1 spatial profile. Only necessary for
        reconstruction of order 2.
    o2_prof : array_like
        Uncontaminated order 2 spatial profile. Only necessary for
        reconstruction of order 3.
    os_factor : int
        Oversampling factor for re-centroiding.
    verbose : int
        level of verbosity.

    Returns
    -------
    new_frame : np.array
        Model of the second order spatial profile with wings reconstructed.
    new_frame_native : np.array
        Reconstructed profile at the native counts level. Only really necessary
        for order 1.
    """

    # Initalize new data frame and get subarray dimensions.
    dimy, dimx = np.shape(residual)
    new_frame = np.zeros((dimy+pad, dimx))
    new_frame_native = np.zeros((dimy, dimx))
    # Get wavelength calibration.
    wavecal_x, wavecal_w = utils.get_wave_solution(order=order)

    first_time = True
    if order == 3:
        maxi = pivot
    else:
        maxi = dimx
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
        cen_o = int(round(cen['order '+str(order)]['Y centroid'][i]*os_factor, 0))
        if cen_o/os_factor + halfwidth > dimy:
            if i < maxi:
                maxi = i
            continue

        # Get a copy of the spatial profile, and normalize it by its max value.
        working_prof = np.copy(residual[:, i])
        lwp = len(working_prof)
        working_prof_os = np.interp(np.linspace(0, (os_factor*lwp-1)/os_factor,
                                                (os_factor*lwp-1)+1),
                                    np.arange(lwp), working_prof)
        max_val = np.nanpercentile(working_prof[(cen_o//os_factor-halfwidth):(cen_o//os_factor+halfwidth)], 99)

        # Simulate the wings.
        if first_time is False:
            verbose = 0
        wing, wing2 = simulate_wings(wave, psfs, halfwidth=halfwidth,
                                     verbose=verbose)
        # Hack to shift left wing over by one pixel to account for the fact
        # that it isn't lining up perfectly with the profile for some reason.
        wing2 = np.pad(wing2, (1, 0), mode='edge')
        wing *= max_val
        lw = len(wing)
        wing_os = np.interp(np.linspace(0, (os_factor*lw-1)/os_factor,
                                        (os_factor*lw-1)+1), np.arange(lw),
                            wing)
        wing2 *= max_val
        lw2 = len(wing2)
        wing2_os = np.interp(np.linspace(0, (os_factor*lw2-1)/os_factor,
                                         (os_factor*lw2-1)+1), np.arange(lw2),
                             wing2)
        first_time = False
        # Concatenate the wings onto the profile core.
        end = int(round((cen_o + halfwidth*os_factor), 0))
        start = int(round((cen_o - halfwidth*os_factor), 0))

        stitch = np.concatenate([wing2_os,
                                 working_prof_os[(start+os_factor):end],
                                 wing_os])
        # Shift the profile back to its correct centroid position
        psf_len = np.shape(psfs['PSF'])[1]*os_factor
        stitch = np.interp(np.arange((dimy+pad)*os_factor),
                           np.arange(psf_len) - psf_len//2 + cen_o,
                           stitch)
        # Interpolate back to native pixel sampling.
        stitch = np.interp(np.arange(dimy+pad),
                           np.linspace(0, (os_factor*(dimy+pad)-1)/os_factor,
                                       (os_factor*(dimy+pad)-1)+1),
                           stitch)
        new_frame[:, i] = stitch
        # Rescale to native flux level.
        new_frame_native[:, i] = stitch[:dimy]

    # For columns where the order 2 core is not distinguishable (due to the
    # throughput dropping near 0, or it being buried in order 1) reuse a
    # profile from order 1 at the same wavelength. The shape of the PSF is
    # completely determined by the optics, and should thus be identical for a
    # given wavelength, irrrespective of the order. The differing
    # tilt/spectral resolution of order 1 vs 2 may have some effect here
    # though.
    if order == 2:
        wavecal_x_o1, wavecal_w_o1 = utils.get_wave_solution(order=1)
        for i in range(pivot):
            wave_o2 = wavecal_w[i]
            co1 = cen['order 1']['Y centroid']
            co2 = cen['order 2']['Y centroid'][i]
            working_prof = utils.interpolate_profile(wave_o2, co2,
                                                     wavecal_w_o1, o1_prof.T,
                                                     co1, os_factor=os_factor)
            new_frame[:, i] = working_prof

        # For columns where the centroid is off the detector, reuse the bluest
        # reconstructed profile.
        # Hard stop for profile reuse - one half width passed where the trace
        # centroid leaves the detector.
        stop = np.where(cen['order 2']['Y centroid'] >= dimy+pad)[0][0]
        stop += halfwidth
        for i in range(maxi, stop):
            anchor_prof = new_frame[:, maxi - 1]
            sc = cen['order '+str(order)]['Y centroid'][maxi - 1]
            ec = cen['order '+str(order)]['Y centroid'][i]
            working_prof = np.interp(np.arange(dimy+pad),
                                     np.arange(dimy+pad) - sc + ec,
                                     anchor_prof)
            new_frame[:, i] = working_prof

    # Do something similar for order 3. Where the throughput is too low, reuse
    # a profile of the same wavelength from order 2.
    if order == 3:
        # Hard stop for profile reuse - one half width passed where the trace
        # centroid leaves the detector.
        stop = np.where(cen['order 3']['Y centroid'] >= dimy+pad)[0][0]
        stop += halfwidth
        wavecal_x_o2, wavecal_w_o2 = utils.get_wave_solution(order=2)
        for i in range(maxi, stop):
            wave_o3 = wavecal_w[i]
            co2 = cen['order 2']['Y centroid']
            co3 = cen['order 3']['Y centroid'][i]
            working_prof = utils.interpolate_profile(wave_o3, co3,
                                                     wavecal_w_o2, o2_prof.T,
                                                     co2, os_factor=os_factor)
            new_frame[:, i] = working_prof

    return new_frame, new_frame_native


def simulate_wings(w, psfs, halfwidth, verbose=0):
    """Extract the wings from a simulated WebbPSF 1D profile.

    Parameters
    ----------
    w : float
        Wavelength of interest (µm).
    psfs : array-like
        Array of simulated SOSS PSFs.
    halfwidth : int
        Half-width of the SOSS trace.
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
    stand = utils.interpolate_profile(w, 0, psfs['Wave'][:, 0], psfs['PSF'],
                                      np.zeros_like(psfs['Wave'][:, 0]))
    psf_size = np.shape(psfs['PSF'])[1]
    # Normalize to a max value of one to match the simulated profile.
    max_val = np.nanpercentile(stand, 99)
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

    spat_prof = EmpiricalProfile(clear_data)
    spat_prof.build_empirical_profile(verbose=1)
