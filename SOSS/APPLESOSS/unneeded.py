#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Apr 21 10:31 2022

@author: MCR

Collection of currenlty unneeded functions, which may possibly come in handy
later?
"""


def get_substrip96_centroids(centroids):
    """For the SUBSTRIP96 subarray, the edgetrig centroiding method cannot
    locate the centroids for orders 2 and 3 as they are not on the detector.
    This function estimates the centroid positions for these orders by
    comparing the edgetrig first order centroids to the centroids in the trace
    table reference file. Analogous to the simple solver, the necessary
    rotation and x/y offsets are calculated to transform the reference first
    order centroids to match the data. All orders are assumed to transform
    rigidly, such that applying the transformation to the reference second, or
    third order centroids gives an estimate of the data centroids.

    Parameters
    ----------
    centroids : dict
        Centroids dictionary, as returned by get_soss_centroids, containing
        only information for the first order.

    Returns
    -------
    centroids : dict
        Centroids dictionary, with second and third order centroids appended.
    """

    # Get first order centroids from the data.
    xcen_dat = centroids['order 1']['X centroid']
    ycen_dat = centroids['order 1']['Y centroid']
    # Get first order centroids in the trace table reference file.
    ttab_file = soss_read_refs.RefTraceTable()
    xcen_ref = ttab_file('X', subarray='SUBSTRIP96')[1]
    # Extend centroids beyond edges of the subarray for more accurate fitting.
    inds = np.where((xcen_ref >= -50) & (xcen_ref < 2098))
    xcen_ref = xcen_ref[inds]
    ycen_ref = ttab_file('Y', subarray='SUBSTRIP96')[1][inds]

    # Fit reference file centroids to the data to determine necessary rotation
    # and offsets.
    guess_params = np.array([0.15, 1., 1.])
    lik_args = [xcen_ref, ycen_ref, xcen_dat, ycen_dat]
    fit = minimize(chi_squared, guess_params, lik_args).x
    rot_ang, x_shift, y_shift = fit

    # Transform centroids to detector frame for orders 2 and 3.
    for order in [2, 3]:
        # Get centroids from the reference file.
        xcen_ref = ttab_file('X', subarray='SUBSTRIP96', order=order)[1]
        # Extend centroids beyond edges of the subarray for more accurate
        # fitting.
        inds = np.where((xcen_ref >= -50) & (xcen_ref < 2098))
        xcen_ref = xcen_ref[inds]
        ycen_ref = ttab_file('Y', subarray='SUBSTRIP96', order=2)[1][inds]
        # Transform reference centroids to the data frame.
        rot_x, rot_y = transform_coords(rot_ang, x_shift, y_shift, xcen_ref,
                                        ycen_ref)
        # Ensure there is a y-centroid for every x-pixel (even if they extend
        # off of the detector).
        rot_y = np.interp(np.arange(2048), rot_x[::-1], rot_y[::-1])
        # Add the transformed centroids to the centroid dict.
        tmp = {['X centroid']: np.arange(2048), ['Y centroid']: rot_y}
        centroids['order {}'.format(order)] = tmp

    return centroids


def get_goodwing(clear96, centroids):
    """Obtain an uncontaminated wing of the first order spatial profile to use
    as a reference wing for SUBSTRIP96 wing reconstruction.

    Parameters
    ----------
    clear96 : np.array
        SUBSTRIP96 2D trace profile dataframe.
    centroids : dict
        Centroids dictionary.

    Returns
    -------
    goodwing : np.array
        Uncontaminated first order wing.
    """

    # Find the column where the first order is lowest on the detector. This
    # ensures that we will never have to extend the goodwing.
    ymin = np.min(centroids['order 1']['Y centroid'])
    ind = np.where(centroids['order 1']['Y centroid'] == ymin)[0][0]
    # Ensure that the second order core is off the detector so there is no
    # contamination.
    while centroids['order 2']['Y centroid'][ind] < 100:
        ind += 1

    # Extract the right wing of the chosen column.
    ycen = int(round(centroids['order 1']['Y centroid'][ind], 0))
    goodwing = np.nanmedian(clear96[(ycen+13):, (ind-2):(ind+2)], axis=1)

    return goodwing


def reconstruct_wings96(profile, ycen, goodwing=None, contamination=False,
                        pad=0, verbose=0, **kwargs):
    """Wing reconstruction for the SUBSTRIP96 subarray. As not enough of the
    first order wings remain on the detector to perform the full wing
    reconstruction, a standard uncontaminated wing profile is used to correct
    second order contamination.

    Parameters
    ----------
    profile : np.array
        Spectral trace spatial profile.
    ycen : list, np.array
        Y-coordinates of the trace centroids. Must include all three
        diffraction orders if contamination is True, or only the first order if
        False.
    goodwing : np.array, None
        Uncontaminated wing profile.
    contamination : bool
        If True, profile is contaminated by the second order.
    pad : int
        Amount to pad each end of the spatial axis (in pixels).
    verbose : int
        Level of verbosity.

    Returns
    -------
    newprof : np.array
        Input spatial profile with reconstructed wings and padding.

    Raises
    ------
    ValueError
        If contamination is True and goodwing is None.
    """

    # Get rid of second order contamination by stitching an uncontaminated wing
    # to the uncontaminated first order core.
    if contamination is True:
        # Stitch together the contaminated profile, and uncontaminated wing.
        ycen = int(round(ycen, 0))
        if goodwing is None:
            errmsg = 'Uncontaminated wing must not be None if profile is \
                      contaminated.'
            raise ValueError(errmsg)
        newprof = np.concatenate([profile[:(ycen+13)], goodwing])
        # Smooth over the joint with a polynomial fit.
        fitprof = newprof[(ycen+10):(ycen+20)]
        fitprof = ma.masked_array(fitprof, mask=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        fitax = ma.masked_array(np.arange(ycen+20 - (ycen+10)) + ycen+10,
                                mask=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        # Fit the pixels on either side of the joint.
        pp = ma.polyfit(fitax, fitprof, 5)
        # Evaluate the above polynomial in the joint region.
        newprof[(ycen+12):(ycen+16)] = np.polyval(pp, np.arange(4)+ycen+12)
        # Trim to the size of SUBSTRIP96
        newprof = newprof[:96]
    else:
        newprof = profile

    # Add padding - padding values are constant at the median of edge pixels.
    padval_r = np.median(newprof[-8:-3])
    padval_l = np.median(newprof[3:8])
    newprof = np.concatenate([np.tile(padval_l, pad+4), newprof[4:-4],
                              np.tile(padval_r, pad+4)])

    if verbose == 3:
        plotting.plot_wing_reconstruction96(profile, newprof, **kwargs)

    return newprof


def construct_order2(clear, order1_rescale, ycens, pad=0, verbose=0):
    """This creates the full order 2 trace profile model. For wavelengths of
    overlap between orders one and two, use the first order wings to correct
    the over-subtracted second order wings. For wavelengths where there is no
    overlap between the orders, use the wings of the closest neighbouring first
    order wavelength. In cases where the second order core is also
    over-subtracted, use the first order core to correct this as well. This
    method implicitly assumes that the spatial profile is determined solely by
    the optics, and is the same for all orders at a given wavelength.

    Parameters
    ----------
    clear : np.array
        NIRISS SOSS CLEAR exposure dataframe.
    order1_rescale : np.array
        Uncontaminated order 1 trace profile.
    ycens : dict
        Dictionary of Y-coordinates for the trace centroids of the first three
        diffraction orders, ie. as returned by get_soss_centroids.
    pad : int
        Number of pixels of padding to add on both ends of the spatial axis.
    verbose : int
        Level of verbosity.

    Returns
    -------
    o2frame : np.array
        Uncontaminated second order trace profile at the native flux level.
    """

    # ========= INITIAL SETUP =========
    # Get wavelength and detector pixel calibration info.
    # TODO : remove local path
    wavecal_o1 = fits.getdata(path+'jwst_niriss_soss-256-ord1_trace.fits', 1)
    wavecal_o2 = fits.getdata(path+'jwst_niriss_soss-256-ord2_trace.fits', 1)
    pp_p = np.polyfit(wavecal_o1['WAVELENGTH'][::-1],
                      wavecal_o1['Detector_Pixels'], 1)
    pp_w2 = np.polyfit(wavecal_o2['Detector_Pixels'],
                       wavecal_o2['WAVELENGTH'][::-1], 1)
    # Get throughput information for each order.
    ttab_file = soss_read_refs.RefTraceTable()
    thpt_o1 = ttab_file('THROUGHPUT', subarray='SUBSTRIP256', order=1)
    thpt_o2 = ttab_file('THROUGHPUT', subarray='SUBSTRIP256', order=2)

    # Useful storage arrays
    ks, pixs = [], []
    notdone = []
    # First estimate of second order is the residuals from subtraction of
    # first order model from original data.
    sub = clear - order1_rescale
    dimy, dimx = np.shape(order1_rescale)
    o2frame = np.zeros((dimy, dimx))

    # ========= RECONSTRUCT SECOND ORDER WINGS =========
    # Determine a coefficient to rescale the first order profile to the flux
    # level of the second order at the same wavelength. Since the
    # wavelengths are the same, we can assume that the spatial profiles are
    # also the same. Thus, use the first order wings to reconstruct the
    # second order wings.
    if verbose != 0:
        print('   Reconstructing oversubtracted wings...', flush=True)
    disable = utils.verbose_to_bool(verbose)
    for o2pix in tqdm(range(dimx), disable=disable):
        # Get the wavelength for each column in order 2.
        o2wave = np.polyval(pp_w2, o2pix)
        # Find the corresponding column in order 1.
        o1pix = int(round(np.polyval(pp_p, o2wave), 0))

        # === FIND CORRECT COLUMNS ===
        if o1pix < dimx - 5:
            # Region where wavelengths in order 1 and order 2 overlap.
            # Get throughput for order 1.
            thpt_1i = np.where(thpt_o1[0] >= o2wave)[0][0]
            ycen_o1i = int(round(ycens['order 1']['Y centroid'][o1pix], 0))
            # Save current order 1 info for special cases below.
            # Since increasing pixel position moves from red to blue, always
            # save the bluest order 1 profile to reuse when lambda<0.8µm.
            o1pix_r = o1pix
            thpt_1i_r = thpt_1i
            marker = True
        else:
            # For region where lambda<0.8µm, there is no profile in order 1.
            # We will reuse the 'bluest' order 1 profile from the above for
            # this region, assuming that the spatial profile does not vary too
            # drastically with wavelength.
            thpt_1i = thpt_1i_r
            o1pix = o1pix_r
            ycen_o1i = int(round(ycens['order 1']['Y centroid'][o1pix_r], 0))
            marker = False
        # Get trace centroid and throughput for the second order.
        thpt_2i = np.where(thpt_o2[0] >= o2wave)[0][0]
        ycen_o2i = int(round(ycens['order 2']['Y centroid'][o2pix], 0))
        # If the centroid is off of the detector, skip the column.
        if ycen_o2i > dimy:
            continue

        # === DETERMINE O1->O2 SCALING ===
        # Find coefficient to scale order 1 to the flux level of order 2.
        # Initial guess for scaling coefficient is ratio of throughputs.
        # Won't be exactly correct due to different spectral resolutions.
        k0 = thpt_2i / thpt_1i
        # Use region +/- 13 pixels around centroid to calculate scaling.
        max1, max2 = np.min([ycen_o1i+14, dimy]), np.min([ycen_o2i+14, dimy])
        min1, min2 = np.max([ycen_o1i-13, 0]), np.max([ycen_o2i-13, 0])
        # Ensure that the order 1 and order 2 arrays are the same size
        if max1 - ycen_o1i != max2 - ycen_o2i:
            mindif = np.min([max1 - ycen_o1i, max2 - ycen_o2i])
            max1, max2 = ycen_o1i + mindif, ycen_o2i + mindif
        if ycen_o1i - min1 != ycen_o2i - min2:
            mindif = np.min([ycen_o1i - min1, ycen_o2i - min2])
            min1, min2 = ycen_o1i - mindif, ycen_o2i - mindif
        # Determine optimal scaling coefficient.
        k = minimize(utils.lik, k0, (sub[min2:max2, o2pix],
                     order1_rescale[min1:max1, o1pix])).x
        if k <= 0:
            notdone.append(o2pix)
            continue
        # Rescale the first order profile to the flux level of order 2.
        o1prof = order1_rescale[:, o1pix]*k
        if marker is True:
            ks.append(k[0])
            pixs.append(o2pix)

        # === RECONSTRUCT SPATIAL PROFILE ===
        # Replace any oversubtracted pixels in o2 with o1.
        inds = np.isnan(np.log10(sub[min2:max2, o2pix]))
        sub[min2:max2, o2pix][inds] = o1prof[min1:max1][inds]
        # Stitch together o2 core with o1 wings.
        newprof = np.concatenate([o1prof[max1:][::-1],
                                  sub[min2:max2, o2pix], o1prof[max1:]])

        # Put the spatial profile on the detector. The order 2 profile is
        # already at the detector scale.
        o2_ycen = ycens['order 2']['Y centroid'][o2pix]
        oldax = np.arange(len(newprof)) - len(newprof)/2 + o2_ycen
        # Find position where 'right' wing is off the detector.
        end = np.where(oldax < dimy)[0][-1]
        ext = 0
        # Find position where 'left' wing is off the detector.
        try:
            start = np.where(oldax < 0)[0][-1]
        except IndexError:
            # If it is never off, repeat the end value to make up space.
            start = 0
            ext = dimy - (end - start)
            val = np.nanmedian(newprof[:5])
            newprof = np.concatenate([np.tile(val, ext), newprof])
        # Put the profile on the detector.
        o2frame[:, o2pix] = newprof[start:(end+ext)]

    # ========= CORRECT OVERSUBTRACTED COLUMNS =========
    # Deal with notdone columns: columns who's scaling coef was <= 0 due to
    # oversubtraction of the first order.
    pixs, ks = np.array(pixs), np.array(ks)
    # Rough sigma clip of huge outliers.
    pixs, ks = utils.sigma_clip(pixs, ks)
    # Fit a polynomial to all positive scaling coefficients, and use the fit to
    # interpolate the correct scaling for notdone columns.
    pp_k = np.polyfit(pixs, ks, 6)
    pp_k = utils.robust_polyfit(pixs, ks, pp_k)
    # Plot the results if necessary.
    if verbose == 3:
        plotting.plot_scaling_coefs(pixs, ks, pp_k)

    if verbose != 0 and len(notdone) != 0:
        print('   Dealing with oversubtracted cores...')
    for o2pix in notdone:
        # Get the order 2 wavelength and corresponding order 1 column.
        o2wave = np.polyval(pp_w2, o2pix)
        o1pix = int(round(np.polyval(pp_p, o2wave), 0))
        # If order 1 pixel is off of the detector, use trailing blue profile.
        if o1pix >= 2048:
            o1pix = o1pix_r
            k = ks[-1]
        else:
            # Interpolate the appropriate scaling coefficient.
            k = np.polyval(pp_k, o2pix)
        # Rescale the order 1 profile to the flux level of order 2.
        o1prof = order1_rescale[:, o1pix]*k
        # Get bounds of the order 1 trace core.
        o1_ycen = ycens['order 1']['Y centroid'][o1pix]
        min1 = np.max([int(round(o1_ycen, 0)) - 13, 0])
        max1 = np.min([int(round(o1_ycen, 0)) + 14, dimy])
        # Use the first order core as well as the wings to reconstruct a first
        # guess of the second order profile.
        newprof = np.concatenate([o1prof[max1:][::-1], o1prof[min1:]])

        # Put the spatial profile on the detector as before.
        o2_ycen = ycens['order 2']['Y centroid'][o2pix]
        oldax = np.arange(len(newprof)) - len(newprof)/2 + o2_ycen
        end = np.where(oldax < dimy)[0][-1]
        ext = 0
        try:
            start = np.where(oldax < 0)[0][-1]
        except IndexError:
            start = 0
            ext = dimy - (end - start)
            val = np.nanmedian(newprof[:5])
            newprof = np.concatenate([np.tile(val, ext), newprof])
        o2frame[:, o2pix] = newprof[start:(end+ext)]

    # ========= SMOOTH DISCONTINUITIES =========
    # Smooth over discontinuities (streaks and oversubtractions) in both the
    # spatial and spectral directions.
    if verbose != 0:
        print('   Smoothing...', flush=True)
    o2frame = smooth_spec_discont(o2frame, verbose=verbose)
    o2frame = smooth_spat_discont(o2frame, ycens)

    # ========= ADD PADDING =========
    # Add padding to the spatial axis by repeating the median of the 5 edge
    # pixels for each column.
    if pad != 0:
        o2frame_pad = np.zeros((dimy+2*pad, dimx))
        padded = np.repeat(np.nanmedian(o2frame[-5:, :, np.newaxis], axis=0),
                           pad, axis=1).T
        o2frame_pad[-pad:] = padded
        padded = np.repeat(np.nanmedian(o2frame[:5, :, np.newaxis], axis=0),
                           pad, axis=1).T
        o2frame_pad[:pad] = padded
        o2frame_pad[pad:-pad] = o2frame
        o2frame = o2frame_pad

    return o2frame


def refine_order1(clear, o2frame, centroids, pad, verbose=0):
    """Refine the first order trace model using a clear exposure with the
    second order subtracted.

    Parameters
    ----------
    clear : np.array
        CLEAR data frame.
    o2frame : np.array
        Model of the second order spatial profile.
    centroids : dict
        Centroid dictionary.
    pad : int
        padding to add to the spatial axis.
    verbose : int
        Verbose level.

    Returns
    -------
    order1_uncontam : np.array
        First order trace model.
    """

    dimy, dimx = np.shape(clear)
    # Create an uncontaminated frame by subtracting the second order model from
    # the CLEAR exposure.
    order1_uncontam_unref = clear - o2frame
    order1_uncontam = np.zeros((dimy+2*pad, dimx))

    # Reconstruct first order wings.
    disable = utils.verbose_to_bool(verbose)
    for i in tqdm(range(dimx), disable=disable):
        ycens = [centroids['order 1']['Y centroid'][i],
                 centroids['order 2']['Y centroid'][i],
                 centroids['order 3']['Y centroid'][i]]
        prof_refine = reconstruct_wings256(order1_uncontam_unref[:, i],
                                           ycens, contamination=[3], pad=pad)
        order1_uncontam[:, i] = prof_refine

    return order1_uncontam


def smooth_spat_discont(o2frame, ycens):
    """Smooth oversubtracted pixels in the spatial direction. If the flux in a
    pixel is >3sigma deviant from the mean value of the trace core in its
    column, it is replaced by a median of flux values over its neighbours in
    the spectral direction.

    Parameters
    ----------
    o2frame : np.array
        Uncontaminated second order trace profile.
    ycens : dict
        Centroids dictionary.

    Returns
    -------
    o2frame : np.array
        Uncontaminated trace profile with oversubtracted pixels interpolated.
    """

    for col in range(o2frame.shape[1]):
        # Get lower and upper bounds of trace core
        start = int(round(ycens['order 2']['Y centroid'][col], 0)) - 11
        end = int(round(ycens['order 2']['Y centroid'][col], 0)) + 11
        # Calculate mean and standard deviation of core.
        mn = np.mean(np.log10(o2frame[start:end, col]))
        sd = np.std(np.log10(o2frame[start:end, col]))
        diff = np.abs(np.log10(o2frame[start:end, col]) - mn)
        # Find where flux values are >3sigma deviant from mean
        inds = np.where(diff > 3*sd)[0]
        for i in inds:
            # Pixel location in spatial profile.
            loc = start+i
            # Replace bad pixel with median of rows.
            start2 = np.max([col-25, 0])
            end2 = np.min([col+25, o2frame.shape[1]])
            rep_val = np.median(o2frame[loc, start2:end2])
            o2frame[loc, col] = rep_val

    return o2frame


def smooth_spec_discont(o2frame, verbose):
    """Smooth over streaks (especially in the contaminated region). If the
    median flux value of a column is >10% deviant from that of the surrounding
    columns, replace it with the median of its neighbours.

    Parameters
    ----------
    o2frame : np.array
        Uncontaminated second order trace profile.
    verbose : int
        Level of verbosity.

    Returns
    -------
    o2frame : np.array
        Uncontaminated trace profile with column-to-column discontinuities
        smoothed.
    """

    # Get mean flux values for each column.
    col_mean = np.nanmedian(o2frame, axis=0)
    # Find where order 2 ends.
    end = np.where(col_mean == 0)[0][0] - 5
    # For each column, find the local mean of the surrounding 6 columns.
    loc_mean = utils.local_mean(col_mean, step=3)
    # Calculate the deviation of each column from the local mean.
    dev_0 = np.abs((col_mean[3:end] - loc_mean[3:end])/loc_mean[3:end])
    # Replace all columns whose mean value is >10% deviant from the local mean
    # with the median of its neighbours.
    dev, iteration = dev_0, 0
    # Iterate until no columns are >10% deviant, or 10 iterations have run.
    while np.any(dev > 0.1) and iteration < 10:
        # Get all >10% deviant columns.
        inds = np.where(dev >= 0.1)[0]
        for i in inds:
            i += 3
            # For each column, calculate the median of the 'local region'.
            # Expand the region by one pixel each iteration.
            local = np.concatenate([o2frame[:, (i - 2 - iteration):i],
                                    o2frame[:, (i+1):(i + 3 + iteration)]],
                                   axis=1)
            o2frame[:, i] = np.nanmedian(local, axis=1)
        # Recalculate the flux deviations as before.
        col_mean = np.nanmean(o2frame, axis=0)
        loc_mean = utils.local_mean(col_mean, step=3)
        dev = np.abs((col_mean[3:end] - loc_mean[3:end])/loc_mean[3:end])
        # Increment iteration.
        iteration += 1

    # Plot the change in flux deviations after all iterations are complete.
    if verbose == 3:
        plotting.plot_flux_deviations(dev_0, dev, iteration)

    return o2frame


def plot_wing_reconstruction96(profile, newprof, text=None):
    """Do diagnostic plotting for the SUBSTRIP96 wing reconstruction.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(profile)), np.log10(profile), ls=':', c='black',
             label='original profile')
    add = len(newprof) - len(profile)
    plt.plot(np.arange(len(newprof))-add/2, np.log10(newprof), c='blue',
             label='reconstructed profile')
    if text is not None:
        plt.text(np.arange(len(newprof))[10], np.min(np.log10(newprof)), text,
                 fontsize=14)

    plt.xlabel('Spatial Pixel', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()


def plot_flux_deviations(dev_init, dev_fin, iteration):
    """Plot the diagnostic results of the order 2 smoothing iterations. The
    upper plot shows the initial column-wise flux deviations from the local
    mean, and the lower plot shows the same variations after the smoothing has
    been completed.
    """

    f, ax = plt.subplots(2, figsize=(15, 6), sharex=True)
    ax[0].plot(dev_init)
    ax[0].axhline(0.1, ls='--', c='black')
    ax[1].plot(dev_fin)
    ax[1].axhline(0.1, ls='--', c='black')

    max_val = np.max([len(dev_init), 2048])
    ax[1].set_xlim(0, max_val)
    ax[1].set_xlabel('Spectral Pixel', fontsize=14)
    ax[0].set_ylabel('Flux Variations (i=0)', fontsize=14)
    ax[1].set_ylabel('Flux Variations (i={})'.format(iteration-1), fontsize=14)
    plt.show()


def plot_scaling_coefs(pixels, k_coefs, pp_k):
    """Do diagnostic plotting for the first-to-second order flux scaling
    relationship.
    """

    plt.figure(figsize=(8, 5))
    plt.scatter(pixels, k_coefs, s=4, c='blue', alpha=0.8,
                label='calculated')
    plt.plot(pixels, np.polyval(pp_k, pixels), ls='--', c='red', label='fit')

    plt.xlabel('Spectral Pixel', fontsize=14)
    plt.ylabel('O1-to-O2 Scaling', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()


def _fit_trace_widths(clear, wave_coefs, verbose=0):
    """Due to the defocusing of the SOSS PSF, the width in the spatial
    direction does not behave as if it is diffraction limited. Calculate the
    width of the spatial trace profile as a function of wavelength, and fit
    with a linear relation to allow corrections of chromatic variations.

    Parameters
    ----------
    clear : np.array
        CLEAR exposure dataframe
    wave_coefs : tuple
        Polynomial coefficients for the wavelength to spectral pixel
        calibration.
    verbose : int
        Level of verbosity.

    Returns
    -------
    wfit_b : tuple
        Polynomial coefficients for a first order fit to the uncontaminated
        trace widths.
    wfit_r : tuple
        Polynomial coefficients for a first order fit to the contaminated trace
        widths.
    """

    # Get subarray dimensions and wavelength to spectral pixel transformation.
    yax, xax = np.shape(clear)
    wax = np.polyval(wave_coefs, np.arange(xax))

    # Determine the width of the trace profile by counting the number of pixels
    # with flux values greater than half of the maximum value in the column.
    trace_widths = []
    for i in range(xax):
        # Oversample by 4 times to get better sub-pixel scale info.
        prof = np.interp(np.linspace(0, yax - 1, yax * 4), np.arange(yax),
                         clear[:, i])
        # Sort the flux values in the profile.
        prof_sort = np.argsort(prof)
        # To mitigate the effects of any outliers, use the median of the 5
        # highest flux values as the maximum.
        inds = prof_sort[-5:]
        maxx = np.nanmedian(prof[inds])
        # Count how many pixels have flux greater than half this value.
        above_av = np.where(prof >= maxx / 2)[0]
        trace_widths.append(len(above_av) / 4)

    # Only fit the trace widths up to the blue anchor (2.1µm) where
    # contamination from the second order begins to be a problem.
    end = np.where(wax < 2.1)[0][0]
    fit_waves_b = np.array(wax[end:])
    fit_widths_b = np.array(trace_widths[end:])
    # Rough sigma clip of huge outliers.
    fit_waves_b, fit_widths_b = utils.sigma_clip(fit_waves_b, fit_widths_b)
    # Robustly fit a straight line.
    pp_b = np.polyfit(fit_waves_b, fit_widths_b, 1)
    wfit_b = utils.robust_polyfit(fit_waves_b, fit_widths_b, pp_b)

    # Separate fit to contaminated region.
    fit_waves_r = np.array(wax[:(end + 10)])
    fit_widths_r = np.array(trace_widths[:(end + 10)])
    # Rough sigma clip of huge outliers.
    fit_waves_r, fit_widths_r = utils.sigma_clip(fit_waves_r, fit_widths_r)
    # Robustly fit a straight line.
    pp_r = np.polyfit(fit_waves_r, fit_widths_r, 1)
    wfit_r = utils.robust_polyfit(fit_waves_r, fit_widths_r, pp_r)

    # Plot the width calibration fit if required.
    if verbose == 3:
        plotting.plot_width_cal((fit_widths_b, fit_widths_r),
                                (fit_waves_b, fit_waves_r), (wfit_b, wfit_r))

    return wfit_r, wfit_b


def plot_width_cal(fit_widths, fit_waves, width_poly):
    """Do the diagnostic plot for the trace width calibration relation.
    """

    plt.figure(figsize=(8, 5))
    plt.scatter(fit_waves[0][::10], fit_widths[0][::10], label='trace widths',
                c='blue', s=12,
                alpha=0.75)
    plt.scatter(fit_waves[1][::10], fit_widths[1][::10], c='blue', s=12,
                alpha=0.75)
    plt.plot(fit_waves[0], np.polyval(width_poly[0], fit_waves[0]), c='red',
             ls='--', label='width relation')
    plt.plot(fit_waves[1], np.polyval(width_poly[1], fit_waves[1]), c='red',
             ls='--')

    plt.xlabel('Wavelength [µm]', fontsize=14)
    plt.ylabel('Trace Spatial Width [pixels]', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()



def simulate_wings_OLD(wavelength, width_coefs, halfwidth=12, verbose=0):
    """Extract the spatial profile wings from a simulated SOSS PSF to
    reconstruct the wings of the second order.

    Parameters
    ----------
    wavelength : float
        Wavelength for which to simulate wings.
    width_coefs : np.array
        Trace width polynomial coefficients.
    halfwidth : int
        Half width of a spatial profile along the spatial axis.
    verbose : int
        Level of verbosty.

    Returns
    -------
    wing : np.array
        Right wing fit.
    wing2 : np.array
        Left wing fit.
    """

    # Open a simulated PSF from which to steal some wings.
    ref_profile = 'Ref_files/SOSS_PSFs/SOSS_os1_256x256_1.000000_0.fits'
    try:
        psf = fits.getdata(ref_profile, 0)
    except FileNotFoundError:
        # If the profile doesn't exist, create it and save it to disk.
        _calibrations.loicpsf([1.0 * 1e-6], save_to_disk=True, oversampling=1,
                              pixel=256, verbose=False, wfe_real=0)
        psf = fits.getdata(ref_profile, 0)
    stand = np.sum(psf, axis=0)
    # Normalize the profile by its maximum.
    max_val = np.nanmax(stand)
    stand /= max_val
    # Scale the profile width to match the current wavelength.
    stand = chromescale(stand, 1, wavelength, 128, width_coefs)

    # Define the edges of the profile 'core'.
    ax = np.arange(256)
    ystart = int(round(256 // 2 - halfwidth, 0))
    yend = int(round(256 // 2 + halfwidth, 0))
    # Get and fit the 'right' wing.
    wing = stand[yend:]
    pp = np.polyfit(ax[yend:], np.log10(wing), 7)
    wing = 10**(np.polyval(pp, ax[yend:]))
    # Get and fit the 'left' wing.
    wing2 = stand[:ystart]
    pp = np.polyfit(ax[:ystart], np.log10(wing2), 7)
    wing2 = 10**(np.polyval(pp, ax[:ystart]))

    # Do diagnostic plot if necessary.
    if verbose == 3:
        plotting.plot_wing_simulation(stand, halfwidth, wing, wing2, ax,
                                      ystart, yend)

    return wing, wing2


def chromescale(profile, wave_start, wave_end, ycen, poly_coef):
    """Correct chromatic variations in the PSF along the spatial direction
    using the polynomial relationship derived in _fit_trace_widths.

    Parameters
    ----------
    profile : np.array
        1D PSF profile to be rescaled.
    wave_start : float
        Starting wavelength.
    wave_end : float
        Wavelength after rescaling.
    ycen : float
        Y-pixel position of the order 1 trace centroid.
    poly_coef : tuple
        1st order polynomial coefficients describing the variation of the trace
        width with wavelength, e.g. as output by _fit_trace_widths.

    Returns
    -------
    prof_rescale : np.array
        Rescaled 1D PSF profile.
    """

    # Get the starting and ending trace widths.
    w_start = np.polyval(poly_coef, wave_start)
    w_end = np.polyval(poly_coef, wave_end)

    # Create a rescaled spatial axis. Shift the Y-centroid to zero so
    # that it does not move during the rescaling.
    xax = np.arange(len(profile)) - ycen
    xax_rescale = xax * (w_end / w_start)
    # Rescale the PSF by interpolating onto the new axis.
    prof_rescale = np.interp(xax, xax_rescale, profile)

    # Ensure the total flux remains the same
    # Integrate the profile with a Trapezoidal method.
    flux_i = np.sum(profile[1:-1]) + 0.5 * (profile[0] + profile[-1])
    flux_f = np.sum(prof_rescale[1:-1]) + 0.5 * (
            prof_rescale[0] + prof_rescale[-1])
    # Rescale the new profile so the total encompassed flux remains the same.
    prof_rescale /= (flux_f / flux_i)

    return prof_rescale


def reconstruct_wings256(profile, ycens=None, contamination=[2, 3], pad=0,
                         halfwidth=14, verbose=0, smooth=True, **kwargs):
    """Masks the second and third diffraction orders and reconstructs the
     underlying wing structure of the first order. Also adds padding in the
     spatial direction if required.

    Parameters
    ----------
    profile : np.array
        Spectral trace spatial profile.
    ycens : list, np.array
        Y-coordinates of the trace centroids. Must include all three
        diffraction orders if contamination is True, or only the first order if
        False.
    contamination : list, np.array, None
        List of contaminating orders present on the detector (not including the
        first order). For an uncontaminated detector, pass None.
    pad : int
        Amount to pad each end of the spatial axis (in pixels).
    halfwidth : int
        Half width of the profile core in pixels.
    verbose : int
        Level of verbosity.
    smooth : bool
        If True, smooths over highly deviant pixels in the spatial profile.

    Returns
    -------
    newprof : np.array
        Input spatial profile with reconstructed wings and padding.

    Raises
    ------
    ValueError
        If centroids are not provided for all three orders when contamination
        is not None.
    """

    dimy = len(profile)
    # Convert Y-centroid positions to indices
    ycens = np.atleast_1d(ycens)
    ycens = np.round(ycens, 0).astype(int)
    if contamination is not None and ycens.size != 3:
        errmsg = 'Centroids must be provided for first three orders ' \
                 'if there is contamination.'
        raise ValueError(errmsg)

    # mask negative and zero values.
    profile[profile <= 0] = np.nan

    # ======= RECONSTRUCT CONTAMINATED WING =======
    # Mask the cores of the first three diffraction orders and fit a straight
    # line to the remaining pixels. Additionally mask any outlier pixels that
    # are >3-sigma deviant from that line. Fit a 7th order polynomial to
    # remaining pixels.
    # Get the right wing of the trace profile in log space and spatial axis.
    working_profile, axis = np.copy(profile), np.arange(dimy)
    working_profile = np.log10(working_profile)

    # === Outlier masking ===
    # Mask the cores of each order.
    for order, ycen in enumerate(ycens):
        order += 1
        if order == 1:
            start = 0
            end = int(ycen + 2.5*halfwidth)
        else:
            start = np.min([ycen - int(1.5*halfwidth), dimy-2])
            end = np.min([ycen + int(1.5*halfwidth), dimy-1])
        # Set core of each order to NaN.
        working_profile[start:end] = np.nan
    # Fit the unmasked part of the wing to determine the mean linear trend.
    inds = np.where(np.isfinite(working_profile))[0]
    pp = utils.robust_polyfit(axis[inds], working_profile[inds], (0, 0))
    wing_mean = pp[1] + pp[0] * axis
    # Calculate the standard dev of unmasked points from the mean trend.
    stddev_m = np.sqrt(np.median((working_profile[inds] - wing_mean[inds])**2))

    # === Wing fit ===
    # Get fresh contaminated wing profile.
    working_profile = np.copy(profile)
    working_profile = np.log10(working_profile)
    # Mask first order core.
    working_profile[:int(ycens[0] + halfwidth)] = np.nan
    # Mask second and third orders.
    if contamination is not None:
        for order, ycen in enumerate(ycens):
            order += 1
            if order in contamination:
                if order == 3:
                    halfwidth -= 2
                if order == 2 and ycens[0] - ycens[1] < 5:
                    if (ycens[1]-0.5*halfwidth)-(ycens[0]+1.5*halfwidth) < 5:
                        start = int(np.max([ycen, 0]))
                    else:
                        start = np.max([ycen - halfwidth, 0]).astype(int)
                else:
                    start = np.max([ycen - 1.5*halfwidth, 0]).astype(int)
                end = np.max([ycen + 1.5*halfwidth, 1]).astype(int)
                # Set core of each order to NaN.
                working_profile[start:end] = np.nan
    halfwidth += 2
    # Find all outliers that are >3-sigma deviant from the mean.
    start = int(ycens[0] + 2.5*halfwidth)
    inds2 = np.where(
        np.abs(working_profile[start:] - wing_mean[start:]) > 3*stddev_m)
    # Mask outliers
    working_profile[start:][inds2] = np.nan
    # Mask edge of the detector.
    working_profile[-3:] = np.nan
    # Indices of all unmasked points in the contaminated wing.
    inds3 = np.isfinite(working_profile)

    # Fit with a 9th order polynomial.
    # To ensure that the polynomial does not start turning up in the padded
    # region, extend the linear fit to the edge of the pad to force the fit
    # to continue decreasing.
    ext_ax = np.arange(25) + np.max(axis[inds3]) + np.max([pad, 25])
    ext_prof = pp[1] + pp[0] * ext_ax
    # Concatenate right-hand profile with the extended linear trend.
    fit_ax = np.concatenate([axis[inds3], ext_ax])
    fit_prof = np.concatenate([working_profile[inds3], ext_prof])
    # Use np.polyfit for a first estimate of the coefficients.
    pp_r0 = np.polyfit(fit_ax, fit_prof, 9)
    # Robust fit using the polyfit results as a starting point.
    pp_r = utils.robust_polyfit(fit_ax, fit_prof, pp_r0)

    # === Stitching ===
    newprof = np.copy(profile)
    # Interpolate contaminated regions.
    if contamination is not None:
        for order in contamination:
            # Interpolate around the trace centroid.
            start = np.max([ycens[order - 1]-1.5*halfwidth, ycens[0]+halfwidth]).astype(int)
            if start >= dimy-1:
                # If order is off of the detector.
                continue
            end = np.min([ycens[order - 1]+1.5*halfwidth, dimy-1]).astype(int)
            # Join interpolations to the data.
            newprof = np.concatenate([newprof[:start],
                                      10**np.polyval(pp_r, axis)[start:end],
                                      newprof[end:]])

    # Interpolate nans and negatives with median of surrounding pixels.
    for pixel in np.where(np.isnan(newprof))[0]:
        minp = np.max([pixel-5, 0])
        maxp = np.min([pixel+5, dimy])
        newprof[pixel] = np.nanmedian(newprof[minp:maxp])

    if smooth is True:
        # Replace highly deviant pixels throughout the wings.
        wing_fit = np.polyval(pp_r, axis[int(ycens[0] + 2.5*halfwidth):])
        # Calculate the standard dev of unmasked points from the wing fit.
        stddev_f = np.sqrt(np.nanmedian((np.log10(newprof[int(ycens[0] + 2.5*halfwidth):])-wing_fit)**2))
        # Find all outliers that are >3-sigma deviant from the mean.
        inds4 = np.where(np.abs(np.log10(newprof[int(ycens[0] + 2.5*halfwidth):])-wing_fit) > 3*stddev_f)
        newprof[int(ycens[0] + 2.5*halfwidth):][inds4] = 10**wing_fit[inds4]

    # Add padding - padding values are constant at the median of edge pixels.
    padval_r = np.median(newprof[-8:-3])
    padval_l = np.median(newprof[3:8])
    newprof = np.concatenate([np.tile(padval_l, pad+4), newprof[4:-4],
                              np.tile(padval_r, pad+4)])

    # Do diagnostic plot if requested.
    if verbose == 3:
        plotting.plot_wing_reconstruction(profile, ycens, axis[inds3],
                                          working_profile[inds3], pp_r,
                                          newprof, pad, **kwargs)

    return newprof


def rescale_model(data, model, centroids, pad=0, verbose=0):
    """Rescale a column normalized trace model to the flux level of an actual
    observation. A multiplicative coefficient is determined via Chi^2
    minimization independently for each column, such that the rescaled model
    best matches the data.

    Parameters
    ----------
    data : np.array
        Observed dataframe.
    model : np.array
        Column normalized trace model.
    centroids : dict
        Centroids dictionary.
    pad : int
        Amount of padding on the spatial axis.
    verbose : int
        Level of verbosity.

    Returns
    -------
    model_rescale : np.array
        Trace model after rescaling.
    """

    # Determine first guess coefficients.
    k0 = np.nanmax(data, axis=0)
    ks = []
    # Loop over all columns - surely there is a more vectorized way to do this
    # whenever I try to minimize all of k at once I get nonsensical results?
    disable = utils.verbose_to_bool(verbose)
    for i in tqdm(range(data.shape[1]), disable=disable):
        # Get region around centroid
        ycen = int(round(centroids['order 1']['Y centroid'][i], 0)) + pad
        start = ycen - 13
        end = ycen + 14
        # Minimize the Chi^2.
        lik_args = (data[(start - pad):(end - pad), i], model[start:end, i])
        k = minimize(utils.lik, k0[i], lik_args)
        ks.append(k.x[0])

    # Rescale the column normalized model.
    ks = np.array(ks)
    model_rescale = ks * model

    return model_rescale


def get_box_weights(centroid, n_pix, shape, cols=None):
    """ Return the weights of a box aperture given the centroid and the width
    of the box in pixels. All pixels will have the same weights except at the
    ends of the box aperture.
    Copy of the same function in soss_boxextract.py of the jwst pipeline to
    circumvent package versioning issues...

    Parameters
    ----------
    centroid : array[float]
        Position of the centroid (in rows). Same shape as `cols`
    n_pix : float
        Width of the extraction box in pixels.
    shape : Tuple(int, int)
        Shape of the output image. (n_row, n_column)
    cols : array[int]
        Column indices of good columns. Used if the centroid is defined
        for specific columns or a sub-range of columns.
    Returns
    -------
    weights : array[float]
        An array of pixel weights to use with the box extraction.
    """

    nrows, ncols = shape

    # Use all columns if not specified
    if cols is None:
        cols = np.arange(ncols)

    # Row centers of all pixels.
    rows = np.indices((nrows, len(cols)))[0]

    # Pixels that are entierly inside the box are set to one.
    cond = (rows <= (centroid - 0.5 + n_pix / 2))
    cond &= ((centroid + 0.5 - n_pix / 2) <= rows)
    weights = cond.astype(float)

    # Fractional weights at the upper bound.
    cond = (centroid - 0.5 + n_pix / 2) < rows
    cond &= (rows < (centroid + 0.5 + n_pix / 2))
    weights[cond] = (centroid + n_pix / 2 - (rows - 0.5))[cond]

    # Fractional weights at the lower bound.
    cond = (rows < (centroid + 0.5 - n_pix / 2))
    cond &= ((centroid - 0.5 - n_pix / 2) < rows)
    weights[cond] = (rows + 0.5 - (centroid - n_pix / 2))[cond]

    # Return with the specified shape with zeros where the box is not defined.
    out = np.zeros(shape, dtype=float)
    out[:, cols] = weights

    return out


def read_interp_coefs(f277w=True, verbose=0):
    """Read the interpolation coefficients from the appropriate reference file.
    If the reference file does not exist, or the correct coefficients cannot be
    found, they will be recalculated.

    Parameters
    ----------
    f277w : bool
        If True, selects the coefficients with a 2.45µm red anchor.
    verbose : int
        Level of verbosity.

    Returns
    -------
    coef_b : np.array
        Blue anchor coefficients.
    coef_r : np.array
        Red anchor coefficients.
    """

    # Attempt to read interpolation coefficients from reference file.
    try:
        df = pd.read_csv('Ref_files/interpolation_coefficients.csv')
        # If there is an F277W exposure, get the coefficients to 2.45µm.
        if f277w is True:
            coef_b = np.array(df['F_blue'])
            coef_r = np.array(df['F_red'])
        # For no F277W exposure, get the coefficients out to 2.9µm.
        else:
            coef_b = np.array(df['NF_blue'])
            coef_r = np.array(df['NF_red'])
    # If the reference file does not exists, or the appropriate coefficients
    # have not yet been generated, call the _calc_interp_coefs function to
    #  calculate them.
    except (FileNotFoundError, KeyError):
        print('No interpolation coefficients found. They will be calculated now.')
        coef_b, coef_r = _calibrations.calc_interp_coefs(f277w=f277w,
                                                         verbose=verbose)

    return coef_b, coef_r


def read_width_coefs(verbose=0):
    """Read the width coefficients from the appropriate reference file.
    If the reference file does not exist, the coefficients will be
    recalculated.

    Parameters
    ----------
    verbose : int
        Level of verbosity.

    Returns
    -------
    wc : np.array
        Width calbration polynomial coefficients.
    """

    # First try to read the width calibration file, if it exists.
    try:
        coef_file = pd.read_csv('Ref_files/width_coefficients.csv')
        wc = np.array(coef_file['width_coefs'])
    # If file does not exist, redo the width calibration.
    except FileNotFoundError:
        print('No width coefficients found. They will be calculated now.')
        wc = _calibrations.derive_width_relations(verbose=verbose)

    return wc


def robust_polyfit(x, y, p0):
    """Wrapper around scipy's least_squares fitting routine implementing the
     Huber loss function - to be more resistant to outliers.

    Parameters
    ----------
    x : list, np.array
        Data describing dependant variable.
    y : list, np.array
        Data describing independent variable.
    p0 : tuple
        Initial guess straight line parameters. The length of p0 determines the
        polynomial order to be fit - i.e. a length 2 tuple will fit a 1st order
        polynomial, etc.

    Returns
    -------
    res.x : list
        Best fitting parameters of the desired polynomial order.
    """

    # Preform outlier resistant fitting.
    res = least_squares(_poly_res, p0, loss='huber', f_scale=0.1, args=(x, y))
    return res.x


def sigma_clip(xdata, ydata, thresh=5):
    """Perform rough sigma clipping on data to remove outliers.

    Parameters
    ----------
    xdata : list, np.array
        Independent variable.
    ydata : list, np.array
        Dependent variable.
    thresh : int
        Sigma threshold at which to clip.

    Returns
    -------
    xdata : np.array
        Independent variable; sigma clipped.
    ydata : np.array
        Dependent variable; sigma clipped.
    """

    xdata, ydata = np.atleast_1d(xdata), np.atleast_1d(ydata)
    # Get mean and standard deviation.
    mean = np.mean(ydata)
    std = np.std(ydata)
    # Points which are >thresh-sigma deviant.
    inds = np.where(np.abs(ydata - mean) < thresh*std)

    return xdata[inds], ydata[inds]


def lik(k, data, model):
    """Utility likelihood function for flux rescaling. Essentially a Chi^2
    multiplied by the data such that wing values don't carry too much weight.
    """
    return np.nansum((data - k*model)**2)


def local_mean(array, step):
    """Calculate the mean of an array in chunks of 2*step.
    """
    running_means = []
    for i in range(-step, step):
        if i == 0:
            continue
        running_means.append(np.roll(array, i))
    loc_mean = np.mean(running_means, axis=0)

    return loc_mean


def _poly_res(p, x, y):
    """Residuals from a polynomial.
    """
    return np.polyval(p, x) - y


def plot_interpmodel(waves, nw1, nw2, p1, p2):
    """Plot the diagnostic results of the derive_model function. Four plots
    are generated, showing the normalized interpolation coefficients for the
    blue and red anchors for each WFE realization, as well as the mean trend
    across WFE for each anchor profile, and the resulting polynomial fit to
    the mean trends.

    Parameters
    ----------
    waves : np.array of float
        Wavelengths at which WebbPSF monochromatic PSFs were created.
    nw1 : np.array of float
        Normalized interpolation coefficient for the blue anchor
        for each PSF profile.
    nw2 : np.array of float
        Normalized interpolation coefficient for the red anchor
        for each PSF profile.
    p1 : np.array of float
        Polynomial coefficients of the fit to the mean interpolation
        coefficients for the blue anchor.
    p2 : np.array of float
        Polynomial coefficients of the fit to the mean interpolation
        coefficients for the red anchor.
    """

    f, ax = plt.subplots(2, 2, figsize=(14, 6))
    for i in range(10):
        ax[0, 0].plot(waves, nw1[i])
        ax[1, 0].plot(waves, nw2[i])

    ax[0, 1].plot(waves, np.mean(nw1, axis=0))
    ax[0, 1].plot(waves, np.mean(nw2, axis=0))

    ax[-1, 0].set_xlabel('Wavelength [µm]', fontsize=14)
    ax[-1, 1].set_xlabel('Wavelength [µm]', fontsize=14)

    y1 = np.polyval(p1, waves)
    y2 = np.polyval(p2, waves)

    ax[1, 1].plot(waves, y1, c='r', ls=':')
    ax[1, 1].plot(waves, y2, c='b', ls=':')
    ax[1, 1].plot(waves, np.mean(nw1, axis=0), c='b', label='Blue Anchor')
    ax[1, 1].plot(waves, np.mean(nw2, axis=0), c='r', label='Red Anchor')
    ax[1, 1].set_xlim(np.min(waves), np.max(waves))
    ax[1, 1].legend(loc=1, fontsize=12)

    f.tight_layout()
    plt.show()


def plot_width_relation(wave_range, widths, wfit, ii):
    """Do diagnostic plot for trace width calibrations.
    """

    ax = np.linspace(wave_range[0], wave_range[-1], 100)
    plt.figure(figsize=(8, 5))
    plt.scatter(wave_range[ii], widths[ii], c='black', alpha=0.5,
                label='Measured Widths')
    plt.plot(ax, np.polyval(wfit, ax), label='Width Fit', c='red')

    plt.xlabel('Spectral Pixel', fontsize=14)
    plt.ylabel('Relative Spatial Profile Width', fontsize=14)
    plt.legend()
    plt.show()


def plot_wing_reconstruction(profile, ycens, axis_r, prof_r2, pp_r, newprof,
                             pad, text=None):
    """Do diagnostic plotting for wing reconstruction.
    """

    dimy = len(profile)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(dimy), np.log10(profile), ls=':', c='black',
             label='original profile')
    for ycen in ycens:
        plt.axvline(ycen, ls=':', c='grey')
    plt.scatter(axis_r, prof_r2, c='orange', s=15, label='unmasked points')
    ax_tot = np.arange(dimy+2*pad) - pad
    plt.plot(ax_tot, np.log10(newprof), c='blue', alpha=1,
             label='reconstructed profile',)
    plt.plot(ax_tot[(ycens[0]+18+pad):-(pad+4)],
             np.polyval(pp_r, ax_tot[(ycens[0]+18+pad):-(pad+4)]), c='red',
             lw=2, ls='--', label='right wing fit')
    if text is not None:
        plt.text(ax_tot[5], np.min(np.log10(newprof)), text, fontsize=14)

    plt.xlabel('Spatial Pixel', fontsize=12)
    plt.xlim(int(ax_tot[0]), int(ax_tot[-1]))
    plt.legend(fontsize=12)
    plt.show()


def plot_f277_rescale(f277_init, f277_rescale, clear_prof):
    """Do diagnoostic plot for F277W rescaling.
    """

    plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0])
    ax1.plot(clear_prof / np.nansum(clear_prof), c='blue')
    ax1.plot(f277_init, c='red', ls='--')
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    ax2 = plt.subplot(gs[1])
    ax2.plot(clear_prof / np.nansum(clear_prof), c='blue')
    ax2.plot(f277_rescale, c='red', ls='--')
    ax2.set_xlabel('Spatial Pixel', fontsize=14)

    plt.subplots_adjust(hspace=0)
    plt.show()


def construct_order1(clear, f277w, ycens, subarray, pad=0, verbose=0):
    """This creates the full order 1 trace profile model. The region
    contaminated by the second order is interpolated from the CLEAR and F277W
    exposures.
    The steps are as follows:
        1. Determine the red and blue anchor profiles for the interpolation.
        2. Construct the interpolated profile at each wavelength in the
           overlap region.
        3. Stitch together the original CLEAR exposure and the interpolated
           trace model (as well as the F277W exposure if provided).
        4. Mask the contamination from the second and third orders, and
           reconstruct the underlying wing structure of the first order -
           including any padding in the spatial direction.

    Parameters
    ----------
    clear : np.array
        NIRISS SOSS CLEAR exposure dataframe.
    f277w : np.array
        NIRISS SOSS F277W filter exposure dataframe.
    ycens : dict
        Dictionary of Y-coordinates for the trace centroids of the first three
        diffraction orders, ie. as returned by get_soss_centroids.
    subarray : str
        Subarray identifier. "SUBSTRIP256", or "FULL".
    pad : int
        Number of pixels of padding to add on both ends of the spatial axis.
    verbose : int
        Level of verbosity.

    Returns
    -------
    o1frame : np.array
        Interpolated order 1 trace model with padding.
    """

    # Get subarray dimensions
    dimx = 2048
    if subarray == 'SUBSTRIP256':
        dimy = 256
    elif subarray == 'FULL':
        dimy = 2048

    # ========= INITIAL SETUP =========
    # Get wavelengh calibration.
    wavecal_x, wavecal_w = utils.get_wave_solution(order=1)
    # Determine how the trace width changes with wavelength.
    if verbose != 0:
        print('   Calibrating trace widths...')
    width_polys = utils.read_width_coefs(verbose=verbose)

    # ========= GET ANCHOR PROFILES =========
    if verbose != 0:
        print('   Getting anchor profiles...')
    # Determine the anchor profiles - blue anchor.
    # Find the pixel position of 2.1µm.
    i_b = np.where(wavecal_w >= 2.1)[0][-1]
    xdb = int(wavecal_x[i_b])
    ydb = ycens['order 1']['Y centroid'][xdb]
    # Extract the 2.1µm anchor profile from the data - take median profile of
    # neighbouring five columns to mitigate effects of outliers.
    bl_anch = np.median(clear[:, (xdb - 2):(xdb + 2)], axis=1)

    # Mask second and third order, reconstruct wing structure and pad.
    cens = [ycens['order 1']['Y centroid'][xdb],
            ycens['order 2']['Y centroid'][xdb],
            ycens['order 3']['Y centroid'][xdb]]
    bl_anch = applesoss.reconstruct_wings256(bl_anch, ycens=cens,
                                             contamination=[2, 3], pad=pad,
                                             verbose=verbose, smooth=True,
                                             **{'text': 'Blue anchor'})
    # Remove the chromatic scaling.
    bl_anch = applesoss.chromescale(bl_anch, 2.1, 2.5, ydb + pad, width_polys)
    # Normalize
    bl_anch /= np.nansum(bl_anch)

    # Determine the anchor profiles - red anchor.
    # If an F277W exposure is provided, only interpolate out to 2.45µm.
    # Red-wards of 2.45µm we have perfect knowledge of the order 1 trace.
    # Find the pixel position of 2.45µm.
    i_r = np.where(wavecal_w >= 2.45)[0][-1]
    xdr = int(wavecal_x[i_r])
    ydr = ycens['order 1']['Y centroid'][xdr]

    # Extract and rescale the 2.45µm profile - take median of neighbouring
    # five columns to mitigate effects of outliers.
    rd_anch = np.median(f277w[:, (xdr - 2):(xdr + 2)], axis=1)
    # Reconstruct wing structure and pad.
    cens = [ycens['order 1']['Y centroid'][xdr]]
    rd_anch = applesoss.reconstruct_wings256(rd_anch, ycens=cens,
                                             contamination=None, pad=pad,
                                             verbose=verbose, smooth=True,
                                             **{'text': 'Red anchor'})
    rd_anch = applesoss.chromescale(rd_anch, 2.45, 2.5, ydr + pad, width_polys)
    # Normalize
    rd_anch /= np.nansum(rd_anch)

    # Interpolation polynomial coefs, calculated via calc_interp_coefs
    coef_b, coef_r = utils.read_interp_coefs(verbose=verbose)

    # Since there will lkely be a different number of integrations for the
    # F277W exposure vs the CLEAR, the flux levels of the two anchors will
    # likely be different. Rescale the F277W anchor to match the CLEAR level.
    rd_anch = rescale_f277(rd_anch, clear[:, xdr], ycen=ydr, pad=pad,
                           verbose=verbose)

    # ========= INTERPOLATE THE CONTAMINATED REGION =========
    # Create the interpolated order 1 PSF.
    map2d = np.zeros((dimy + 2 * pad, dimx)) * np.nan
    # Get centroid pixel coordinates and wavelengths for interpolation region.
    cenx_d = np.arange(xdb - xdr).astype(int) + xdr
    ceny_d = ycens['order 1']['Y centroid'][cenx_d]
    lmbda = wavecal_w[cenx_d]

    # Create an interpolated 1D PSF at each required position.
    if verbose != 0:
        print('   Interpolating trace...', flush=True)
    disable = utils.verbose_to_bool(verbose)
    for i, vals in tqdm(enumerate(zip(cenx_d, ceny_d, lmbda)),
                        total=len(lmbda), disable=disable):
        cenx, ceny, lbd = int(round(vals[0], 0)), vals[1], vals[2]
        # Evaluate the interpolation polynomials at the current wavelength.
        wb_i = np.polyval(coef_b, lbd)
        wr_i = np.polyval(coef_r, lbd)
        # Recenter the profile of both anchors on the correct Y-centroid.
        bax = np.arange(dimy + 2 * pad) - ydb + ceny
        bl_anch_i = np.interp(np.arange(dimy + 2 * pad), bax, bl_anch)
        rax = np.arange(len(rd_anch)) - ydr + ceny
        rd_anch_i = np.interp(np.arange(dimy + 2 * pad), rax, rd_anch)
        # Construct the interpolated profile.
        prof_int = (wb_i * bl_anch_i + wr_i * rd_anch_i)
        # Re-add the lambda/D scaling.
        prof_int_cs = applesoss.chromescale(prof_int, 2.5, lbd, ceny + pad,
                                            width_polys)
        # Put the interpolated profile on the detector.
        map2d[:, cenx] = prof_int_cs

        # Note detector coordinates of the edges of the interpolated region.
        bl_end = cenx
        if i == 0:
            # 2.9µm (i=0) limit will be off the end of the detector.
            rd_end = cenx

    # ========= RECONSTRUCT THE FIRST ORDER WINGS =========
    if verbose != 0:
        print('   Stitching data and reconstructing wings...', flush=True)
    # Stitch together the interpolation and data.
    o1frame = np.zeros((dimy + 2 * pad, dimx))
    # Insert interpolated data.
    o1frame[:, rd_end:bl_end] = map2d[:, rd_end:bl_end]
    # Bluer region is known from the CLEAR exposure.
    disable = utils.verbose_to_bool(verbose)
    for col in tqdm(range(bl_end, dimx), disable=disable):
        cens = [ycens['order 1']['Y centroid'][col],
                ycens['order 2']['Y centroid'][col],
                ycens['order 3']['Y centroid'][col]]
        # Mask contamination from second and third orders, reconstruct
        # wings and add padding.
        o1frame[:, col] = applesoss.reconstruct_wings256(clear[:, col],
                                                         ycens=cens, pad=pad,
                                                         smooth=True)
    # Add on the F277W frame to the red of the model.
    disable = utils.verbose_to_bool(verbose)
    for col in tqdm(range(rd_end), disable=disable):
        cens = [ycens['order 1']['Y centroid'][col]]
        o1frame[:, col] = applesoss.reconstruct_wings256(f277w[:, col],
                                                         ycens=cens,
                                                         contamination=None,
                                                         pad=pad, smooth=True)

        o1frame[:, col] = rescale_f277(o1frame[:, col], clear[:, col],
                                       ycen=cens[0], verbose=0)

    # Column normalize - necessary for uniformity as anchor profiles are
    # normalized whereas stitched data is not.
    o1frame /= np.nansum(o1frame, axis=0)

    return o1frame


def rescale_f277(f277_prof, clear_prof, ycen=71, width=50, max_iter=10, pad=0,
                 verbose=0):
    """As the F277W and CLEAR exposures will likely have different flux levels
    due to different throughputs as well as possibly a different number of
    integrations. Rescale the F277W profile to the flux level of the CLEAR
    profile such that both profiles can be used as end points for interplation.

    Parameters
    ----------
    f277_prof : np.array
        Spatial profile in the F277W fiter.
    clear_prof : np.array
        Spatial profile in the CLEAR filter.
    ycen : float
        Y-Centroid position of the F277W profile.
    width : int
        Pixel width on either end of the centroid to consider.
    max_iter : int
        Maximum number of iterations.
    pad : int
        Amount of padding included in the spatial profiles.
    verbose : int
        Level of verbosity

    Returns
    -------
    f277_rescale : np.array
        F277W profile rescaled to the flux level of the CLEAR profile.
    """

    dimy = len(f277_prof)
    start, end = int(ycen - width), int(ycen + width)
    # Iterate over different rescalings + vertical shifts to minimize the
    # Chi^2 between the F277W and CLEAR exposures.
    # Calculate starting Chi^2 value.
    chi2 = np.nansum(((f277_prof[pad:(dimy + pad)] /
                       np.nansum(f277_prof[pad:(dimy + pad)]))[start:end] -
                      (clear_prof / np.nansum(clear_prof))[start:end]) ** 2)
    # Append Chi^2 and normalize F277W profile to arrays.
    chi2_arr = [chi2]
    anchor_arr = [f277_prof / np.nansum(f277_prof)]

    niter = 0
    while niter < max_iter:
        # Subtract an offset so the floors of the two profilels match.
        offset = np.nanpercentile(clear_prof / np.nansum(clear_prof), 1) - \
                 np.nanpercentile(f277_prof / np.nansum(f277_prof), 1)

        # Normalize offset F277W.
        f277_prof = f277_prof / np.nansum(f277_prof) + offset

        # Calculate new Chi^2.
        chi2 = np.nansum(((f277_prof[pad:(dimy + pad)] /
                           np.nansum(f277_prof[pad:(dimy + pad)]))[start:end] -
                          (clear_prof / np.nansum(clear_prof))[
                          start:end]) ** 2)
        chi2_arr.append(chi2)
        anchor_arr.append(f277_prof / np.nansum(f277_prof))

        niter += 1

    # Keep the best fitting F277W profile.
    min_chi2 = np.argmin(chi2_arr)
    f277_rescale = anchor_arr[min_chi2]

    if verbose == 3:
        plotting.plot_f277_rescale(anchor_arr[0][pad:(dimy + pad)],
                                   f277_rescale[pad:(dimy + pad)],
                                   clear_prof)

    return f277_rescale


def calc_interp_coefs(f277w=True, verbose=0):
    """Function to calculate the interpolation coefficients necessary to
    construct a monochromatic PSF profile at any wavelength between
    the two 1D PSF anchor profiles. Linear combinations of the blue and red
    anchor are iteratively fit to each intermediate wavelength to find the
    best fitting combination. The mean linear coefficients across the 10 WFE
    error realizations are returned for each wavelengths.
    When called, 2D monochromatic PSF profiles will be generated and saved
    to disk if the user does not already have them available.
    This should not need to be called by the end user except in rare cases.

    Parameters
    ----------
    f277w : bool
        Set to False if no F277W exposure is available for the observation.
        Finds coefficients for the entire 2.1 - 2.9µm region in this case.
    verbose : int
        Level of verbosity.

    Returns
    -------
    pb : np.array of float
        Polynomial coefficients of the interpolation index fits for the
        blue anchor.
    pr : np.array of float
        Polynomial coefficients of the interpolation index fits for the
        red anchor.
    """

    if verbose != 0:
        print('Calculating interpolation coefficients.')
    # Red anchor is 2.9µm without an F277W exposure.
    if f277w is False:
        wave_range = np.linspace(2.1, 2.9, 7)
    # Red anchor is 2.45µm with F277W exposure.
    else:
        wave_range = np.linspace(2.1, 2.45, 7)

    # Read in monochromatic PSFs generated by WebbPSF.
    psf_list = []
    # Loop over all 10 available WFE realizations.
    for i in range(10):
        psf_run = []
        # Import the PSFs,
        for w in wave_range:
            # If the user already has the PSFs generated, import them.
            infile = 'Ref_files/SOSS_PSFs/SOSS_os10_128x128_{0:.6f}_{1:.0f}.fits'.format(w, i)
            try:
                psf_run.append(fits.getdata(infile, 0))
            # Generate missing PSFs if necessary.
            except FileNotFoundError:
                errmsg = ' No monochromatic PSF found for {0:.2f}µm and WFE '\
                         'realization {1:.0f}. Creating it now.'.format(w, i)
                print(errmsg)
                loicpsf(wavelist=[w*1e-6], wfe_real=i, verbose=False)
                psf_run.append(fits.open(infile)[0].data)
        psf_list.append(psf_run)

    # Determine specific interpolation coefficients for all WFEs
    wb, wr = [], []
    for E in range(10):
        # Generate the blue wavelength anchor.
        # The width of the 1D PSF has lambda/D dependence, so rescale all
        # profiles to a common wavelength to remove these chromatic effects.
        rngeb = np.linspace(0, round(1280*(2.5/2.1), 0) - 1, 1280)
        offsetb = rngeb[640] - 640
        newb = np.interp(np.arange(1280), rngeb - offsetb,
                         np.sum(psf_list[E][0][600:700, :], axis=0))

        # Generate the red wavelength anchor.
        if f277w is False:
            # Use 2.85µm for CLEAR.
            rnger = np.linspace(0, round(1280*(2.5/2.9), 0) - 1, 1280)
        else:
            # Or 2.42µm for F277W.
            rnger = np.linspace(0, round(1280*(2.5/2.45), 0) - 1, 1280)
        offsetr = rnger[640] - 640
        # Remove lambda/D scaling.
        newr = np.interp(np.arange(1280), rnger - offsetr,
                         np.sum(psf_list[E][6][600:700, :], axis=0))

        # Loop over all monochromatic PSFs to determine interpolation
        # coefficients.
        for f, wave in enumerate(wave_range):
            # Lists for running counts of indices and model residuals.
            resid, ind_i, ind_j = [], [], []

            # Rescale monochromatic PSF to remove lambda/D.
            newrnge = np.linspace(0, round(1280*(2.5/wave), 0) - 1, 1280)
            newoffset = newrnge[640] - 640
            newpsf = np.interp(np.arange(1280), newrnge - newoffset,
                               np.sum(psf_list[E][f][600:700, :], axis=0))

            # Brute force the coefficient determination.
            for i in range(1, 100):
                for j in range(1, 100):
                    i /= 10
                    j /= 10
                    # Create a test profile which is a mix of the two anchors.
                    mix = (i*newb + j*newr) / (i + j)

                    # Save current iteration in running counts.
                    resid.append(np.sum(np.abs(newpsf - mix)[450:820]))
                    ind_i.append(i)
                    ind_j.append(j)

            # Determine which combination of indices minimizes model residuals.
            ind = np.where(resid == np.min(resid))[0][0]
            # Save the normalized indices for this wavelength.
            wb.append(ind_i[ind] / (ind_i[ind] + ind_j[ind]))
            wr.append(ind_j[ind] / (ind_i[ind] + ind_j[ind]))

    wb = np.reshape(wb, (10, 7))
    wr = np.reshape(wr, (10, 7))

    # Fit a second order polynomial to the mean of the interpolation indices.
    pb = np.polyfit(wave_range, np.mean(wb, axis=0), 2)
    pr = np.polyfit(wave_range, np.mean(wr, axis=0), 2)

    # Show the diagnostic plot if necessary.
    if verbose == 3:
        plotting.plot_interpmodel(wave_range, wb, wr, pb, pr)

    # Save the coefficients to disk so that they can be accessed by the
    # empirical trace construction module.
    try:
        df = pd.read_csv('Ref_files/interpolation_coefficients.csv')
    except FileNotFoundError:
        # If the interpolation coefficients file does not already exist, create
        # a new dictionary.
        df = {}
    # Replace the data for F277W or no F277W depending on which was run.
    if f277w is True:
        df['F_red'] = pr
        df['F_blue'] = pb
    else:
        df['NF_red'] = pr
        df['NF_blue'] = pb
    # Write to file.
    df = pd.DataFrame(data=df)
    df.to_csv('Ref_files/interpolation_coefficients.csv', index=False)

    return pb, pr


def derive_width_relations(no_wave=25, wave_start=0.6, wave_end=2.9,
                           verbose=0):
    """Due to the defocusing of the SOSS PSF, its width does not simply evolve
    as it would if diffraction limited. Use the WebbPSF package to simulate
    SOSS PSFs at different wavelengths, and fit a Gaussian profile o each PSF
    to estimate the evolution in the widths.
    When called, 2D monochromatic PSF profiles will be generated and saved
    to disk if the user does not already have them available.
    This should not need to be called by the end user except in rare cases.

    Parameters
    ----------
    no_wave : int
        Number of wavelengths to include.
    wave_start : float
        Smallest wavelength to consider in microns.
    wave_end : float
       vLargest wavelength to consider in microns.
    verbose : int
        Level of verbosity

    Returns
    -------
    wfit : np.array
        Polynomial coefficients describing how the spatial profile width
        changes with wavelength.
    """

    # Get wavelength range to consider
    wave_range = np.linspace(wave_start, wave_end, no_wave)
    psfs = []
    # Generate PSFs for each wavelength.
    for w in wave_range:
        # If the user already has the PSFs generated, import them.
        infile = 'Ref_files/SOSS_PSFs/SOSS_os10_128x128_{0:.6f}_{1:.0f}.fits'.format(w, 0)
        try:
            psf = fits.getdata(infile, 0)
        # Generate missing PSFs if necessary.
        except FileNotFoundError:
            errmsg = ' No monochromatic PSF found for {0:.2f}µm and WFE ' \
                     'realization {1:.0f}. Creating it now.'.format(w, 0)
            print(errmsg)
            loicpsf(wavelist=[w * 1e-6], wfe_real=0, verbose=False)
            psf = fits.getdata(infile, 0)
        psfs.append(np.sum(psf[600:700, :], axis=0))
    psfs = np.array(psfs)

    # Define the gaussian model.
    def gauss(x, *p):
        amp, mu, sigma = p
        return amp * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))
    # Initial guess parameters
    p0 = [0.01, 600., 200.]

    # Fit the Gaussian to the PSF at each wavelength and save the width.
    widths = []
    for i in range(len(wave_range)):
        coeff, var_matrix = curve_fit(gauss, np.arange(1280), psfs[i], p0=p0)
        widths.append(np.abs(coeff[2]))
    widths = np.array(widths)

    # Normalize widths to their max value, and remove spurious smalll-valued
    # outliers.
    widths /= np.nanmax(widths)
    ii = np.where(widths > 0.9)

    # Fit a polynomial to the widths.
    pp_init = np.polyfit(wave_range[ii], widths[ii], 5)
    wfit = utils.robust_polyfit(wave_range[ii], widths[ii], pp_init)

    # Do debugging plot.
    if verbose == 3:
        plotting.plot_width_relation(wave_range, widths, wfit, ii)

    # Save the polynomial ceofficients to file so that this doesn't need to be
    # repeated.
    df = {'width_coefs': wfit}
    df = pd.DataFrame(data=df)
    df.to_csv('Ref_files/width_coefficients.csv', index=False)

    return wfit


def loicpsf(wavelist=None, wfe_real=None, save_to_disk=True, oversampling=10,
            pixel=128, verbose=True):
    """Calls the WebbPSF package to create monochromatic PSFs for NIRISS
    SOSS observations and save them to disk.

    Parameters
    ----------
    wavelist : list
        List of wavelengths (in meters) for which to generate PSFs.
    wfe_real : int
        Index of wavefront realization to use for the PSF (if non-default
        WFE realization is desired).
    save_to_disk : bool
        Whether to save PSFs to disk, or return them from the function.
    oversampling : int
        Oversampling pixels scale for the PSF.
    pixel : int
        Width of the PSF in native pixels.
    verbose : bool
        Whether to print explanatory comments.

    Returns
    -------
    psf_list : list
        If save_to_disk is False, a list of the generated PSFs.
    """

    # Create PSF storage array.
    psf_list = []

    if wavelist is None:
        # List of wavelengths to generate PSFs for
        wavelist = np.linspace(0.5, 5.2, 95) * 1e-6

    # Select the NIRISS instrument
    niriss = webbpsf.NIRISS()

    # Override the default minimum wavelength of 0.6 microns
    niriss.SHORT_WAVELENGTH_MIN = 0.5e-6
    # Set correct filter and pupil wheel components
    niriss.filter = 'CLEAR'
    niriss.pupil_mask = 'GR700XD'

    # Change the WFE realization if desired
    if wfe_real is not None:
        niriss.pupilopd = ('OPD_RevW_ote_for_NIRISS_predicted.fits.gz',
                           wfe_real)

    # Loop through all wavelengths to generate PSFs
    first_time = True
    for wave in wavelist:
        if verbose is True:
            print('Calculating PSF at wavelength ',
                  round(wave/1e-6, 2), ' microns')
        psf = niriss.calc_psf(monochromatic=wave, fov_pixels=pixel,
                              oversample=oversampling, display=False)
        psf_list.append(psf)

        if save_to_disk is True:
            filepath = 'Ref_files/SOSS_PSFs/'
            if first_time is True:
                # PSFs will be saved to a SOSS_PSFs directory. If it does not
                # already exist, create it.
                if os.path.exists(filepath):
                    pass
                else:
                    os.mkdir(filepath)
                first_time = False
            # Save psf realization to disk
            text = '{0:5f}'.format(wave*1e+6)
            fpars = [oversampling, pixel, text, wfe_real]
            outfile = filepath+'SOSS_os{0}_{1}x{1}_{2}_{3}.fits'.format(*fpars)
            psf.writeto(outfile, overwrite=True)

    if save_to_disk is False:
        return psf_list

def pad_orders2_and_3(dataframe, cen, pad, order):
    """Add padding to the spatial axis of an order 2 or 3 dataframe. Since
    these orders curve almost vertically at short wavelengths, we must take
    special care to properly extend the spatial profile

    Parameters
    ----------
    dataframe : array-like
        A dataframe of order 2 or 3.
    cen : dict
        Centroids dictionary.
    pad : int
        Amount of padding to add to the spatial axis.
    order : int
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
