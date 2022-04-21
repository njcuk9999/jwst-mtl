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