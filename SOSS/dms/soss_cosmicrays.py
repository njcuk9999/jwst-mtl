from astropy.io import fits, ascii
import sys
import os
#sys.path.insert(1, '/Users/albert/NIRISS/SOSSpipeline/jwst-mtl/')
from jwst import datamodels
from astropy.io import fits
import numpy as np

def mediandev(x, axis=None):
    med = np.nanmedian(x, axis=axis)

    return np.nanmedian(np.abs(x - med), axis=axis) / 0.67449

def cosmicrays_action(ramp, nsig=5, correction_noisy=True,
                      integration_to_correct=None, save_diagnostics=False):
    # integration_to_correct:
    # should either be None or a scalar integer corresponding to the
    # integration index
    if integration_to_correct is None:
        apply_to_all_integrations = True
        print('cosmicrays_action - apply_to_all_integrations is True')
    else:
        apply_to_all_integrations = False
        currentint = integration_to_correct
        print('cosmicrays_action - apply_to_all_integrations is False')

    # apply_to_all_integrations
    # This key determines if the cosmic ray correction is applied to only the
    # central integration or to all integrations. To have the effect of a
    # running correction, False should be used and the function called at each
    # integration step.

    # ramp = np.copy(inputramp)
    nint, ng, dimy, dimx = np.shape(ramp)
    # print(nint, ng, dimy, dimx)

    ## Create a CDS cube corresponding to the ramp model
    # cds = np.zeros((nint, ng-1, dimy, dimx))
    # for i in range(nint):
    #    cds[i,:,:,:] = ramp[i,1:,:,:] - ramp[i,0:-1,:,:]

    # random number generator
    rng = np.random.default_rng()

    # Compute the median of each group across the integrations
    ramp_med = np.median(ramp, axis=0)
    ramp_mdv = mediandev(ramp, axis=0)

    diframp = ramp - ramp_med
    # hdu = fits.PrimaryHDU(difsci)
    # hdu.writeto('difsci.fits', overwrite=True)

    # difference from median expressed in units of mediandev error
    diframpnsig = (ramp - ramp_med) / ramp_mdv
    # hdu = fits.PrimaryHDU(difscinsig)
    # hdu.writeto('difscinsig.fits', overwrite=True)

    # scan the first group of each integration to spot positive outliers
    # and use the flux to correct the first cds.
    print('cosmicrays_action - Correct first group')
    # firstgroup = ramp[:,0,:,:]
    # print(firstimage.shape)
    # outliers = (diframpnsig[:,0,:,:] > nsig) | (diframpnsig[:,0,:,:] < nsig)
    # print(outliers.shape)

    if apply_to_all_integrations is True:
        firstgroup_correction = np.zeros((nint, dimy, dimx))
        for i in range(nint):
            firstgroup = ramp[i, 0, :, :]
            # print(firstimage.shape)
            outliers = diframpnsig[i, 0, :, :] > nsig
            # print(outliers.shape)
            # print(outliers)
            # Here is therefore the correction to the first group image
            # Add also noise to that correction to mimick the group image noise
            firstgroup_correction[i][outliers] = firstgroup[outliers] \
                                                 - ramp_med[0][outliers] \
                                                 + rng.normal(size=outliers.sum()) * ramp_mdv[0][outliers]
        # TODO: Change above loop for this and test if it works:
        # for i in range(nint):
        #    outliers = diframpnsig[i,0,:,:] > nsig
        #    # Here is therefore the correction to the first group image
        #    # Add also noise to that correction to mimick the group image noise
        #    firstgroup_correction[i][outliers] = ramp[i,0,:,:][outliers] \
        #               - ramp_med[0][outliers] \
        #               + rng.normal(size=outliers.sum()) * ramp_mdv[0][outliers]

        # hdu = fits.PrimaryHDU(firstimage_correction)
        # hdu.writeto('firstimage_correction.fits', overwrite=True)

        # Apply correction to all subsequent groups in the ramp
        for g in range(ng):
            ramp[:, g, :, :] = ramp[:, g, :, :] - firstgroup_correction[:, :, :]

        # hdu = fits.PrimaryHDU(sci)
        # hdu.writeto('sci_im1iteration.fits', overwrite=True)

        # Now that the first group image is corrected, compute
        # the g - (g-1) cds and repeat the correction on the gth group image.
        for n in range(ng - 1):
            g = n + 1
            print('cosmicrays_action - Correct group {:}'.format(g + 1))
            cdsgg1 = np.zeros((nint, dimy, dimx))
            for i in range(nint):
                cdsgg1[i, :, :] = ramp[i, g, :, :] - ramp[i, g - 1, :, :]
            cdsgg1_med = np.median(cdsgg1, axis=0)
            cdsgg1_mdv = mediandev(cdsgg1, axis=0)

            diframpnsig = (cdsgg1 - cdsgg1_med) / cdsgg1_mdv
            if save_diagnostics:
                hdu = fits.PrimaryHDU(diframpnsig)
                hdu.writeto(outdir+'/diframpnsig_cdsgg1.fits', overwrite=True)

                # gth_group = cdsgg1[:,:,:]
            # print(gth_group.shape)
            # outliers = diframpnsig[:,:,:] > nsig
            # print(outliers.shape)

            gth_group_correction = np.zeros((nint, dimy, dimx))
            for i in range(nint):
                gth_group = cdsgg1[i, :, :]
                # print(firstimage.shape)
                outliers = diframpnsig[i, :, :] > nsig
                # print(outliers.shape)
                # print(outliers)
                # gth_group_correction[i][outliers] = gth_group[outliers]
                # Here is therefore the correction to the first sci image
                # Add also noise to that correction to mimick the sci image noise
                gth_group_correction[i][outliers] = gth_group[outliers] - cdsgg1_med[outliers] + rng.normal(
                    size=outliers.sum()) * cdsgg1_mdv[outliers]

            if save_diagnostics:
                hdu = fits.PrimaryHDU(gth_group_correction)
                hdu.writeto(outdir+'/gth_group_correction.fits', overwrite=True)

                # Apply the correction to all subsequent groups of the ramp
            for m in range(ng - g):
                ramp[:, m + g, :, :] = ramp[:, m + g, :, :] - gth_group_correction[:, :, :]

            # hdu = fits.PrimaryHDU(sci)
            # hdu.writeto('sci_im_gth_iteration.fits', overwrite=True)

    else:  # apply correction to only the current integration

        nint, ng, dimy, dimx = np.shape(ramp)

        # random number generator
        rng = np.random.default_rng()

        # Compute the median of each group across the integrations, rejecting
        # the current integration
        rampmasked = ramp * 1
        rampmasked[currentint, :, :, :] = np.nan
        ramp_med = np.nanmedian(rampmasked, axis=0)
        ramp_mdv = mediandev(rampmasked, axis=0)

        # difference from median expressed in units of mediandev error
        diframpnsig = (ramp - ramp_med) / ramp_mdv

        firstgroup_correction = np.zeros((dimy, dimx))
        firstgroup = ramp[currentint, 0, :, :] * 1

        outliers = np.abs(diframpnsig[currentint, 0, :, :]) > nsig

        # Here is therefore the correction to the first group image
        # Add also noise to that correction to mimick the group image noise
        firstgroup_correction[outliers] = firstgroup[outliers] \
                                          - ramp_med[0][outliers] \
                                          + rng.normal(size=outliers.sum()) * ramp_mdv[0][outliers]
        # TODO: Change above loop for this and test if it works:
        # for i in range(nint):
        #    outliers = diframpnsig[i,0,:,:] > nsig
        #    # Here is therefore the correction to the first group image
        #    # Add also noise to that correction to mimick the group image noise
        #    firstgroup_correction[i][outliers] = ramp[i,0,:,:][outliers] \
        #               - ramp_med[0][outliers] \
        #               + rng.normal(size=outliers.sum()) * ramp_mdv[0][outliers]

        # Apply correction to all subsequent groups in the ramp
        for g in range(ng):
            ramp[currentint, g, :, :] = ramp[currentint, g, :, :] - firstgroup_correction[:, :]

        # Now that the first group image is corrected, compute
        # the g - (g-1) cds and repeat the correction on the gth group image.
        for n in range(ng - 1):
            g = n + 1
            print('cosmicrays_action - Correct group {:}'.format(g + 1))
            cdsgg1 = np.zeros((nint, dimy, dimx))

            for i in range(nint):
                # CDS for all ints of this particular group
                cdsgg1[i, :, :] = ramp[i, g, :, :] - ramp[i, g - 1, :, :]
            # Median CDS, median deviation and nsigma outlier map for this group
            cdsgg1masked = cdsgg1 * 1
            cdsgg1masked[currentint, :, :] = np.nan
            cdsgg1_med = np.nanmedian(cdsgg1masked, axis=0)
            cdsgg1_mdv = mediandev(cdsgg1masked, axis=0)
            cdsgg1_nsig = (cdsgg1 - cdsgg1_med) / cdsgg1_mdv

            gth_group_correction = np.zeros((dimy, dimx))
            gth_group = cdsgg1[currentint, :, :]
            outliers = np.abs(cdsgg1_nsig[currentint, :, :]) > nsig

            # Here is therefore the correction to the gth group image
            # Add also noise to that correction to mimick the group image noise
            gth_group_correction[outliers] = gth_group[outliers] \
                                             - cdsgg1_med[outliers] \
                                             + rng.normal(size=outliers.sum()) \
                                             * cdsgg1_mdv[outliers]

            # Apply the correction to all subsequent groups of the ramp
            for m in range(ng - g):
                ramp[currentint, m + g, :, :] = ramp[currentint, m + g, :, :] - gth_group_correction[:, :]

    return ramp


def cosmicrays_step(input_data, ninthalf=3, save_diagnostics=False,
                    outdir=None, save_results=False):
    # If the input_data is a list of rampdata files on disk, then assume they are
    # segments of a TSO and perform the cosmic ray for this segment with a few
    # integrations to pad the beginning/end of the cube.
    # If the input_data is a rampdata then directly launch the cosmic ray
    # step on it.

    # Determine what the input_type is
    shape = np.shape(input_data)
    sz = np.size(shape)
    if sz >= 3:
        input_type = 'rampdata'
        rampdata = input_data
    else:
        input_type = 'segmentlist'
    print('cosmicrays_step - detected input_type is ', input_type)

    # Check the size of the input_data to branch correctly
    if input_type == 'segmentlist':
        nsegments = np.size(input_data)
        if nsegments == 1:
            # Supposed to be a list but really is a single segment.
            # Read it and switch to the rampdata input type.
            rampdata = datamodels.open(input_data)
            input_type = 'rampdata'
        else:
            # Check if the list of segments is sorted
            arg_sorted = np.argsort(input_data)
            arg_input = np.arange(np.size(input_data))
            rezz = arg_sorted == arg_input
            if rezz.all() == False:
                print('cosmicrays_step - ERROR. list of segments is not sorted. Abort')
                sys.exit()

    # =========================================================================
    # INPUT is a list of contiguous segments found on disk. They will be read
    # out and the corrected segments, written back to disk.
    # =========================================================================
    if input_type == 'segmentlist':

        print('cosmicrays_step - input_type = segmentlist')
        nsegments = np.size(input_data)

        # Reading the first jwst datamodel file
        jwstmodel = datamodels.open(input_data[0])
        input_filename = jwstmodel.meta.filename

        # Determine the outdir and basename for later saving
        #basename = os.path.splitext(jwstmodel.meta.filename)[0]
        #basename = basename.split('_nis')[0] + '_nis'
        if outdir == None:
            outdir = os.path.dirname(input_data[0])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Determine the number of integrations from the first file data
        nint, ng, dimy, dimx = np.shape(jwstmodel.data)
        del jwstmodel

        # pad the current segment at both ends using the previous and next segments
        # Iteratively send each segment to the cosmicrays_action() function
        # But taking care to pad the ends of each segment with neighboring
        # integrations.
        for s in range(nsegments):
            print('cosmicrays_step - segment {:} of {:}'.format(s + 1, nsegments))
            # SEGMENT 1 -------------------------------------------------------
            if s == 0 and nsegments > 1:
                print('cosmicrays_step - first segment')
                jwstmodel = datamodels.open(input_data[0])
                jwstmodelnext = datamodels.open(input_data[1])

                # Number of integrations in each segment (deal with possibly nint=1)
                nprev = 0
                ncurr = np.shape(jwstmodel.data)[0]
                nnext = np.shape(jwstmodelnext.data)[0]

                # Determine the number of integrations in the padded cube
                padprev = 0
                padnext = min([nnext, ninthalf])
                nintpad = padprev + ncurr + padnext

                 # Create a jwstmodel.data with integration pads at the end
                ramppadded = np.zeros((nintpad, ng, dimy, dimx))
                # Fill that padded jwstmodel.data
                ramppadded[:ncurr, :, :, :] = np.copy(jwstmodel.data[:, :, :, :])
                # handle the case where the next ramp is a single integration
                if nnext == 1:
                    ramppadded[-1:, :, :, :] = np.copy(jwstmodelnext.data[:, :, :])
                else:
                    ramppadded[-padnext:, :, :, :] = np.copy(jwstmodelnext.data[:padnext, :, :, :])
                print('cosmicrays_step - number of integrations in this segment: {:}'.format(ncurr))
                print('cosmicrays_step - integrations padding pre/current/next is {:}/{:}/{:}'.format(padprev, ncurr, padnext))
                print('cosmicrays_step - number of integrations with padding: {:}'.format(nintpad))

                # Now call cosmicrays_action in the usual way but with added
                # padding.
                for i in range(nintpad):
                    print('cosmicrays_step - Padded integration {:} of {:}'.format(i + 1, nintpad))
                    if i < ninthalf:
                        # integration to correct is close to the beginning of the series
                        n = np.copy(i)
                        cosmicrays_action(ramppadded[:i + ninthalf + 1], integration_to_correct=n,
                                          save_diagnostics=save_diagnostics)
                    elif nintpad - i < ninthalf:
                        # integration to correct is close to the end of the series
                        n = np.copy(ninthalf)
                        cosmicrays_action(ramppadded[i - ninthalf:], integration_to_correct=n,
                                          save_diagnostics=save_diagnostics)
                    else:
                        # integration to correct is comfortably within the series
                        n = np.copy(ninthalf)
                        cosmicrays_action(ramppadded[i - ninthalf:i + ninthalf + 1], integration_to_correct=n,
                                          save_diagnostics=save_diagnostics)

                # Save the result
                jwstmodel.data[:,:,:,:] = ramppadded[:ncurr, :, :, :] * 1
                jwstmodel.groupdq[:,:,:,:] = 0
                jwstmodel.pixeldq[:,:] = 0
                if save_results == True:
                    # Determine the basename for before saving
                    basename = os.path.splitext(jwstmodel.meta.filename)[0]
                    basename = basename.split('_nis')[0] + '_nis'
                    jwstmodel.meta.filename = basename + '_cosmicraystep.fits'
                    print('cosmicrays_step - saving ', jwstmodel.meta.filename)
                    jwstmodel.write(outdir + '/' + jwstmodel.meta.filename)
                else:
                    jwstmodel.meta.filename = input_filename

                # Delete big arrays in the hope to free memory
                del jwstmodel
                del jwstmodelnext
                del ramppadded

            # SEGMENTS 2 to N-1 -----------------------------------------------
            elif s >= 1 and s <= nsegments - 2:
                print('cosmicrays_step - 2 < segment < before last')
                jwstmodelprev = datamodels.open(input_data[s - 1])
                jwstmodel = datamodels.open(input_data[s])
                jwstmodelnext = datamodels.open(input_data[s + 1])

                # Number of integrations in each segment (deal with possibly nint=1)
                nprev = np.shape(jwstmodelprev.data)[0]
                ncurr = np.shape(jwstmodel.data)[0]
                nnext = np.shape(jwstmodelnext.data)[0]

                # Determine the number of integrations in the padded cube
                padprev = ninthalf
                padnext = min([ninthalf,nnext])
                nintpad = padprev + ncurr + padnext

                # Create a jwstmodel.data with integration pads at both ends
                ramppadded = np.zeros((nintpad, ng, dimy, dimx))
                # Fill it with the integrations padding from the previous segment
                ramppadded[:padprev, :, :, :] = np.copy(jwstmodelprev.data[-padprev:, :, :, :])
                # Fill it with the integrations from the current segment
                ramppadded[padprev:-padnext, :, :, :] = np.copy(jwstmodel.data[:, :, :, :])
                # Fill it with the integrations padding from the next segment
                if nnext == 1:
                    ramppadded[-1:, :, :, :] = np.copy(jwstmodelnext.data[:, :, :])
                else:
                    ramppadded[-padnext:, :, :, :] = np.copy(jwstmodelnext.data[:padnext, :, :, :])

                print('cosmicrays_step - number of integrations in this segment: {:}'.format(ncurr))
                print('cosmicrays_step - integrations padding pre/current/next is {:}/{:}/{:}'.format(padprev, ncurr, padnext))
                print('cosmicrays_step - number of integrations with padding: {:}'.format(nintpad))

                # Now call cosmicrays_action in the usual way but with added
                # padding.
                for i in range(nintpad):
                    print('cosmicrays_step - Padded integration {:} of {:}'.format(i + 1, nintpad))
                    if i < ninthalf:
                        # integration to correct is close to the beginning of the series
                        n = np.copy(i)
                        cosmicrays_action(ramppadded[:i + ninthalf + 1], integration_to_correct=n,
                                          save_diagnostics=save_diagnostics)
                    elif nintpad - i < ninthalf:
                        # integration to correct is close to the end of the series
                        n = np.copy(ninthalf)
                        cosmicrays_action(ramppadded[i - ninthalf:], integration_to_correct=n,
                                          save_diagnostics=save_diagnostics)
                    else:
                        # integration to correct is comfortably within the series
                        n = np.copy(ninthalf)
                        cosmicrays_action(ramppadded[i - ninthalf:i + ninthalf + 1], integration_to_correct=n,
                                          save_diagnostics=save_diagnostics)

                # Save the result
                jwstmodel.data[:, :, :, :] = ramppadded[padprev:-padnext, :, :, :] * 1
                jwstmodel.groupdq[:,:,:,:] = 0
                jwstmodel.pixeldq[:,:] = 0
                if save_results == True:
                    # Determine the basename for before saving
                    basename = os.path.splitext(jwstmodel.meta.filename)[0]
                    basename = basename.split('_nis')[0] + '_nis'
                    jwstmodel.meta.filename = basename + '_cosmicraystep.fits'
                    print('cosmicrays_step - saving ', jwstmodel.meta.filename)
                    jwstmodel.write(outdir + '/' + jwstmodel.meta.filename)
                else:
                    jwstmodel.meta.filename = input_filename

                # Delete big arrays in the hope to free memory
                del jwstmodelprev
                del jwstmodel
                del jwstmodelnext
                del ramppadded

            # LAST SEGMENT ----------------------------------------------------
            elif s == nsegments - 1:
                print('cosmicrays_step - last segment')
                jwstmodelprev = datamodels.open(input_data[s - 1])
                jwstmodel = datamodels.open(input_data[s])

                # Number of integrations in each segment (deal with possibly nint=1)
                nprev = np.shape(jwstmodelprev.data)[0]
                ncurr = np.shape(jwstmodel.data)[0]
                nnext = 0

                # Determine the number of integrations in the padded cube
                padprev = ninthalf
                padnext = 0
                nintpad = padprev + ncurr + padnext

                # Create a rampdata with integration pads at the beginning
                ramppadded = np.zeros((nintpad, ng, dimy, dimx))
                # Fill it with the integrations from the previous then current ramps
                ramppadded[:padprev, :, :, :] = np.copy(jwstmodelprev.data[-padprev:, :, :, :])
                # Fill padded cube with current segment - handle case of
                # current having a single integrations.
                if ncurr == 1:
                    ramppadded[-1:, :, :, :] = np.copy(jwstmodel.data[:, :, :])
                else:
                    ramppadded[-ncurr:, :, :, :] = np.copy(jwstmodel.data[:ncurr, :, :, :])
                print('cosmicrays_step - number of integrations in this segment: {:}'.format(ncurr))
                print('cosmicrays_step - integrations padding pre/current/next is {:}/{:}/{:}'.format(padprev, ncurr, padnext))
                print('cosmicrays_step - number of integrations with padding: {:}'.format(nintpad))

                # Now call cosmicrays_action in the usual way but with added
                # padding.
                for i in range(nintpad):
                    print('cosmicrays_step - Padded integration {:} of {:}'.format(i + 1, nintpad))
                    if i < ninthalf:
                        # integration to correct is close to the beginning of the series
                        n = np.copy(i)
                        cosmicrays_action(ramppadded[:i + ninthalf + 1], integration_to_correct=n,
                                          save_diagnostics=save_diagnostics)
                    elif nintpad - i < ninthalf:
                        # integration to correct is close to the end of the series
                        n = np.copy(ninthalf)
                        cosmicrays_action(ramppadded[i - ninthalf:], integration_to_correct=n,
                                          save_diagnostics=save_diagnostics)
                    else:
                        # integration to correct is comfortably within the series
                        n = np.copy(ninthalf)
                        cosmicrays_action(ramppadded[i - ninthalf:i + ninthalf + 1], integration_to_correct=n,
                                          save_diagnostics=save_diagnostics)

                # Save the result
                if ncurr == 1:
                    jwstmodel.data[:, :, :] = ramppadded[-1, :, :, :] * 1
                    jwstmodel.groupdq[:, :, :] = 0
                    jwstmodel.pixeldq[:, :] = 0
                else:
                    jwstmodel.data[:, :, :, :] = ramppadded[-ncurr:, :, :, :] * 1
                    jwstmodel.groupdq[:, :, :, :] = 0
                    jwstmodel.pixeldq[:, :] = 0
                if save_results == True:
                    # Determine the basename for before saving
                    basename = os.path.splitext(jwstmodel.meta.filename)[0]
                    basename = basename.split('_nis')[0] + '_nis'
                    jwstmodel.meta.filename = basename + '_cosmicraystep.fits'
                    print('cosmicrays_step - saving ', jwstmodel.meta.filename)
                    jwstmodel.write(outdir + '/' + jwstmodel.meta.filename)
                else:
                    jwstmodel.meta.filename = input_filename

                # Delete big arrays in the hope to free memory
                del jwstmodelprev
                del jwstmodel
                del ramppadded

            # THAT would be ODD...
            else:
                print('cosmicrays_step - not supposed to reach this! Abort!')
                #TODO: will reach this if there is a single segment! Do something.
                sys.exit()

    # =========================================================================
    # INPUT is a rampdata.data segment already in memory.
    # =========================================================================
    elif input_type == 'rampdata':
        print('cosmicrays_step - input_type = rampdata')
        nint, ng, dimy, dimx = np.shape(rampdata)

        # test = np.arange(nint)
        for i in range(nint):

            if i == 1:
                save_diagnostics = True
            else:
                save_diagnostics = False

            if i < ninthalf:
                # integration to correct is close to the beginning of the series
                n = np.copy(i)
                # print(i, n, test[:i+ninthalf+1])
                cosmicrays_action(rampdata[:i + ninthalf + 1], integration_to_correct=n,
                                  save_diagnostics=save_diagnostics)
            elif nint - i < ninthalf:
                # integration to correct is close to the end of the series
                n = np.copy(ninthalf)
                # print(i, n, test[i-ninthalf:])
                cosmicrays_action(rampdata[i - ninthalf:], integration_to_correct=n,
                                  save_diagnostics=save_diagnostics)
            else:
                # integration to correct is comfortably within the series
                n = np.copy(ninthalf)
                # print(i, n, test[i-ninthalf:i+ninthalf+1])
                cosmicrays_action(rampdata[i - ninthalf:i + ninthalf + 1], integration_to_correct=n,
                                  save_diagnostics=save_diagnostics)

        hdu = fits.PrimaryHDU(rampdata)
        hdu.writeto(outdir+'/test_rampdata_corrected.fits', overwrite=True)

    # =========================================================================
    # INPUT does not make sense...
    # =========================================================================
    else:
        print('cosmicrays_step - input_type not defined. Abort.')
        sys.exit()

    return
