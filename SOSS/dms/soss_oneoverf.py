import numpy as np
from astropy.io import fits
from jwst import datamodels
import os
import SOSS.dms.soss_centroids as soss_centroids



def mediandev(x, axis=None):
    med = np.nanmedian(x, axis=axis)

    return np.nanmedian(np.abs(x - med), axis=axis) / 0.67449

def stack_multisegments(postsaturationstep_list, outdir=None, save_results=False,
                        stack_nblocks=None):
    '''
    Read files from disk (those output by the Saturation Step) to create a time-series
    wide deep stack of each group (frame). To minimize memory usage, allow this to
    operate on blocks of columns rather than full image.
    '''

    if stack_nblocks == None:
        # any divider of 2048 would do
        nblocks = np.size(postsaturationstep_list)*2
    else:
        nblocks = np.copy(stack_nblocks)
    #nblocks = 2  # any divider of 2048 would do
    #blocksize = 2048 // nblocks
    blocksize = int(np.ceil(2048 / nblocks)) # no need for comon divider of 2048

    print('1/f stacking of all {:} segments in the time-series in {:} blocks of {:} columns'.format(
        np.size(postsaturationstep_list), nblocks, blocksize))

    for b in range(nblocks):
        print('Block {:}'.format(b+1))
        # Set the x-axis limits of this block of columns
        firstcol, lastcol = b * blocksize, (b + 1) * blocksize
        # Check that the last column is never above 2048
        lastcol = np.min([2048, lastcol])
        currentblocksize = lastcol - firstcol
        print('Curent block is between firstcol={:} and lastcol={:} and has size of {:} columns'.format(
            firstcol, lastcol, currentblocksize))

        # For each block of columns, loop over all segments
        for segment in range(np.size(postsaturationstep_list)):
            # Fill the data cube and groupdq cube for block b
            seg = datamodels.open(postsaturationstep_list[segment])
            i_start, i_end = seg.meta.exposure.integration_start, seg.meta.exposure.integration_end
            if (i_start == None) & (i_end == None):
                # it means that this is a time-series NOT split into segments
                i_start, i_end = 1, seg.meta.exposure.nints
            print('i_start = {:}, i_end = {:}'.format(i_start, i_end))
            if segment == 0:
                # First segment, initialize cubes of proper size
                _, ngroups, dimy, dimx = np.shape(seg.data)
                nints = seg.meta.exposure.nints
                data = np.zeros((nints, ngroups, dimy, currentblocksize)) * np.nan
                mask = np.zeros((nints, ngroups, dimy, currentblocksize)) * np.nan

            # the current segment data, group DQ and pixel DQ
            data[i_start-1:i_end, :, :, :] = np.copy(seg.data[:, :, :, firstcol:lastcol])
            gdq = np.copy(seg.groupdq[:, :, :, firstcol:lastcol])
            pdq = np.copy(seg.pixeldq[:, firstcol:lastcol])
            # Add to the mask the group dq (4 dimensional)
            segmask = np.where(gdq != 0, np.nan, 1)
            mask[i_start-1:i_end, :, :, :] = np.copy(segmask)
            # Add to the mask the pixel dq (2 dimensional)
            segmask = np.where(pdq != 0, np.nan, 1)
            mask[i_start-1:i_end, :] = mask[i_start-1:i_end, :] * segmask
        # Stack that block, all bad pixels are NaNs
        if b == 0:
            # First block, initialize the final products with proper size
            deepstack = np.zeros((ngroups, dimy, dimx)) * np.nan
            rms = np.zeros((ngroups, dimy, dimx)) * np.nan
        block_stack = np.nanmedian(data * mask, axis=0)
        block_rms = mediandev(data * mask - block_stack, axis=0)
        deepstack[:, :, firstcol:lastcol] = np.copy(block_stack)
        rms[:, :, firstcol:lastcol] = np.copy(block_rms)

    if save_results:
        # Recover names and directory
        segment1name = postsaturationstep_list[0]
        if outdir == None:
            outdir = os.path.dirname(segment1name)
        basename = os.path.basename(os.path.splitext(segment1name)[0])
        basename_ts = basename.split('-seg')[0]
        # Save as fits files
        hdu = fits.PrimaryHDU(deepstack)
        hdu.writeto(outdir+'/oof_deepstack_'+basename_ts+'.fits', overwrite=True)
        hdu = fits.PrimaryHDU(rms)
        hdu.writeto(outdir+'/oof_rms_'+basename_ts+'.fits', overwrite=True)
        #hdu = fits.PrimaryHDU(mask)
        #hdu.writeto(outdir+'/oof_lastblockmask_'+basename_ts+'.fits', overwrite=True)


    return deepstack, rms




def stack(cube, deepstack_custom=None, outliers_map=None):

    if deepstack_custom is None:
        if outliers_map is None:
            deepstack = np.nanmedian(cube, axis=0)
        else:
            deepstack = np.nanmedian(cube * outliers_map, axis=0)
    else:
        deepstack = np.copy(deepstack_custom)

    if outliers_map is None:
        rms = mediandev(cube - deepstack, axis=0)
    else:
        rms = mediandev((cube - deepstack) * outliers_map, axis=0)

    return deepstack, rms


def make_trace_mask(trace_table_ref, subarray_name, aphalfwidth=[13,13,13],
                    outdir=None):

    dimx = 2048
    norders = 3
    if subarray_name == 'FULL': dimy = 2048
    if subarray_name == 'SUBSTRIP256': dimy = 256
    if subarray_name == 'SUBSTRIP96':
        dimy = 96
        norders = 1

    # We assume that a valid trace table reference file was passed. Read it.
    ref = fits.open(trace_table_ref)
    x_o1, y_o1, wv_o1 = np.array(ref[1].data['X']), np.array(ref[1].data['Y']), np.array(ref[1].data['WAVELENGTH'])
    x_o2, y_o2, wv_o2 = np.array(ref[2].data['X']), np.array(ref[2].data['Y']), np.array(ref[2].data['WAVELENGTH'])
    x_o3, y_o3, wv_o3 = np.array(ref[3].data['X']), np.array(ref[3].data['Y']), np.array(ref[3].data['WAVELENGTH'])
    # Assumption is made later that x are integers from 0 to 2047
    # sort order 1
    x = np.arange(2048)
    sorted = np.argsort(x_o1)
    x_o1, y_o1, wv_o1 = x_o1[sorted], y_o1[sorted], wv_o1[sorted]
    y_o1 = np.interp(x, x_o1, y_o1)
    wv_o1 = np.interp(x, x_o1, wv_o1)
    x_o1 = x
    # sort order 2
    x = np.arange(2048)
    sorted = np.argsort(x_o2)
    x_o2, y_o2, wv_o2 = x_o2[sorted], y_o2[sorted], wv_o2[sorted]
    y_o2 = np.interp(x, x_o2, y_o2)
    w_o2 = np.interp(x, x_o2, wv_o2)
    x_o2 = x
    # sort order 3
    x = np.arange(2048)
    sorted = np.argsort(x_o3)
    x_o3, y_o3, w_o3 = x_o3[sorted], y_o3[sorted], wv_o3[sorted]
    y_o3 = np.interp(x, x_o3, y_o3)
    wv_o3 = np.interp(x, x_o3, wv_o3)
    x_o3 = x

    # Create a cube containing the mask for all orders
    maskcube = np.zeros((norders, dimy, dimx))

    for m in range(norders):
        if m == 0: ordercen = np.copy(y_o1)
        if m == 1: ordercen = np.copy(y_o2)
        if m == 2: ordercen = np.copy(y_o3)

        mask_trace = soss_centroids.build_mask_trace(ordercen, subarray=subarray_name,
                                                     halfwidth=aphalfwidth[m],
                                                     extend_below=False,
                                                     extend_above=False)
        mask = np.zeros(np.shape(mask_trace))
        mask[mask_trace == True] = 1

        maskcube[m,:,:] = np.copy(mask_trace)

    # crunch the orders into a single stack
    trace_mask = np.nansum(maskcube, axis=0)
    trace_mask[trace_mask >= 1] = 1

    if outdir != None:
        hdu = fits.PrimaryHDU(trace_mask)
        hdu.writeto(outdir+'/trace_mask.fits', overwrite=True)

    return trace_mask

def applycorrection(uncal_rampmodel, output_dir=None, save_results=False,
                    deepstack_custom=None, return_intermediates=None,
                    outlier_map=None, trace_mask=None, trace_table_ref=None):

    '''
    uncal_rampmodel is a 4D ramp (not a rate)
    '''

    print('Custom 1/f correction step. Generating a deep stack for each frame using all integrations...')

    # Forge output directory where data may be written
    basename = os.path.splitext(uncal_rampmodel.meta.filename)[0]
    basename = basename.split('_nis')[0]+'_nis'
    print('basename {:}'.format(basename))
    if output_dir == None:
        output_dir = os.path.curdir+'/'
    if not os.path.exists(output_dir):
        if save_results == True: os.makedirs(output_dir)
    output_supp = output_dir+'/supplemental_'+basename+'/'
    if not os.path.exists(output_supp):
        if save_results == True: os.makedirs(output_supp)

    # The readout setup

    ngroup = uncal_rampmodel.meta.exposure.ngroups
    #nint = uncal_rampmodel.meta.exposure.nints # does not work on segments
    intstart = uncal_rampmodel.meta.exposure.integration_start
    intend = uncal_rampmodel.meta.exposure.integration_end
    if (intstart == None) & (intend == None):
        # it means that this is a time-series NOT split into segments
        intstart, intend = 1, uncal_rampmodel.meta.exposure.nints
    nint = intend-intstart+1
    dimx = np.shape(uncal_rampmodel.data)[-1]

    # Generate the deep stack and rms of it. Both 3D (ngroup, dimy, dimx)
    deepstack, rms = stack(uncal_rampmodel.data, deepstack_custom=deepstack_custom)

    print('shape of rms ', np.shape(rms))
    # Weighted average to determine the 1/F DC level
    w = np.copy(uncal_rampmodel.data)
    print('shape of w ', np.shape(w))
    w[:] = 1 / rms ** 2  # weight
    # Make sure that constant pixels with rms=0 don't get assign a weight
    w[:, rms <= 0] = 0
    # Make sure that pixels with suspiciously small rms get zero weight as well
    #med = np.median(rms, axis=0)
    #dev = mediandev(rms, axis=0)
    #pct1 = np.percentile(rms, 1, axis=(1,2))
    #for i in range(ngroup):
    #    w[i, rms[i,:,:] < pct1[i]] = 0
    # Make sure that bad pixels don't have any weight
    print('shape of groupdq ', np.shape(uncal_rampmodel.groupdq))
    dq = np.median(uncal_rampmodel.groupdq, axis=0)
    dq = np.copy(uncal_rampmodel.groupdq)
    w[dq != 0] = 0

    # TODO: add odd even correction
    print(np.shape(w))
    print(np.shape(w * uncal_rampmodel.data[0]))

    # DO NOT USE THIS OUTLIER MAP. PRODUCES WORST RESULTS IF USED.
    if False:
        # Read in the outlier map -- a (nints, dimy, dimx) 3D cube
        print('outlier_map = {:}'.format(outlier_map))
        print('is None', (outlier_map == None))
        print('is not a file?', (not os.path.isfile(outlier_map)))
        if (outlier_map == None) | (not os.path.isfile(outlier_map)):
            print('Warning - No outlier map passed or the outlier map passed as input does not exist on disk - no outlier map used!')
            #outliers = np.zeros((nint, np.shape(uncal_rampmodel.data)[-2], dimx))
        else:
            print('Using an existing cosmic ray outlier map named {:}'.format(outlier_map))
            outliers = fits.getdata(outlier_map)
            # The outlier is 0 where good and >0 otherwise
            #outliers = np.where(outliers == 0, 1, np.nan)
            # Modify the weight map with these outliers
            wtmp = w.swapaxes(0,1)
            print('shape of wtmp', np.shape(wtmp))
            wtmp[:,(outliers == 0) | (outliers == 2)] = 0
            w = wtmp.swapaxes(0,1)
            print('shape of w', np.shape(w))


    if False:
        # Read the trace mask to mask the traces from the 1/f estimate
        print('trace_mask = {:}'.format(trace_mask))
        print('is None', (trace_mask == None))
        print('is not a file?', (not os.path.isfile(trace_mask)))
        if (trace_mask == None) |  (not os.path.isfile(trace_mask)):
            print('Warning - No trace mask passed or the trace mask passed as input does not exist on disk - no trace mask used!')
        else:
            print('Using an existing trace mask named {:}'.format(trace_mask))
            tmask = fits.getdata(trace_mask)
            w[:, (tmask == 0) | (~np.isfinite(tmask))] = 0
    else:
        # Create a trace mask from scratch using the trace reference file
        tmask = make_trace_mask(trace_table_ref, uncal_rampmodel.meta.subarray.name,
                                outdir=output_supp)
        # Update the weight based on that mask
        print(np.shape(tmask))
        print(np.shape(w))
        w[:, :, (tmask == 1) | (~np.isfinite(tmask))] = 0

    # Write these on disk in a sub folder
    if save_results == True:
        hdu = fits.PrimaryHDU(deepstack)
        print(output_supp+'/deepstack1.fits')
        hdu.writeto(output_supp+'/deepstack1.fits', overwrite=True)
        hdu = fits.PrimaryHDU(rms)
        hdu.writeto(output_supp+'/rms1.fits', overwrite=True)
        hdu = fits.PrimaryHDU(w)
        hdu.writeto(output_supp+'/weight1.fits', overwrite=True)

    print('Applying the 1/f correction.')
    dcmap = np.copy(uncal_rampmodel.data)
    subcorr = np.copy(uncal_rampmodel.data)
    sub = np.copy(uncal_rampmodel.data)
    outliers = np.copy(uncal_rampmodel.data)
    for i in range(nint):
        # The actual integration in the original time series (not in this segment)
        actualint = intstart + i
        sub[i] = uncal_rampmodel.data[i] - deepstack
        for g in range(ngroup):
            # sigma map allowing for a possible median offset w.r.t. the deepstack
            sigmap = np.abs((sub[i,g,:,:] - np.nanmedian(sub[i, g, :, :])) / rms[g,:,:])
            outliers[i,g,:,:] = np.where(sigmap > 3, np.nan, 1)
            sub[i, g, :, :] = sub[i, g, :, :] * outliers[i, g, :, :]
            # Mask the noisy rolling two-rows
            if uncal_rampmodel.meta.subarray.name == 'SUBSTRIP256':
                noisyrows = [256 - actualint, 256 - actualint + 1]
            if uncal_rampmodel.meta.subarray.name == 'SUBSTRIP96':
                noisyrows = [96 - actualint, 96 - actualint + 1]
            if uncal_rampmodel.meta.subarray.name == 'SUBSTRIP256':
                noisyrows = [2048 - actualint, 2048 - actualint + 1]
            sub[i, g, noisyrows[0]:noisyrows[1]+1,:] = np.nan
            # Make sure to not subtract an overall bias
            # TODO: Is it legit to subtract the median??? Don't think so.
            #sub[i,g,:,:] = sub[i,g, :, :] - np.nanmedian(sub[i,g,:,:])
        #if save_results == True:
        #    hdu = fits.PrimaryHDU(sub)
        #    hdu.writeto(output_supp+'/sub.fits', overwrite=True)
        if uncal_rampmodel.meta.subarray.name == 'SUBSTRIP256':
            dc = np.nansum(w[i] * sub[i], axis=-2) / np.nansum(w[i], axis=-2)
            # make sure no NaN will corrupt the whole column
            dc = np.where(np.isfinite(dc), dc, 0)
            # dc is 2-dimensional - expand to the 3rd (columns) dimension
            dcmap[i, :, :, :] = np.repeat(dc, 256).reshape((ngroup, 2048, 256)).swapaxes(1,2)
            subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
        elif uncal_rampmodel.meta.subarray.name == 'SUBSTRIP96':
            dc = np.nansum(w[i] * sub[i], axis=-2) / np.nansum(w[i], axis=-2)
            # make sure no NaN will corrupt the whole column
            dc = np.where(np.isfinite(dc), dc, 0)
            # dc is 2-dimensional - expand to the 3rd (columns) dimension
            dcmap[i,:,:,:] = np.repeat(dc, 96).reshape((ngroup, 2048, 96)).swapaxes(1,2)
            subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
        elif uncal_rampmodel.meta.subarray.name == 'SUB80':
            dc = np.nansum(w[i] * sub[i], axis=-2) / np.nansum(w[i], axis=-2)
            # make sure no NaN will corrupt the whole column
            dc = np.where(np.isfinite(dc), dc, 0)
            # dc is 2-dimensional - expand to the 3rd (columns) dimension
            dcmap[i,:,:,:] = np.repeat(dc, 80).reshape((ngroup, 80, 80)).swapaxes(1,2)
            subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
        elif uncal_rampmodel.meta.subarray.name == 'FULL':
            for amp in range(4):
                yo = amp*512
                dc = np.nansum(w[:, :, yo:yo+512, :] * sub[:, :, yo:yo+512, :], axis=1) / np.nansum(w[:, :, yo:yo+512, :], axis=1)
                # make sure no NaN will corrupt the whole column
                dc = np.where(np.isfinite(dc), dc, 0)
                # dc is 2-dimensional - expand to the 3rd (columns) dimension
                dcmap[i, :, yo:yo+512, :] = np.repeat(dc, 512).reshape((ngroup, 2048, 512)).swapaxes(1,2)
                subcorr[i, :, yo:yo+512, :] = sub[i, :, yo:yo+512, :] - dcmap[i, :, yo:yo+512, :]

    # Make sure no nan is in DC map
    dcmap = np.where(np.isfinite(dcmap), dcmap, 0)

    # Subtract the DC map from a copy of the data model
    rampmodel_corr = uncal_rampmodel.copy()
    rampmodel_corr.data = uncal_rampmodel.data - dcmap

    if save_results == True:
        hdu = fits.PrimaryHDU(sub)
        hdu.writeto(output_supp+'/sub.fits', overwrite=True)
        hdu = fits.PrimaryHDU(subcorr)
        hdu.writeto(output_supp+'/subcorr.fits', overwrite=True)
        hdu = fits.PrimaryHDU(dcmap)
        hdu.writeto(output_supp+'/noisemap.fits', overwrite=True)
        hdu = fits.PrimaryHDU(rampmodel_corr.data)
        hdu.writeto(output_supp+'/corrected.fits', overwrite=True)
        hdu = fits.PrimaryHDU(outliers)
        hdu.writeto(output_supp+'/outliers.fits', overwrite=True)

    if save_results == True:
        print('Custom 1/f output name: {:}'.format(output_dir+'/'+basename+'_custom1overf.fits'))
        rampmodel_corr.write(output_dir+'/'+basename+'_custom1overf.fits')

    rampmodel_corr.meta.filename = uncal_rampmodel.meta.filename

    # Allow returning intermediate results
    if return_intermediates is None:
        return rampmodel_corr
    else:
        return rampmodel_corr, deepstack, rms, outliers


if __name__ == "__main__":
    ################ MAIN ###############
    # Open the uncal time series that needs 1/f correction
    exposurename = '/genesis/jwst/userland-soss/loic_review/GTO/wasp52b/IDTSOSS_clear_noisy.fits'
    outdir = '/genesis/jwst/userland-soss/loic_review/oneoverf/'
    uncal_datamodel = datamodels.open(exposurename)

    # Run the 1/f correction step
    #map = applycorrection(uncal_datamodel, exposurename)
    map = applycorrection(uncal_datamodel, output_dir=outdir, save_results=True)

    # Free up the input time series
    uncal_datamodel.close()

    # Write down the output corrected time series
    map.write('/genesis/jwst/userland-soss/loic_review/oneoverf/uncal_corrected.fits')
