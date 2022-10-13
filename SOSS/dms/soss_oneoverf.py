import numpy as np
from astropy.io import fits
from jwst import datamodels
import os


def mediandev(x, axis=None):
    med = np.nanmedian(x, axis=axis)

    return np.nanmedian(np.abs(x - med), axis=axis) / 0.67449

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


def applycorrection(uncal_rampmodel, output_dir=None, save_results=False,
                    deepstack_custom=None, return_intermediates=None):

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
    nint = intend-intstart+1
    dimx = np.shape(uncal_rampmodel.data)[-1]

    # Generate the deep stack and rms of it. Both 3D (ngroup, dimy, dimx)
    deepstack, rms = stack(uncal_rampmodel.data, deepstack_custom=deepstack_custom)

    # Weighted average to determine the 1/F DC level
    w = 1 / rms ** 2  # weight
    # Make sure that constant pixels with rms=0 don't get assign a weight
    w[rms <= 0] = 0
    # Make sure that pixels with suspiciously small rms get zero weight as well
    #med = np.median(rms, axis=0)
    #dev = mediandev(rms, axis=0)
    #pct1 = np.percentile(rms, 1, axis=(1,2))
    #for i in range(ngroup):
    #    w[i, rms[i,:,:] < pct1[i]] = 0
    # Make sure that bad pixels don't have any weight
    dq = np.median(uncal_rampmodel.groupdq, axis=0)
    w[dq != 0] = 0

    # Write these on disk in a sub folder
    if save_results == True:
        hdu = fits.PrimaryHDU(deepstack)
        print(output_supp+'/deepstack1.fits')
        hdu.writeto(output_supp+'/deepstack1.fits', overwrite=True)
        hdu = fits.PrimaryHDU(rms)
        hdu.writeto(output_supp+'/rms1.fits', overwrite=True)
        hdu = fits.PrimaryHDU(w)
        hdu.writeto(output_supp+'/weight1.fits', overwrite=True)

    # TODO: 1/rms2 should be used
    # TODO: filter out the rms=0 pixels
    # TODO: add odd even correction
    print(np.shape(w))
    print(np.shape(w * uncal_rampmodel.data[0]))

    if False:
        # Read in the outlier map -- a (nints, dimy, dimx) 3D cube
        print('outlier_map = {:}'.format(outlier_map))
        print('is None', (outlier_map == None))
        print('is not a file?', (not os.path.isfile(outlier_map)))
        if (outlier_map == None) | (not os.path.isfile(outlier_map)):
            print('Warning - the outlier map passed as input does not exist on disk - no outlier map used!')
            outliers = np.zeros((nint, np.shape(uncal_rampmodel.data)[-2], dimx))
        else:
            print('Using an existing cosmic ray outlier map named {:}'.format(outlier_map))
            outliers = fits.getdata(outlier_map)
        # The outlier is 0 where good and >0 otherwise
        outliers = np.where(outliers == 0, 1, np.nan)

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
            dc = np.nansum(w * sub[i], axis=1) / np.nansum(w, axis=1)
            # make sure no NaN will corrupt the whole column
            dc = np.where(np.isfinite(dc), dc, 0)
            # dc is 2-dimensional - expand to the 3rd (columns) dimension
            dcmap[i, :, :, :] = np.repeat(dc, 256).reshape((ngroup, 2048, 256)).swapaxes(1,2)
            subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
        elif uncal_rampmodel.meta.subarray.name == 'SUBSTRIP96':
            dc = np.nansum(w * sub[i], axis=1) / np.nansum(w, axis=1)
            # make sure no NaN will corrupt the whole column
            dc = np.where(np.isfinite(dc), dc, 0)
            # dc is 2-dimensional - expand to the 3rd (columns) dimension
            dcmap[i,:,:,:] = np.repeat(dc, 96).reshape((ngroup, 2048, 96)).swapaxes(1,2)
            subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
        elif uncal_rampmodel.meta.subarray.name == 'SUB80':
            dc = np.nansum(w * sub[i], axis=1) / np.nansum(w, axis=1)
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
