import numpy as np
from astropy.io import fits
from jwst import datamodels
import os

def stack(cube):
    deepstack = np.nanmedian(cube, axis=0)
    rms = np.nanstd(cube, axis=0)
    return deepstack, rms

def makemask(stack, rms):
    # Exact strategy TBD

    return

def applycorrection(uncal_datamodel, output_dir=None, save_results=False):

    print('Custom 1/f correction step. Generating a deep stack for each frame using all integrations...')

    # Forge output directory where data may be written
    basename = os.path.splitext(uncal_datamodel.meta.filename)[0]
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
    ngroup = uncal_datamodel.meta.exposure.ngroups
    #nint = uncal_datamodel.meta.exposure.nints # does not work on segments
    nint = np.shape(uncal_datamodel.data)[0]
    dimx = np.shape(uncal_datamodel.data)[-1]

    # Generate the deep stack and rms of it
    deepstack, rms = stack(uncal_datamodel.data)

    # Write these on disk in a sub folder
    if save_results == True:
        hdu = fits.PrimaryHDU(deepstack)
        print(output_supp+'/deepstack1.fits')
        hdu.writeto(output_supp+'/deepstack1.fits', overwrite=True)
        hdu = fits.PrimaryHDU(rms)
        hdu.writeto(output_supp+'/rms1.fits', overwrite=True)

    # Weighted average to determine the 1/F DC level
    w = 1/rms # weight
    print(np.shape(w))
    print(np.shape(w * uncal_datamodel.data[0]))

    print('Applying the 1/f correction.')
    dcmap = np.copy(uncal_datamodel.data)
    subcorr = np.copy(uncal_datamodel.data)
    for i in range(nint):
        sub = uncal_datamodel.data[i] - deepstack
        # Make sure to not subtract an overall bias
        #sub = sub - np.nanmedian(sub)
        if save_results == True:
            hdu = fits.PrimaryHDU(sub)
            hdu.writeto(output_supp+'/sub.fits', overwrite=True)
        if uncal_datamodel.meta.subarray.name == 'SUBSTRIP256':
            dc = np.nansum(w * sub, axis=1) / np.nansum(w, axis=1)
            # dc is 2-dimensional - expand to the 3rd (columns) dimension
            dcmap[i,:,:,:] = np.repeat(dc, 256).reshape((ngroup, 2048, 256)).swapaxes(1,2)
            subcorr[i, :, :, :] = sub - dcmap[i, :, :, :]
        elif uncal_datamodel.meta.subarray.name == 'SUBSTRIP96':
            dc = np.nansum(w * sub, axis=1) / np.nansum(w, axis=1)
            # dc is 2-dimensional - expand to the 3rd (columns) dimension
            dcmap[i,:,:,:] = np.repeat(dc, 256).reshape((ngroup, 2048, 256)).swapaxes(1,2)
            subcorr[i, :, :, :] = sub - dcmap[i, :, :, :]
        elif uncal_datamodel.meta.subarray.name == 'FULL':
            for amp in range(4):
                yo = amp*512
                dc = np.nansum(w[:, :, yo:yo+512, :] * sub[:, :, yo:yo+512, :], axis=1) / np.nansum(w[:, :, yo:yo+512, :], axis=1)
                # dc is 2-dimensional - expand to the 3rd (columns) dimension
                dcmap[i, :, yo:yo+512, :] = np.repeat(dc, 512).reshape((ngroup, 2048, 512)).swapaxes(1,2)
                subcorr[i, :, yo:yo+512, :] = sub[:, yo:yo+512, :] - dcmap[i, :, yo:yo+512, :]

    # Subtract the DC map from a copy of the data model
    datamodel_corr = uncal_datamodel.copy()
    datamodel_corr.data = uncal_datamodel.data - dcmap

    if save_results == True:
        hdu = fits.PrimaryHDU(subcorr)
        hdu.writeto(output_supp+'/subcorr.fits', overwrite=True)
        hdu = fits.PrimaryHDU(dcmap)
        hdu.writeto(output_supp+'/noisemap.fits', overwrite=True)
        hdu = fits.PrimaryHDU(datamodel_corr.data)
        hdu.writeto(output_supp+'/corrected.fits', overwrite=True)

    if save_results == True:
        print('Custom 1/f output name: {:}'.format(output_dir+'/'+basename+'_custom1overf.fits'))
        datamodel_corr.write(output_dir+'/'+basename+'_custom1overf.fits')

    datamodel_corr.meta.filename = uncal_datamodel.meta.filename

    return datamodel_corr


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
