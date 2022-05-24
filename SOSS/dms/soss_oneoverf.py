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

def applycorrection(uncal_datamodel, uncal_filename):

    print('Custom 1/f correction step. Generating a deep stack for each frame using all integrations...')

    # Forge output directory where data may be written
    basename = os.path.basename(os.path.splitext(uncal_filename)[0])
    outdir = os.path.dirname(uncal_filename)+'/oneoverf_'+basename+'/'
    # Create the output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # The readout setup
    ngroup = uncal_datamodel.meta.exposure.ngroups
    nint = uncal_datamodel.meta.exposure.nints
    dimx = np.shape(uncal_datamodel.data)[-1]

    # Generate the deep stack and rms of it
    deepstack, rms = stack(uncal_datamodel.data)

    # Write these on disk in a sub folder
    hdu = fits.PrimaryHDU(deepstack)
    print(outdir+'/deepstack1.fits')
    hdu.writeto(outdir+'/deepstack1.fits', overwrite=True)
    hdu = fits.PrimaryHDU(rms)
    hdu.writeto(outdir+'/rms1.fits', overwrite=True)

    # Weighted average to determine the 1/F DC level
    w = 1/rms # weight
    print(np.shape(w))
    print(np.shape(w * uncal_datamodel.data[0]))

    print('Applying the 1/f correction.')
    dcmap = np.copy(uncal_datamodel.data)
    subcorr = np.copy(uncal_datamodel.data)
    for i in range(nint):
        sub = uncal_datamodel.data[i] - deepstack
        hdu = fits.PrimaryHDU(sub)
        hdu.writeto(outdir+'/sub.fits', overwrite=True)
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

    hdu = fits.PrimaryHDU(subcorr)
    hdu.writeto(outdir+'/subcorr.fits', overwrite=True)
    hdu = fits.PrimaryHDU(dcmap)
    hdu.writeto(outdir+'/noisemap.fits', overwrite=True)

    corrected = uncal_datamodel.data - dcmap
    hdu = fits.PrimaryHDU(corrected)
    hdu.writeto(outdir+'/corrected.fits', overwrite=True)

    datamodel_corr = uncal_datamodel.copy()
    datamodel_corr.data = corrected

    return datamodel_corr


if __name__ == "__main__":
    ################ MAIN ###############
    # Open the uncal time series that needs 1/f correction
    exposurename = '/genesis/jwst/userland-soss/loic_review/GTO/wasp52b/IDTSOSS_clear_noisy.fits'
    uncal_datamodel = datamodels.open(exposurename)

    # Run the 1/f correction step
    map = applycorrection(uncal_datamodel, exposurename)

    # Free up the input time series
    uncal_datamodel.close()

    # Write down the output corrected time series
    map.write('/genesis/jwst/userland-soss/loic_review/oneoverf/uncal_corrected.fits')
