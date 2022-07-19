from astropy.io import fits
import bottleneck as bn
from jwst import datamodels
import numpy as np
import os
from tqdm import tqdm
import warnings


def stack(cube): #, outlier_map=None):

    #if outlier_map is None:
    deepstack = bn.nanmedian(cube, axis=0)
    rms = bn.nanstd(cube, axis=0)
    #else:
    #    deepstack = np.nanmedian(cube * outlier_map, axis=0)
    #    rms = np.nanstd(cube * outlier_map, axis=0)

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


def michael_applycorrection(datafiles, output_dir=None, save_results=False,
                    outlier_map=None):
    """Custom 1/f correction routine to be applied at the group level.

    Parameters
    ----------
    datafiles : list[str]
        List of paths to data files for each segment of the TSO. Should be 4D
        ramps and not rate files.
    output_dir : str
        Directory to which to save results.
    save_results : bool
        If True, save results to disk.
    outlier_map :

    Returns
    -------
    corrected_rampmodels : list
        Ramp models for each segment corrected for 1/f noise.
    """

    print('Starting UdeM custom 1/f correction step.')

    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    data, fileroots = [], []
    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        currentfile = datamodels.open(file)
        data.append(currentfile)
        # Hack to get filename root.
        filename_split = file.split('/')[-1].split('_')
        fileroot = ''
        for seg, segment in enumerate(filename_split):
            if seg == len(filename_split) - 1:
                break
            segment += '_'
            fileroot += segment
        fileroots.append(fileroot)

        # To create the deepstack, join all segments together.
        if i == 0:
            cube = currentfile.data
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)

    # Get total file root, with no segment info.
    working_name = fileroots[0]
    if 'seg' in working_name:
        parts = working_name.split('seg')
        part1, part2 = parts[0][:-1], parts[1][3:]
        fileroot_noseg = part1+part2
    else:
        fileroot_noseg = fileroots[0]

    # Generate the deep stack and rms of it. Both 3D (ngroup, dimy, dimx).
    print('Generating a deep stack for each frame using all integrations...')
    deepstack, rms = stack(cube)
    # Save these to disk if requested.
    if save_results is True:
        hdu = fits.PrimaryHDU(deepstack)
        hdu.writeto(output_dir+fileroot_noseg+'udem1overfstep_deepstack.fits',
                    overwrite=True)
        hdu = fits.PrimaryHDU(rms)
        hdu.writeto(output_dir+fileroot_noseg+'udem1overfstep_rms.fits',
                    overwrite=True)

    corrected_rampmodels = []
    for n, datamodel in enumerate(data):
        print('Starting segment {} of {}.'.format(n + 1, len(data)))

        # The readout setup
        ngroup = datamodel.meta.exposure.ngroups
        nint = np.shape(datamodel.data)[0]
        dimx = np.shape(datamodel.data)[-1]

        # Weighted average to determine the 1/f DC level
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            w = 1 / rms

        # Read in the outlier map -- a (nints, dimy, dimx) 3D cube
        # print('outlier_map = {}'.format(outlier_map))
        # print('is None', (outlier_map == None))
        # print('is not a file?', (not os.path.isfile(outlier_map)))
        if outlier_map is not None:
            if not os.path.isfile(outlier_map):
                msg = 'the outlier map passed as input does not exist on ' \
                      'disk - no outlier map used!'
                warnings.warn(msg)
                outliers = np.zeros((nint, np.shape(datamodel.data)[-2], dimx))
            else:
                print('Using cosmic ray outlier map {}'.format(outlier_map))
                outliers = fits.getdata(outlier_map)
        else:
            outliers = np.zeros((nint, np.shape(datamodel.data)[-2], dimx))
        # The outlier is 0 where good and >0 otherwise. As this will be
        # applied multiplicatively, replace 0s with 1s and others with NaNs.
        outliers = np.where(outliers == 0, 1, np.nan)

        dcmap = np.copy(datamodel.data)
        subcorr = np.copy(datamodel.data)
        sub = np.copy(datamodel.data)
        for i in tqdm(range(nint)):
            sub[i] = datamodel.data[i] - deepstack
            for g in range(ngroup):
                sub[i, g, :, :] *= outliers[i]
                # Make sure to not subtract an overall bias
                sub[i, g, :, :] -= np.nanmedian(sub[i, g, :, :])
            if datamodel.meta.subarray.name == 'SUBSTRIP256':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    dc = np.nansum(w * sub[i], axis=1) / np.nansum(w, axis=1)
                # make sure no NaN will corrupt the whole column
                dc = np.where(np.isfinite(dc), dc, 0)
                # dc is 2D - expand to the 3rd (columns) dimension
                dcmap[i, :, :, :] = np.repeat(dc, 256).reshape((ngroup, 2048, 256)).swapaxes(1, 2)
                subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
            elif datamodel.meta.subarray.name == 'SUBSTRIP96':
                dc = np.nansum(w * sub[i], axis=1) / np.nansum(w, axis=1)
                # make sure no NaN will corrupt the whole column
                dc = np.where(np.isfinite(dc), dc, 0)
                # dc is 2D - expand to the 3rd (columns) dimension
                dcmap[i, :, :, :] = np.repeat(dc, 256).reshape((ngroup, 2048, 256)).swapaxes(1, 2)
                subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
            elif datamodel.meta.subarray.name == 'FULL':
                for amp in range(4):
                    yo = amp*512
                    dc = np.nansum(w[:, :, yo:yo+512, :] * sub[:, :, yo:yo+512, :], axis=1) / np.nansum(w[:, :, yo:yo+512, :], axis=1)
                    # make sure no NaN will corrupt the whole column
                    dc = np.where(np.isfinite(dc), dc, 0)
                    # dc is 2D - expand to the 3rd (columns) dimension
                    dcmap[i, :, yo:yo+512, :] = np.repeat(dc, 512).reshape((ngroup, 2048, 512)).swapaxes(1, 2)
                    subcorr[i, :, yo:yo+512, :] = sub[i, :, yo:yo+512, :] - dcmap[i, :, yo:yo+512, :]

        # Make sure no NaNs are in the DC map
        dcmap = np.where(np.isfinite(dcmap), dcmap, 0)

        # Subtract the DC map from a copy of the data model
        rampmodel_corr = datamodel.copy()
        rampmodel_corr.data = datamodel.data - dcmap

        # Save results to disk if requested.
        if save_results is True:
            hdu = fits.PrimaryHDU(sub)
            hdu.writeto(output_dir + fileroots[n] + 'udem1overfstep_diffim.fits',
                        overwrite=True)
            hdu = fits.PrimaryHDU(subcorr)
            hdu.writeto(output_dir + fileroots[n] + 'udem1overfstep_diffimcorr.fits',
                        overwrite=True)
            hdu = fits.PrimaryHDU(dcmap)
            hdu.writeto(output_dir + fileroots[n] + 'udem1overfstep_noisemap.fits',
                        overwrite=True)

            corrected_rampmodels.append(rampmodel_corr)
            rampmodel_corr.write(output_dir + fileroots[n] + 'udem1overfstep.fits')

        datamodel.close()

    return corrected_rampmodels


if __name__ == "__main__":
    # Open the uncal time series that needs 1/f correction
    exposurename = '/genesis/jwst/userland-soss/loic_review/GTO/wasp52b/IDTSOSS_clear_noisy.fits'
    outdir = '/genesis/jwst/userland-soss/loic_review/oneoverf/'

    # Run the 1/f correction step
    map = applycorrection([exposurename], output_dir=outdir, save_results=True)[0]

    # Write down the output corrected time series
    map.write('/genesis/jwst/userland-soss/loic_review/oneoverf/uncal_corrected.fits')
