import os.path

import matplotlib.pyplot as plt

import numpy as np

import scipy.interpolate

from scipy.signal import medfilt

import glob

from astropy.io import fits, ascii

from astropy.stats import sigma_clip

from jwst import datamodels

import SOSS.dms.soss_centroids as soss_centroids

import sys

from jwst import datamodels

from scipy.ndimage import rotate

from scipy.signal import savgol_filter

from scipy.signal import medfilt

import SOSS.trace.tracepol as tracepol

from astropy.nddata.bitmask import bitfield_to_boolean_mask

from jwst.datamodels import dqflags

from scipy.optimize import least_squares


def mediandev(x, axis=None):
    med = np.nanmedian(x, axis=axis)

    return np.nanmedian(np.abs(x - med), axis=axis) / 0.67449


def stack_rateints(rateints, outdir=None):

    '''
    Stack the CubeModel into a ImageModel and write to disk
    '''

    basename = os.path.splitext(rateints.meta.filename)[0]
    if outdir == None:
        outdir = './'
    stackname = outdir+'/'+basename+'_stack.fits'


    sci = np.copy(rateints.data)
    dq = np.copy(rateints.dq)
    err = np.copy(rateints.err)

    # Initiate output ImageModel stack
    dimy, dimx = rateints.meta.subarray.ysize, rateints.meta.subarray.xsize
    rate = datamodels.ImageModel((dimy, dimx))
    rate.update(rateints)
    rate.meta.exposure.nints = 1

    # Fill with median, make sure err is added in quadrature
    rate.data = np.nanmedian(sci, axis=0)
    mask = (dq > 0)
    err[mask] = np.nan
    rate.err = np.sqrt(np.nansum(err**2, axis=0)) / np.nansum(err*0+1, axis=0)
    bad = np.isfinite(rate.err) == False
    rate.err[bad] = 0.0
    rate.dq = np.nanmin(dq, axis=0)

    rate.write(stackname)

    return rate


def stack_datamodel(datamodel):

    '''
    Stack a time series.
    First put all non-zero DQ pixels to NaNs.
    Nan median and nan std along integrations axis.
    Remaining NaNs on the stack really are common bad pixels for all integrations.
    '''


    #import matplotlib.pyplot as plt


    # Mask pixels that have the DO_NOT_USE data quality flag set, EXCEPT the saturated pixels (because the ramp
    # fitting algorithm handles saturation).
    donotuse = bitfield_to_boolean_mask(datamodel.dq, ignore_flags=dqflags.pixel['DO_NOT_USE'], flip_bits=True)
    saturated = bitfield_to_boolean_mask(datamodel.dq, ignore_flags=dqflags.pixel['SATURATED'], flip_bits=True)
    #mask = donotuse & ~saturated
    mask = donotuse # Just this is enough to not flag the saturated pixels in the trace
    # Copy the datamodel to tmp
    #tmp_data = np.copy(datamodel.data)
    #tmp_data[mask] = np.nan

    #deepstack = np.nanmedian(tmp_data, axis=0)
    #rms = mediandev(tmp_data, axis=0)
    #dq = np.copy(deepstack) * 0
    #nan = ~np.isfinite(deepstack) | ~np.isfinite(rms)
    #dq[nan] = 1

    # prior to the HATP14b analysis, this worked:
    #bad = (datamodel.dq != 0) | ~np.isfinite(datamodel.dq)
    # bad if:
    # 1) DO_NOT_USE
    # OR
    # 2) different than zero but allowing for saturated only
    # OR
    # 3) NaN
    bad = (datamodel.dq % 2 == 1) | ((datamodel.dq != 0) & (datamodel.dq != 2)) | ~np.isfinite(datamodel.dq)
    tmp_data = datamodel.data * 1
    tmp_data[bad] = np.nan

    deepstack = np.nanmedian(tmp_data, axis=0)
    rms = mediandev(tmp_data, axis=0)
    dq = np.copy(deepstack) * 0
    nan = ~np.isfinite(deepstack) | ~np.isfinite(rms)
    dq[nan] = 1


    return deepstack, rms, dq


def build_mask_contamination(order, x, y, halfwidth=15, subarray='SUBSTRIP256',
                             output_boolean=False):
    """Mask out a contaminating trace from a field contaminant. Based on the
    order specified, a model of it is moved by x,y for orders 1 to 3.
    In the case of order 0, x,y is the desired position to mask, not a relative offset from nominal.
    """
    # Constrain within subarray
    if subarray == 'SUBSTRIP256':
        dimy = 256
    elif subarray == 'SUBSTRIP96':
        dimy = 96
    else:
        dimy = 2048

    if order >= 1:

        # Use the optics model calibrated on CV3 has the nominal trace positions
        wave = np.linspace(0.6,5.0,1000)
        tracepars = tracepol.get_tracepars('/Users/albert/NIRISS/SOSSpipeline/jwst-mtl/SOSS/trace/NIRISS_GR700_trace_extended.csv')
        x_ref, y_ref, _ = tracepol.wavelength_to_pix(wave, tracepars, m=order, subarray=subarray, oversample=1)

        # on the subarray real estate, the x,y trace is:
        xtrace = np.arange(2048)
        funky = scipy.interpolate.interp1d(x_ref + x, y_ref + y)
        # This is to prevent extrapolating (with a crash to call funky)
        bound = (xtrace > np.min(x_ref + x)) & (xtrace < np.max(x_ref + x))
        # ytrace is potentially narrower than 2048 pixels
        ytrace = funky(xtrace[bound])

        if False:
            plt.plot(x_ref, y_ref)
            plt.plot(x_ref+x, y_ref+y)
            plt.plot(xtrace[bound], ytrace)
            plt.show()

        # the y limits for the aperture:
        y_min, y_max = ytrace - halfwidth, ytrace + halfwidth

        # Create a coordinate grid.
        #xgrid, ygrid = np.meshgrid(np.arange(2048), np.arange(dimy))
        # This is modified to account for potentially narrower x size
        xgrid, ygrid = np.meshgrid(np.arange(np.size(ytrace)), np.arange(dimy))


        # Mask the pixels within a halfwidth of the trace center.
        mask_trace = np.abs(ygrid - ytrace) < halfwidth
        mask_trace = mask_trace.astype('float')

        # This is to put the potentially narrower mask into the full size
        final_mask_trace = np.zeros((dimy, 2048))
        final_mask_trace[:, bound] = mask_trace
        mask_trace = final_mask_trace

    elif order == 0:
        width = 5
        mask_trace = np.zeros((dimy, 2048))
        ymin = (y-halfwidth)
        if ymin < 0: ymin = 0
        ymax = (y+halfwidth)
        if ymax > dimy: ymax = dimy
        xmin = x - width
        if xmin < 0: xmin = 0
        xmax = x + width
        if xmax > 2048: xmax = 2048

        mask_trace[ymin:ymax, xmin:xmax] = 1

    hdu = fits.PrimaryHDU(mask_trace)
    hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/titi.fits', overwrite=True)

    if output_boolean:
        mask_trace = mask_trace == 1

    return mask_trace

def localbackground_subtraction(datamodel, trace_table_ref_file_name, width=25, back_offset=-25):

    '''
    For development purposes. Instead of using atoca's box extraction, have this which allows to play with
    the recipe (e.g. a background subtraction from pixels near the trace)
    '''
    from jwst.extract_1d.soss_extract.soss_boxextract import get_box_weights, box_extract

    print('Subtracting a nearby sky value from the data... Only order 1 but order 2 will be affected.')

    # Prepare the output data model
    # The background will be applied on the image pixels to then easily run the rest of the pipeline
    # as before instead of having to recreate extracted timeseries.
    #result = np.copy(datamodel)

    nintegrations, dimy, dimx = np.shape(datamodel.data)

    # Read the trace table reference file for this data set
    ref = fits.open(trace_table_ref_file_name)
    x_o1, y_o1, w_o1 = np.array(ref[1].data['X']), np.array(ref[1].data['Y']), np.array(ref[1].data['WAVELENGTH'])
    x_o2, y_o2, w_o2 = np.array(ref[2].data['X']), np.array(ref[2].data['Y']), np.array(ref[2].data['WAVELENGTH'])
    x_o3, y_o3, w_o3 = np.array(ref[3].data['X']), np.array(ref[3].data['Y']), np.array(ref[3].data['WAVELENGTH'])
    # Assumption is made later that x are integers from 0 to 2047
    x = np.arange(2048)
    sorted = np.argsort(x_o1)
    x_o1, y_o1, w_o1 = x_o1[sorted], y_o1[sorted], w_o1[sorted]
    y_o1 = np.interp(x, x_o1, y_o1)
    w_o1 = np.interp(x, x_o1, w_o1)
    x_o1 = x

    # Create an aperture mask based on those positions
    subarray_shape = (dimy, dimx)
    aper = get_box_weights(y_o1, width, subarray_shape, cols=x_o1)

    # Create a background aperture
    # the back_offset is how much in y the aperture is move relative to the center of the trace
    backap = get_box_weights(y_o1 + back_offset, width, subarray_shape, cols=x_o1)

    # Loop over images.
    for i in range(nintegrations):
        scidata = np.copy(datamodel.data[i])
        scierr = np.copy(datamodel.err[i])
        scimask = datamodel.dq[i] > 0
        # Measure the flux in the aperture
        cols, flux, flux_err, npix = box_extract(scidata, scierr, scimask, aper, cols = x_o1)
        # Measure the flux in the background aperture
        bcols, bflux, bflux_err, bnpix = box_extract(scidata, scierr, scimask, backap, cols = x_o1)

        #print(i, npix, bnpix)
        #plt.plot(npix)
        #plt.plot(bnpix)
        #plt.show()

        # Subtract background from aperture
        bflux_perpix = (bflux / bnpix)
        flux_corr = flux - npix * bflux_perpix

        #plt.plot(x_o1, flux)
        #plt.plot(x_o1, flux_corr)
        #plt.plot(x_o1, bflux)
        #plt.show()

        # To simplify the test for now, subtract the image pixels directly
        backdc = np.tile(bflux_perpix, dimy).reshape((dimy, dimx))
        #print(np.shape(datamodel.data))
        datamodel.data[i] = datamodel.data[i] - backdc

    return datamodel

def aperture_from_scratch(datamodel, norders=3, aphalfwidth=[40,20,20], outdir=None, datamodel_isfits=False,
                          verbose=False, trace_table_ref=None):
    '''Builds a mask centered on the traces from scratch using the edge centroid and build_mask_trace

    INPUT : a jwst datamodel. Or, if datamodel_isfits==True: 3D rateints or CDS, or a 2D high SNR stack
    OUTPUT : a cube of mask for each order
    '''

    if outdir == None:
        # Forge output directory where data may be written
        basename = os.path.basename(os.path.splitext(datamodel)[0])
        outdir = os.path.dirname(datamodel)+'/centroiding_'+basename+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if datamodel_isfits == True:
        if verbose: print('Reading a fits file')
        rateints = fits.getdata(datamodel)
        dims = np.shape(rateints)
        if dims[-2] == 2048: subarray = 'FULL'
        elif dims[-2] == 256: subarray = 'SUBSTRIP256'
        elif dims[-2] == 96: subarray = 'SUBSTRIP96'
        else:
            print('error should be size full ss96 or ss256')
            sys.exit()
        if verbose: print('Subarray is ', subarray)
    else:
        if verbose: print('A datamodel was passed.')
        rateints = datamodel.data
        subarray = datamodel.meta.subarray.name

    # Build a deep stack from all integrations
    dims = np.shape(rateints)
    if np.size(dims) == 3:
        stack = np.nanmedian(rateints, axis=0)
    elif np.size(dims) == 2:
        stack = rateints
    else:
        print('ERROR. Use a datamodel with 2 or 3 dimensions.')
        sys.exit()

    if trace_table_ref == None:
        # If no trace position reference file is passed then determine those using soss_centroids
        centroids = soss_centroids.get_soss_centroids(stack, mask=None, subarray=subarray, halfwidth=2,
                                                  poly_orders=None, apex_order1=None, calibrate=True, verbose=verbose,
                                                  outdir=outdir)
        x_o1, y_o1 = centroids['order 1']['X centroid'], centroids['order 1']['Y centroid']
        x_o2, y_o2 = centroids['order 2']['X centroid'], centroids['order 2']['Y centroid']
        x_o3, y_o3 = centroids['order 3']['X centroid'], centroids['order 3']['Y centroid']
        #w_o1 = centroids['order 1']['trace widths']
        #w_o2 = centroids['order 2']['trace widths']
        #w_o3 = centroids['order 3']['trace widths']
    else:
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


    #with open(outdir+'centroids_o1.txt', 'w') as the_file:
    #    the_file.write('Order, x, y, width\n')
    #    for i in range(len(x_o1)):
    #        the_file.write('{:} {:} {:} {:}\n'.format(1, x_o1[i], y_o1[i], w_o1[i]))
    #with open(outdir+'centroids_o2.txt', 'w') as the_file:
    #    the_file.write('Order, x, y, width\n')
    #    for i in range(len(x_o2)):
    #        the_file.write('{:} {:} {:} {:}\n'.format(2, x_o2[i], y_o2[i], w_o2[i]))
    #with open(outdir+'centroids_o3.txt', 'w') as the_file:
    #    the_file.write('Order, x, y, width\n')
    #    for i in range(len(x_o3)):
    #        the_file.write('{:} {:} {:} {:}\n'.format(3, x_o3[i], y_o3[i], w_o3[i]))


    # Create a cube containing the mask for all orders
    maskcube = np.zeros((norders,dims[-2],dims[-1]))

    for m in range(norders):
        if m == 0: ordercen = np.copy(y_o1)
        if m == 1: ordercen = np.copy(y_o2)
        if m == 2: ordercen = np.copy(y_o3)

        mask_trace = soss_centroids.build_mask_trace(ordercen, subarray=subarray,
                                                     halfwidth=aphalfwidth[m], extend_below=False, extend_above=False)
        mask = np.zeros(np.shape(mask_trace))
        mask[mask_trace == True] = 1

        maskcube[m,:,:] = np.copy(mask_trace)

        hdu = fits.PrimaryHDU(mask)
        hdu.writeto(outdir+'/trace_mask_order{:}.fits'.format(m+1), overwrite=True)

    return maskcube


def interp_badpix(image, noise, dq=None):
    '''
    Interpolates bad pixels of soss for a single image (e.g. a deepstack for instance)
    Define a rectangle as the kernel to determine the interpolating value from.
    '''

    kx, ky = 3, 1 # semi widths (rectangle size = 2*kx+1, 2*ky+1)
    #kernel = np.ones((2*ky+1, 2*kx+1))

    # Use only light sensitive pixels
    dimy, dimx = np.shape(image)
    if dimy == 96:
        ymin, ymax = 0, 96
    elif dimy == 2048:
        ymin, ymax = 5, 2043
    else:
        ymin, ymax = 0, 251
    x, y = np.meshgrid(np.arange(dimx), np.arange(dimy))
    notrefpix = ~((x <= 5) | (x >=2043) | (y <= ymin) | (y >= ymax))
    #plt.imshow(refpix)
    #plt.show()

    # Positions of the bad pixels
    bady, badx = np.where(~np.isfinite(image) & notrefpix)

    # If an optional DQ map is passed, add it to the list of bad pixels
    if dq != None:
        dq_y, dq_x = np.where(dq != 0)
        badx += dq_x
        bady += dq_y

    nbad = np.size(badx)
    for i in range(nbad):
        image[bady[i], badx[i]] = np.nanmedian(image[bady[i]-ky:bady[i]+ky+1, badx[i]-kx:badx[i]+kx+1])
        # error is the standard deviations of pixels that were used in the median. Not division by sqrt(n).
        #noise[bady[i], badx[i]] = np.nanstd(image[bady[i]-ky:bady[i]+ky+1, badx[i]-kx:badx[i]+kx+1])
        # That above yields pixels with huge errors, > x10 the surrounding pixels.
        # Instead: Error is the median of the errors for the pixels used in the interpolation.
        noise[bady[i], badx[i]] = np.nanmedian(noise[bady[i]-ky:bady[i]+ky+1, badx[i]-kx:badx[i]+kx+1])

    return image, noise

def add_manual_badpix(datamodel):

    pid = datamodel.meta.observation.program_number
    obs = datamodel.meta.observation.observation_number

    print('Adding bad pixels manually based on the program ID {:} and observation number {:}'.format(pid, obs))

    if pid == '01091':
        # manually selected bad pixels in ds9 and saved as 2-cols ascii region file
        ds9reg = '/Users/albert/NIRISS/SOSSpipeline/jwst-mtl/SOSS/Commissioning/files/manual_badpix_01091.reg'
        table = ascii.read(ds9reg)
        x = np.array(table['col1'])
        y = np.array(table['col2'])
    if pid == '01201' and obs == '008':
        # WASP107b
        ds9reg = '/Users/albert/NIRISS/SOSSpipeline/jwst-mtl/SOSS/Commissioning/files/manual_badpix_01201008.reg'
        table = ascii.read(ds9reg)
        x = np.array(table['col1'])
        y = np.array(table['col2'])
    elif pid == '01541':
        # manually selected bad pixels in ds9 and saved as 2-cols ascii region file
        ds9reg = '/Users/albert/NIRISS/SOSSpipeline/jwst-mtl/SOSS/Commissioning/files/manual_badpix_01541.reg'
        table = ascii.read(ds9reg)
        x = np.array(table['col1'])
        y = np.array(table['col2'])
    elif pid == '02589':
        # manually selected bad pixels in ds9 and saved as 2-cols ascii region file
        ds9reg = '/Users/albert/NIRISS/SOSSpipeline/jwst-mtl/SOSS/Commissioning/files/manual_badpix_02589_obs03.reg'
        table = ascii.read(ds9reg)
        x = np.array(table['col1'])
        y = np.array(table['col2'])
    else:
        x = np.array([])
        y = np.array([])

    # Remove 1 so that first pixel is 0,0
    xlist = list(x - 1)
    ylist = list(y - 1)

    print('x : ', x)
    print('y : ', y)

    print('DQ before:')
    print(datamodel.dq[0, ylist, xlist])
    datamodel.dq[:, ylist, xlist] = 1
    print('DQ after:')
    print(datamodel.dq[0, ylist, xlist])



    return datamodel


def soss_interp_badpix(modelin, outdir, save_results=False):

    # Create a deep stack from the time series.
    # Interpolate on that deep stack.
    # Then use the deepstack as pixel replacement values in single integrations.

    # Create a deep stack from the time series
    stack, stackrms, stackdq = stack_datamodel(modelin)
    hdu = fits.PrimaryHDU([stack, stackrms, stackdq])
    hdu.writeto(outdir+'/test_pre_interpbadpstack.fits', overwrite=True)

    # Interpolate the deep stack
    cleanstack, cleanstack_rms = interp_badpix(stack, stackrms)
    hdu = fits.PrimaryHDU([cleanstack, cleanstack_rms])
    hdu.writeto(outdir+'/test_post_interpbadpstack.fits', overwrite=True)

    # Apply clean stack pixel values to each integration's bad pixels
    #
    # Use only light sensitive pixels
    nint, dimy, dimx = np.shape(modelin.data)
    if dimy == 96:
        ymin, ymax = 0, 96
    elif dimy == 2048:
        ymin, ymax = 5, 2043
    else:
        ymin, ymax = 0, 251
    x, y = np.meshgrid(np.arange(dimx), np.arange(dimy))
    notrefpix = ~((x <= 5) | (x >=2043) | (y <= ymin) | (y >= ymax))
    #
    # Loop over all integrations
    for i in range(nint):
        # Mask pixels that have the DO_NOT_USE data quality flag set
        donotuse = bitfield_to_boolean_mask(modelin.dq[i], ignore_flags=dqflags.pixel['DO_NOT_USE'], flip_bits=True)
        ind = notrefpix & ((~np.isfinite(modelin.data[i])) | (~np.isfinite(modelin.err[i])) | donotuse)
        # Initial method - Replace bad pixel here by the clean stack value
        #modelin.data[i][ind] = np.copy(cleanstack[ind])
        #modelin.err[i][ind] = np.copy(cleanstack_rms[ind])
        # Final method - Replace bad pixel by a realization of the clean stack
        modelin.data[i][ind] = np.random.default_rng().normal(cleanstack[ind], cleanstack_rms[ind])
        modelin.err[i][ind] = np.copy(cleanstack_rms[ind])
        # Set the DQ map to good for all pixels except ref pixels
        modelin.dq[i][notrefpix] = 0

    basename = os.path.splitext(modelin.meta.filename)[0]
    basename = basename.split('_nis')[0] + '_nis'
    if save_results:
        modelin.write(outdir+'/'+basename+'_badpixinterp.fits')
    modelin.meta.filename = basename

    #hdu = fits.PrimaryHDU([modelin.data, modelin.dq])
    #hdu.writeto(outdir+'/test_interpolated_segment.fits', overwrite=True)

    return modelin


def remove_nans(datamodel, outdir=None, save_results=False):
    # Checks that the JWST Data Model does not contains NaNs
    # This is really a final check (bad pixels were already interpolated)
    # in order to not have extract_1d crash because of NaNs

    # Forge output directory where data may be written
    basename = os.path.splitext(datamodel.meta.filename)[0]
    basename = basename.split('_nis')[0]+'_nis'
    print('basename {:}'.format(basename))
    if outdir == None:
        outdir = os.path.curdir+'/'
    if not os.path.exists(outdir):
        if save_results == True: os.makedirs(outdir)
    output_supp = outdir+'/supplemental_'+basename+'/'
    if not os.path.exists(output_supp):
        if save_results == True: os.makedirs(output_supp)

    modelout = datamodel.copy()

    ind = (~np.isfinite(datamodel.data)) | (~np.isfinite(datamodel.err))
    modelout.data[ind] = 0
    modelout.err[ind] = np.nanmedian(datamodel.err)*10
    modelout.dq[ind] += 1

    # Check that the exposure type is NIS_SOSS
    modelout.meta.exposure.type = 'NIS_SOSS'

    if save_results:
        filename = modelout.meta.filename
        modelout.write(output_supp+'/datamodel_nanfree_before_extract1d.fits')
        modelout.meta.filename = filename

    return modelout


def box_extraction(datamodel, ref_spectrace, width=30):
    '''
    Box extraction to bypass ATOCA's
    '''




    return



def background_subtraction(datamodel, aphalfwidth=[40,30,30], outdir=None, verbose=False,
                           contamination_mask=None, trace_table_ref=None, save_results=False):

    nint, dimy, dimx = np.shape(datamodel.data)

    basename = os.path.splitext(datamodel.meta.filename)[0]
    basename = basename.split('_nis')[0] + '_nis'
    if outdir == None:
        outdir = './'
        cntrdir = './backgroundsub_'+basename+'/'
    else:
        cntrdir = outdir+'/backgroundsub_'+basename+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(cntrdir):
        os.makedirs(cntrdir)
    maskeddata = np.copy(datamodel.data)

    print('New background subtraction algorithm (as of Aug 4 2022) - Modelling the 1D profile')

    # Apply bad pixel masking to the input data
    print('Masking the bad pixels !=0 in the DQ map')
    maskeddata[datamodel.dq != 0] = np.nan
    if contamination_mask is not None:
        print('Masking the contaminating traces from field stars (orders 0 to 2) using the passed mask.')
        contmask = fits.getdata(contamination_mask)
        contmask = np.where(contmask >= 1, 1, 0)
        # add the contamintion masked pixels
        contpix = contmask == 1
        maskeddata[:, contpix] = np.nan

    # Make a mask of the traces
    print('Masking the spectral traces')
    maskcube = aperture_from_scratch(datamodel, aphalfwidth=aphalfwidth, outdir=cntrdir, verbose=verbose,
                                     trace_table_ref=trace_table_ref)

    # Crunch the cube (one order per slice) to a 2D mask
    mask = np.sum(maskcube, axis=0, dtype='bool')
    # Apply aperture masking to the input data
    maskeddata[:, mask] = np.nan
    hdu = fits.PrimaryHDU(maskeddata)
    print('Saving the mask used for background estimation as background_mask')
    hdu.writeto(cntrdir+'background_mask.fits', overwrite=True)

    # Construct the background fit
    background_model = construct_background(maskeddata, tilt=-1.8, isafitsfile=False, metric='10pct',
                               savetest=True, outdir=cntrdir)


    # Perform the subtraction on the output data model
    output = datamodel.copy()
    output.data = datamodel.data - background_model

    if save_results:
        output.write(outdir+'/'+basename+'_backsubstep.fits')
        #hdu = fits.PrimaryHDU(output.data)
        #hdu.writeto(outdir+'/'+basename+'_backsubtracted.fits', overwrite=True)

    # Make sure filename is back to normal
    output.meta.filename = basename

    return output


def background_subtraction_v1(datamodel, aphalfwidth=[30,30,30], outdir=None, verbose=False,
                           applyonintegrations=False, contamination_mask=None, override_background=None,
                           trace_table_ref=None):

    nint, dimy, dimx = np.shape(datamodel.data)

    basename = os.path.splitext(datamodel.meta.filename)[0]
    if outdir == None:
        outdir = './'
        cntrdir = './backgroundsub_'+basename+'/'
    else:
        cntrdir = outdir+'/backgroundsub_'+basename+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(cntrdir):
        os.makedirs(cntrdir)
    maskeddata = np.copy(datamodel.data)

    print('Hello')

    if False:
        if contamination_mask is not None:
            contmask = fits.getdata(contamination_mask)
            contmask = np.where(contmask >= 1, 1, 0)
            print('contmask shape: ', np.shape(contmask))
            maskeddata[:, contmask] = np.nan
            print(np.shape(maskeddata[:, contmask]))
            # mark contamination masked as do not use
            # do_not_use (apply to input and return as output)
            for i in range(nint):
                tmp = datamodel.dq[i]
                tmp[contmask] = tmp[contmask] + 1
                datamodel.dq[i] = np.copy(tmp)
            print('Allo')
    # Apply bad pixel masking to the input data
    maskeddata[datamodel.dq != 0] = np.nan
    # add the contamintion masked pixels
    contpix = contmask >= 1
    for i in range(nint):
        maskeddata[i][contmask] = np.nan

    print('Hi')
    # Make a mask of the traces
    maskcube = aperture_from_scratch(datamodel, aphalfwidth=aphalfwidth, outdir=cntrdir, verbose=verbose,
                                     trace_table_ref=trace_table_ref)

    print('Ni Hao')
    # Crunch the cube (one order per slice) to a 2D mask
    mask = np.sum(maskcube, axis=0, dtype='bool')
    # Apply aperture masking to the input data
    maskeddata[:, mask] = np.nan
    hdu = fits.PrimaryHDU(maskeddata)
    hdu.writeto(cntrdir+'background_mask.fits', overwrite=True)

    # Bottom portion not usable in the FULL mode and skews levels estimates
    if datamodel.meta.subarray.name == 'FULL': maskeddata[:, 0:1024, :] = np.nan

    # Identify pixels free of astrophysical signal (e.g. 15th percentile)
    if applyonintegrations == True:
        # Measure level on each integration
        levels = np.nanpercentile(maskeddata, 15, axis=1)
        print('levels shape', np.shape(levels))
        nlevels = nint
    else:
        # Measure level on deep stack of all integrations (default)
        maskeddata = np.nanmedian(maskeddata, axis=0)
        levels = np.nanpercentile(maskeddata, 15, axis=0)
        nlevels = 1

    # Open the reference background image
    if override_background is None:
        ### need to handle default background downlaoded from CRDS, eventually
        print('ERROR: no background specified. Crash!')
        sys.exit()
    else:
        backref = fits.getdata(override_background)

    # Check that the backref is 2048x2048
    bdimy, bdimx = np.shape(backref)
    if bdimy == 256:
        backref_full = np.zeros((2048,2048)) * np.nan
        backref_full[-256:, :] = np.copy(backref)
        backref = np.copy(backref_full)
    elif bdimy == 2048:
        # nothing to do
        print('')
    else:
        print('ERROR: the override_background passed as input is not of 256x2048 nor 2048x2048!')
        sysy.exit()

    # Find the single scale that best matches the reference background
    if datamodel.meta.subarray.name == 'FULL':
        rows = slice(0, 2048) #not all contain background!
        maskrows = slice(1024, 2048)
    elif datamodel.meta.subarray.name == 'SUBSTRIP256':
        rows = slice(1792, 2048)
        maskrows = slice(1792, 2048)
    elif datamodel.meta.subarray.name == 'SUBSTRIP96':
        rows = slice(1802, 1898)
        maskrows = slice(1802, 1898)
    else:
        raise ValueError('SUBARRAY must be one of SUBSTRIP96, SUBSTRIP256 or FULL')

    # stablish the reference background level when crunched into a single row
    reflevel = np.nanmedian(backref[maskrows, :], axis=0)
    reflevel[0:5] = np.nan
    reflevel[2044:2048] = np.nan

    # Scaling to apply to the ref level to bring it to that measured
    corrscale = np.nanmedian(levels / reflevel, axis=-1)
    print('Scaling of the ref background (should not be negative):', corrscale)
    print('Background correction scale shape ', np.shape(corrscale))

    # plot the measured background levels and ref level
    if applyonintegrations == True:
        for i in range(nlevels):
            plt.scatter(np.arange(2048), levels[i,:], marker='.')
            plt.plot(np.arange(2048), reflevel * corrscale[i], color='red', ls='dotted')
        plt.plot(np.arange(2048), reflevel, color='black', label='Background Ref File')

    else:
        plt.scatter(np.arange(2048), levels, marker='.', label='Observed background')
        plt.plot(np.arange(2048), reflevel, color='black', label='Background Ref File')
        plt.plot(np.arange(2048), reflevel*corrscale, color='red', ls='dotted', label='Ref scaled to fit Obs')

    plt.legend()
    plt.title('Background fitting')
    #plt.show()
    plt.savefig(cntrdir+'background_fitting.png', overwrite=True)

    # Construct the backgroud subarray from the reference file
    backtosub = backref[rows, :]
    # Perform the subtraction on the output data model
    output = datamodel.copy()
    if applyonintegrations == True:
        for i in range(nlevels):
            if corrscale[i] > 0:
                output.data[i, :, :] = datamodel.data[i, :, :] - corrscale[i] * backtosub
            else:
                print('Warning. The background scaling was not performed because it was negative. For integration ', i+1)
    else:
        if corrscale > 0:
            output.data = datamodel.data - corrscale * backtosub
        else:
            print('Warning. The background scaling was not performed because it was negative.')

    #output.write(outdir+'/'+basename+'_backsubtracted.fits', overwrite=True)
    hdu = fits.PrimaryHDU(output.data)
    hdu.writeto(outdir+'/'+basename+'_backsubtracted.fits', overwrite=True)

    return output


def stack_multi_spec(multi_spec, quantities=('WAVELENGTH', 'FLUX', 'FLUX_ERROR')):
    ''' Convert jwst.datamodels.MultiSpecModel to a dictionnary of 2d-arrays with
    the first axis being the integration.

    Example
    -------
    To get the flux from order 2, simply,
    >>> quantity = 'FLUX'
    >>> order = 2
    >>> all_spec = stack_multi_spec(multi_spec_object)
    >>> all_flux_array = all_spec[order][quantity]

    Author: Antoine Darveau-Bernier
    '''

    all_spec = {sp_ord: {quantity: [] for quantity in quantities}
                for sp_ord in [1, 2, 3]}
    for spec in multi_spec.spec:
        sp_ord = spec.spectral_order
        for quantity in quantities:
            all_spec[sp_ord][quantity].append(spec.spec_table[quantity])

    for sp_ord in all_spec:
        for key in all_spec[sp_ord]:
            all_spec[sp_ord][key] = np.array(all_spec[sp_ord][key])

    return all_spec


def plot_timeseries(spectrum_file, outdir=None, norder=3):

    # Forge output directory where plots will be written
    if outdir == None:
        basename = os.path.basename(os.path.splitext(spectrum_file)[0])
        basename = basename.split('_nis')[0] + '_nis'
        outdir = os.path.dirname(spectrum_file)+'/supplemental_'+basename+'/'
    # Create the output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Start processing the spectra file
    multispec = datamodels.open(spectrum_file)

    # spectra are stored at indice 1 (order 1), then 2 (order2) then 3 (order 3) then 4 (order 1, 2nd time step), ...
    # TODO Manage nint and norder better
    #nint = multispec.meta.exposure.nints
    norder = 3
    nint = int(np.shape(multispec.spec)[0] / norder)
    #norder = int(np.shape(multispec.spec)[0] / nint)

    print('nint = {:}, norder= {:}'.format(nint, norder))

    # format differently
    wavelength = np.zeros((nint, norder, 2048))
    flux = np.zeros((nint, norder, 2048))
    fluxerr = np.zeros((nint, norder, 2048))
    for i in range(nint):
        for m in range(norder):
            nnn = i * norder + m
            print(i, m, nnn)
            wavelength[i, m, :] = multispec.spec[nnn].spec_table['wavelength']
            flux[i, m, :] = multispec.spec[nnn].spec_table['flux']
            fluxerr[i, m, :] = multispec.spec[nnn].spec_table['flux_error']

    # Normalize each wavelength
    fluxnorm = flux / np.nanmedian(flux, axis=0)

    # Write flux vs column as a fits image
    hdu = fits.PrimaryHDU(flux.transpose(1, 0, 2))
    hdu.writeto(outdir+'timeseries_greyscale_rawflux.fits', overwrite=True)

    # Produce a wavelength calibrated spectrum time-series
    for m in range(norder):
        if m == 0:
            dy = 0.05
        elif m == 1:
            dy = 0.1
        else:
            dy = 0.2
        vmin, vmax = 0, np.nanpercentile(flux[:,m,:], 95)
        yamp = vmax - vmin
        vmaxall = vmax+nint*dy*yamp
        print(vmin, vmax, yamp, vmaxall)

        fig = plt.figure(figsize=(6,6*0.5*(1+nint*dy)))
        for i in range(nint):
            plt.plot(wavelength[i, m, :], flux[i, m, :]+i*dy*yamp, color='black')
        plt.ylim((vmin, vmaxall))
        plt.title('Extracted Order {:}'.format(m+1))
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('extracted Flux (DN/sec)')
        plt.savefig(outdir+'extractedflux{:}.png'.format(i+1))
        #plt.show()
        plt.close()



    # Produce that Raw extracted flux greyscale
    for i in range(norder):
        fig = plt.figure(figsize=(8,8))
        plt.imshow(flux[:,i,:], origin='lower')
        plt.title('Order {:} Raw Extracted Flux'.format(i+1))
        plt.xlabel('Column')
        plt.ylabel('Integration Number')
        plt.savefig(outdir+'timeseries_greyscale_rawflux_order{:}.png'.format(i+1))
        plt.close()

    # Write flux vs column as a fits image
    hdu = fits.PrimaryHDU(fluxnorm.transpose(1, 0, 2))
    hdu.writeto(outdir+'timeseries_greyscale_normalizedflux.fits', overwrite=True)

    # Produce that Normalized flux greyscale
    for i in range(norder):
        fig = plt.figure(figsize=(8,8))
        plt.imshow(fluxnorm[:,i,:], origin='lower')
        plt.title('Order {:} Normalized Flux'.format(i+1))
        plt.xlabel('Column')
        plt.ylabel('Integration Number')
        plt.savefig(outdir+'timeseries_greyscale_normalizedflux_order{:}.png'.format(i+1))
        plt.close()

    #plt.figure()
    #for i in range(nint):
    #    plt.plot(wavelength[i, 0], flux[i, 0] + 0.02 * i)
    #plt.show()

    return


def check_atoca_residuals(fedto_atoca_filename, atoca_model_filename, outdir=None):
    '''
    Check that the ATOCA modelling went well by looking at residual maps for each order.
    '''



    # Forge output directory where plots will be written
    if outdir == None:
        basename = os.path.basename(os.path.splitext(fedto_atoca_filename)[0])
        outdir = os.path.dirname(fedto_atoca_filename)+'/atoca_'+basename+'/'
    # Create the output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Read the file fed to ATOCA
    obser = datamodels.open(fedto_atoca_filename)
    #<CubeModel(20, 256, 2048)>
    nints = obser.meta.exposure.nints
    if obser.meta.subarray.name == 'SUBSTRIP96':
        norders = 2
    else:
        norders = 3

    # Read the traces modelled by ATOCA
    model = datamodels.open(atoca_model_filename)
    #<SossExtractModel>

    # Residual map after subtracting both orders' model
    cube = obser.data - model.order1 - model.order2
    hdu = fits.PrimaryHDU(cube)
    hdu.writeto(outdir+'residualmap_bothorders.fits', overwrite=True)

    # Residual map after subtracting order 1 model
    cube = obser.data - model.order1
    hdu = fits.PrimaryHDU(cube)
    hdu.writeto(outdir+'residualmap_order2.fits', overwrite=True)

    # Residual map after subtracting order 2 model
    cube = obser.data - model.order2
    hdu = fits.PrimaryHDU(cube)
    hdu.writeto(outdir+'residualmap_order1.fits', overwrite=True)

    return

def make_mask_nis17():

    stackname = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/stack_flux.fits'
    checkname = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/check.fits'
    maskname = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/mask_contamination.fits'
    stack = fits.getdata(stackname)

    mask = build_mask_contamination(0, 1376, 111)
    mask += build_mask_contamination(0, 1867, 75)
    mask += build_mask_contamination(1, -680, 153)

    hdu = fits.PrimaryHDU([stack, mask])
    hdu.writeto(checkname, overwrite=True)

    hdu = fits.PrimaryHDU(mask)
    hdu.writeto(maskname, overwrite=True)

    return

def make_mask_nis18obs2():

    stackname = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/stack_clear_reobservation.fits'
    checkname = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/check.fits'
    maskname = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/mask_contamination.fits'
    stack = fits.getdata(stackname)

    #mask = build_mask_contamination(0, 1376, 111)
    #mask += build_mask_contamination(0, 1867, 75)
    mask = build_mask_contamination(1, 60, 136)
    mask += build_mask_contamination(0, 1839, 220)

    hdu = fits.PrimaryHDU([stack, mask])
    hdu.writeto(checkname, overwrite=True)

    hdu = fits.PrimaryHDU(mask)
    hdu.writeto(maskname, overwrite=True)

    return

def make_mask_02589_obs001():
    # T1 obs 1 contamination mask

    stackname = '/Users/albert/NIRISS/Commissioning/analysis/T1/stack_t1obs1.fits'
    checkname = '/Users/albert/NIRISS/Commissioning/analysis/T1/check.fits'
    maskname = '/Users/albert/NIRISS/Commissioning/analysis/T1/mask_contamination.fits'
    stack = fits.getdata(stackname)

    mask = build_mask_contamination(2, -900, -140)
    mask += build_mask_contamination(0, 866, 175)
    mask += build_mask_contamination(0, 1003, 204)
    mask += build_mask_contamination(0, 1034, 79)
    mask += build_mask_contamination(0, 840, 12)
    mask += build_mask_contamination(0, 1363, 74)
    mask += build_mask_contamination(0, 1519, 210)
    mask += build_mask_contamination(0, 1795, 192)

    hdu = fits.PrimaryHDU([stack, mask])
    hdu.writeto(checkname, overwrite=True)

    hdu = fits.PrimaryHDU(mask)
    hdu.writeto(maskname, overwrite=True)


    return

def make_mask_02589_obs003():
    # T1 obs 3 contamination mask

    stackname = '/Users/albert/NIRISS/Commissioning/analysis/T1_4/stack_t1obs4.fits'
    checkname = '/Users/albert/NIRISS/Commissioning/analysis/T1_4/check.fits'
    maskname = '/Users/albert/NIRISS/Commissioning/analysis/T1_4/mask_contamination.fits'
    stack = fits.getdata(stackname)
    stack = stack[0] # has 2 slices in the stack



    #mask = build_mask_contamination(2, -900, -140)
    mask = build_mask_contamination(0, 1851, 181)
    mask += build_mask_contamination(0, 1702, 164)
    mask += build_mask_contamination(0, 1616, 132)
    mask += build_mask_contamination(0, 2000, 225)
    mask += build_mask_contamination(0, 1363, 78)
    mask += build_mask_contamination(0, 1216, 183)
    mask += build_mask_contamination(0, 1161, 183)
    mask += build_mask_contamination(0, 1870, 2)
    mask += build_mask_contamination(0, 1576, 174)
    mask += build_mask_contamination(0, 948, 171)
    mask += build_mask_contamination(0, 1111, 210)

    # further mask order 2 right of x=1500
    #mask[100:175,1500:] = 1

    hdu = fits.PrimaryHDU([stack, mask])
    hdu.writeto(checkname, overwrite=True)

    hdu = fits.PrimaryHDU(mask)
    hdu.writeto(maskname, overwrite=True)


    return


def combine_segments(prefix):
    print()
    return


def combine_multi_spec(wildcard, outputname):
    """
    Example:
    #>>> filename_list = ['spec_part1.fits', 'spec_part2.fits', 'etc.fits']
    #>>> multi_spec_list = [datamodels.open(fname) for fname in filename_list]
    #>>> combined = combine_multi_spec(multi_spec_list)
    #>>> combined.save('spec_combined.fits')
    """
    filename_list = sorted(glob.glob(wildcard))
    print('List of extract1d spectra (check here that they are ordered).')
    print(filename_list)
    # read the files
    print('Opening the multispec - takes a while...')
    multi_spec_list = [datamodels.open(fname) for fname in filename_list]

    # Init output spec
    combined = datamodels.MultiSpecModel()

    # Update meta data based on first MultiSpecModel in the list
    combined.update(multi_spec_list[0])

    # Do not take the same filename though
    combined.meta.filename = None

    # Iterate over all objects in the list
    print('Combining all spectra. Takes a while...')
    for multi_spec_obj in multi_spec_list:
        print('A segment starts...')
        # Iterate over all SingleSpecModel
        for single_spec_obj in multi_spec_obj.spec:
            # Append single spec to output
            combined.spec.append(single_spec_obj)

    combined.save(outputname)

    return


## Example
#filename_list = ['spec_part1.fits', 'spec_part2.fits', 'etc.fits']
#multi_spec_list = [datamodels.open(fname) for fname in filename_list]
#combined = combine_multi_spec(multi_spec_list)
#combined.save('spec_combined.fits')

def combine_timeseries(wildcard, outputname_normalized, outputname_rawflux):

    nisfiles= sorted(glob.glob(wildcard))
    print(nisfiles)

    nints = 0
    for ts_file in nisfiles:
        im = fits.getdata(ts_file)
        norder, ninteg, dimx = np.shape(im)
        nints = nints + ninteg

    ts = np.zeros((3, nints, 2048))
    current = 0
    for ts_file in nisfiles:
        im = fits.getdata(ts_file)
        norder, ninteg, dimx = np.shape(im)
        ts[:, current:current+ninteg, :] = np.copy(im)
        current += ninteg

    # Save the raw flux (no normalization)
    hdu = fits.PrimaryHDU(ts)
    hdu.writeto(outputname_rawflux, overwrite=True)

    print(np.shape(ts))
    # normalize each wavelength
    ts = ts.transpose(1, 0, 2)
    ts = ts / np.nanmedian(ts, axis=0)
    ts = ts.transpose(1, 0, 2)

    hdu = fits.PrimaryHDU(ts)
    hdu.writeto(outputname_normalized, overwrite=True)

    return

def median_absolute_spectrum(photomstep_spectrum, outputname):
    # Reads the time series of spectra calibrated in absolute flux (from photomstep)
    # and combine them to output the median, combined spectrum, along with rms.

    # Start processing the spectra file
    multispec = datamodels.open(photomstep_spectrum)

    # spectra are stored at indice 1 (order 1), then 2 (order2) then 3 (order 3) then 4 (order 1, 2nd time step), ...
    # TODO Manage nint and norder better
    # nint = multispec.meta.exposure.nints
    norder = 3
    nint = int(np.shape(multispec.spec)[0] / norder)
    # norder = int(np.shape(multispec.spec)[0] / nint)

    print('nint = {:}, norder= {:}'.format(nint, norder))

    # format differently
    wavelength = np.zeros((nint, norder, 2048))
    flux = np.zeros((nint, norder, 2048))
    fluxerr = np.zeros((nint, norder, 2048))
    for i in range(nint):
        for m in range(norder):
            nnn = i * norder + m
            print(i, m, nnn)
            wavelength[i, m, :] = multispec.spec[nnn].spec_table['wavelength']
            flux[i, m, :] = multispec.spec[nnn].spec_table['flux']
            fluxerr[i, m, :] = multispec.spec[nnn].spec_table['flux_error']

    # make a quick white light to reject in-transit spectra later
    wl = np.nansum(flux[:,0,:], axis=-1)
    wl = wl/np.nanmedian(wl, axis=0)
    wlmed = np.nanmedian(wl)
    wldev = mediandev(wl)
    print('whitelight = ', wl)
    print('whitelight median = ', wlmed)
    print('whitelight deviations = ', wldev)
    oot = wl > (wlmed - 3 * wldev)

    # Start building the output fits file.
    hdul = list()
    hdu = fits.PrimaryHDU()
    hdu.header['DESCRIP'] = ('Median Out-of-transit spectrum', 'Desription of the file')
    hdu.header['AUTHOR'] = ('Loic Albert', 'Author of the file')
    hdul.append(hdu)

    for m in range(norder):
        spec_median = np.nanmedian(flux[oot,m,:], axis=0)
        spec_rms = mediandev(flux[oot,m,:], axis=0)

        # Create the order 1 extension.
        # Order 1 table.
        col1 = fits.Column(name='micron', format='F', array=wavelength[0, m, :])
        col2 = fits.Column(name='spectrum', format='F', array=spec_median)
        col3 = fits.Column(name='rms', format='F', array=spec_rms)
        cols = fits.ColDefs([col1, col2, col3])

        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header['ORDER'] = (m+1, 'Spectral order.')
        hdu.header['EXTNAME'] = 'ORDER {:}'.format(m+1)
        hdul.append(hdu)

    hdul = fits.HDUList(hdul)
    hdul.writeto(outputname, overwrite=True)





    return



def performance_fluxcal(spectrum_file, outdir=None, title=''):
    '''
    Analysis of the Flux Calibration time series to assess performance.
    Compare flux with expected level of noise.
    '''

    print('WARNING! Please delete the old extracted_spectra_*.fits files if you are running this on new data.')
    print('WARNING! Please delete the old extracted_spectra_*.fits files if you are running this on new data.')
    print('WARNING! Please delete the old extracted_spectra_*.fits files if you are running this on new data.')
    print('WARNING! Please delete the old extracted_spectra_*.fits files if you are running this on new data.')

    # Start processing the spectra file
    print('Reading the file containing the spectra Data Model...')
    if os.path.isfile(outdir+'extracted_spectra_wavelength.fits') & \
        os.path.isfile(outdir + 'extracted_spectra_flux.fits') & \
        os.path.isfile(outdir + 'extracted_spectra_fluxerr.fits'):
        print('Opening already formatted (wavelength, flux, fluxerr) fits files.')
        wavelength = fits.getdata(outdir+'extracted_spectra_wavelength.fits')
        flux = fits.getdata(outdir+'extracted_spectra_flux.fits')
        fluxerr = fits.getdata(outdir + 'extracted_spectra_fluxerr.fits')
        nint, norder, _ = np.shape(flux)
    else:
        multispec = datamodels.open(spectrum_file)

        # spectra are stored at indice 1 (order 1), then 2 (order2) then 3 (order 3) then 4 (order 1, 2nd time step), ...
        # TODO Manage nint and norder better
        #nint = multispec.meta.exposure.nints
        norder = 3
        nint = int(np.shape(multispec.spec)[0] / norder)
        print('nint = {:}, norder= {:}'.format(nint, norder))

        # format differently
        print('Format the data as numpy arrays...')
        wavelength = np.zeros((nint, norder, 2048))
        flux = np.zeros((nint, norder, 2048))
        fluxerr = np.zeros((nint, norder, 2048))
        for i in range(nint):
            if i % 10 == 0: print('Reading integration {:} of {:}'.format(i+1, nint))
            for m in range(norder):
                nnn = i * norder + m
                #print(i, m, nnn)
                wavelength[i, m, :] = multispec.spec[nnn].spec_table['wavelength']
                flux[i, m, :] = multispec.spec[nnn].spec_table['flux']
                fluxerr[i, m, :] = multispec.spec[nnn].spec_table['flux_error']

    if outdir != None:
        if os.path.isfile(outdir + 'extracted_spectra_wavelength.fits') & \
            os.path.isfile(outdir + 'extracted_spectra_flux.fits') & \
            os.path.isfile(outdir + 'extracted_spectra_fluxerr.fits'):
            print('')
        else:
            print('Saving properly formatted spectra.')
            hdu = fits.PrimaryHDU(wavelength)
            hdu.writeto(outdir+'extracted_spectra_wavelength.fits', overwrite=True)
            hdu = fits.PrimaryHDU(flux)
            hdu.writeto(outdir+'extracted_spectra_flux.fits', overwrite=True)
            hdu = fits.PrimaryHDU(fluxerr)
            hdu.writeto(outdir + 'extracted_spectra_fluxerr.fits', overwrite=True)

    if True:
        # rms across all integrations
        rms = mediandev(flux, axis=0)
        # error from the DMS
        err = np.median(fluxerr, axis=0)
        # Subtract the "white light" curve, i.e. the systematics along the TS integrations
        systematics = np.nansum(flux, axis=2)
        systematics /= np.nanmedian(systematics, axis=0)
        flux_syscorr = flux - systematics[:,:,np.newaxis]
        #flux_syscorr = np.copy(flux)
        #for i in range(2048):
        #    flux_syscorr[:,:,i] = flux[:,:,i] - systematics[:,:]
        rms_syscorr = mediandev(flux_syscorr, axis=0)


        # Infer the level of noise unaccounted for
        aperwidth = 25
        othernoise_per_col_per_integ = np.sqrt(rms**2 - err**2)
        othernoise_per_pixel = np.sqrt(3*5.494) * othernoise_per_col_per_integ / np.sqrt(aperwidth) / np.sqrt(3)
        plt.scatter(wavelength[0,0,:], othernoise_per_col_per_integ[0,:])
        plt.scatter(wavelength[0,1,:], othernoise_per_col_per_integ[1,:])
        plt.scatter(wavelength[0,2,:], othernoise_per_col_per_integ[2,:])
        plt.grid()
        plt.show()

        plt.scatter(wavelength[0,0,:], rms[0,:], label='rms')
        plt.scatter(wavelength[0,0,:], err[0,:], label='err')
        plt.scatter(wavelength[0,0,:], np.sqrt(rms[0,:]**2-err[0,:]**2), label='sqrt(rms**2-err**2)')
        plt.ylabel('Variance')
        plt.xlabel('Wavelength [microns]')
        plt.legend()
        plt.grid()
        plt.show()


        # Figure for the SOSS mode paper
        rat = (rms/err)
        norders, ncols = np.shape(rat)
        print(norders, ncols)
        binsize = 64
        nbins = 2048 // binsize
        # Group the 2048 columns in bins.
        # Reshape in 4D then sum the additional dimension.
        tmp = rat[:, 0:binsize * nbins].reshape((norders,nbins,binsize))
        rat_binned = np.nanmedian(tmp, axis=-1)
        tmp = wavelength[0,:, 0:binsize * nbins].reshape((1,norders,nbins,binsize))
        wavelength_binned = np.nanmean(tmp, axis=-1)
        fig = plt.figure(figsize=(6,5))
        plt.scatter(wavelength[0,0,:], rms[0,:]/err[0,:], marker='.', s=2, color='black')#, label='Order 1')
        plt.plot(wavelength_binned[0,0,:], rat_binned[0,:], color='black', linewidth=3, #marker='o',
                 markerfacecolor='none', markeredgewidth=2.0, label='Order 1')
        plt.scatter(wavelength[0,1,:], rms[1,:]/err[1,:], marker='.', s=2, color='blue')#, label='Order 2')
        plt.plot(wavelength_binned[0,1,:], rat_binned[1,:], color='blue', linewidth=3, #marker='o',
                 markerfacecolor='none', markeredgewidth=2.0, label='Order 2')
        plt.scatter(wavelength[0,2,:], rms[2,:]/err[2,:], marker='.', s=2, color='orange')#, label='Order 3')
        plt.plot(wavelength_binned[0,2,:], rat_binned[2,:], color='orange', linewidth=3, #marker='o',
                 markerfacecolor='none', markeredgewidth=2.0, label='Order 3')
        plt.plot([0.5,2.9],[1.0,1.0], linewidth=3, color='black', linestyle='dashed', label='Photon + Readout')
        plt.xlabel('Wavelength [microns]')
        plt.ylabel('Measured Scatter / DMS Expected Scatter')
        plt.yticks(np.linspace(0.9,2.0,12))
        plt.xticks(np.linspace(0.6,2.8,12))
        plt.grid()
        plt.legend()
        plt.ylim((0.8,2.1))
        plt.xlim((0.5,2.9))
        #plt.show()
        plt.savefig(outdir+'/A0_performance_square.png')

        # Figure for the SOSS mode paper
        tint = np.arange(nint) * (3+1)*5.494/3600
        norders = 3
        binsize = 18
        nbins = nint // binsize
        tmp = systematics[0:binsize * nbins, :].reshape((norders,nbins,binsize))
        sys_binned = np.nanmedian(tmp, axis=-1)
        print(nint, nbins, binsize)
        print(np.shape(tmp))
        print(np.shape(sys_binned))
        tmp = tint[0:binsize*nbins].reshape((nbins,binsize))
        tint_binned = np.nanmean(tmp, axis=-1)
        print(np.shape(tint_binned))

        tmp = systematics[0:binsize * nbins, :].reshape((nbins,binsize,norders))
        sys_binned = np.nanmedian(tmp, axis=-2)
        print(np.shape(sys_binned))

        fig = plt.figure(figsize=(6,5))
        plt.scatter(tint, (systematics[:,0]-1)*1e+6, marker='.', s=2, color='black')#, label='Order 1')
        plt.plot(tint_binned, (sys_binned[:,0]-1)*1e+6, color='black', linewidth=3, label='Order 1', alpha=0.7)
        plt.scatter(tint, (systematics[:,1]-1)*1e+6, marker='.', s=2, color='blue')#, label='Order 2')
        plt.plot(tint_binned, (sys_binned[:,1]-1)*1e+6, color='blue', linewidth=3, label='Order 2', alpha=0.7)
        #plt.scatter(tint, (systematics[:,2]-1)*1e+6, marker='.', s=2, color='orange')#, label='Order 3 x0.1')
        #plt.plot(tint_binned, (sys_binned[:,2]-1)*1e+6, color='orange', linewidth=3, label='Order 3 x0.1', alpha=0.7)

        #plt.scatter(np.arange(nint)*tint, systematics[:,2], marker='.', s=2, color='orange', label='Order 3')
        plt.xlabel('Time (hours)')
        plt.ylabel('White Light Trend [ppm]')
        #plt.yticks(np.linspace(0.9,2.0,12))
        #plt.xticks(np.linspace(0.6,2.8,12))
        plt.grid()
        plt.legend()
        plt.ylim((-600,600))
        #plt.xlim((0.5,2.9))
        #plt.show()
        plt.savefig(outdir+'/A0_whitelight_square.png')

        sys.exit()




        fig, frames = plt.subplots(4)
        frames[0].scatter(wavelength[0,0,:], rms[0,:], marker='.', s=2, color='black', label='Measured RMS along TS - Order 1')
        frames[0].scatter(wavelength[0,1,:], rms[1,:], marker='.', s=2, color='blue', label='Measured RMS along TS - Order 2')
        frames[0].scatter(wavelength[0,2,:], rms[2,:], marker='.', s=2, color='red', label='Measured RMS along TS - Order 3')
        frames[0].scatter(wavelength[0,0,:], err[0,:], marker='.', s=2, color='grey', label='DMS RMS along TS - Order 1')
        frames[0].scatter(wavelength[0,1,:], err[1,:], marker='.', s=2, color='cyan', label='DMS RMS along TS - Order 2')
        frames[0].scatter(wavelength[0,2,:], err[2,:], marker='.', s=2, color='orange', label='DMS RMS along TS - Order 3')
        frames[0].set_title('Measured and Expected Noises')
        frames[0].set_xlabel('Wavelength [microns]')
        frames[0].set_ylabel('Noise [e-]')
        frames[0].grid()
        frames[0].legend()
        frames[1].scatter(wavelength[0,0,:], rms[0,:]/err[0,:], marker='.', s=2, color='black', label='Ratio Measured/Expected - Order 1')
        frames[1].scatter(wavelength[0,1,:], rms[1,:]/err[1,:], marker='.', s=2, color='blue', label='Ratio Measured/Expected - Order 2')
        frames[1].scatter(wavelength[0,2,:], rms[2,:]/err[2,:], marker='.', s=2, color='red', label='Ratio Measured/Expected - Order 3')
        frames[1].set_ylim((0.8,4))
        frames[1].set_title('Noise Ratio')
        frames[1].set_xlabel('Wavelength [microns]')
        frames[1].set_ylabel('Measured vs. Expected Noise Ratio')
        frames[1].grid()
        frames[1].legend()
        frames[2].scatter(np.arange(nint), systematics[:,0], marker='.', s=2, color='black', label='Systematics - Order 1')
        frames[2].scatter(np.arange(nint), systematics[:,1], marker='.', s=2, color='blue', label='Systematics - Order 2')
        frames[2].scatter(np.arange(nint), systematics[:,2], marker='.', s=2, color='red', label='Systematics - Order 3')
        frames[2].set_title('White Light systematics along Time-Series')
        frames[2].set_xlabel('Integration Number')
        frames[2].set_ylabel('Relative Flux')

        frames[3].scatter(wavelength[0,0,:], rms_syscorr[0,:]/err[0,:], marker='.', s=2, color='black', label='Ratio Measured/Expected - Systematics corrected - Order 1')
        frames[3].scatter(wavelength[0,1,:], rms_syscorr[1,:]/err[1,:], marker='.', s=2, color='blue', label='Ratio Measured/Expected - Systematics corrected - Order 2')
        frames[3].scatter(wavelength[0,2,:], rms_syscorr[2,:]/err[2,:], marker='.', s=2, color='red', label='Ratio Measured/Expected - Systematics corrected - Order 3')
        frames[3].set_title('Noise Ratio for a detrended TS')
        frames[3].set_xlabel('Wavelength [microns]')
        frames[3].set_ylabel('Measured vs. Expected Noise Ratio')
        frames[3].set_ylim((0.8,4))
        frames[3].grid()
        frames[3].legend()
        plt.show()

    if True:

        # Make a RMS vs BIN size plot
        # Bin along the integrations axis
        binsize = np.arange(1, nint/5, 1, dtype=int)
        nbins = np.array(np.floor(nint / binsize), dtype=int)

        ppm = np.zeros((np.size(binsize),norder))
        for i in range(np.size(binsize)):
            # Group the 2048 columns in bins.
            # Reshape in 4D then sum the additional dimension.
            print(nint, norder, binsize[i], nbins[i])
            tmp = flux[0:binsize[i]*nbins[i],:,:].reshape((nbins[i],binsize[i],norder,2048))
            #print(binsize[i])
            #print(tmp.shape)
            flux_binned = np.nansum(tmp, axis=1)
            #print(flux_binned.shape)
            binned_whitelight = np.nansum(flux_binned, axis=-1)
            #print(binned_whitelight.shape)
            dev_whitelight = mediandev(binned_whitelight, axis=0)
            med_whitelight = np.nanmedian(binned_whitelight, axis=0)
            ppm[i,:] = dev_whitelight/med_whitelight*1e+6
            print('ppm = {:}'.format(ppm[i,:]))

        plt.scatter(np.arange(np.size(binsize)), ppm[:,0], marker='s', color='black', label='Order 1')
        plt.scatter(np.arange(np.size(binsize)), ppm[:,1], marker='s', color='blue', label='Order 2')
        plt.scatter(np.arange(np.size(binsize)), ppm[:,2], marker='s', color='orange', label='Order 3')
        plt.loglog()
        plt.ylabel('Scatter [ppm]')
        plt.xlabel('Bin Size [pixel columns]')
        plt.grid()
        plt.show()



    #TODO: Here we are
    sys.exit()

    return


def greyscale_rms(ts_greyscale, title=''):
    '''
    Removes outliers from the greyscale and perform standard deviation
    '''

    outdir = os.path.dirname(ts_greyscale)
    a = fits.getdata(ts_greyscale)
    norder, nint, dimx = np.shape(a)

    rms = np.zeros((norder, dimx)) * np.nan
    plt.figure(figsize=(8,5))
    for m in range(3):
        # For each order, generate the median profile (to account for the transit)
        transit_profile = np.nanmedian(a[m, :, :], axis=-1)
        for x in range(dimx):
            # Measured column
            col = a[m, :, x]
            # Skip columns that only contain NaNs
            if any(np.isfinite(col)):
                # Measured column with transit signal removed
                colnotransit = col - transit_profile
                colnotransitnonan = colnotransit[np.where(np.isfinite(colnotransit))[0]]
                sigmaclipping = sigma_clip(colnotransitnonan, sigma=3)
                maskgood = sigmaclipping.mask == False
                rms[m, x] = np.std(colnotransitnonan[maskgood])
        plt.plot(1/rms[m,:], label='Order {:}'.format(m+1))
    plt.legend()
    plt.xlabel('Column (pixels)')
    plt.ylabel('SNR')
    plt.ylim((0, 800))
    plt.title(title)
    plt.grid()
    plt.savefig(outdir+'/greyscale_rms.png')
    #plt.show()
    plt.close()



    for m in range(3):
        # White light
        white = np.nanmedian(a[m, :, :], axis=-1)
        dev = white[1:] - white[:-1]
        plt.figure(figsize=(8, 5))
        #plt.plot(dev, label='Order {:}'.format(m+1))
        plt.plot(white, label='Order {:}'.format(m+1))
        plt.legend()
        plt.title(title)
        #plt.ylim((0.95,1.05))
        plt.savefig(outdir+'/whitelight_order{:}.png'.format(m+1))
        plt.close()

    return

def test_backgrounds():
    from astropy.io import fits, ascii
    import numpy as np
    import matplotlib.pyplot as plt

    hdu = fits.open('jw02589001001_04101_00001-seg002_nis_rate.fits')
    clear = hdu[1].data
    hdu = fits.open('jw02589001001_04102_00001-seg001_nis_rate.fits')
    f277 = hdu[1].data

    slice_c = np.percentile(clear, 20, axis=0)
    slice_f = np.percentile(f277, 20, axis=0)

    scale = slice_c / slice_f
    av_scale = np.median(scale)
    nestor = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/pipelineprep/calibrations/model_background256.fits')
    slice_n = np.median(nestor, axis=0)
    nestor_scale = slice_c / slice_n
    av_scale_nestor = np.nanmedian(nestor_scale)
    red_scale_nestor = np.nanmedian(nestor_scale[0:650])
    blue_scale_nestor = np.nanmedian(nestor_scale[750:])

    slice_n_red = slice_n * red_scale_nestor
    slice_n_blue = slice_n * blue_scale_nestor


    plt.plot(slice_c, label='clear')
    plt.plot(slice_f, label='f277w')
    plt.plot(slice_f * av_scale, label='scaled f277w')
    plt.plot(slice_n, label='Nestor model')
    plt.plot(slice_n * av_scale_nestor, label='scaled Nestor')
    plt.plot(slice_n_red[0:750], color='brown', label='scaled Nestor - red only')
    plt.plot(np.arange(2048)[650:], slice_n_blue[650:], color='black', label='scaled Nestor - blue only')
    plt.legend()
    plt.grid()
    plt.xlabel('Detector column (pixels)')
    plt.ylabel('Counts (e-/sec)')
    plt.title('T1 obs1 Background versus its F277W observation and Nestors model')
    plt.show()

    return

#a = test_backgrounds()


def robust_polyfit(x, y, order, maxiter=5, nstd=3.):
    """Perform a robust polynomial fit.
    Paremeters
    ----------
    x : array, list
        Independent fitting variable.
    y : array, list
        Dependent fitting variable.
    order : int
        Polynomial order to fit.
    maxiter : int
        Number of iterations for outlier rejection.
    nstd : float
        Number of standard deviations to consider a value an outlier.
    Returns
    -------
    res : array[float]
        The best fitting polynomial parameters.
    """

    def _poly_res(p, x, y):
        """Residuals from a polynomial.
        """
        return np.polyval(p, x) - y

    mask = np.ones_like(x, dtype='bool')
    for niter in range(maxiter):

        # Fit the data and evaluate the best-fit model.
        param = np.polyfit(x[mask], y[mask], order)
        yfit = np.polyval(param, x)

        # Compute residuals and mask ouliers.
        res = y - yfit
        stddev = np.std(res)
        mask = np.abs(res) <= nstd*stddev

    res = least_squares(_poly_res, param, loss='huber', f_scale=0.1,
                        args=(x[mask], y[mask])).x

    return res

def measure_background_tilt(input_image, isafitsfile=True, method='halfbumpvalue',
                            outputdir='/Users/albert/Downloads/', return_all=False,
                            comment=''):

    if isafitsfile == True:
        input_image = '/Users/albert/NIRISS/Commissioning/analysis/T1/backgroundsub_jw02589001001_04101_00001-seg001_nis_customrateints_flatfieldstep/background_mask.fits'
        input_image = '/Users/albert/NIRISS/Commissioning/analysis/T1_2/backgroundsub_jw02589002001_04101_00001-seg001_nis_customrateints_flatfieldstep/background_mask.fits'
        image = fits.getdata(input_image)
    else:
        image = np.copy(input_image)

    # if it is a cube, stack it
    if np.size(image.shape) == 3:
            image = np.nanmedian(image, axis=0)

    #hdu = fits.PrimaryHDU(image)
    #hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/T1_masked_stacked.fits')

    if method == 'halfbumpvalue':
        # Compute the background median on the left and on the right of the bump
        # in a clean region (top rows). Then, for each row, find the x position where
        # the value is clostest to intermediate left/right background. Each row is
        # smoothed to improve SNR.

        # Columns and angles to explore
        cmin, cmax = 650, 750
        rowmin, rowmax = 0, 256

        # xtract a sub image
        subim = image[rowmin:rowmax, cmin:cmax]
        dimy, dimx = np.shape(subim)

        if True:
            # Running median of 2*medhalf+1 rows
            medhalf = 2
            smoothing = 5
            rows = np.arange(dimy)
            rows = rows[medhalf:-medhalf]
            subimsm = np.copy(subim)*np.nan
            val_left, val_right, val_break = [], [], []
            c_break = []
            allcols = np.linspace(cmin, cmax, dimx)

            rown = 0
            # Fill missed rows because of row medianing
            for i in range(medhalf):
                c_break.append(np.nan)
                val_left.append(np.nan)
                val_right.append(np.nan)
                val_break.append(np.nan)
            for row in rows:
                # median 2*medhalf+1 rows
                rowrunningmed = np.nanmedian(subim[row - medhalf:row + medhalf + 1, :], axis=0)
                # Smooth along columns
                subimsm[row, :] = medfilt(rowrunningmed, smoothing)
                val_left.append(np.nanmedian(rowrunningmed[:35]))
                val_right.append(np.nanmedian(rowrunningmed[-35:]))
                val_break.append((val_left[rown] + val_right[rown]) / 2)
                r = np.abs(rowrunningmed - val_break[rown])
                ind = np.argmin(r)
                print(ind)
                # Remove rows where the background is definitely not pure
                if (val_break[rown] > 5.5-2) & (val_break[rown] < 5.5+2):
                    c_break.append(allcols[ind])
                else:
                    c_break.append(np.nan)
                rown = rown+1
            # Fill missed rows because of row medianing
            for i in range(medhalf):
                c_break.append(np.nan)
                val_left.append(np.nan)
                val_right.append(np.nan)
                val_break.append(np.nan)
            # convert to numpy arrays
            c_break = np.array(c_break)
            val_left = np.array(val_left)
            val_right = np.array(val_right)

        if False:
            # smooth to improve SNR
            subimsm = np.copy(subim)
            val_left, val_right, val_break = [], [], []
            c_break = []
            allcols = np.linspace(cmin, cmax, dimx)
            for i in range(dimy):
                subimsm[i, :] = medfilt(subim[i, :], 5)
                val_left.append(np.nanmedian(subim[i, :35]))
                val_right.append(np.nanmedian(subim[i, -35:]))
                val_break.append((val_left[i] + val_right[i]) / 2)
                r = np.abs(subimsm[i, :] - val_break[i])
                ind = np.argmin(r)
                print(ind)
                # Remove rows where the background is definitely not pure
                if (val_break[i] > 5.5-2) & (val_break[i] < 5.5+2):
                    c_break.append(allcols[ind])
                else:
                    c_break.append(np.nan)
            c_break = np.array(c_break)
            val_left = np.array(val_left)
            val_right = np.array(val_right)
        # Further remove rows where the background position is > 10 pixels off
        #ind = np.abs(c_break - 697) > 10
        #c_break[ind] = np.nan
        cols = np.arange(dimy)

        ind = np.isfinite(c_break)
        cols = cols[ind]
        c_break = c_break[ind]
        val_left = val_left[ind]
        val_right = val_right[ind]

        pars = robust_polyfit(cols, c_break, 1, maxiter=5, nstd=3.)
        print(pars)
        tilt = np.rad2deg(np.arctan(pars[0]))
        print('tilt = {:}'.format(tilt))
        c_fit = np.polyval(pars, cols)

        if return_all == True:
            return tilt, pars, cols, c_break, val_left, val_right
        else:
            return tilt

        if True:
            plt.scatter(cols, c_break-700, label='col break')
            plt.scatter(cols, val_left, label='val left')
            plt.scatter(cols, val_right, label='val right')
            plt.plot(cols, c_fit-700, label='Fit')
            plt.title(comment)

            plt.legend()
            plt.show()


    if method == 'stackgradients':
        # The gradient is computed for all pixels of each angle.
        # For each angle, the result is crunched into a 1D slice.
        # Whose maximum is measured.
        # The best fit tilt is the angle producing maximum measurement.

        # Columns and angles to explore
        cmin, cmax = 650, 740
        #rowmin, rowmax = 1100, 2044
        rowmin, rowmax = 0,256
        thetalist = np.linspace(-2.5, -0.5, 101)

        # Obtain the derivative along the columns
        subim = image[rowmin:rowmax, cmin:cmax]
        #plt.imshow(subim)
        #plt.show()
        dimy, dimx = np.shape(subim)
        print(dimx, dimy)
        dfdx = np.gradient(subim, axis=1)
        print(dfdx)
        hdu = fits.PrimaryHDU(dfdx)
        hdu.writeto(outputdir+'dfdx.fits', overwrite=True)
        # fill nans with zero
        dfdx = np.where(np.isfinite(dfdx), dfdx, 0)

        # Initialize output arrays
        dfdxcube = np.zeros((np.size(thetalist), dimy, dimx))
        tmax = np.zeros(np.size(thetalist))

        # Loop over angles
        for theta in range(np.size(thetalist)):
            dfdxrot = rotate(dfdx, thetalist[theta], axes=(1, 0), reshape=False)
            hdu = fits.PrimaryHDU(dfdxrot)
            hdu.writeto(outputdir+'dfdxrot.fits', overwrite=True)
            dfdxcube[theta,:,:] = np.copy(dfdxrot)

            # Crunch as a single slice and measure max
            # mask zeros back to nan
            dfdxrot[dfdxrot < 1e-6] = np.nan
            # crunch as a slice
            trace = np.nanmedian(dfdxrot, axis=0)
            tmax[theta] = np.max(medfilt(trace,5))
            if theta % 3 == 0:
                plt.plot(trace+0.01*theta)
                plt.plot(medfilt(trace+0.01*theta,5))
        plt.grid()
        #plt.show()
        print(thetalist, tmax)

        # Optimal angle is
        #tilt = thetalist[np.argmax(medfilt(tmax,3))]
        tilt = thetalist[np.argmax(tmax)]

        print('Tilt angle is ', tilt)

        plt.plot(thetalist, tmax)
        plt.plot(thetalist, medfilt(tmax,3))
        plt.grid()
        #plt.show()

        hdu = fits.PrimaryHDU(dfdxcube)
        hdu.writeto(outputdir+'dfdxrot_cube.fits', overwrite=True)

    if method == 'stackprofiles':
        # For each angle, the image is crunched into a 1D slice.
        # On that 1D profile, the gradient is measured.
        # The best fit tilt is the angle producing maximum measurement.

        # Columns and angles to explore
        cmin, cmax = 650, 740
        rowmin, rowmax = 850, 2044
        rowmin, rowmax = 0,256
        thetalist = np.linspace(-3, -0.5, 51)

        # xtract a sub image
        subim = image[rowmin:rowmax, cmin:cmax]
        dimy, dimx = np.shape(subim)

        # fill nans with zero
        subim = np.where(np.isfinite(subim), subim, 0)
        hdu = fits.PrimaryHDU(subim)
        hdu.writeto(outputdir+'test.fits', overwrite=True)

        # Initialize output arrays
        subimcube = np.zeros((np.size(thetalist), dimy, dimx))
        gradientmap = np.zeros((np.size(thetalist),dimx))
        tmax = np.zeros(np.size(thetalist))

        # Loop over angles
        for theta in range(np.size(thetalist)):
            subimrot = rotate(subim, thetalist[theta], axes=(1, 0), reshape=False)
            hdu = fits.PrimaryHDU(subimrot)
            hdu.writeto(outputdir+'subimrot.fits', overwrite=True)
            subimcube[theta, :, :] = np.copy(subimrot)

            # Crunch as a single slice and measure max
            # mask zeros back to nan
            subimrot[subimrot < 1e-6] = np.nan
            # crunch as a slice
            trace = np.nanmedian(subimrot, axis=0)
            # Obtain the gradient
            gradient = np.gradient(trace)
            # insert in the gradient map
            gradientmap[theta,:] = np.copy(gradient)
            # Identify the maximum gradient for that angle
            tmax[theta] = np.max(gradient)
            if theta % 5 == 0:
                plt.plot(gradient)
        plt.grid()
        plt.show()
        print(thetalist, tmax)

        # Optimal angle is
        # tilt = thetalist[np.argmax(medfilt(tmax,3))]
        tilt = thetalist[np.argmax(tmax)]

        print('Tilt angle is ', tilt)

        plt.plot(thetalist, tmax)
        plt.plot(thetalist, medfilt(tmax,3))
        plt.grid()
        #plt.show()

        hdu = fits.PrimaryHDU(subimcube)
        hdu.writeto(outputdir+'subimrot_cube.fits', overwrite=True)
        hdu = fits.PrimaryHDU(gradientmap)
        hdu.writeto(outputdir+'gradientmap.fits', overwrite=True)

    if method == 'crosscorel':
        # Take the n rows at top to produce a profile. Then
        # cross correlate those with each individual row

        # Columns and angles to explore
        cmin, cmax = 650, 740
        rowmin, rowmax = 0, 256
        thetalist = np.linspace(-3, -0.5, 51)

        # xtract a sub image
        subim = image[rowmin:rowmax, cmin:cmax]
        dimy, dimx = np.shape(subim)

        # Oversample pixels in the x direction but not in y
        osf = 4 # oversampling factor
        subimos = np.zeros((dimy,dimx*osf))
        for i in range(dimy):
            subimos[i,:] = np.interp(np.arange(dimx*osf)/osf, np.arange(dimx), subim[i,:])


        # generate the oversampled kernel. Skip the top 5 rows (ref pixels)
        kernel = np.nanmedian(subimos[-11:-6,:], axis=0)
        # handle NaNs
        replacement_NaN = np.nanmedian(subim)
        kernel[~np.isfinite(kernel)] = replacement_NaN

        #submethod = 'full' # kernel is full row
        submethod = 'short' # kernel is a short row around the background bump
        if submethod == 'full':
            rows = np.arange(dimy)
            rows = rows[2:-2]
            ccf = np.zeros((np.size(rows),dimx*osf))
            rown = 0
            for row in rows:
                # median 2*medhalf+1 rows
                medhalf = 2
                runningmed = np.nanmedian(subimos[row-medhalf:row+medhalf+1,:], axis=0)
                # handle NaNs
                runningmed[~np.isfinite(runningmed)] = replacement_NaN
                # Do the CC
                ccf[rown, :] = np.correlate(runningmed, kernel, mode='same')
                # Smooth the CCF
                #ccf[rown, :] = medfilt(ccf[rown, :], 5)
                rown = rown + 1
        elif submethod == 'short':
            # select a short extract of the kernel
            kernelshort = kernel[dimx*osf//2-(20*osf):dimx*osf//2+(20*osf)]
            ccfdim = dimx*osf - np.size(kernelshort) + 1
            rows = np.arange(dimy)
            rows = rows[2:-2]
            #ccf = np.zeros((np.size(rows),dimx*osf))
            ccf = np.zeros((np.size(rows),ccfdim))

            rown = 0
            for row in rows:
                # median 2*medhalf+1 rows
                medhalf = 2
                runningmed = np.nanmedian(subimos[row-medhalf:row+medhalf+1,:], axis=0)
                # handle NaNs
                runningmed[~np.isfinite(runningmed)] = replacement_NaN
                # Do the CC
                ccf[rown, :] = np.correlate(runningmed, kernelshort, mode='valid')
                # Smooth the CCF
                #ccf[rown, :] = medfilt(ccf[rown, :], 5)
                rown = rown + 1

        hdu = fits.PrimaryHDU(ccf)
        hdu.writeto(outputdir+'testccf'+comment+'.fits', overwrite=True)





def construct_background(input_image, tilt=-1.8, isafitsfile=False, metric='10pct',
                         savetest=False, outdir=None):

    if isafitsfile == True:
        imagein = fits.getdata(input_image)
        if outdir == None:
            outdir = os.path.dirname(input_image)
    else:
        imagein = np.copy(input_image)

    # if it is a cube, stack it
    if np.size(imagein.shape) == 3:
            imagein = np.nanmedian(imagein, axis=0)

    if (savetest == True) & (outdir == None):
        print('WARNING. To save the test plots/files, outdir must be set. No saving will occur.')

    dimy, dimx = np.shape(imagein)

    # Handle the masked pixels during rotation by rotating the mask
    mask = ~np.isfinite(imagein)
    badmap = imagein*0
    badmap[mask] = 1.0
    badmaprot = rotate(badmap, tilt, axes=(1, 0), reshape=False, order=1, cval=1)
    if (savetest == True) & (outdir != None):
        hdu = fits.PrimaryHDU(badmaprot)
        hdu.writeto(outdir+'/badmap_rotated.fits', overwrite=True)
    mask = badmaprot > 0.05 # the smaller the threshold, the largest the mask

    # Rotate the image cleaned of nans then put them back in based on above mask
    image = np.where(np.isfinite(imagein), imagein, 0)
    imrot = rotate(image, tilt, axes=(1, 0), order=1, reshape=False)
    imrot[mask] = np.nan
    if (savetest == True) & (outdir != None):
        hdu = fits.PrimaryHDU(imrot)
        hdu.writeto(outdir+'/image_rotated.fits', overwrite=True)

    # profile of the background - the plain median
    bgd1d_plainmedian = np.nanmedian(imrot, axis=0)

    # profile of the background - 20th percentile
    bgd1d_10pct = np.nanpercentile(imrot, 10, axis=0)
    bgd1d_20pct = np.nanpercentile(imrot, 20, axis=0)
    bgd1d_30pct = np.nanpercentile(imrot, 30, axis=0)

    # profile of the background - only the top of the image
    bgd1d_mediantop = np.nanmedian(imrot[-50:,:], axis=0)

    # Smooth the profile
    #bgd1d_savgolfit = savgol_filter(bgd1d_20pct, 100, 4)#, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0

    # Polynomial fitting - Use the below version of the background
    if metric == '10pct':
        background = bgd1d_10pct
    elif metric == '20pct':
        background = bgd1d_20pct
    elif metric == '30pct':
        background = bgd1d_30pct
    elif metric == 'median':
        background = bgd1d_plainmedian
    elif metric == 'mediantop':
        background = bgd1d_mediantop
    else:
        print('ERROR - metric does not exist: ', metric)
        sys.exit()
    #x = np.arange(dimx)
    #masknan = np.isfinite(background)
    #pars = np.polyfit(x[masknan], background[masknan], 15)
    #bgd1d_polyfit = np.polyval(pars, x)

    # 3-segment fit (on each side of the step)
    padding = 50
    xpad = np.arange(dimx+2*padding)-padding
    x = np.arange(dimx)
    cutleft, cutright = 695, 712
    # Initialize the output array (with padding)
    bgd1d_3segmentfit = np.zeros(dimx+2*padding)
    bgd1d_3segmentfit[padding:-padding] = np.copy(background)
    # Fit the left side of the curve
    maskleft = np.isfinite(background) & (x < cutleft)
    parsleft = np.polyfit(x[maskleft], background[maskleft], 3)
    bgd1d_3segmentfit[xpad < cutleft] = np.polyval(parsleft, xpad[xpad < cutleft])
    # Fit the right side of the curve
    maskright = np.isfinite(background) & (x > cutright)
    parsright = np.polyfit(x[maskright], background[maskright], 14)
    bgd1d_3segmentfit[xpad > cutright] = np.polyval(parsright, xpad[xpad > cutright])

    # Plot the results
    if (savetest == True) & (outdir != None):
        plt.figure(figsize=(10,6))
        plt.plot(bgd1d_plainmedian, label='Plain median')
        plt.plot(bgd1d_10pct, label='10th percentile')
        plt.plot(bgd1d_20pct, label='20th percentile')
        plt.plot(bgd1d_30pct, label='30th percentile')
        plt.plot(bgd1d_mediantop, label='Median of the 50 pixels at the top')
        #plt.plot(bgd1d_savgolfit, label='Savgol fit')
        #plt.plot(bgd1d_polyfit, label='Polynomial fit')
        plt.plot(bgd1d_3segmentfit[padding:-padding], label='3-segment fit')
        plt.grid()
        plt.legend()
        plt.savefig(outdir+'/bgd_plot.png')
        #plt.show()

    # Then project back this model across in 2D and derotate
    dimxpad = dimx + 2*padding
    dimypad = dimy + 2*padding
    bgd2drotpad = np.tile(bgd1d_3segmentfit, dimypad).reshape(dimypad, dimxpad)
    bgd2dpad = rotate(bgd2drotpad, -tilt, axes=(1, 0), order=1, reshape=False)
    # remove the padding
    bgd2d = np.copy(bgd2dpad[padding:-padding,padding:-padding])

    if (savetest == True) & (outdir != None):
        hdu = fits.PrimaryHDU(bgd2d)
        hdu.writeto(outdir+'/background_fitted.fits', overwrite=True)

    # Subtract background model from the input image
    imagecorr = imagein - bgd2d
    if (savetest == True) & (outdir != None):
        hdu = fits.PrimaryHDU(imagecorr)
        hdu.writeto(outdir+'/bgd_corrected.fits', overwrite=True)

    return bgd2d




def test_back_construct():

    from astropy.io import fits
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import rotate
    from scipy.signal import savgol_filter

    input_image = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/T1_masked_stacked.fits')

    tilt = -1.8

    dimy, dimx = np.shape(input_image)

    # Handle the masked pixels during rotation by rotating the mask
    mask = ~np.isfinite(input_image)
    badmap = input_image*0
    badmap[mask] = 1.0
    badmaprot = rotate(badmap, tilt, axes=(1, 0), reshape=False, order=1, cval=1)
    hdu = fits.PrimaryHDU(badmaprot)
    hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/badmap_rotated.fits', overwrite=True)
    mask = badmaprot > 0.05 # the smaller the threshold, the largest the mask

    # Rotate the image cleaned of nans then put them back in based on above mask
    image = np.where(np.isfinite(input_image), input_image, 0)
    imrot = rotate(image, tilt, axes=(1, 0), order=1, reshape=False)
    imrot[mask] = np.nan
    hdu = fits.PrimaryHDU(imrot)
    hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/image_rotated.fits', overwrite=True)

    # profile of the background - the plain median
    bgd1d_plainmedian = np.nanmedian(imrot, axis=0)

    # profile of the background - 20th percentile
    bgd1d_10pct = np.nanpercentile(imrot, 10, axis=0)
    bgd1d_20pct = np.nanpercentile(imrot, 20, axis=0)
    bgd1d_30pct = np.nanpercentile(imrot, 30, axis=0)

    # profile of the background - only the top of the image
    bgd1d_mediantop = np.nanmedian(imrot[-50:,:], axis=0)

    # Smooth the profile
    bgd1d_savgolfit = savgol_filter(bgd1d_20pct, 100, 4)#, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0

    # Polynomial fitting - Use the below version of the background
    background = bgd1d_20pct
    x = np.arange(dimx)
    masknan = np.isfinite(background)
    pars = np.polyfit(x[masknan], background[masknan], 15)
    bgd1d_polyfit = np.polyval(pars, x)

    # 3-segment fit (on each side of the step)
    padding = 50
    xpad = np.arange(dimx+2*padding)-padding
    x = np.arange(dimx)
    cutleft, cutright = 695, 712
    # Initialize the output array (with padding)
    bgd1d_3segmentfit = np.zeros(dimx+2*padding)
    bgd1d_3segmentfit[padding:-padding] = np.copy(background)
    # Fit the left side of the curve
    maskleft = np.isfinite(background) & (x < cutleft)
    parsleft = np.polyfit(x[maskleft], background[maskleft], 3)
    bgd1d_3segmentfit[xpad < cutleft] = np.polyval(parsleft, xpad[xpad < cutleft])
    # Fit the right side of the curve
    maskright = np.isfinite(background) & (x > cutright)
    parsright = np.polyfit(x[maskright], background[maskright], 14)
    bgd1d_3segmentfit[xpad > cutright] = np.polyval(parsright, xpad[xpad > cutright])

    # Plot the results
    plt.plot(bgd1d_plainmedian, label='Plain median')
    plt.plot(bgd1d_10pct, label='10th percentile')
    plt.plot(bgd1d_20pct, label='20th percentile')
    plt.plot(bgd1d_30pct, label='30th percentile')
    plt.plot(bgd1d_mediantop, label='Median of the 50 pixels at the top')
    #plt.plot(bgd1d_savgolfit, label='Savgol fit')
    #plt.plot(bgd1d_polyfit, label='Polynomial fit')
    plt.plot(bgd1d_3segmentfit[padding:-padding], label='3-segment fit')
    plt.grid()
    plt.legend()
    plt.show()

    # Then project back this model across in 2D and derotate
    dimxpad = dimx + 2*padding
    dimypad = dimy + 2*padding
    bgd2drotpad = np.tile(bgd1d_3segmentfit, dimypad).reshape(dimypad, dimxpad)
    plt.imshow(bgd2drotpad, origin='lower')
    plt.show()
    bgd2dpad = rotate(bgd2drotpad, -tilt, axes=(1, 0), order=1, reshape=False)
    # remove the padding
    bgd2d = np.copy(bgd2dpad[padding:-padding,padding:-padding])

    hdu = fits.PrimaryHDU(bgd2d)
    hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/background_model_fitted.fits', overwrite=True)

    # Subtract background model from the input image
    imagecorr = input_image - bgd2d
    hdu = fits.PrimaryHDU(imagecorr)
    hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/bgd_corrected.fits', overwrite=True)

    return

#a = test_back_construct()



def test_back_scaling():

    from astropy.io import fits, ascii
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import rotate
    from scipy.signal import medfilt

    nestor = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/pipelineprep/calibrations/model_background256.fits')

    image = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/T1/backgroundsub_jw02589001001_04101_00001-seg001_nis_customrateints_flatfieldstep/background_mask.fits')
    image = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/T1_2/backgroundsub_jw02589002001_04101_00001-seg001_nis_customrateints_flatfieldstep/background_mask.fits')

    image = np.nanmedian(image, axis=0)
    hdu = fits.PrimaryHDU(image)
    #hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/T1_masked_stacked.fits')

    # Direct scaling alone
    ratio = image / nestor
    #hdu = fits.PrimaryHDU(ratio)
    #hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/T1_stack_to_nestor_scaling.fits', overwrite=True)
    # residuals shows an oversubtraction

    # Try a 2-parameter fit. i.e. treat te left side of the image as a DC additive offset and
    # skipped for now

    # Try constructing a 1-D background model.

    # A) Determine the step angle
    cmin, cmax = 680, 720
    thetalist = np.linspace(-3, -0.5, 51)

    subim = image[:,cmin:cmax]
    dimy, dimx = np.shape(subim)
    dfdx = np.gradient(subim, axis=1)
    hdu = fits.PrimaryHDU(dfdx)
    hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/dfdx.fits', overwrite=True)
    # fill nans with zero
    dfdx = np.where(np.isfinite(dfdx), dfdx, 0)


    dfdxcube = np.zeros((np.size(thetalist), dimy, dimx))
    tmax = np.zeros(np.size(thetalist))
    for theta in range(np.size(thetalist)):
        dfdxrot = rotate(dfdx, thetalist[theta], axes=(1, 0), reshape=False)
        #hdu = fits.PrimaryHDU(dfdxrot)
        #hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/dfdxrot.fits', overwrite=True)
        dfdxcube[theta,:,:] = np.copy(dfdxrot)

        # Crunch as a single slice and measure max
        # mask zeros back to nan
        dfdxrot[dfdxrot < 1e-6] = np.nan
        # crunch as a slice
        trace = np.nanmedian(dfdxrot, axis=0)
        tmax[theta] = np.max(medfilt(trace,5))
        if theta % 3 == 0:
            plt.plot(trace+0.01*theta)
            plt.plot(medfilt(trace+0.01*theta,5))
    plt.grid()
    plt.show()
    print(thetalist, tmax)

    # Optimal angle is
    #tilt = thetalist[np.argmax(medfilt(tmax,3))]
    tilt = thetalist[np.argmax(tmax)]

    print('Tilt angle is ', tilt)

    plt.plot(thetalist, tmax)
    #plt.plot(thetalist, medfilt(tmax,3))
    plt.grid()
    plt.show()

    hdu = fits.PrimaryHDU(dfdxcube)
    hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/documenting_steps/dfdxrot_cube.fits', overwrite=True)

    return

#a = test_back_scaling()


def return_ecliptic_latitude(alpha, delta):

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c_icrs = SkyCoord(ra=alpha * u.degree, dec=delta * u.degree, frame='icrs')

    return c_icrs.geocentricmeanecliptic.lat.value

def return_ecliptic_longitude(alpha, delta):

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c_icrs = SkyCoord(ra=alpha * u.degree, dec=delta * u.degree, frame='icrs')

    return c_icrs.geocentricmeanecliptic.lon.value

def characterize_background():
    # Characterize the background position for a dozen SOSS TSO
    ratelist = sorted(
        glob.glob("/Users/albert/NIRISS/SOSSpipeline/2023_brainstorming/mastDownload/JWST/jw*-seg001_nis/*rate.fits"))
    pwcpos, filename = [], []
    for i in range(np.size(ratelist)):
        a = datamodels.open(ratelist[i])
        # a.info(max_rows=200)
        im = a.data
        position = a.meta.instrument.pupil_position
        substrip = a.meta.subarray.name
        if substrip == 'SUBSTRIP256':
            pwcpos.append(position)
            filename.append(ratelist[i])

    filename = np.array(filename)
    pwcpos = np.array(pwcpos)
    print(filename, pwcpos, np.size(filename))

    # good measurements
    good = [3, 4, 5, 6, 8, 9, 10]

    # Select one by one, which plot to generate
    plot_version = 'allpositions'
    #plot_version = 'single_breaks'
    #plot_version = 'leftright'
    #plot_version = 'eclipticrelation'

    tilt, yintercept, obsid = [], [], []
    eclipticlat, eclipticlon, left_level, right_level = [], [], [], []
    for i in range(np.size(filename)):
        a = datamodels.open(filename[i])
        im = a.data
        dq = a.err

        # print(a.info(max_rows=1000))
        thisobsid = a.meta.observation.visit_id
        obsid.append(thisobsid)
        eclipticlat.append(return_ecliptic_latitude(a.meta.target.ra, a.meta.target.dec))
        eclipticlon.append(return_ecliptic_longitude(a.meta.target.ra, a.meta.target.dec))

        # ind = ~np.isfinite(im)
        # im[ind] = np.nanmedian(im)
        # hdu = fits.PrimaryHDU(im)
        # hdu.writeto('/Users/albert/NIRISS/SOSSpipeline/2023_brainstorming/test.fits', overwrite=True)
        backtilt, pars, cols, c_break, val_left, val_right = measure_background_tilt(im,
                                                                isafitsfile=False, method='halfbumpvalue',
                                                                comment=thisobsid,
                                                                outputdir='/Users/albert/NIRISS/SOSSpipeline/2023_brainstorming/',
                                                                return_all=True)
        tilt.append(backtilt)
        yintercept.append(pars[1])
        xcol = np.arange(256)
        c_fit = np.polyval(pars, xcol)

        # Establish the representative level left and right
        left_level.append(np.nanpercentile(val_left, 10))
        right_level.append(np.nanpercentile(val_right, 10))

        if plot_version == 'allpositions':
            if set([i]).isdisjoint(set(good)) == False:
                plt.plot(c_fit, xcol, label=thisobsid)
            else:
                plt.plot(c_fit, xcol, label=thisobsid, ls='dashed')
        if plot_version == 'single_breaks':
            plt.scatter(c_break, cols, color='black', label='Measurements')
            plt.plot(c_fit, xcol, color='red', label='Linear Fit')
            plt.gca().invert_yaxis()
            plt.xlabel('SUBSTRIP256 Row [pixel]')
            plt.ylabel('Background Break Column [pixel]')
            plt.legend()
            plt.title(thisobsid)
            plt.savefig('/Users/albert/NIRISS/SOSSpipeline/2023_brainstorming/background_'+thisobsid+'.png')
            plt.show()
        if plot_version == 'leftright':
            # Do the plot
            plt.scatter(cols, val_left, color='blue', label='Left of Break Level')
            plt.scatter(cols, val_right, color='orange', label='Right of Break Level')
            plt.hlines(left_level[i], 0, np.max(cols), color='cyan', label='Left 10% percentile level')
            plt.hlines(right_level[i], 0, np.max(cols), color='red', label='Right 10% percentile level')
            #plt.plot(c_fit, xcol, color='red', label='Linear Fit')
            #plt.gca().invert_yaxis()
            plt.ylabel('Background Level (e-/sec)')
            plt.ylim((2,10))
            plt.xlabel('Detector Row [Y-pixel]')
            plt.legend()
            plt.title(thisobsid)
            plt.savefig('/Users/albert/NIRISS/SOSSpipeline/2023_brainstorming/background_leftright_'+thisobsid+'.png')
            plt.show()

    if plot_version == 'eclipticrelation':
        plt.scatter(eclipticlat, left_level, label='Left Level')
        plt.scatter(eclipticlat, right_level, label='Right Level')
        plt.xlabel('Ecliptic Latitude (degrees)')
        plt.ylabel('Background Level (e-/sec)')
        plt.title('Background vs Ecliptic Latitude')
        plt.legend()
        plt.savefig('/Users/albert/NIRISS/SOSSpipeline/2023_brainstorming/background_eclipticlat_.png')
        plt.show()
        plt.scatter(eclipticlon, left_level, label='Left Level')
        plt.scatter(eclipticlon, right_level, label='Right Level')
        plt.xlabel('Ecliptic Longitude (degrees)')
        plt.ylabel('Background Level (e-/sec)')
        plt.title('Background vs Ecliptic Longitude')
        plt.legend()
        plt.savefig('/Users/albert/NIRISS/SOSSpipeline/2023_brainstorming/background_eclipticlon_.png')
        plt.show()



    tilt = np.array(tilt)
    yintercept = np.array(yintercept)

    if plot_version == 'allpositions':
        plt.gca().invert_yaxis()
        plt.xlabel('Pixel Columns (X)')
        plt.ylabel('Pixel Rows (Y)')
        plt.xlim((695, 710))
        plt.title('Background Break Fitted Position')
        plt.legend()
        plt.savefig('/Users/albert/NIRISS/SOSSpipeline/2023_brainstorming/background_fitted_position.png')
        plt.show()





    # Good sample - Tilt figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    #plt.scatter(pwcpos, tilt, label='Tilt (all Data)')
    ax.scatter(pwcpos[good], tilt[good])#, label='Tilt (Good Data)')
    ax.vlines(245.7616, 2.05, 1.65, ls='dashed', color='red')
    ax.set_ylim((2.05,1.65))
    ax.set_aspect('equal', 'box')
    ax.set_title('SOSS Background Break Tilt')
    ax.set_xlabel('Pupil Wheel Position (PWCPOS - degrees)')
    ax.set_ylabel('Tilt Angle from vertical (degrees)')
    plt.tight_layout()
    plt.savefig('/Users/albert/NIRISS/SOSSpipeline/2023_brainstorming/background_tilt_vs_pwcpos.png')
    plt.show()

    # Good sample - Y intercept figure
    params = np.polyfit(pwcpos[good], yintercept[good], 1)
    pwangle = np.linspace(np.min(pwcpos), np.max(pwcpos), 100)
    yfit = np.polyval(params, pwangle)

    plt.figure(figsize=(9,6))
    plt.scatter(pwcpos[good], yintercept[good], label='Measurements')
    plt.plot(pwangle, yfit, label='Fit - slope = {:7.3F}'.format(params[0]))
    plt.vlines(245.7616, 695, 699, ls='dashed', color='red', label='Nominal GR700XD')
    plt.title('SOSS Background Break Position')
    plt.xlabel('Pupil Wheel Position (PWCPOS - degrees)')
    plt.ylabel('X Position at the Top of the SUBSTRIP256')
    plt.legend()
    plt.savefig('/Users/albert/NIRISS/SOSSpipeline/2023_brainstorming/background_yintercept_vs_pwcpos.png')
    plt.show()

    return

def cds(rampmodel, outdir=None, verbose=False):

    #nint, dimy, dimx = np.shape(datamodel.data)

    basename = os.path.splitext(rampmodel.meta.filename)[0]
    if outdir == None:
        outdir = './'
        #cntrdir = './backgroundsub_'+basename+'/'
    #else:
    #    cntrdir = outdir+'/backgroundsub_'+basename+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    #if not os.path.exists(cntrdir):
    #    os.makedirs(cntrdir)

    ngroup = rampmodel.meta.exposure.ngroups
    intstart = rampmodel.meta.exposure.integration_start
    intend = rampmodel.meta.exposure.integration_end
    if (intstart == None) & (intend == None):
        # it means that this is a time-series NOT split into segments
        intstart, intend = 1, rampmodel.meta.exposure.nints
    #nint = intend-intstart+1
    #dimx = np.shape(rampmodel.data)[-1]

    superbias = fits.getdata(SUPERBIAS_NAME)
    cds = rampmodel.data[:,ngroup-1,:,:] - superbias

    frametime =  rampmodel.meta.exposure.frame_time
    exptime = ngroup * frametime
    cds = cds / exptime


if __name__ == "__main__":

    if True:
        characterize_background()
        sys.exit()
    if False:
        # Make contamination mask for T1_4
        a = make_mask_02589_obs003()
        sys.exit()

    if True:
        # Debuf the performance script
        outdir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/'
        spectra = 'extracted_spectrum.fits'
        a = performance_fluxcal(outdir+spectra, outdir=outdir)

        sys.exit()

#    # Test 1/f step from Thomas
#    from SOSS.dms import oneoverf_step
#    outdir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/'
#    input_image = outdir+'jw01091002001_03101_00001-seg001_nis_customrateints_flatfieldstep.fits'

    # Test outlier detection
    import SOSS.dms.soss_outliers as soss_outliers
    outdir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/'
    input_image = outdir+'jw01091002001_03101_00001-seg001_nis_customrateints_flatfieldstep.fits'
    outmap = outdir+'outmap.fits'
    result = datamodels.open(input_image)
    result = soss_outliers.flag_outliers(result, window_size=(3, 11), n_sig=9,
                                         verbose=True, outdir=outdir,
                                         save_diagnostic=True)
    result.close()
    sys.exit()



    input_image = '/Users/albert/NIRISS/Commissioning/analysis/T1/backgroundsub_jw02589001001_04101_00001-seg001_nis_customrateints_flatfieldstep/background_mask.fits'
    #input_image = '/Users/albert/NIRISS/Commissioning/analysis/T1_2/backgroundsub_jw02589002001_04101_00001-seg001_nis_customrateints_flatfieldstep/background_mask.fits'
    outdir = os.path.dirname(input_image)
    bkg = construct_background(input_image, tilt=-1.8, isafitsfile=True, metric='10pct',
                               savetest=True, outdir=outdir)
    sys.exit()


    a = make_mask_02589_obs001()
    sys.exit()
    #datamodel = datamodels.open('/Users/albert/NIRISS/Commissioning/analysis/HATP14b/jw01541001001_04101_00001-seg003_nis_customrateints_flatfieldstep.fits')
    #trace_table_ref_file_name = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/ref_files/SOSS_ref_trace_table_SUBSTRIP256.fits'
    #datamodel = commutils.remove_nans(datamodel)
    #rien = localbackground_subtraction(datamodel, trace_table_ref_file_name, width=25, back_offset=-25)
    #sys.exit()

    #greyscale_rms('/Users/albert/NIRISS/Commissioning/analysis/HATP14b/timeseries_combined_20220610.fits', title='No 1/f correction')
    #greyscale_rms('/Users/albert/NIRISS/Commissioning/analysis/HATP14b/timeseries_combined_1f_ap25_20220711.fits.gz', title='With Loic 1/f correction')
    #greyscale_rms('/Users/albert/NIRISS/Commissioning/analysis/HATP14b/timeseries_combined_1f_ap25_atoca_20220713.fits.gz', title='With Loic 1/f correction and atoca modelling')
    greyscale_rms('/Users/albert/NIRISS/Commissioning/analysis/HATP14b/timeseries_greyscale.fits', title='20220719_1937')

    sys.exit()
    a = make_mask_nis17()
    a = make_mask_nis18obs2()
    sys.exit()
    outdir = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/'
    wildcard = outdir+'supplemental_jw01541001001_04101_00001-seg00?_nis/timeseries_greyscale_rawflux.fits'
    a = combined_timeseries(wildcard, outdir)