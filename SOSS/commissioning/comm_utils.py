import os.path

import matplotlib.pyplot as plt

import numpy as np
import scipy.interpolate

import glob

from astropy.io import fits

from scipy.stats import sigmaclip

from jwst import datamodels

import SOSS.dms.soss_centroids as soss_centroids

import sys

from jwst import datamodels

from astropy.nddata.bitmask import bitfield_to_boolean_mask

from jwst.datamodels import dqflags

import SOSS.trace.tracepol as tracepol

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

    from astropy.nddata.bitmask import bitfield_to_boolean_mask

    from jwst.datamodels import dqflags

    #import matplotlib.pyplot as plt


    # Mask pixels that have the DO_NOT_USE data quality flag set, EXCEPT the saturated pixels (because the ramp
    # fitting algorithm handles saturation).
    donotuse = bitfield_to_boolean_mask(datamodel.dq, ignore_flags=dqflags.pixel['DO_NOT_USE'], flip_bits=True)
    #saturated = bitfield_to_boolean_mask(datamodel.dq, ignore_flags=dqflags.pixel['SATURATED'], flip_bits=True)
    #mask = donotuse & ~saturated
    mask = donotuse # Just this is enough to not flag the saturated pixels in the trace
    datamodel.data[mask] = np.nan

    hdu = fits.PrimaryHDU(datamodel.data)
    hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/HATP14b/test_masked.fits', overwrite=True)

    #plt.imshow(donotuse[8])
    #plt.show()
    #plt.imshow(saturated[8])
    #plt.show()
    #plt.imshow(mask[8])
    #plt.show()

    deepstack = np.nanmedian(datamodel.data, axis=0)
    rms = np.nanstd(datamodel.data, axis=0)

    #hdu = fits.PrimaryHDU(deepstack)
    #hdu.writeto('/Users/albert/NIRISS/Commissioning/analysis/HATP14b/test_pre_interpbadpstack.fits', overwrite=True)

    return deepstack, rms


def build_mask_contamination(order, x, y, halfwidth=15, subarray='SUBSTRIP256'):
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
        rateints = fits.getdata(datamodel)
        dims = np.shape(rateints)
        if dims[-1] == 2048: subarray = 'FULL'
        elif dims[-1] == 256: subarray = 'SUBSTRIP256'
        elif dims[-1] == 96: subarray = 'SUBSTRIP96'
        else:
            print('error should be size full ss96 or ss256')
            sys.exit()
    else:
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


def interp_badpix(image, noise):
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
    nbad = np.size(badx)
    for i in range(nbad):
        image[bady[i], badx[i]] = np.nanmedian(image[bady[i]-ky:bady[i]+ky+1, badx[i]-kx:badx[i]+kx+1])
        # error is the standard deviations of pixels that were used in the median. Not division by sqrt(n).
        noise[bady[i], badx[i]] = np.nanstd(image[bady[i]-ky:bady[i]+ky+1, badx[i]-kx:badx[i]+kx+1])

    return image, noise


def soss_interp_badpix(modelin, outdir):

    # Create a deep stack from the time series.
    # Interpolate on that deep stack.
    # Then use the deepstack as pixel replacement values in single integrations.

    # Create a deep stack from the time series
    stack, stackrms = stack_datamodel(modelin)
    hdu = fits.PrimaryHDU(stack)
    hdu.writeto(outdir+'/test_pre_interpbadpstack.fits', overwrite=True)

    # Interpolate the deep stack
    clean_stack, stack_noise = interp_badpix(stack, stackrms)
    hdu = fits.PrimaryHDU(stack)
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
        # Replace bad pixel here by the clean stack value
        modelin.data[i][ind] = np.copy(clean_stack[ind])
        modelin.err[i][ind] = np.copy(stack_noise[ind])
        # Set the DQ map to good for all pixels except NaNs
        # TODO: this should be more clever so that ATOCA has knowledge of interpolated pixels
        modelin.dq[i][notrefpix] = 0

    hdu = fits.PrimaryHDU(modelin.data)
    hdu.writeto(outdir+'/test_interpolated_segment.fits', overwrite=True)

    return modelin


def remove_nans(datamodel):
    # Checks that the JWST Data Model does not contains NaNs

    #if True:
    #    jwstmodel = datamodels.open(jwstmodel)

    modelout = datamodel.copy()

    ind = (~np.isfinite(datamodel.data)) | (~np.isfinite(datamodel.err))
    modelout.data[ind] = 0
    modelout.err[ind] = np.nanmedian(datamodel.err)*10
    modelout.dq[ind] += 1

    # Replace occasional outlier by the median of the TSO at that pixel


    # Interpolate remaining bad pixels with surrounding pixels
    #modelout = interp_badpix(modelout, ind)


    # Check that the exposure type is NIS_SOSS
    modelout.meta.exposure.type = 'NIS_SOSS'

    return modelout


def background_subtraction(datamodel, aphalfwidth=[30,30,30], outdir=None, verbose=False,
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

def combine_timeseries(wildcard, outputname):

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


    print(np.shape(ts))
    # normalize each wavelength
    ts = ts.transpose(1, 0, 2)
    ts = ts / np.nanmedian(ts, axis=0)
    ts = ts.transpose(1, 0, 2)

    hdu = fits.PrimaryHDU(ts)
    hdu.writeto(outputname, overwrite=True)

    return


def greyscale_rms(ts_greyscale, title=''):
    '''
    Removes outliers form the greayscale and perform standard deviation
    '''

    outdir = os.path.dirname(ts_greyscale)
    #basename = os.path.basename(os.path.splitext(ts_greyscale)[0])
    #basename = basename.split('_nis')[0]+'_nis'
    #stackbasename = basename+'_stack'

    #a = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/HATP14b/timeseries_combined_1f_ap25_20220711.fits.gz')
    #a = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/HATP14b/timeseries_combined_1f_ap25_atoca_20220713.fits.gz')
    #a = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/HATP14b/timeseries_combined_20220610.fits')
    a = fits.getdata(ts_greyscale)
    norder, nint, dimx = np.shape(a)


    rms = np.zeros((norder, dimx)) * np.nan
    fig = plt.figure(figsize=(8,5))
    for m in range(3):
        for x in range(dimx-8):
            col = a[m, :, x+4]
            colnonan = col[np.where(np.isfinite(col))[0]]
            colclipped, low, high = sigmaclip(colnonan, low=3, high=3)
            rms[m, x+4] = np.std(colclipped)

        plt.plot(1/rms[m,:], label='Order {:}'.format(m+1))
    plt.legend()
    plt.xlabel('Column (pixels)')
    plt.ylabel('RMS noise (ppm)')
    plt.ylim((0,500))
    plt.title(title)
    plt.grid()
    plt.savefig(outdir+'/greyscale_rms.png')
    #plt.show()
    return



if __name__ == "__main__":
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