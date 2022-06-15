import os.path

import matplotlib.pyplot as plt

import numpy as np
import scipy.interpolate

import glob

from astropy.io import fits

from jwst import datamodels

import SOSS.dms.soss_centroids as soss_centroids

import sys

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


    deepstack = np.nanmedian(cube, axis=0)
    rms = np.nanstd(cube, axis=0)
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




def aperture_from_scratch(datamodel, norders=3, aphalfwidth=[30,20,20], outdir=None, datamodel_isfits=False,
                          verbose=False):
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

    centroids = soss_centroids.get_soss_centroids(stack, mask=None, subarray=subarray, halfwidth=2,
                                                  poly_orders=None, apex_order1=None, calibrate=True, verbose=verbose,
                                                  outdir=outdir)

    # exemple of the dictionnary: x_o2 = centroids['order 1']['X centroid']
    #o2_dict['X centroid'] = x_o2
    #o2_dict['Y centroid'] = y_o2
    #o2_dict['trace widths'] = w_o2
    #o2_dict['poly coefs'] = par_o2
    #centroids['order 2'] = o2_dict
    x_o1, y_o1 = centroids['order 1']['X centroid'], centroids['order 1']['Y centroid']
    x_o2, y_o2 = centroids['order 2']['X centroid'], centroids['order 2']['Y centroid']
    x_o3, y_o3 = centroids['order 3']['X centroid'], centroids['order 3']['Y centroid']
    w_o1 = centroids['order 1']['trace widths']
    w_o2 = centroids['order 2']['trace widths']
    w_o3 = centroids['order 3']['trace widths']


    with open(outdir+'centroids_o1.txt', 'w') as the_file:
        the_file.write('Order, x, y, width\n')
        for i in range(len(x_o1)):
            the_file.write('{:} {:} {:} {:}\n'.format(1, x_o1[i], y_o1[i], w_o1[i]))
    with open(outdir+'centroids_o2.txt', 'w') as the_file:
        the_file.write('Order, x, y, width\n')
        for i in range(len(x_o2)):
            the_file.write('{:} {:} {:} {:}\n'.format(2, x_o2[i], y_o2[i], w_o2[i]))
    with open(outdir+'centroids_o3.txt', 'w') as the_file:
        the_file.write('Order, x, y, width\n')
        for i in range(len(x_o3)):
            the_file.write('{:} {:} {:} {:}\n'.format(3, x_o3[i], y_o3[i], w_o3[i]))


    # Create a cube containing the mask for all orders
    maskcube = np.zeros((norders,dims[-2],dims[-1]))

    for m in range(norders):
        if m == 0: ordercen = centroids['order 1']
        if m == 1: ordercen = centroids['order 2']
        if m == 2: ordercen = centroids['order 3']

        mask_trace = soss_centroids.build_mask_trace(ordercen['Y centroid'], subarray=subarray,
                                                     halfwidth=aphalfwidth[m], extend_below=False, extend_above=False)
        mask = np.zeros(np.shape(mask_trace))
        mask[mask_trace == True] = 1

        maskcube[m,:,:] = np.copy(mask_trace)

        hdu = fits.PrimaryHDU(mask)
        hdu.writeto(outdir+'/trace_mask_order{:}.fits'.format(m+1), overwrite=True)

    return maskcube




def remove_nans(datamodel):
    # Checks that the JWST Data Model does not contains NaNs

    #if True:
    #    jwstmodel = datamodels.open(jwstmodel)

    modelout = datamodel.copy()

    ind = (~np.isfinite(datamodel.data)) | (~np.isfinite(datamodel.err))
    modelout.data[ind] = 0
    modelout.err[ind] = np.nanmedian(datamodel.err)*10
    modelout.dq[ind] += 1

    # Check that the exposure type is NIS_SOSS
    modelout.meta.exposure.type = 'NIS_SOSS'

    return modelout


def background_subtraction(datamodel, aphalfwidth=[30,30,30], outdir=None, verbose=False, override_background=None,
                           applyonintegrations=False, contamination_mask=None):

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

    if contamination_mask is not None:
        contmask = fits.getdata(contamination_mask)
        contmask = np.where(contmask >= 1, 1, 0)
        print(np.shape(contmask))
        maskeddata[:, contmask] = np.nan
        print(np.shape(maskeddata[:, contmask]))
        # do_not_use (apply to input and return as output)
        datamodel.dq = datamodel.dq[:, contmask] + 1
        print('Allo')
    # Apply bad pixel masking to the input data
    maskeddata[datamodel.dq != 0] = np.nan

    print('Hi')
    # Make a mask of the traces
    maskcube = aperture_from_scratch(datamodel, aphalfwidth=aphalfwidth, outdir=cntrdir, verbose=verbose)

    print('Ni Hao')
    # Crunch the cube (one order per slice) to a 2D mask
    mask = np.sum(maskcube, axis=0, dtype='bool')
    # Apply aperture masking to the input data
    maskeddata[:, mask] = np.nan
    hdu = fits.PrimaryHDU(maskeddata)
    hdu.writeto(cntrdir+'background_mask.fits', overwrite=True)

    # Bottom portion not usable in the FULL mode and skews levels estimates
    if datamodel.meta.subarray.name == 'FULL': maskeddata[:, 0:1024, :] = np.nan

    # Identify pixels free of astrophysical signal (e.g. 25th percentile)
    if applyonintegrations == True:
        # Measure level on each integration
        levels = np.nanpercentile(maskeddata, 25, axis=1)
        nlevels = 1
    else:
        # Measure level on deep stack of all integrations (default)
        maskeddata = np.nanmedian(maskeddata, axis=0)
        levels = np.nanpercentile(maskeddata, 25, axis=0)
        nlevels = datamodel.meta.exposure.nints

    # Open the reference background image
    if override_background is None:
        ### need to handle default background downlaoded from CRDS, eventually
        print('ERROR: no background specified. Crash!')
        sys.exit()
    else:
        backref = fits.getdata(override_background)

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

    output.write(outdir+'/'+basename+'_backsubtracted.fits', overwrite=True)

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


def combine_segments(prefix):
    print()
    return


def combine_multi_spec(wildcard, outdir):
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

    combined.save(outdir+'/extracted_spectra_combined.fits')

    return


## Example
#filename_list = ['spec_part1.fits', 'spec_part2.fits', 'etc.fits']
#multi_spec_list = [datamodels.open(fname) for fname in filename_list]
#combined = combine_multi_spec(multi_spec_list)
#combined.save('spec_combined.fits')

def combine_timeseries(wildcard, outdir):

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
    hdu.writeto(outdir+'/timeseries_combined.fits', overwrite=True)

    return









if __name__ == "__main__":
    #a = make_mask_nis17()
    outdir = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/'
    wildcard = outdir+'supplemental_jw01541001001_04101_00001-seg00?_nis/timeseries_greyscale_rawflux.fits'
    a = combined_timeseries(wildcard, outdir)