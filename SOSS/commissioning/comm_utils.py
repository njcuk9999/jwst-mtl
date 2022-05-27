import os.path

import matplotlib.pyplot as plt

import numpy as np

from astropy.io import fits

from jwst import datamodels

import SOSS.dms.soss_centroids as soss_centroids

import sys



def stack_datamodel(datamodel):


    deepstack = np.nanmedian(cube, axis=0)
    rms = np.nanstd(cube, axis=0)
    return deepstack, rms

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

    ind = (np.isfinite(datamodel.data) == False) | (np.isfinite(datamodel.err) == False)
    modelout.data[ind] = 0
    modelout.err[ind] = 1000000

    # Check that the exposure type is NIS_SOSS
    modelout.meta.exposure.type = 'NIS_SOSS'

    return modelout


def background_subtraction(datamodel, aphalfwidth=[30,30,30], outdir=None, verbose=False, override_background=None,
                           applyonintegrations=False):

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

    # Make a mask of the traces
    maskcube = aperture_from_scratch(datamodel, aphalfwidth=aphalfwidth, outdir=cntrdir, verbose=verbose)
    # Crunch the cube (one order per slice) to a 2D mask
    mask = np.sum(maskcube, axis=0, dtype='bool')
    # Apply aperture masking to the input data
    maskeddata[:, mask] = np.nan
    # Apply bad pixel masking to the input data
    maskeddata[datamodel.dq != 0] = np.nan
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
    plt.show()

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
        outdir = os.path.dirname(spectrum_file)+'/plots_'+basename+'/'
    # Create the output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Start processing the spectra file
    multispec = datamodels.open(spectrum_file)

    # spectra are stored at indice 1 (order 1), then 2 (order2) then 3 (order 3) then 4 (order 1, 2nd time step), ...
    nint = multispec.meta.exposure.nints
    norder = int(np.shape(multispec.spec)[0] / nint)

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

    plt.figure()
    for i in range(nint):
        plt.plot(wavelength[i, 0], flux[i, 0] + 0.02 * i)
    plt.show()

    return


def check_atoca(fedto_atoca_filename, atoca_model_filename, outdir=None):
    '''
    Check that the ATOCA modelling went well by looking at residual maps for each order.
    '''

    obser = datamodels.open(fedto_atoca_filename)
    #<CubeModel(20, 256, 2048)>
    nints = obser.meta.exposure.nints
    if obser.meta.subarray.name == 'SUBSTRIP96':
        norders = 2
    else:
        norders = 3

    model = datamodels.open(atoca_model_filename)
    #<SossExtractModel>

    # Residual map after subtracting both orders' model
    cube = obser.data - model.order1 - model.order2

    # Forge output directory where plots will be written
    if outdir == None:
        basename = os.path.basename(os.path.splitext(fedto_atoca_filename)[0])
        outdir = os.path.dirname(fedto_atoca_filename)+'/atoca_'+basename+'/'
    # Create the output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hdu = fits.PrimaryHDU(cube)
    hdu.writeto(outdir+'residualmap.fits', overwrite=True)

    return