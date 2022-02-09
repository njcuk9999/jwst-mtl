from astropy.io import fits
import numpy as np
from scipy.stats import binned_statistic

def read_soss_spectra(spectra_file):
    '''
    Reads the output extracted spectra fits file produced by the DMS
    :param spectra_file:
    :return:
    wavelength[norder,nintegration,ntimestep]
    same for flux and err
    '''

    data = fits.open(spectra_file)

    # spectra are stored at indice 1 (order 1), then 2 (order2) then 3 (order 3) then 4 (order 1, 2nd time step), ...

    integ = []
    order = []
    for i in enumerate(data[0:-1]):
        extname = data[i[0]+1].header['EXTNAME']
        if extname == 'EXTRACT1D':
            integ.append(data[i[0]+1].header['INT_NUM'])
            order.append(data[i[0] + 1].header['SPORDER'])
    nint = np.max(integ)
    norder = np.max(order)

    print('Number of spectral orders ({:}) and integrations ({:})'.format(norder, nint))

    # Read the spectra into variables
    wavelength = np.zeros((norder, nint, 2048))
    flux = np.zeros((norder, nint, 2048))
    err = np.zeros((norder, nint, 2048))

    for i in enumerate(data[0:-1]):
        extname = data[i[0] + 1].header['EXTNAME']
        if extname == 'EXTRACT1D':
            intg = data[i[0] + 1].header['INT_NUM']
            m = data[i[0] + 1].header['SPORDER']
            wavelength[m-1, intg-1, :] = data[i[0] + 1].data['WAVELENGTH']
            flux[m-1, intg-1, :] = data[i[0] + 1].data['FLUX']
            err[m-1, intg-1, :] = data[i[0] + 1].data['FLUX_ERROR']

    return wavelength, flux, err



#a, b, c, = read_soss_spectra('/genesis/jwst/userland-soss/loic_review/extracted_spectra.fits')



def bin_lightcurves(wavelength, flux, flux_err, nbins):
    '''

    :param wavelength: [norder, nintegration, ncolumns]
    :param flux: same format
    :param flux_err:
    :param nbins: is a scalar applicable to all orders

    ###NO!!! array os same size as number of orders (2 or 3)
    :return:
    '''

    norders, nints, ncols = np.shape(wavelength)

    #if len(nbins) == 1:
    #    nbins_array = []
    #    for m in range(norders):
    #        nbins_array.append(nbins)
    #    # replace scalar nbins by array of nbins (one for each order)
    #    nbins = nbins_array

    # Number of cols per bin, last bin contains equal or more
    bin_width = []
    for m in range(norders):
        bin_width.append(ncols // nbins)

    # TODO: be more clever than using ncols for order 2 which has many nan cols

    # Initialize binned arrays
    bin_flux = np.zeros((norders, nints, nbins)) * np.nan
    bin_flux_err = np.zeros((norders, nints, nbins)) * np.nan
    bin_wavelength = np.zeros((norders, nints, nbins)) * np.nan

    # Bin. i.e. Sum fluxes, Sum errors in quadrature, average wavelengths
    for m in range(norders):
        for i in range(nints):
            for n in range(nbins):
                f = flux[m, i, n*bin_width[m]:(n+1)*bin_width[m]]
                df = flux_err[m, i, n*bin_width[m]:(n+1)*bin_width[m]]
                w = wavelength[m, i, n*bin_width[m]:(n+1)*bin_width[m]]
                good = np.isfinite(f) & np.isfinite(df)
                bin_flux[m, i, n] = np.sum(f[good])
                bin_flux_err[m, i, n] = np.sqrt(np.sum(np.power(df[good],2)))
                bin_wavelength[m, i,n] = np.mean(w[good])

    return bin_wavelength, bin_flux, bin_flux_err