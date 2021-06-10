'''
Example for performing spectrum extraction
'''


from astropy.io import fits
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import trace.tracepol as tp
import itsosspipeline as soss

# Assume that we have a fit file exposure



# Read the exposure
# Extract trace position and determine offset wrt ref file or model (x, y, w relations)
# Option - generate a new set of ref files based on observed set
# Perform classic box extraction column by column
# Perform decontamination
# Perform box extraction on decontaminated images

debug = False

'''
Read in the rateint exposure (processed through DMS)
'''
plt.figure()

for method in range(3):
    if method == 0:
        exposure_name = '/genesis/jwst/userland-soss/loic_review/test_clear_noisy_rateints.fits'
        #exposure_name = '/genesis/jwst/userland-soss/loic_review/timeseries_20210604_formeeting/test_clear_noisy_rateints.fits'
        handle = fits.open(exposure_name)
        data = handle['SCI'].data
        dataerr = handle['ERR'].data
        datavar = dataerr**2
        datadq = handle['DQ'].data
        nint, dimy, dimx = np.shape(data)
        os = 1
        method_label = 'Noisy DMS (rateint)'
        if nint == 1:
            data = data[0,:,:]
            dataerr = dataerr[0,:,:]
            dataver = datavar[0,:,:]
            datadq = datadq[0,:,:]
    elif method == 1:
        exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_000000.fits'
        data = fits.getdata(exposure_name)
        data = np.flipud(data[0,:,:])
        print(np.shape(data))
        dimy, dimx = np.shape(data)
        nint = 1
        os = 4
        method_label = 'Noiseless Convolved Trace, os={:}'.format(os)
    elif method == 2:
        exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_trace_000000.fits'
        data = fits.getdata(exposure_name)
        data = np.flipud(data[0,:,:])
        print(np.shape(data))
        dimy, dimx = np.shape(data)
        nint = 1
        os = 4
        method_label = 'Noiseless Seed Trace, os={:}'.format(os)
    else:
        print('hein')
        sys.exit()

    '''
    Establish the trace position relations: y(x), w(x), x(w), y(w)
    Method 1 - reading reference files
    Method 2 - reading the optics model
    Method 3 - measuring the trace, model and figuring conversion from measured vs modelled
    '''

    #a = tp.test_tracepol()
    #sys.exit()

    # Method 1 - reading reference files
    #TODO: how to read the ref files to get y(x), w(x) , x(w), y(w)

    # Method 2 - reading the optics model

    if False:
        trace_filename = '/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv'
        pars = tp.get_tracepars(trace_filename, disable_rotation=True)
        # The following way of retrieving the trace center, y, as a function of integer x position DOES
        # NOT work. The retrieved x after converting to wavelength is off by up to 1.2 pixels from
        # the seeded x. Issue is with how tracepol gets the conversion specpix_to_wavelength() by
        # refitting the points rather than by really doing an inversion. Anyway. UNtil that is fixed,
        # resort on interpolating the wave--->specpix at integer x values.
        # x indices
        x_index1 = np.linspace(0,2047,2048)
        w, mask = tp.specpix_to_wavelength(x_index1, pars, m=1)
        x_index2, y_index, mask = tp.wavelength_to_pix(w, pars, m=1, subarray='SUBSTRIP256')
        dx = x_index2 - x_index1
        for i in range(np.size(x_index1)):
            print(i,x_index1[i], x_index2[i], dx[i], y_index[i], w[i])
        plt.figure()
        plt.imshow(data[0,:,:], vmin=0, vmax=1000, origin='lower')
        plt.plot(x_index, y_index, color='black')
        plt.show()

    trace_filename = '/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv'
    pars = tp.get_tracepars(trace_filename, disable_rotation=False)
    w = np.linspace(0.7,3.0,10000)
    x, y, mask = tp.wavelength_to_pix(w, pars, m=1, oversample=os, subarray='SUBSTRIP256')
    x_index = np.arange(2048*os)
    # np.interp needs ordered x
    ind = np.argsort(x)
    x, w = x[ind], w[ind]
    wavelength = np.interp(x_index, x, w)
    y_index = np.interp(x_index, x, y)

    for i in range(np.size(x_index)):
        print(i, x_index[i], y_index[i])

    if False:
        plt.figure()
        plt.scatter(x, y, color='red')
        plt.scatter(x_index, y_index, color='blue')
        plt.show()


    # Method 3 - measure the trace, compare with ref files and apply necessary adjustments
    #TODO: create the method 3 process


    '''
    Perform box aperture spectrum extraction
    '''

    box_aperture = soss.box_aperture(data, x_index, y_index, os=os, box_width=75.0)
    hdu = fits.PrimaryHDU()
    hdu.data = box_aperture
    hdu.writeto('/genesis/jwst/userland-soss/loic_review/box_aperture.fits', overwrite=True)
    flux = soss.aperture_extract(data, x_index, box_aperture, mask=None)

    # From flux [adu/pixel/sec] to Flambda [J/sec/m2/um]
    h = 6.62606957e-34
    c = 3e+8
    gain = 1.6 # e-/adu
    Area = 25.0 # m2
    Eph = h * c / (wavelength*1e-6) # J/photon
    dispersion = np.zeros_like(wavelength) # um/pixel
    dispersion[1:] = np.abs(wavelength[1:]-wavelength[0:-1])
    dispersion[0] = dispersion[1]
    Flambda = flux * gain * Eph / Area / dispersion

    plt.plot(wavelength, Flambda, label=method_label)
    #plt.plot(wavelength, flux)
    #plt.plot(wavelength, Eph)
    #plt.plot(wavelength, dispersion)

plt.xlabel('Wavelength [microns]')
plt.legend()
plt.savefig('/genesis/jwst/userland-soss/loic_review/trace_investigation.png')
plt.show()

sys.exit()








median_spectrum = np.median(flux[0:40,:], axis=0)

plt.figure()
for i in range(nint):
    plt.plot(flux[i,:]/median_spectrum, linewidth=0.02, alpha=0.2, color='navy')
plt.xlabel('Pixels')
plt.ylabel('Relative flux')
plt.ylim((0.97,1.01))
plt.title('Time-Series Extracted Spectra')

plt.savefig('/genesis/jwst/userland-soss/loic_review/allspectra.png')
plt.show()

white = np.sum(flux[:,1375:1600], axis=1)

plt.figure()
plt.plot(white)
plt.title('White light curve')
plt.xlabel('Time')
plt.ylabel('Total flux')
plt.savefig('/genesis/jwst/userland-soss/loic_review/whitelight.png')
plt.show()


#print(np.shape(flux))





'''
Write extracted spectra on disk in the format readable by our light curve analysis tool.
BJD, flux [e- or Flambda], flux_err, exptime [sec], wavelength [um], channel [integer for each sample]
FITS header keyword JDOFFSET = 2459518 subtracted from BJD to retain good precision on our floats.
'''

MJD = np.ones(nint)
dw = wavelength*0
integtime = np.ones(nint)
flux_err = flux*0.01
soss.write_spectrum(wavelength, dw, MJD, integtime, flux, flux_err,
                    '/genesis/jwst/userland-soss/loic_review/spectrum_order1.fits')

print('Extraction example 1 completed successfully.')