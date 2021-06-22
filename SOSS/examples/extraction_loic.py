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

def readsim(exposure_name):
    handle = fits.open(exposure_name)
    hdr = handle[0].header
    os = int(hdr['NOVRSAMP'])
    xpad = int(os * hdr['XPADDING'])
    ypad = int(os * hdr['YPADDING'])
    data = handle[0].data
    #data = np.flipud(data[0,:,:])
    data = data[0,:,:]
    data = data[ypad:-ypad,xpad:-ypad]
    dimy, dimx = np.shape(data)
    print(np.shape(data))

    return data, dimy, dimx, os

def readtrace(os):
    trace_filename = '/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv'
    pars = tp.get_tracepars(trace_filename, disable_rotation=False)
    w = np.linspace(0.7, 3.0, 10000)
    x, y, mask = tp.wavelength_to_pix(w, pars, m=1, oversample=os, subarray='SUBSTRIP256')
    x_index = np.arange(2048 * os)
    # np.interp needs ordered x
    ind = np.argsort(x)
    x, w = x[ind], w[ind]
    wavelength = np.interp(x_index, x, w)
    y_index = np.interp(x_index, x, y)

    return x_index, y_index, wavelength


# Start of debugging code


exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_grayseed.fits'
data1, dimy1, dimx1, os1 = readsim(exposure_name)
nint1 = 1
method_label1 = 'Noiseless Convolved Trace, grey pixels, os={:}'.format(os1)
x1, y1, w1 = readtrace(os1)
box_aperture = soss.box_aperture(data1, x1, y1, os=os1, box_width=75.0)
hdu = fits.PrimaryHDU()
hdu.data = box_aperture
hdu.writeto('/genesis/jwst/userland-soss/loic_review/box_aperture.fits', overwrite=True)
flux1 = soss.aperture_extract(data1, x1, box_aperture, mask=None)
Flambda1 = soss.elecflux_to_flambda(flux1, w1)


exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_trace_grayseed.fits'
data2, dimy2, dimx2, os2 = readsim(exposure_name)
nint2 = 1
method_label2 = 'Noiseless Seed Trace, grey pixels, os={:}'.format(os2)
x2, y2, w2 = readtrace(os2)
box_aperture = soss.box_aperture(data2, x2, y2, os=os2, box_width=75.0)
hdu = fits.PrimaryHDU()
hdu.data = box_aperture
hdu.writeto('/genesis/jwst/userland-soss/loic_review/box_aperture.fits', overwrite=True)
flux2 = soss.aperture_extract(data2, x2, box_aperture, mask=None)
Flambda2 = soss.elecflux_to_flambda(flux2, w2)

exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_binaryseed.fits'
data3, dimy3, dimx3, os3 = readsim(exposure_name)
nint3 = 1
method_label3 = 'Noiseless Convolved Trace, binary pixels, os={:}'.format(os3)
x3, y3, w3 = readtrace(os3)
box_aperture = soss.box_aperture(data3, x3, y3, os=os3, box_width=75.0)
hdu = fits.PrimaryHDU()
hdu.data = box_aperture
hdu.writeto('/genesis/jwst/userland-soss/loic_review/box_aperture.fits', overwrite=True)
flux3 = soss.aperture_extract(data3, x3, box_aperture, mask=None)
Flambda3 = soss.elecflux_to_flambda(flux3, w3)

exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_trace_binaryseed.fits'
data4, dimy4, dimx4, os4 = readsim(exposure_name)
nint4 = 1
method_label4 = 'Noiseless Seed Trace, binary pixels, os={:}'.format(os4)
x4, y4, w4 = readtrace(os4)
box_aperture = soss.box_aperture(data4, x4, y4, os=os4, box_width=75.0)
hdu = fits.PrimaryHDU()
hdu.data = box_aperture
hdu.writeto('/genesis/jwst/userland-soss/loic_review/box_aperture.fits', overwrite=True)
flux4 = soss.aperture_extract(data4, x4, box_aperture, mask=None)
Flambda4 = soss.elecflux_to_flambda(flux4, w4)

# convolve binned to native
data5 = soss.rebin(data1, os1)
os5 = 1
method_label5 = 'Noiseless Convolved Trace, grey pixels, rebin from os={:} to native'.format(os1)
x5, y5, w5 = readtrace(os5)
box_aperture = soss.box_aperture(data5, x5, y5, os=os5, box_width=75.0)
hdu = fits.PrimaryHDU()
hdu.data = box_aperture
hdu.writeto('/genesis/jwst/userland-soss/loic_review/box_aperture.fits', overwrite=True)
flux5 = soss.aperture_extract(data5, x5, box_aperture, mask=None)
Flambda5 = soss.elecflux_to_flambda(flux5, w5)


exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_yspread.fits'
exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_000000.fits'
data6, dimy6, dimx6, os6 = readsim(exposure_name)
nint6 = 1
method_label6 = 'Noiseless Convolved Trace, grey pixels, yspread, os={:}'.format(os6)
x6, y6, w6 = readtrace(os6)
box_aperture = soss.box_aperture(data6, x6, y6, os=os6, box_width=75.0)
hdu = fits.PrimaryHDU()
hdu.data = box_aperture
hdu.writeto('/genesis/jwst/userland-soss/loic_review/box_aperture.fits', overwrite=True)
flux6 = soss.aperture_extract(data6, x6, box_aperture, mask=None)
Flambda6 = soss.elecflux_to_flambda(flux6, w6)

exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_trace_yspread.fits'
exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_trace_000000.fits'
data7, dimy7, dimx7, os7 = readsim(exposure_name)
nint7 = 1
method_label7 = 'Noiseless Seed Trace, grey pixels, yspread, os={:}'.format(os7)
x7, y7, w7 = readtrace(os7)
box_aperture = soss.box_aperture(data7, x7, y7, os=os7, box_width=75.0)
hdu = fits.PrimaryHDU()
hdu.data = box_aperture
hdu.writeto('/genesis/jwst/userland-soss/loic_review/box_aperture.fits', overwrite=True)
flux7 = soss.aperture_extract(data7, x7, box_aperture, mask=None)
Flambda7 = soss.elecflux_to_flambda(flux7, w7)

plt.figure()
plt.plot(w1, Flambda1, label=method_label1)
plt.plot(w2, Flambda2, label=method_label2)
#plt.plot(w3, Flambda3, label=method_label3)
#plt.plot(w4, Flambda4, label=method_label4)
plt.plot(w5, Flambda5, label=method_label5)
plt.plot(w6, Flambda6, label=method_label6)
plt.plot(w7, Flambda7, label=method_label7)
plt.xlabel('Wavelength [microns]')
plt.legend()
plt.savefig('/genesis/jwst/userland-soss/loic_review/trace_investigation.png')
plt.show()



sys.exit()

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
        handle = fits.open(exposure_name)
        hdr = handle[0].header
        os = int(hdr['NOVRSAMP'])
        xpad = int(os * hdr['XPADDING'])
        ypad = int(os * hdr['YPADDING'])
        data = handle[0].data
        data = np.flipud(data[0,:,:])
        data = data[ypad:-ypad,xpad:-ypad]
        print(np.shape(data))
        dimy, dimx = np.shape(data)
        nint = 1
        #os = 4
        method_label = 'Noiseless Convolved Trace, os={:}'.format(os)
    elif method == 2:
        exposure_name = '/genesis/jwst/userland-soss/loic_review/tmp/clear_trace_000000.fits'
        handle = fits.open(exposure_name)
        hdr = handle[0].header
        os = int(hdr['NOVRSAMP'])
        xpad = int(os * hdr['XPADDING'])
        ypad = int(os * hdr['YPADDING'])
        data = handle[0].data
        data = fits.getdata(exposure_name)
        data = np.flipud(data[0,:,:])
        data = data[ypad:-ypad,xpad:-ypad]
        print(np.shape(data))
        dimy, dimx = np.shape(data)
        nint = 1
        #os = 4
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

    #for i in range(np.size(x_index)):
    #    print(i, x_index[i], y_index[i])

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
    Flambda = soss.elecflux_to_flambda(flux, wavelength)


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