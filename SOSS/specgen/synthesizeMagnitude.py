#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:06:06 2020

@author: albert
"""
# kona version

from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys

# Import trace position library
import trace.tracepol as tp





def expected_flux_calibration(filtername, magnitude, model_angstrom, model_flux,
                              list_orders, subarray='SUBSTRIP256',
                              convert_to_adupersec=False,
                              F277=None, verbose=None,
                              trace_file = None,
                              response_file = None,
                              pathfilter = None,
                              pathvega = None):
    
    # This function establishes what the absolute electron flux (e- per second)
    # is when integrating the flux over the image (each order being treated
    # separately). It accepts a filter name and magnitude as the normalization
    # anchor. The input stellar spectrum needs to be in the following units:
    # 1) wavelengths in microns; 2) flux in energy per wavelength bin (Flambda)
    # (NOT per frequency bin - Fnu). Another input is the list_orders array
    # which lists the orders to make the calculations for. As of June 26 2020,
    # only orders 0 to 3 are supported.
    #
    # The steps are the following:
    # 1) Use the binned model_um and binned model_flux to calibrate the flux
    #    based on a a filter magnitude
    # 2) Integrate the calibrated flux within the SOSS Order 1 wavelength span
    #    Use Geert Jan's p2w to determine the wavelength coverage.
    
    # model_um is the stellar model in microns, it should be sorted.
    # model_flux i sthe stellar model in F lambda (energy flux per wavelength)
    # filtername is an ascii representing the filter to normalize with. The 
    #     choice of filter name is large, see readFilter below.
    # magnitude is the magnitude in the Vega or AB system used for that filter
    #
    # OUTPUT:
    # The output is an array containing as many entries as there are orders
    # in list_orders. The output is the expected count rate (electrons per
    # second) integrated over the full 256x2048 subarray.
    # A few assumptions are made:
    # 1) The JWST surface area is assumed to be 25 m2
    # 2) The SOSS throughput from the reference file is used
    # 3) The quantum yield from the reference file is used
    # 4) NO detector gain is applied (e-/sec to adu/sec is to performed) 
    #    by default
    # Note: the count rate is by default in electrons per second. The optional
    #       keyword convert_to_adupersec will trigger output to be in adu/sec.
    

    # To return the expectd counts of an F277W + GR700XD exposure rather than
    # the default CLEAR + F277W exposure, use the F277=True option. It will
    # calibrate based on the model flux (without F277) but compute the expected
    # counts based on the model flux being multiplied by the F277 transmission
    # curve.

    # Output text to let the user know where we are
    print('Entering expected_flux_calibration')


    
    #filtername = 'J'
    #magnitude = 8.5
    
    # Choice of filter names can be found here: 
    # synthesizeMagnitude.py --> readFilter
    # When you embed in specgen, replace paths by relative path in the github
    # hierarchy.
    if pathvega == None: pathvega = './specgen/FilterSVO/'
    if pathfilter == None: pathfilter = './specgen/FilterSVO/'
    if trace_file == None: trace_file = './trace/NIRISS_GR700_trace_extended.csv'
    if response_file == None: response_file = './tables/NIRISS_Throughput_20210318.fits'
    
    # Some more constants
    hc = 6.62607015e-34 * 299792458.0 # J * m
    JWST_area = 25.0 # m2
    detector_gain = 1.6 # e-/adu

    # Use wavelength in microns internally
    model_um = model_angstrom / 10000.

    # initialize some variables
    lba = np.copy(model_um)
    
    # Prepare output array
    output_elec_per_sec = np.zeros(np.size(list_orders))
    
    # Do the synthetic magnitude call on the model spectrum
    # waves should be in microns
    # Flux should be Flambda, not Fnu (Make sure that is the case)
    syntmag = syntMag(model_um,model_flux,filtername,
                        path_filter_transmission=pathfilter,
                        path_vega_spectrum=pathvega,verbose=verbose)
    if verbose is True:
        print(syntmag)
        print('Synthetic magnitude of input spectrum: {:6.2f}'.format(syntmag))
    
    # The normalization scale is the difference between synthetic mag and
    # requested mag. So Flba is the flux normalized expressed in W/m2/um
    model_flux_norm = model_flux * 10**(-0.4*(magnitude-syntmag))
    
    # Check that this worked by re synthesizing the magnitude
    syntmag_check = syntMag(model_um,model_flux_norm,filtername, 
                            path_filter_transmission=pathfilter,
                            path_vega_spectrum=pathvega,verbose=verbose)
    if verbose is True:
        print('Synthetic magnitude of normalized spectrum: {:6.2f}'.format(syntmag_check))
    
    
    
    
    # Now that our spectrum is normalized in flux (W/m2/um), 
    # compute what the flux count (in electrons) should be for each order.

    # Get the trace parameters, function found in tracepol imported above
    tracepars = tp.get_tracepars(trace_file)
    
    # Loop for each spectral order
    for eachorder in range(np.size(list_orders)):
        m = list_orders[eachorder]
        
        # Get wavelength (in um) of first and last pixel of the Order m trace
        #TODO: Fix bug for order 3 in tp.subarray_wavelength_bounds, tracepol.py line 222
        (lbabound1, lbabound2), (pixbound1, pixbound2) = tp.subarray_wavelength_bounds(tracepars, subarray=subarray, m=m)
        print('CHECK lba bounds. m={:}, lbabound1={:}, lbabound2={:}'.format(m,lbabound1,lbabound2))
        print('CHECK specpix bounds. pixbound1={:}, pixbound2={:}'.format(pixbound1, pixbound2))

        # Make sure that integration wavelengths are on the subarray
        if np.isfinite(lbabound1) & np.isfinite(lbabound2) & np.isfinite(pixbound1) & np.isfinite(pixbound2):
        
            # The number of photons per second integrated over that range is
            # (Flambda: J/sec/m2/um requires dividing by photon energy to get counts)
            # Ephoton = h*c/lambda
            #
            # Model spectrum wavelength step size (in microns).
            # The only assumption is arrays is sorted but not nec. equal steps.
            wv_sampling_width = np.copy(model_um)
            wv_sampling_width[1:] = model_um[1:]-model_um[0:-1]
            wv_sampling_width[0] = wv_sampling_width[1]
            if verbose is True:
                plt.figure(figsize=(15,8))
                plt.scatter(model_um,wv_sampling_width,marker='.')
                plt.xlabel('Wavelength [angstroms]')
                plt.ylabel('Sample width [angstroms]')
                plt.title('Binned Stellar Spectrum going into Trace Seeding')

            # Wavelength indices of all pixels in Order m
            order_m_range = (model_um >= lbabound1) & (model_um <= lbabound2)

            # Get the throughput (response) and quantum yield
            T_um, Throughput = read_throughput(m, response_file)
            QY_um, QuantumYield = read_quantumyield(response_file)

            # In case an F277W filter exposure is needed. Read the F277W transmis-
            # sion curve and resample it on the same grid as the model flux.
            if F277 is True:
                f277_um,f277_t,f277_system = \
                    readFilter('NIRISS.F277W',
                               path_filter_transmission=pathfilter,
                               keepPeakAbsolute=True,returnWidth=False,
                               verbose = False)
                # Resample the transmission curve on the model grid
                f277_transmission = np.interp(model_um,f277_um,f277_t)

            # NOW. CONVERT THE NORMALIZED SPECTRUM TO PHOTON COUNTS ON THE DETECTOR
            # THAT REQUIRES ADDITIONAL ASSUMPTIONS: AREA, QUANTUM YIELD, THROUGHPUT

            # Photon energy depends on wavelength (lba is in um, so convert to m)
            Ephot = hc / (lba * 1e-6)

            # e-/sec = area [m2] x SUM(  Flambda [J/s/m2/um]
            #                          x Throughput [no dim]
            #                          x Quantum_yield [e-/photon]
            #                          x dlambda [um]
            #                          / E_photon [J]  )
            A = np.copy(JWST_area) # surface area of JWST in m2
            lba = np.copy(model_um) # wavelengths in um
            if F277 is True:
                Flba = model_flux_norm * f277_transmission
            else:
                Flba = np.copy(model_flux_norm) # Flambda in J/sec/m2/um
            dlba = np.copy(wv_sampling_width) # wavelengths sampling steps in um
            QY = np.interp(model_um, QY_um, QuantumYield)
            T = np.interp(model_um, T_um, Throughput)
            mi = np.copy(order_m_range)
            elec_per_sec = A * np.sum(Flba[mi]*T[mi]*QY[mi]*dlba[mi]/Ephot[mi])
            if verbose is True:
                print('Best calculations for the estimated electron counts for the whole order {:} trace:'.format(m))
                print('elec_per_sec', elec_per_sec)

            # Assign value to output array
            output_elec_per_sec[eachorder] = np.copy(elec_per_sec)
        else:
            # Case where no wavelength falls onto the detector area
            output_elec_per_sec[eachorder] = 0

    if convert_to_adupersec == True:
        if verbose is True:
            print('expected_flux_calibration returns adu/sec using a detector gain of {:} e-/adu.'.format(detector_gain))
        return(output_elec_per_sec/detector_gain)
    else:
        if verbose is True:
            print('expected_flux_calibration returns e-/sec. Use convert_to_adupersec if you want adu/sec instead.')
        return(output_elec_per_sec)

    
def measure_actual_flux(imagename, xbounds=[0,2048], ybounds=[0,256],
                        noversample=1):
    '''
    Measures the integrated flux in a spectral order on actual
    images.

    :param imagename:
    :param xbounds: assumes native pixels boundaries
    :param ybounds: assumes native pixels boundaries
    :param noversample:
    :return:
    '''

    # Convert to numpy arrays and oversampled coordinates
    xbounds_os = np.array(xbounds, dtype=np.int)*noversample
    ybounds_os = np.array(ybounds, dtype=np.int)*noversample

    # Read the cube on disk assuming a (norder, dimy, dimx) shape
    image = fits.getdata(imagename)
    norder, dimy, dimx = np.shape(image)
    image_cropped = image[:, ybounds_os[0]:ybounds_os[1], xbounds_os[0]:xbounds_os[1]]
    print('shape of the image on which flux is measured:', np.shape(image_cropped))

    # Measure the flux on a single order at a time
    measured_flux = []
    for i in range(norder):
        measured_flux.append(np.sum(image_cropped[i,:,:]))
    measured_flux = np.array(measured_flux)

    # Correct for the oversampling
    measured_flux = measured_flux / noversample**2

    return np.array(measured_flux)


def read_throughput(order, throughput_file=None):
    if throughput_file is None:
        throughput_file = './tables/NIRISS_Throughput_20210318.fits'
    a = fits.open(throughput_file)
    lba = np.array(a[1].data['LAMBDA']) # in nanometers
    lba_um = lba / 1000.0
    order0 = np.array(a[1].data['SOSS_ORDER0'])
    order1 = np.array(a[1].data['SOSS_ORDER1'])
    order2 = np.array(a[1].data['SOSS_ORDER2'])
    order3 = np.array(a[1].data['SOSS_ORDER3'])
    
    if order == 0:
        return(lba_um, order0)
    elif order == 1:
        return(lba_um, order1)
    elif order == 2:
        return(lba_um, order2)    
    elif order == 3:
        return(lba_um, order3)
    elif order == -1:
        # not known, assum 10% of order=1
        return(lba_um, 0.1 * order1)
    else:
        print('The reference throughput file only accepts order -1 to 3. For order -1, assume 10% of order 1.')
        sys.exit()

def read_quantumyield(throughput_file=None):
    if throughput_file is None:
        throughput_file = './tables/NIRISS_Throughput_20210318.fits'
    a = fits.open(throughput_file)
    lba = np.array(a[1].data['LAMBDA']) # in nanometers
    lba_um = lba / 1000.0
    quantum_yield = np.array(a[1].data['YIELD'])
    
    return(lba_um, quantum_yield)


def anchor_spectrum(wave_micron, flux_W_m2_micron, filtername, magnitude,
                    path_filter_transmission, verbose=False):
    '''
    Anchors a spectrum to the flux measured in a photometric band.
    :param wave_micron:
    :param flux_W_m2_micron:
    :param filtername:
    :param magnitude:
    :param path_filter_transmission:
    :param path_vega_spectrum:
    :return:
    '''

    rawmag = syntMag(wave_micron, flux_W_m2_micron, filtername,
                        path_filter_transmission = path_filter_transmission,
                        path_vega_spectrum = path_filter_transmission, verbose=verbose)

    if verbose:
        print('Normalizing spectrum to magnitude {:6.2f} in filter {:}'.format(magnitude, filtername))

    flux_normalized = flux_W_m2_micron * 10**(-0.4*(magnitude-rawmag))

    return flux_normalized


def example_1(wave_micron, flux_W_m2_micron):
    
    print('Beginning of example_1')
    # Synthesize magnitudes
    filterlist = ['WISE2','Johnson-U','MKO-J'] # keep as array, an array of name is expected
    pathvega = '/Users/albert/NIRISS/SOSSpipeline/sandbox/'
    pathfilter = '/Users/albert/filterSVO/'
    
    filtermag = syntMag(wave_micron,flux_W_m2_micron,filterlist,
                    path_filter_transmission=pathfilter,
                    path_vega_spectrum=pathvega)
    # take index 0 to get the magnitude of the first filter
    WISE2 = filtermag[0]
    print('WISE2 magnitude of input spectrum: {:6.2f}'.format(WISE2))
    mag = 15.0
    print('To be normalized at magnitude {:6.2f}'.format(mag))
    # Set the WISE2 magnitude to 15.0
    flux_normalized = flux_W_m2_micron * 10**(-0.4*(mag-WISE2))
    filtermag = syntMag(wave_micron,flux_normalized,filterlist,
                    path_filter_transmission=pathfilter,
                    path_vega_spectrum=pathvega)
    WISE2ok = filtermag[0]
    print('WISE2 magnitude of normalized spectrum: {:6.2f}'.format(WISE2ok))
    
    plt.plot(wave_micron,flux_normalized)
    plt.ylabel('Flambda (W/m2/micron)')
    plt.xlabel('Wavelength (micron)')
    plt.show()    
    
    print('End of example_1')
    return()

def example_2():
    
    print('Beginning of example_2')

    pathvega = '/Users/albert/NIRISS/SOSSpipeline/sandbox/'
    pathfilter = '/Users/albert/filterSVO/'

    # read some spectrum and wavelength in wave_micron, flux_W_m2_micron
    #wave_micron, flux_W_m2_micron = read_some_spectrum_not_implemented()
    # For the example, lets read Vega
    wave_micron, flux_W_m2_micron = readVega(path_vega_spectrum=pathvega)
    # change the flux by 5 magnitudes, scale by 100
    flux_W_m2_micron = flux_W_m2_micron * 100
    
    filterlist = ['Johnson-V', 'KIC-r','MKO-J','WIRCam-Ks']
    calibrationmag_KICr = 8.5

    # Get the uncalibrated magnitude through a filter
    filtermag = syntMag(wave_micron,flux_W_m2_micron,filterlist,
                               path_filter_transmission=pathfilter,
                               path_vega_spectrum=pathvega)
    print('KIC-r magnitude of uncalibrated spectrum: {:6.2f}'.format(filtermag[1]))
    # Normalize spectrum to the desired magnitude
    flux_W_m2_micron_normalized = flux_W_m2_micron * 10**(-0.4*(calibrationmag_KICr-filtermag[1]))
    # Check that it worked
    filtermag = syntMag(wave_micron,flux_W_m2_micron_normalized,filterlist,
                             path_filter_transmission=pathfilter,
                             path_vega_spectrum=pathvega)
    print('KIC-r magnitude of calibrated spectrum: {:6.2f}'.format(filtermag[1]))
    
    plt.figure(figsize=(12,6))
    plt.loglog(wave_micron,flux_W_m2_micron_normalized)
    plt.title('Normalized Vega spectrum to KIC-r = 8.5')
    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Flambda (W/m2/micron)')
    plt.show()
    
    # Now, divide by the photon energy and integrate over 25 m2 (JWST) and by 
    # the pixel dispersion (in microns per pixel: e.g. ~0.001 micron/pixel in SOSS m=1)
    #Fnu_W_m2_Hz = FlambdaToFnu(wave_micron,flux_W_m2_micron_normalized)
    h = 6.62607004e-34 # m2 kg / sec
    c = 299792458.0 #m/sec
    photon_energy = h*c/(wave_micron*1e-6) # Joule
    dispersion_micron_pixel = 0.001 # microns/pixel, could be an array matching the wave_micron
    area = 25.0
    
    # number of electrons per pixel ***per second*** is:
    nelectron = area * dispersion_micron_pixel * flux_W_m2_micron_normalized / photon_energy
    
    plt.figure(figsize=(12,6))   
    plt.loglog(wave_micron,nelectron)
    plt.title('Electron flux with JWST assuming constant dispersion of 1 nm/pixel')
    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Electron/sec/pixel')
    plt.ylim((1e-2,1e+7))
    plt.show()

    print('End of example_2')

    return(nelectron)

def FlambdaToFnu(wave,Flambda):
    #wave has to be in microns
    #Flambda has to be in W/m2/micron
    #Physical constants
    c = 299792458.0 #m/sec
    wave_m = 1.0e-6*wave
    Flambda_W_m2_m = 1.0e+6*Flambda
    #Convert Flambda to Fnu (W/M2/Hz)
    Fnu_W_m2_Hz = np.power(wave_m,2.0) * Flambda_W_m2_m / c
    #Convert to Jansky using 1 Jy = 10^26 W/m2/Hz
    #Jy = 1.0e+26 * Fnu_W_m2_Hz
    return(Fnu_W_m2_Hz)


def jansky_to_AB(jansky):

    # Convert from Jansky to Fnu 
    
    #The AB Magnitude constant
    #ABconstant = 48.594 # advocated by Arnouts
    ABconstant = 48.600
    
    # by definition:
    Fnu_W_m2_Hz = 1.0e-26 * jansky
    #Convert Fnu (metric) to Fnu (cgs) (erg/sec/cm2/Hz) using 1 erg = 1.0e-7 J and 1 m2 = 10^4 cm2
    Fnu_erg_sec_cm2_Hz = 1.0e+3 * Fnu_W_m2_Hz
    #Convert to AB magnitude using magnitude = -2.5*alog10(flux) - 48.60
    ABmag = -2.5 * np.log10(Fnu_erg_sec_cm2_Hz) - ABconstant

    return(ABmag)

def AB_to_jansky(magAB):
    
    # Convert from AB magnitude to Jansky
    
    #The AB Magnitude constant
    #ABconstant = 48.594 # advocated by Arnouts
    ABconstant = 48.600
    
    # First convert from mag AB to Fnu in cgs units: erg/s/cm2/Hz
    fnu_cgs = np.power(10,-(magAB+ABconstant)/2.5)
    # Then convert cgs to Jy (1 Jy = 10-23 erg/s/cm2/Hz)
    jansky = fnu_cgs * 1e+23
    
    return(jansky)
    

def ABmag_to_Vegamag(magAB, filterlist, path_filter_transmission=None,
            path_vega_spectrum=None,verbose=None):
    # Convert AB magnitude to Vega magnitudes. That requires a filter name.
    #
    # Handles 3 cases:
    # 1) if magAB and filterlist are both scalars, then the function also
    #    returns a scalar.
    # 2) if magAB is an array but the filterlist is a scalar, then the
    #    function returns an array of same length as magAB.
    # 3) if both magAB and filterlist are arrays, then the function returns
    #    a matrix of shape nmag x nfilter, e.g. mat[:,0] is all mag thru 
    #    one filter.
    
    if verbose:
        print('shapes of input parameters:')
        print('magAB:', np.shape(magAB))
        print('filterlist', np.shape(filterlist))
    
    #Check if a path for the Vega spectrum was passed. If not, assume some
    # local path
    if path_vega_spectrum == None:
        path_vega_spectrum = '/Users/albert/NIRISS/SOSSpipeline/syntMagCode/'
    if path_filter_transmission == None:
        path_filter_transmission = '/Users/albert/filterSVO/'

    # Initialize the filters array
    if np.size(filterlist) == 1:
        filters = np.array(np.reshape(filterlist,-1))
    else:
        filters = np.array(filterlist)

    # Initialize matrix of AB to Vega magnitude offsets
    magoffset = np.empty((np.size(magAB),np.size(filters)), dtype=np.float)
    # Initialize input magAB into a matrix spanning filters across axis=2
    magABmatrix = np.empty((np.size(magAB),np.size(filters)), dtype=np.float)
    for f in range(np.size(filters)):
        magABmatrix[:,f] = magAB

    # Read the Vega and AB spectra. Both share the same wavelength sampling
    # (that of the Vega spectrum).
    wave_Vega, Flambda_Vega = readVega(path_vega_spectrum=path_vega_spectrum)
    lba = np.copy(wave_Vega)
    wave_AB, Flambda_AB = readAB(wave_sampling=lba)
    
    # Each wavelength sample has a width, determined here: 
    dlba = sample_width(lba)

    for f in range(np.size(filters)):
        #Get the filter transmission curve for that range at same sampling
        filter_wave, filter_t, magsystem = readFilter(filters[f],wave_sampling=lba,
                                                      path_filter_transmission=path_filter_transmission)
        
        Flux_Vega = np.sum(Flambda_Vega * filter_t * dlba) / np.sum(filter_t * dlba)
        Flux_AB = np.sum(Flambda_AB * filter_t * dlba) / np.sum(filter_t * dlba)
        #magVega[f] = magAB -2.5*np.log10(Flux_AB/Flux_Vega)
        magoffset[:,f] = -2.5*np.log10(Flux_AB/Flux_Vega)
 
    # Apply offset to input AB magnitudes
    magVega = magABmatrix + magoffset
    
    # Manage output because at this point, it is a matrix (nfilter x nmag)
    if (np.size(filterlist) == 1) and (np.size(magAB) == 1):
        # a single magnitude through a single filter was requested.
        # return a scalar:
        return(magVega[0,0])
    elif (np.size(filterlist) == 1) and (np.size(magAB) > 1):
        # an array of magnitudes was passed, through a single filter.
        # return an array of magnitudes, not a matrix
        return(np.reshape(magVega[:,0],-1))
    elif (np.size(filterlist) > 1) and (np.size(magAB) == 1):
        # magAB is a scalr but filterlist is an array as input.
        # return an array of size nfilter.
        return(np.reshape(magVega[0,:],-1))
    else:
        # magnitudes and filters were both arrays as input.
        # return a matrix
        return(magVega)


def Vegamag_to_ABmag(magVega, filterlist, path_filter_transmission=None,
            path_vega_spectrum=None,verbose=None):
    
    # Convert Vega magnitude to AB magnitude.
    # refer to ABmag_to_Vegamag for explanations
    
    # Determine the AB to Vega magnitude offset for each filter.
    # Send zero thru the AB --> Vega converter (Vega mag will have lower values
    # so offset will be less than zero for most filters)
    offset = ABmag_to_Vegamag(0,filterlist, 
                              path_filter_transmission=path_filter_transmission,
                              path_vega_spectrum=path_vega_spectrum,
                              verbose=verbose)
    # Subtract (rather than add) the offsets to get ABmags
    if (np.size(magVega) > 1) and (np.size(filterlist)) > 1:
        magAB = np.zeros((np.size(magVega),np.size(filterlist)))
        for f in range(np.size(filterlist)):
            magAB[:,f] = magVega - offset[f]
    else:
        magAB = magVega - offset
        
    return(magAB)


def sample_width(lba):
    # Given an array of wavelength, not necessarily sampled equally spaced
    # and BUT necessarily sorted, return the wavelength width spanned by each
    # sample.
    
    # Find the indices of sorted array of wavelengths
    indsort = np.argsort(lba)
    
    if np.array_equal(lba, lba[indsort]) is False:
        print('Error. The input array needs to be sorted before entering this function. Stop.')
        stop
        
    # Devise the width of each wavelength sample   
    dlba = lba*0.0
    dlba[0:-1] = lba[1:]-lba[0:-1]
    # Make the last index the same as previous last
    dlba[-1] = dlba[-2]*1.0   
    
    return(dlba)


def syntMag(lba,Flba,filterlist,path_filter_transmission=None,
            path_vega_spectrum=None,verbose=None):
    # Computes the synthetic magnitude of a spectrum through an input list of filters
    
    # Make sure that the input spectrum has its wavelengths sorted.
    indsorted = np.argsort(lba)
    if np.array_equal(lba,lba[indsorted]) is False:
        print('Input spectrum to syntMag has its wavelengths not sorted in increasing order.')
        stop

    #Check if a path for the Vega spectrum was passed. If not, assume some
    # local path
    if path_vega_spectrum == None:
        path_vega_spectrum = '/Users/albert/NIRISS/SOSSpipeline/syntMagCode/'
    if path_filter_transmission == None:
        path_filter_transmission = '/Users/albert/filterSVO/'
    
    # Initialize array of output magnitudes
    mag = np.arange(np.size(filterlist), dtype=np.float)

    # Read the Vega and AB spectra first, so it is done only once
    wave_Vega, Flambda_Vega = readVega(wave_sampling=lba, path_vega_spectrum=path_vega_spectrum)
    wave_AB, Flambda_AB = readAB(wave_sampling=lba)

    # Each wavelength sample has a width, determined here: 
    dlba = sample_width(lba)

    for f in range(np.size(filterlist)):
        #Get the filter transmission curve for that range at same sampling
        filter_wave, filter_t, magsystem = readFilter(filterlist[f],wave_sampling=lba,
                                                      path_filter_transmission=path_filter_transmission)
        if magsystem == 'Vega':
            #Do the Vega spectrum
            #Energy_Vega = np.sum(Flambda_Vega * filter_t) / np.sum(filter_t)
            #Energy_filter = np.sum(Flba * filter_t) / np.sum(filter_t)
            #mag[f] = -2.5*np.log10(Energy_filter/Energy_Vega)
            Flux_Vega = np.sum(Flambda_Vega * filter_t * dlba) / np.sum(filter_t * dlba)
            Flux_filter = np.sum(Flba * filter_t * dlba) / np.sum(filter_t * dlba)
            mag[f] = -2.5*np.log10(Flux_filter/Flux_Vega)
        if magsystem == 'AB':
            #Repeat with the AB spectrum
            #Energy_AB = np.sum(Flambda_AB * filter_t) / np.sum(filter_t)
            #Energy_filter = np.sum(Flba * filter_t) / np.sum(filter_t)
            #mag[f] = -2.5*np.log10(Energy_filter/Energy_AB)
            Flux_AB = np.sum(Flambda_AB * filter_t * dlba) / np.sum(filter_t * dlba)
            Flux_filter = np.sum(Flba * filter_t * dlba) / np.sum(filter_t * dlba)
            mag[f] = -2.5*np.log10(Flux_filter/Flux_AB)

    if np.size(mag) == 1:
        return(mag[0])
    else:
        return(mag)




def syntNphoton(lba, Flba, filterlist, path_filter_transmission=None,
                path_vega_spectrum=None):
    
    # Computes the number of photons/sec/m2 produced by a spectrum when 
    # integrated in a filter band pass.
    #
    # INPUTS:
    # lba : wavelength (microns)
    # Flba : Flux in Flambda units (energy/time/area/wavelength)
    # filterlist : an array of filter shortnames (to do 1 or more filters)
    #
    # OPTIONAL INPUTS:
    # path_filter_transmission : The path of the directory where the filter
    #                            transmission curves can be found.
    # path_vega_spectrum : The path of the directory where the Vega spectrum
    #                      can be found.

    # physical constants
    c = 299792458.0 # m/sec
    h = 6.62607004e-34 # m2 kg / sec

    # Check if a path to the filter transmission files was passed. 
    # If not, assume some local path
    #if path_vega_spectrum == None:
    #    path_vega_spectrum = '/Users/albert/NIRISS/SOSSpipeline/syntMagCode/'
    if path_filter_transmission == None:
        path_filter_transmission = '/Users/albert/filterSVO/'        
    
    # Devise the width of each wavelength sample (in microns)     
    dlba = sample_width(lba)
    
    # Initialize array of output magnitudes
    nphot = np.arange(np.size(filterlist), dtype=np.float)

    # the photon energy is (need to convert lba from microns to meters)
    Ephot = h * c / (lba * 1e-6)

    for f in range(np.size(filterlist)):
        #Get the filter transmission curve for that range at same sampling
        filter_wave, filter_t, magsystem = readFilter(filterlist[f],wave_sampling=lba,
                                                      path_filter_transmission=path_filter_transmission)
        # Ephoton = h * c / lambda
        # Nphoton = deltalambda * Flambda / Ephoton = 
        #         = deltalambda * lambda * Flambda / (h * c)
        # where deltalambda is the width of each wavelength sample in microns

        # The energy contained in one spectrum sample, after correcting for 
        # transmission, is (DO NOT convert dlba from microns to meters because
        # the spectrum is per micron).
        Espec = dlba * Flba * filter_t
        
        # The number of photons is the sum over all samples of the spectrum
        # energy divided by a photon energy.
        nphot[f] = np.sum(Espec/Ephot)
        
    return(nphot)

def readVega(wave_sampling=None,path_vega_spectrum=None):
    if path_vega_spectrum != None:
        vegafile = path_vega_spectrum+'VegaLCB.sed'
    else:
        vegafile = '/Users/albert/filterSVO/VegaLCB.sed'
    tab = ascii.read(vegafile, names=['wave_nm', 'Flambda'],data_start=52, 
                     format='fixed_width', delimiter=' ',col_starts=(0, 10))
    #Lejeune spectrum is in angstrom, Flambda (erg/s/cm2/angstrom)
    l = tab['wave_nm']  # First column
    f = tab['Flambda']  # Second column
    #Make sure that wavelengths are sorted
    ind = np.argsort(l)
    l = l[ind]
    f = f[ind]
    # Convert to microns and W/m2/um
    l_micron = l / 10000.
    Flambda_uncal = f * 1.0e-7 * 1.0e+4 * 10000.0    #to W/m2/um

    #Convert read out Vega spectrum to Flambda - NO NO NO, the Lejeune 
    # spectrum is in Flambda after all
    # Flambda_uncal [W/m2/um] 
    # Flambda_uncal = 1e-6 * K.c * Fnu_uncal / np.power(l_micron*1e-6,2) #or equivalently:
    # Flambda_uncal = 1e+6 * K.c * Fnu_uncal / np.power(l_micron,2)
    Flambda_anchor_spectrum = np.interp([0.5556], l_micron, Flambda_uncal)

    # The reference anchor for Vega in Flambda is given by Fukugita 1995
    # for Vega of 3.44+/-0.05 e-9 erg/cm2/s/angstrom at 5556 ang 
    # (Fukugita 1995)
    Flambda_anchor_Fukugita = 3.44e-9 * 1.0e-7 * 1.0e+4 * 10000.0  #to W/m2/um

    #print('Flambda_anchor_spectrum =', Flambda_anchor_spectrum)
    #print('Flambda_anchor_Fukugita =', Flambda_anchor_Fukugita)

    # The Lejeune spectrum is fine, nothing to do
    Flambda_calibrated = Flambda_uncal * 1.0

    # double check with pysynphot (it matches exactly)
    #vega_file = os.path.join(os.environ['PYSYN_CDBS'], 'calspec', 'alpha_lyr_stis_005.fits')
    #vega = S.FileSpectrum(vega_file)
    #or
    #sp = S.Vega
    #print(np.array(sp.flux).size)
    #sp.convert('um')
    #sp.convert('flam')    #erg/s/cm2/ang --> X10 to get W/m2/um
    #lambda_pysynphot = np.array(sp.wave)
    #Flambda_pysynphot = np.array(sp.flux*10.)

    #fig = plt.figure(figsize=(15, 6))
    #plt.loglog(lambda_pysynphot, Flambda_pysynphot)
    #plt.loglog(l_micron, Flambda_calibrated)
    #plt.xlim(0.3,3.0)
    #plt.grid(True)
    #plt.show()

    if (wave_sampling is None):
        return(l_micron,Flambda_calibrated)
    else:
        Flambda_calibrated_resamp = np.interp(wave_sampling, l_micron, Flambda_calibrated)
        # make sure that extremes are set to zero
        ind = (np.where((wave_sampling < np.min(l_micron)) | (wave_sampling > np.max(l_micron))))[0]
        nind = np.size(ind)
        if nind >= 1:
            Flambda_calibrated_resamp[ind] = 0.0
        return(wave_sampling,Flambda_calibrated_resamp)

    return()

def readAB(wave_sampling=None):

    #wave_sampling can be passed as input. It should be in units of microns
    l_micron = np.arange(0.1,100,0.01)
    c = 299792458.0
    #Convert AB magnitude to Fnu
    Fnu_erg_sec_cm2_Hz = np.power(10.0,-0.4*48.594)
    Fnu_W_m2_Hz = 1.0e-3 * Fnu_erg_sec_cm2_Hz
    #Convert Fnu to Flambda (W/m2/m)
    Flambda_W_m2_m = c * Fnu_W_m2_Hz / np.power((l_micron * 1e-6),2.0)
    Flambda_W_m2_um = Flambda_W_m2_m * 1e-6
    Flambda_calibrated = Flambda_W_m2_um * 1.0

    if (wave_sampling is None):
        return(l_micron,Flambda_calibrated)
    else:
        Flambda_calibrated_resamp = np.interp(wave_sampling, l_micron, 
                                              Flambda_calibrated)
        # make sure that extremes are set to zero
        ind = (np.where((wave_sampling < np.min(l_micron)) | (wave_sampling > np.max(l_micron))))[0]
        nind = np.size(ind)
        if nind >= 1:
            Flambda_calibrated_resamp[ind] = 0.0
        return(wave_sampling,Flambda_calibrated_resamp)

    return()


def readFilter(requestedFilterName,path_filter_transmission=None,
               wave_sampling=None,keepPeakAbsolute=True,returnWidth=None,
               verbose = None):

    # What is missing and would be nice is if there was a function to
    # list the available supported filters.

    if path_filter_transmission == None:
        path = '/Users/albert/filterSVO/'
    else:
        path = path_filter_transmission+'/'
    if verbose is True: print('path to filter transmission curve: {:}'.format(path))
    
    # dictionnary for filter definition
    dico = []
    dico.append(('TYCHO_TYCHO.B_MvB.dat.txt','Vega',['Tycho-B']))
    dico.append(('TYCHO_TYCHO.V_MvB.dat.txt','Vega',['Tycho-V']))
    dico.append(('SLOAN_SDSS.u.dat.txt','AB',['SDSS-u','KIC-u','u']))
    dico.append(('SLOAN_SDSS.g.dat.txt','AB',['SDSS-g','KIC-g','g']))
    dico.append(('SLOAN_SDSS.r.dat.txt','AB',['SDSS-r','KIC-r','r']))
    dico.append(('SLOAN_SDSS.i.dat.txt','AB',['SDSS-i','KIC-i','i']))
    dico.append(('SLOAN_SDSS.z.dat.txt','AB',['SDSS-z','KIC-z','z','Pesto-z']))
    dico.append(('Kepler_Kepler.K.dat.txt','AB',['Kepler']))
    dico.append(('Generic_Johnson.U.dat.txt','Vega',['Johnson-U','U']))
    dico.append(('Generic_Johnson.B.dat.txt','Vega',['Johnson-B','B']))
    dico.append(('Generic_Johnson.V.dat.txt','Vega',['Johnson-V','V']))
    dico.append(('Generic_Johnson.R.dat.txt','Vega',['Johnson-R','R']))
    dico.append(('Generic_Johnson.I.dat.txt','Vega',['Johnson-I','I']))
    dico.append(('Generic_Johnson.J.dat.txt','Vega',['Johnson-J']))
    dico.append(('PAN-STARRS_PS1.g.dat.txt','AB',['PS-g']))
    dico.append(('PAN-STARRS_PS1.r.dat.txt','AB',['PS-r']))
    dico.append(('PAN-STARRS_PS1.i.dat.txt','AB',['PS-i']))
    dico.append(('PAN-STARRS_PS1.z.dat.txt','AB',['PS-z']))
    dico.append(('PAN-STARRS_PS1.y.dat.txt','AB',['PS-y']))
    dico.append(('2MASS_2MASS.J.dat.txt','Vega',['2MASS-J','2M-J','2MJ']))
    dico.append(('2MASS_2MASS.H.dat.txt','Vega',['2MASS-H','2M-H','2MH']))
    dico.append(('2MASS_2MASS.Ks.dat.txt','Vega',['2MASS-Ks','2MASS-K','2M-K','2MK']))
    dico.append(('CFHT_Wircam.J.dat.txt','Vega',['WIRCam-J','MKO_J','MKO-J','J']))
    dico.append(('CFHT_Wircam.H.dat.txt','Vega',['WIRCam-H','MKO_H','MKO-H','H']))
    dico.append(('CFHT_Wircam.Ks.dat.txt','Vega',['WIRCam-Ks','MKO_Ks','MKO-Ks','K']))
    dico.append(('CFHT_Wircam.W.dat','Vega',['WIRCam-W']))
    dico.append(('UKIRT_UKIDSS.Z.dat.txt','Vega',['UKIDSS-Z']))
    dico.append(('UKIRT_UKIDSS.Y.dat.txt','Vega',['UKIDSS-Y']))
    dico.append(('UKIRT_UKIDSS.J.dat.txt','Vega',['UKIDSS-J']))
    dico.append(('UKIRT_UKIDSS.H.dat.txt','Vega',['UKIDSS-H']))
    dico.append(('UKIRT_UKIDSS.K.dat.txt','Vega',['UKIDSS-K']))
    dico.append(('HST_NICMOS1.F110W.dat.txt','Vega',['NICMOS1.F110W']))
    dico.append(('HST_NICMOS1.F190N.dat.txt','Vega',['NICMOS1.F190N']))
    dico.append(('HST_NICMOS1.F170M.dat.txt','Vega',['NICMOS1.F170M']))
    # JWST - NIRISS
    dico.append(('JWST_NIRISS.F277W.dat','Vega',['NIRISS.F277W']))
    dico.append(('JWST_NIRISS.F380M.dat.txt','Vega',['NIRISS.F380M']))
    dico.append(('JWST_NIRISS.F430M.dat.txt','Vega',['NIRISS.F430M']))
    dico.append(('JWST_NIRISS.F480M.dat.txt','Vega',['NIRISS.F480M']))
    # JWST - NIRCam
    dico.append(('JWST_NIRCam.F070W.dat','Vega',['NIRCam.F070W']))
    dico.append(('JWST_NIRCam.F090W.dat','Vega',['NIRCam.F090W']))
    dico.append(('JWST_NIRCam.F115W.dat','Vega',['NIRCam.F115W']))
    dico.append(('JWST_NIRCam.F140M.dat','Vega',['NIRCam.F140M']))
    dico.append(('JWST_NIRCam.F150W.dat','Vega',['NIRCam.F150W']))
    dico.append(('JWST_NIRCam.F150W2.dat','Vega',['NIRCam.F150W2']))
    dico.append(('JWST_NIRCam.F162M.dat','Vega',['NIRCam.F162M']))
    dico.append(('JWST_NIRCam.F164N.dat','Vega',['NIRCam.F164N']))
    dico.append(('JWST_NIRCam.F182M.dat','Vega',['NIRCam.F182M']))
    dico.append(('JWST_NIRCam.F187N.dat','Vega',['NIRCam.F187N']))
    dico.append(('JWST_NIRCam.F200W.dat','Vega',['NIRCam.F200W']))
    dico.append(('JWST_NIRCam.F210M.dat','Vega',['NIRCam.F210M']))
    dico.append(('JWST_NIRCam.F212N.dat','Vega',['NIRCam.F212N']))
    dico.append(('JWST_NIRCam.F250M.dat','Vega',['NIRCam.F250M']))
    dico.append(('JWST_NIRCam.F277W.dat','Vega',['NIRCam.F277W']))
    dico.append(('JWST_NIRCam.F300M.dat','Vega',['NIRCam.F200M']))
    dico.append(('JWST_NIRCam.F322W2.dat','Vega',['NIRCam.F322W2']))
    dico.append(('JWST_NIRCam.F323N.dat','Vega',['NIRCam.F323N']))
    dico.append(('JWST_NIRCam.F335M.dat','Vega',['NIRCam.F335M']))
    dico.append(('JWST_NIRCam.F356W.dat','Vega',['NIRCam.F356W']))
    dico.append(('JWST_NIRCam.F360M.dat','Vega',['NIRCam.F360M']))
    dico.append(('JWST_NIRCam.F405N.dat','Vega',['NIRCam.F405N']))
    dico.append(('JWST_NIRCam.F410M.dat','Vega',['NIRCam.F410M']))
    dico.append(('JWST_NIRCam.F430M.dat','Vega',['NIRCam.F430M']))
    dico.append(('JWST_NIRCam.F444W.dat','Vega',['NIRCam.F444W']))
    dico.append(('JWST_NIRCam.F460M.dat','Vega',['NIRCam.F460M']))
    dico.append(('JWST_NIRCam.F466N.dat','Vega',['NIRCam.F466N']))
    dico.append(('JWST_NIRCam.F470N.dat','Vega',['NIRCam.F470N']))
    dico.append(('JWST_NIRCam.F480M.dat','Vega',['NIRCam.F480M']))
    # JWST - MIRI 
    dico.append(('JWST_MIRI.F560W.dat','Vega',['MIRI.F560W']))
    dico.append(('JWST_MIRI.F770W.dat','Vega',['MIRI.F770W']))
    dico.append(('JWST_MIRI.F1000W.dat','Vega',['MIRI.F1000W']))
    dico.append(('JWST_MIRI.F1130W.dat','Vega',['MIRI.F1130W']))
    dico.append(('JWST_MIRI.F1280W.dat','Vega',['MIRI.F1280W']))
    dico.append(('JWST_MIRI.F1500W.dat','Vega',['MIRI.F1500W']))
    dico.append(('JWST_MIRI.F1800W.dat','Vega',['MIRI.F1800W']))
    dico.append(('JWST_MIRI.F2100W.dat','Vega',['MIRI.F2100W']))
    dico.append(('JWST_MIRI.F2550W.dat','Vega',['MIRI.F2550W']))
    # Gaia DR2    
    dico.append(('GAIA_GAIA2.G.dat','Vega',['GAIA.DR2.G']))
    dico.append(('GAIA_GAIA2.Grp.dat','Vega',['GAIA.DR2.Grp']))
    dico.append(('GAIA_GAIA2.Gbp.dat','Vega',['GAIA.DR2.Gbp']))
    dico.append(('TESS_TESS.Red.dat','Vega',['TESS']))
    dico.append(('WISE_WISE.W1.dat.txt','Vega',['WISE1','W1']))
    dico.append(('WISE_WISE.W2.dat.txt','Vega',['WISE2','W2']))
    dico.append(('WISE_WISE.W3.dat.txt','Vega',['WISE3','W3']))
    dico.append(('WISE_WISE.W4.dat.txt','Vega',['WISE4','W4']))
    dico.append(('Spitzer_IRAC.I1.dat.txt','Vega',['IRAC1']))
    dico.append(('Spitzer_IRAC.I2.dat.txt','Vega',['IRAC2']))
    dico.append(('Spitzer_IRAC.I3.dat.txt','Vega',['IRAC3']))
    dico.append(('Spitzer_IRAC.I4.dat.txt','Vega',['IRAC4']))
    dico.append(('Paranal_NACO.J.dat.txt','Vega',['NACO-J']))
    dico.append(('Paranal_NACO.Ks.dat.txt','Vega',['NACO-K']))
    dico.append(('Gemini_Flamingos2.J.dat.txt','Vega',['FLAMINGOS2-J']))
    dico.append(('GMOS-south-z_and_detector.txt','Vega',['GMOS-south-z']))
    dico.append(('Gemini_NIRI.CH4short-G0228.dat.txt','Vega',['NIRI-CH4short']))
    #special case, need to convert the file
    #elif filterName == 'Espadons-guider':
    #    #print('Reading filter curve for the Espadons-guider')
    #    fileName = 'Espadons-guider.txt'
    #    path = './FilterSVO/'
    #    name = path+fileName
    #    tab = ascii.read(name, names=['wave', 'transmission'],data_start=0, delimiter=' ')
    #    l = tab['wave']  # First column
    #    t = tab['transmission']  # Second column
    #    wave_micron = l / 1000.    # THIS ONE WAS GENERATED BY ME, not SVO.
    if verbose is True: print('Dictionnary of filter names/system:',dico)

    
    # Check which filter was requested
    nfilter = len(dico) # number of filters available
    found = False # flag to True when the correct match between requested and current is reached
    n = 0 # iterate over the filters using n
    # Iterate over filter names
    while (found is False) and (n < nfilter):
        # For each filter name, a list of short names or synonims exists, loop over
        nsynonyms = np.size(dico[n][2])
        s = 0
        while (found is False) and (s < nsynonyms):
            # Check if current filter short name is the requested short name.
            if verbose is True: print('Is {:} == {:} True? {:}'.format(requestedFilterName, 
                                                                       dico[n][2][s], 
                                                                       requestedFilterName == dico[n][2][s]))
            if requestedFilterName == dico[n][2][s]:
                found = True
                filename = path+dico[n][0]
                if verbose: print('Reading file named {:}'.format(filename))
                tab = ascii.read(filename, names=['wave', 'transmission'],data_start=0, delimiter=' ')
                l = tab['wave']  # First column
                t = tab['transmission']  # Second column
                wave_micron = l / 10000.
                magsystem = dico[n][1]
            s = s+1
        # no short name for the current filter matched the requested name
        n = n+1
       
    if found is False:
        print(' requestedFilterName = {:} not in list of existing filters. Abort.'.format(requestedFilterName))
        print(' Download the filter transmission curve from http://svo2.cab.inta-csic.es/theory/fps/')
        print(' And add it in the transmission filter directory. Then edit the python code to include it.')
        stop

    # pad each side of the profile by T values at zero so that eventual
    # out of range estimates are done with zero rather than non-zero 
    # transmission.
    t = np.array(list([0.0,0.0,0.0])+list(t)+list([0.0,0.0,0.0]))
    wavemin, wavemax = np.min(wave_micron), np.max(wave_micron)
    wave_low = np.linspace(wavemin-0.0003,wavemin-0.0001,3)
    wave_high = np.linspace(wavemax+0.0001,wavemax+0.0003,3)
    wave_micron = np.array(list(wave_low)+list(wave_micron)+list(wave_high))

    #replace negative values by 0
    t[t <= 0] = 0.0
 
    #Make sure the wavelengths are sorted!!!

    #fig = plt.figure(figsize=(15,6))
    #plt.plot(wave_micron,t)
    #plt.show()

    # Scale the peak of the transmission curve to 1
    if keepPeakAbsolute != True:
        t = t/np.max(t)

    if (wave_sampling is None):
        transmission = np.copy(t)
    else:
        transmission = np.interp(wave_sampling, wave_micron, t)
        # make sure that extremes are set to zero
        ind = (np.where((wave_sampling < np.min(wave_micron)) | (wave_sampling > np.max(wave_micron))))[0]
        nind = np.size(ind)
        if nind >= 1:
            transmission[ind] = 0.0
        wave_micron = wave_sampling
        #plt.plot(wave_sampling,transmission,color='red')
        #for i in transmission:
        #    print(i)
    
    #plt.show()

    # define an array describing the width of each waveleneght sampling
    wave_delta = wave_micron[1:] - wave_micron[0:-1]
    wave_delta = np.append(wave_delta[0],wave_delta)


    if returnWidth is True:
        width = np.sum(transmission*wave_delta)/np.max(transmission)
        return(wave_micron,transmission, magsystem, width)
    else:
        return(wave_micron,transmission, magsystem)


def throughput_withF277W(response, path_filter_transmission):
    """
    Convolves the GR700XD + CLEAR throughput by the F277W filter to o get
    the GR700XD + F277W throughput.
    :param response: GR700XD + CLEAR input response
    :param path_filter_transmission: The path to the directory where filter curves are stored
    :return: GR700XD + F277W output throughput
    """
    # Initialize a copy of the input response
    import copy
    response_f277 = copy.deepcopy(response)

    # Get the F277W filter transmission curve
    response_micron = response.wv / 10000 # angstrom to micron
    filter_wave, filter_t, magsystem = readFilter('NIRISS.F277W',
                                                  wave_sampling=response_micron,
                                                  path_filter_transmission=path_filter_transmission,
                                                  verbose=True)
    #plt.figure(figsize=(10, 6))
    #plt.plot(filter_wave, filter_t)
    #plt.xlabel('Wavelength [$\mu$m')
    #plt.show()

    for m in response.response_order:
        # Multiply the CLEAR response by the F277W transmission curve
        response_f277.response[m] = response.response[m] * filter_t

    return response_f277