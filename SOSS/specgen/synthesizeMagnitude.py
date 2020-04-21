#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:06:06 2020

@author: albert
"""

from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt



def example_1(wave_micron, flux_W_m2_micron):
    
    print('Beginning of example_1')
    # Synthesize magnitudes
    filterlist = ['WISE2','Johnson-U','MKO-J'] # keep as array, an array of name is expected
    pathvega = '/Users/albert/NIRISS/SOSSpipeline/sandbox/'
    pathfilter = '/Users/albert/NIRISS/SOSSpipeline/sandbox/filterSVO/'
    
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
    pathfilter = '/Users/albert/NIRISS/SOSSpipeline/sandbox/filterSVO/'

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


def syntMag(lba,Flba,filterlist,path_filter_transmission=None,path_vega_spectrum=None):
    # Computes the synthetic magnitude of a spectrum through an input list of filters

    # Initialize array of output magnitudes
    mag = np.arange(np.size(filterlist), dtype=np.float)

    # Read the Vega and AB spectra first, so it is done only once
    wave_Vega, Flambda_Vega = readVega(wave_sampling=lba, path_vega_spectrum=path_vega_spectrum)
    wave_AB, Flambda_AB = readAB(wave_sampling=lba)

    for f in range(np.size(filterlist)):
        #Get the filter transmission curve for that range at same sampling
        filter_wave, filter_t, magsystem = readFilter(filterlist[f],wave_sampling=lba,
                                                      path_filter_transmission=path_filter_transmission)
        if magsystem == 'Vega':
            #Do the Vega spectrum
            Energy_Vega = np.sum(Flambda_Vega * filter_t) / np.sum(filter_t)
            Energy_filter = np.sum(Flba * filter_t) / np.sum(filter_t)
            mag[f] = -2.5*np.log10(Energy_filter/Energy_Vega)
        if magsystem == 'AB':
            #Repeat with the AB spectrum
            Energy_AB = np.sum(Flambda_AB * filter_t) / np.sum(filter_t)
            Energy_filter = np.sum(Flba * filter_t) / np.sum(filter_t)
            mag[f] = -2.5*np.log10(Energy_filter/Energy_AB)

    return(mag)


def readVega(wave_sampling=None,path_vega_spectrum=None):
    if path_vega_spectrum != None:
        vegafile = path_vega_spectrum+'VegaLCB.sed'
    else:
        vegafile = './VegaLCB.sed'
    tab = ascii.read(vegafile, names=['wave_nm', 'Flambda'],data_start=52, format='fixed_width', delimiter=' ',col_starts=(0, 10))
    #Lejeune spectrum is in angstrom, Flambda (erg/s/cm2/angstrom)
    l = tab['wave_nm']  # First column
    f = tab['Flambda']  # Second column
    l_micron = l / 10000.
    Flambda_uncal = f * 1.0e-7 * 1.0e+4 * 10000.0    #to W/m2/um

    #Convert read out Vega spectrum to Flambda - NO NO NO, the Lejeune spectrum is in Flambda after all
    # Flambda_uncal [W/m2/um] 
    # Flambda_uncal = 1e-6 * K.c * Fnu_uncal / np.power(l_micron*1e-6,2) #or equivalently:
    # Flambda_uncal = 1e+6 * K.c * Fnu_uncal / np.power(l_micron,2)
    Flambda_anchor_spectrum = np.interp([0.5556], l_micron, Flambda_uncal)

    # The reference anchor for Vega in Flambda is given by Fukugita 1995
    # for Vega of 3.44+/-0.05 e-9 erg/cm2/s/angstrom at 5556 ang (Fukugita 1995)
    Flambda_anchor_Fukugita = 3.44e-9 * 1.0e-7 * 1.0e+4 * 10000.0    #to W/m2/um

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
        Flambda_calibrated_resamp = np.interp(wave_sampling, l_micron, Flambda_calibrated)
        # make sure that extremes are set to zero
        ind = (np.where((wave_sampling < np.min(l_micron)) | (wave_sampling > np.max(l_micron))))[0]
        nind = np.size(ind)
        if nind >= 1:
            Flambda_calibrated_resamp[ind] = 0.0
        return(wave_sampling,Flambda_calibrated_resamp)

    return()


def readFilter(requestedFilterName,path_filter_transmission=None,
               wave_sampling=None,keepPeakAbsolute=None,returnWidth=None,
               verbose = None):

    # What is missing and would be nice is if there was a function to
    # list the available supported filters.
    
    if path_filter_transmission == None:
        path = './FilterSVO/'
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
    dico.append(('CFHT_Wircam.W.dat.txt','Vega',['WIRCam-W']))
    dico.append(('UKIRT_UKIDSS.Z.dat.txt','Vega',['UKIDSS-Z']))
    dico.append(('UKIRT_UKIDSS.Y.dat.txt','Vega',['UKIDSS-Y']))
    dico.append(('UKIRT_UKIDSS.J.dat.txt','Vega',['UKIDSS-J']))
    dico.append(('UKIRT_UKIDSS.H.dat.txt','Vega',['UKIDSS-H']))
    dico.append(('UKIRT_UKIDSS.K.dat.txt','Vega',['UKIDSS-K']))
    dico.append(('HST_NICMOS1.F110W.dat.txt','Vega',['NICMOS1.F110W']))
    dico.append(('HST_NICMOS1.F190N.dat.txt','Vega',['NICMOS1.F190N']))
    dico.append(('HST_NICMOS1.F170M.dat.txt','Vega',['NICMOS1.F170M']))
    dico.append(('JWST_NIRISS.F380M.dat.txt','Vega',['NIRISS.F380M']))
    dico.append(('JWST_NIRISS.F430M.dat.txt','Vega',['NIRISS.F430M']))
    dico.append(('JWST_NIRISS.F480M.dat.txt','Vega',['NIRISS.F480M']))
    dico.append(('JWST_NIRCam.F210M.dat','Vega',['NIRCam.F210M']))
    dico.append(('JWST_NIRCam.F360M.dat','Vega',['NIRCam.F360M']))
    dico.append(('JWST_NIRCam.F410M.dat','Vega',['NIRCam.F410M']))
    dico.append(('JWST_NIRCam.F430M.dat','Vega',['NIRCam.F430M']))
    dico.append(('JWST_NIRCam.F460M.dat','Vega',['NIRCam.F460M']))
    dico.append(('JWST_NIRCam.F480M.dat','Vega',['NIRCam.F480M']))
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

    # Scale the peak of the transmission curve to 1
    if keepPeakAbsolute != True:
        t = t/np.max(t)

    if (wave_sampling is None):
        transmission = t
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
        return(wave_micron,transmission,magsystem,width)
    else:
        return(wave_micron,transmission,magsystem)