


# This script is to generate a tilt table reference file specifically
# for the simulations using the webbPSF monochromatic PSFs.

import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt

import scipy.signal as signal

from scipy.optimize import curve_fit

import sys
sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/SOSS/')
sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/SOSS/trace/')
sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/SOSS/specgen/')
sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/SOSS/detector/')
sys.path.insert(0, '/genesis/jwst/jwst-ref-soss/fortran_lib/')

def CCF_gaus(x, c, a, x0, sigma):
    return(c+a*np.exp(-(x-x0)**2/(2*sigma**2)))

def CCF_parabola(x,xo,yo,sigma):
    return(yo -((x-xo)/sigma)**2)

def fitCCF(ref, cur, fitfunc='gauss', fitradius=3, makeplot=False):
    if fitradius == False:
        maxneighbors = False
    else:
        # fitradius needs to be an integer.
        # It represents the number of data points in addition to the peak
        # value, on both sides of the peak. So fitradius=2 will fit 5 points
        maxneighbors = True
        npix = fitradius

    # fitfunc = 'parabola'
    # fitfunc = 'gauss'

    # Perform median filtering through the data to remove low-frequencies
    if False:
        ref = ref - signal.medfilt(ref, 25)
        cur = cur - signal.medfilt(cur, 25)

    # Perform the cross correlation function
    delta = np.linspace(-100, 100, 201, dtype='int')
    # print('delta = ',delta)
    delta_fine = np.linspace(-100, 100, 200)
    CCF = np.zeros(len(delta))
    for i in range(len(delta)):
        # print(i, 'delta=', delta)
        # a = np.correlate(cur,np.roll(ref,delta[i]))
        # print('CCF=',a)
        # if len(a) == 1:
        #    CCF[i] = a
        # else:
        #    CCF[i] = np.nan
        CCF[i] = np.correlate(cur, np.roll(ref, delta[i]), mode='valid')

    # Perform a fit of the CCF peak position
    # Either limit the fit to the data points very close to the peak
    if maxneighbors == True:
        # Select data points near the CCF peak only to fit
        indmax = np.where(CCF == np.max(CCF))[0]
        indfit = np.linspace(indmax[0] - npix, indmax[0] + npix, npix * 2 + 1, dtype='int')
        # Fit model function, either a gaussian or a parabola
        if fitfunc == 'parabola':
            popt, pcov = curve_fit(CCF_parabola, delta[indfit], CCF[indfit], p0=[0.0, np.max(CCF[indfit]), 4e-4])
            dx = popt[0]
        else:
            popt, pcov = curve_fit(CCF_gaus, delta[indfit], CCF[indfit],
                                   p0=[np.min(CCF), np.max(CCF) - np.min(CCF), 0, 1])
            dx = popt[2]
    # Or perform the fit on all available data points
    # (not recommended because fo CCF asymmetry)
    else:
        popt, pcov = curve_fit(CCF_gaus, delta, CCF, p0=[np.min(CCF), np.max(CCF) - np.min(CCF), 0, 1])
        dx = popt[2]

    if makeplot is True:
        print('popt=', popt)
        # print('pcov=',pcov)
        fig = plt.figure(figsize=(15, 4))
        plt.plot(delta, CCF, marker='s')
        plt.scatter(delta[indfit], CCF[indfit], marker='.', color='red', zorder=3)
        if fitfunc == 'parabola':
            plt.plot(delta_fine, CCF_parabola(delta_fine, popt[0], popt[1], popt[2]), color='orange')
            # plt.xlim((np.min(dy_fine),np.max(dy_fine)))
            # plt.ylim((np.min(CCF_parabola(dy_fine,popt[0],popt[1],popt[2])),np.max(CCF)))
        else:
            plt.plot(delta_fine, CCF_gaus(delta_fine, popt[0], popt[1], popt[2], popt[3]), color='orange')
            # plt.xlim((np.min(dy_fine),np.max(dy_fine)))
            # plt.ylim((np.min(CCF_gaus(dy_fine,popt[0],popt[1],popt[2],popt[3])),np.max(CCF)))
        # plt.xlim((-npix*2,npix*2))

    return (dx)


WORKING_DIR = '/genesis/jwst/jwst-user-soss/loic_review/'

PSF_DIR = '/genesis/jwst/jwst-ref-soss/monochromatic_PSFs/'

wave = np.linspace(0.5,5.2,48)
tilt = np.copy(wave)*0.0
print(wave)

col = np.linspace(511,762,252) # pixel column number
row = np.arange(1280) # the pixel row number
com = np.copy(col)*0.0 # center of mass along the y axis
pek = np.copy(col)*0.0 # peak value along y
ccf = np.copy(col)*0.0 # CCF gauss fit along y

for i in range(len(wave)):
#for i in range(1):
    # Open each PSF fits file
    psfname = PSF_DIR + 'SOSS_os10_128x128_{:8.6F}.fits'.format(wave[i])
    a = fits.open(psfname)
    psf = a[0].data

    if False:
        refslice = np.sum(psf[640-100:640+100, int(col[0])-5:int(col[0])+5],axis=1)
        # For each column between x=511 and x=762, find the centroid position
        for j in range(len(col)):
            # center of mass
            x = int(np.copy(col[j]))
            com[j] = np.sum(row * psf[:,x]) / np.sum(psf[:,x])
            # peak value
            pek[j] = row[np.argmax(psf[:,x])]
            # gauss CCF fit
            curslice = np.sum(psf[640-100:640+100,x-5:x+5],axis=1)
            ccf[j] = fitCCF(refslice, curslice, fitfunc='gauss', fitradius=40, makeplot=False)
            refslice = np.sum(psf[640-100:640+100,x-5:x+5],axis=1)
        tilt[i] = np.rad2deg(np.arctan(np.median(ccf)))
    # Two peaks CCF
    if True:
        leftslice = np.sum(psf[640-100:640+100,518:604],axis=1)
        righslice = np.sum(psf[640-100:640+100,697:765],axis=1)
        # The number of columns between both slices
        dx = np.mean([697,765]) - np.mean([604,518])
        #print(dx)
        #plt.figure()
        #plt.plot(leftslice)
        #plt.plot(righslice)
        #plt.show()
        ccf2 = fitCCF(leftslice, righslice, fitfunc='gauss', fitradius=40, makeplot=False)
        tilt[i] = np.rad2deg(np.arctan(ccf2/dx))

    print(wave[i],tilt[i])

plt.figure()
plt.scatter(wave,tilt)
plt.savefig('/genesis/jwst/userland-soss/loic_review/soss_simtilt.pdf')
plt.show()

wavefine = np.linspace(0.5,3.0,2501)
tiltfine = np.interp(wavefine,wave,tilt)


from astropy.table import Table

meta = {'description': 'This file was created by soss_characterize_sim_tilt.py'}
formats = {'Wavelength':'{:.4f}','order1':'{:.4f}','order2':'{:.4f}','order3':'{:.4f}'}
tab = Table([wavefine,tiltfine,tiltfine,tiltfine], names=formats, meta=meta)
tab.write('SOSS_wavelength_dependent_tilt_sim.ecsv', formats=formats)

