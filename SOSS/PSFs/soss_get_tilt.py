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



def soss_get_tilt(psfname):
    # Read PSF fits file
    a = fits.open(psfname)
    psf = a[0].data
    # Define two arrays representing vertical cuts through both peaks of the PSF
    leftslice = np.sum(psf[640 - 100:640 + 100, 518:604], axis=1)
    righslice = np.sum(psf[640 - 100:640 + 100, 697:765], axis=1)
    # The number of columns between both slices
    dx = np.mean([697, 765]) - np.mean([604, 518])
    # Perform a cross-correlation correlation and fit its peak using a gaussian
    ccf2 = fitCCF(leftslice, righslice, fitfunc='gauss', fitradius=40, makeplot=False)
    # The monochromatic tilt is then:
    tilt = np.rad2deg(np.arctan(ccf2 / dx))

    return(tilt)