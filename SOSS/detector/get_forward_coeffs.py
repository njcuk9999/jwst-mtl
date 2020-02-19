#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:17:45 2020

@author: caroline

Get forward coefficients to apply non-linearity on a linear ramp
based on the correction coefficients available on the CRDS website
"""

# Import modules 

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import fits
from copy import deepcopy
import pdb
import time

#%% Function(s)

def get_forward(correc_coeffs,iRow=1380,iCol=55,range_calibration= [0.,100e3],npoints=100,poly_deg=4,plot=False):
    '''
    Fit forward coefficients for a pixel pixel
    
    range_calibration: range of counts for which the forward coefficients are calculated
    npoints: number of points to use for calibration
    poly_deg: degree of the polynomial to fit for the forward coefficients
    plot: plot the result of the polynomial fit
    '''

    fluxes_calibration = np.linspace(range_calibration[0],range_calibration[1],npoints)
        
    flux_with_nonlin = np.zeros_like(fluxes_calibration)
    
    # use root finding method to calculate the fluxes with added non-linearity
    for i in range(fluxes_calibration.size):
        coeffs = -correc_coeffs[:,iCol,iRow][::-1]
        coeffs[-1] = coeffs[-1]+fluxes_calibration[i]
        cr_nonlin = np.roots(coeffs)
        flux_with_nonlin[i] = np.real(cr_nonlin[np.isreal(cr_nonlin)])[0]
    
    # fit a polynomial to the fluxes with non-linearity
    fwd_coeffs = np.polyfit(fluxes_calibration,flux_with_nonlin,poly_deg)
    p = np.poly1d(fwd_coeffs)
    
    if plot:
        # make figure of the fluxes with non-linearity as a function of the input "perfect" fluxes
        fig, ax = plt.subplots(1,1)
        ax.set_title('Example: Col '+str(iCol)+', Row '+str(iRow))
        ax.plot(fluxes_calibration,flux_with_nonlin,marker='.',ls='',color='k',label='Root-finding results')
        ax.plot(fluxes_calibration,p(fluxes_calibration),color='r',label='Polynomial fit')
        ax.plot(fluxes_calibration,fluxes_calibration,zorder=-20,color='gray',ls='--',label='1:1')
        ax.set_xlabel('Ideal flux')
        ax.set_ylabel('Flux with non-linearity')
        ax.legend(loc=2)
    
    return fwd_coeffs
    
def calc_forward_coeffs_array(correc_coeffs,range_calibration= [0.,100e3],npoints=100,poly_deg=4,printEveryNCol=5):
    '''
    Calculate the ndarray of (poly_deg+1)*ncols*nrows coefficients

    correc_coeffs: correction coefficients from CRDS file
    range_calibration: range of counts for which the forward coefficients are calculated
    npoints: number of points to use for calibration
    poly_deg: degree of the polynomial to fit for the forward coefficients
    printEveryNCol: if None, does nothing. if = a number, will print to the screen every 
                    * this number of columns has been processed
    '''
    ncols=256
    nrows=2048
    forward_coeffs = np.zeros((poly_deg+1,ncols,nrows))
    
    start_time = time.time()
    for c in range(ncols):
        if printEveryNCol is not None:
            if c%printEveryNCol==0:
                print('Column '+str(c+1)+'/'+str(ncols))
        for r in range(nrows):
            forward_coeffs[:,c,r] = get_forward(correc_coeffs,r,c)
            if c==0 and r==0:
                time_first = time.time()-start_time
                print('The first calculation took',time_first,'seconds.')
                print('Estimated total run time:',time_first*nrows*ncols,'seconds.')
    
    return forward_coeffs

#%% Read in the CRDS file of correction polynomial coefficients

def main(argv): 
    '''
    Example call:
    
    python get_forward_coeffs.py path/to/CRDS/file.fits 
    or
    python get_forward_coeffs.py path/to/CRDS/file.fits 0 100000 100 4
        
    '''
    print('Input arguments: ', argv)

    l = len(argv)
    if l>1: # a minimum of 1 argument must be specified for a command-line call
        crdsPath  = argv[1]    # path to the fits file containing the CRDS file
        if l>2:
            range_calibration=[]
            range_calibration.append(float(argv[2]))
            range_calibration.append(float(argv[3]))
            npoints = int(argv[4])
            poly_deg = int(argv[5])
        else:
            # setup for calculation
            range_calibration= [0.,100e3]
            npoints = 100
            poly_deg = 4
            
    else: # if not called from the command line
        crdsPath  = 'files/jwst_niriss_linearity_0011.fits'
        
        # setup for calculation
        range_calibration= [0.,100e3]
        npoints = 100
        poly_deg = 4  
        
        
    correc_coeffs = fits.open(crdsPath)[1].data[:,1792:,:] # subsection for SOSS detector
    
    
    fwd_coeffs = calc_forward_coeffs_array(correc_coeffs,range_calibration=range_calibration,npoints=npoints,poly_deg=poly_deg)
    np.save(crdsPath[:-5].replace('/','_')+'_range_'+str(range_calibration[0])\
            +'_'+str(range_calibration[0])+'_npoints_'+str(npoints)+'_polydeg_'\
            +str(poly_deg)+'.npy',fwd_coeffs)
    
    
#%%

if __name__ == "__main__":
    fit = main(sys.argv)
    