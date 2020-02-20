"""
Created on Tue Jan 21 14:21:35 2020

@author: caroline

Introduce the detector response in the simulated images
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import fits
from pkg_resources import resource_filename
from copy import deepcopy
import timeseries
import pdb

    
#%% Main Function

def main(argv): 
    '''
    python detector.py path/to/fits/file.fits 1
        
    '''
    print('Input arguments: ', argv)

    l = len(argv)
    if l>2: # a minimum of 2 arguments must be specified for a command-line call 
        imaPath         = argv[1]                   # path to the fits file containing the image (Jason's output)        
        addNonLinearity = bool(int(argv[2]))        # include effect of non-linearity in detector response if True

    else: # if not called from the command line
        imaPath         = '/Users/caroline/Research/GitHub/SOSS/jwst-mtl/SOSS/detector/data/jw00001001001_0110100001_NISRAPID_cal_c.fits'
        addNonLinearity = True
    
        print(imaPath,addNonLinearity)
    
    
    ts = timeseries.TimeSeries(imaPath)
    
    # adding Poisson noise to the images prior to non-linearity correction
    ts.addPoissonNoise()
    
    # modify time series for non-linearity effects
    if addNonLinearity:
        
        # File path containing the forward coefficients for the fit 
        
        # forward coefficients calculated using non-linearity data from CRDS website
        non_linearity = np.load('files/files_jwst_niriss_linearity_0011_range_0_100000_npoints_100_polydeg_4.npy')
                
        ts.addNonLinearity(non_linearity)
            
    ts.writeToFits() # write modified time series observations to a new file 
    
    
    
#%%

if __name__ == "__main__":
    fit = main(sys.argv)
    
    

