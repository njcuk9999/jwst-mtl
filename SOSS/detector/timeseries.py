
"""
Created on Sun Jan 26 16:39:05 2020

@author: caroline

TimeSeries objects for simulations of SOSS observations
"""


from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import fits
from pkg_resources import resource_filename
from copy import deepcopy
import pdb
from scipy.optimize import minimize

#%%
class TimeSeries(object):
    def __init__(self,imaPath,singleImage=True):
        '''
        Make a TimeSeries object from a single synthetic detector image 
        '''
        
        if singleImage:
            self.imaPath    = imaPath
            
            hdu_ideal   = fits.open(imaPath) # read in fits file
            header      = hdu_ideal[1].header
            
            self.hdu_ideal  = hdu_ideal
            self.data       = hdu_ideal[1].data # image to be altered
            self.nrows      = header['NAXIS1']
            self.ncols      = header['NAXIS2']
            self.ngroups    = header['NAXIS3'] # number of groups per integration
            self.nintegs    = header['NAXIS4'] # number of integrations in time series observations
            
            self.modifStr   = '_mod' # string encoding the modifications

        
    def addNonLinearity(self,non_linearity):
        '''
        Add non-linearity on top of the linear integration-long ramp
        non_linearity: array of polynomial coefficients
        offset: removed prior to correction and put back after
        '''
        
        # Add non-linearity to each ramp
        for i in range(self.nintegs):
            # select part of the time series lying within this integration (i.e., ith sequence of ngroups groups)
            integ = self.data[i,:,:,:]
                        
            # Apply offset before introducing non-linearity
            new_integ = deepcopy(integ)
            
            # Iterate over groups
            for g in range(self.ngroups):
                frame = deepcopy(new_integ[g,:,:])

                ncoeffs = non_linearity.shape[0]
                corr = non_linearity[ncoeffs-1,:,:]
                for k in range(1,ncoeffs):
                    corr = corr + non_linearity[-k-1,:,:] * frame**k
                
                new_integ[g,:,:]    = corr

            new_integ[np.where(new_integ<0.)]==0.
            
            self.data[i,:,:,:] = deepcopy(new_integ)

        self.modifStr   = self.modifStr+'_nonlin'
    
        
    def writeToFits(self,filename=None):
        '''
        write to a fits file the new header and data 
        '''
        
        hdu_new             = self.hdu_ideal
        hdu_new[1].data     = self.data
        
        if filename is None:
            filename = self.imaPath[:-5]+self.modifStr+'.fits'
            hdu_new.writeto(filename,overwrite=True)
        print('Writing to file: '+filename)
        
    def plotImage(self,iGroup=0,iInteg=0,log=False,reverse_y=True,
                  save=False,filename=None):
        '''
        Plot the detector image for a chosen frame
        '''
        
        fig, ax = plt.subplots(1,1,figsize=(8,3))
        img     = self.data[iInteg,iGroup,:,:]
        if log:
            im      = ax.imshow(np.log10(img))
            if reverse_y:
                ax.invert_yaxis()
            ax.set_title('log10 Group '+str(iGroup)+'; Integ '+str(iInteg))
        else:
            im      = ax.imshow(img)
            if reverse_y:
                ax.invert_yaxis()
            ax.set_title('Group '+str(iGroup)+'; Integ '+str(iInteg))

        fig.colorbar(im,ax=ax,orientation='horizontal')
        plt.tight_layout()
        
        # option to save the image
        if save:
            if filename is None:
                filename = 'image_G'+str(iGroup)+'_I'+str(iInteg)+'.png'
            fig.savefig(filename)
    
    def plotPixel(self,iRow=1380,iCol=55,marker='o',color='b',
                  plotOnIm=True,save=False,filename=None):
        '''
        Plot the flux in a given pixel as a function of Frame #
        '''
        
        # to distinguish integrations and groups in plotting
        colors  = ['b','orange','g','red']
        markers = ['o','^','*']
        count   = 0
        
        if plotOnIm: # if True, plot location of pixel on the first image
            fig, (ax2,ax) = plt.subplots(2,1,figsize=(7,5))
        else:
            fig, ax = plt.subplots(1,1,figsize=(7,3))
        
        for i in range(self.nintegs):
            for j in range(self.ngroups):
                ax.plot(count,self.data[i,j,iCol,iRow],marker=markers[j%3],color=colors[i%4],ls='')
                count = count + 1
        
        ax.set_xlabel('Frames')
        ax.set_ylabel('Pixel count')
        
        ax.set_title('Row '+str(iRow)+'; Column '+str(iCol))
        
        # ---- In addition, plot location of pixel on image --- #
        if plotOnIm:
            img     = self.data[0,0,:,:]
            im      = ax2.imshow(img)
            ax2.plot(iRow,iCol,marker='x',color='r')
            ax2.invert_yaxis()
            ax2.set_title('Group '+str(0)+'; Integ '+str(0))
        
        # -----
        
        # option to save the image
        if save:
            if filename is None:
                filename = 'pixel_'+str(iRow)+'_'+str(iCol)+'.png'
            fig.savefig(filename)
        