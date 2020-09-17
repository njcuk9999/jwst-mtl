#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:27:00 2020

@author: albert
"""

# This script writes DMS ready fits files. It uses as input a simulated
# noiseless image to which it adds the required minimum set of FITS keywords
# required in order to minimally run the STScI pipeline. Based on
# the Fortran code specgen/utils/writefits_phdu.f90
#
# INPUTS:
# image - is a numpy array with 2,3 or 4 dimensions
#         4 dim --> nint,ngroup,dimy,dimx are the dimensions expected
#         3 dim --> ngroup,dimy,dimx are the dimensions expected
#         2 dim --> dimy,dimx are the dimensions expected
#         The input image is expected to be in the DMS orientation already
# filename - is the name of the fits file that will be written on disk
# optional inputs:
# os - is the oversampling integer 1 or more of the input image.
# input_frame is either dms, native or sim. represents the coorinate frame of
#         input image.
import sys
from astropy.io import fits
import numpy as np
from skimage.transform import downscale_local_mean, resize

def write_dmsready_fits(image, filename, os=1, input_frame='sim'):
    print('Start of write_dmsready_fits')
    # Expects a numpy array with n dimensions =2 or 3, representing a single
    # integration. The size of each dimension is left open to accomodate
    # oversampled images or oversized images (to include order 0 for example).

    # First deal with possibly oversampled input images.
    # At the end of this exercice, a 4-dim array at native pixel size should exist.
    if (os >= 1) & (os % 1 == 0) :
        size = np.shape(image)
        if len(size) == 4:
            # 4 dimensional. Assume nint,ngroup,dimy,dimx. Bin dimy,dimx dimensions.    
            nint,ngroup,dimy,dimx = size
            # Create a 4-dimensional array with native pixel size dimensions
            data = np.zeros((nint,ngroup,int(dimy/os),int(dimx/os)))
            # For each int and group, bin the image to native pixel size (handling flux properly)
            for i in range(nint):
                for j in range(ngroup):
                    data[i,j,:,:] = downscale_local_mean(image[i,j,:,:],(os,os))*os**2
        elif len(size) == 3:
            nint = 1
            ngroup,dimy,dimx = size
            data = np.zeros((nint,ngroup,int(dimy/os),int(dimx/os)))
            for j in range(ngroup):
                data[0,j,:,:] = downscale_local_mean(image[j,:,:],(os,os))*os**2
        elif len(size) == 2:
            nint = 1
            ngroup = 1
            dimy,dimx = size
            data = np.zeros((nint,ngroup,int(dimy/os),int(dimx/os)))
            data[0,0,:,:] = downscale_local_mean(image,(os,os))*os**2
        else:
            print('There is a problem with the image passed to write_dmsread_fits.')
            print('Needs to have 2 to 4 dimensions.')
            sys.exit()
        # Reset the dimx, dimy parameters now that all is in native pixel size
        nint,ngroup,dimy,dimx = np.shape(data)
    else:
        print('The oversampling of the input image should be 1 or higher integer. Stop')
        sys.exit()

    # Handle the case where the input_frame optional keyword is set
    if input_frame == 'dms':
        # Then the array is already in the desired coordinate frame, i.e: dms
        print('') #
    elif input_frame == 'sim':
        # Need to flip along dimy axis
        data = np.flip(data,axis=2)
        nint,ngroup,dimy,dimx = np.shape(data)
    elif input_frame == 'native':
        # Need to mirror and rotate
        data = np.flip(np.flip(np.swapaxes(data,2,3),axis=3),axis=2)
        nint,ngroup,dimy,dimx = np.shape(data)
    else:
        # bad choice
        print('input_frame should be dms, sim or native.')
        sys.exit()

    # The format of a JWST file is:
    # A primary header, containing only a header with all important keywords.
    # An extension, EXTNAME=SCI, containing a minimal header and the data
    #
    # Here is an example thar runs successfully in the STScI pipeline:
    #SIMPLE  =                    T / file does conform to FITS standard             
    #BITPIX  =                  -32 / number of bits per data pixel                  
    #NAXIS   =                    1 / number of data axes                            
    #NAXIS1  =                    0 / length of data axis 1                          
    #EXTEND  =                    T / FITS dataset may contain extensions            
    #COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
    #COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H 
    #NRSTSTRT=                    1 / / the number of resets at the start of the expo
    #NRESETS =                    1 / / the number of resets between integrations    
    #DATE    = '2019-12-05T11:09:28.097' / / [yyyy-mm-ddThh:mm:ss.ss] UTC date file c
    #FILENAME= 'jw00001001001_0110100001_NISRAPID_cal_c.fits' / / Name of the file   
    #DATAMODL= 'RampModel'          / / Type of data model                           
    #TELESCOP= 'JWST    '           / / Telescope used to acquire the data           
    #DATE-OBS= '2020-02-05'         / / [yyyy-mm-dd] Date of observation             
    #TIME-OBS= '11:08:45.000'       / / [hh:mm:ss.sss] UTC time at start of exposure 
    #TARG_RA =            188.38685 / / Target RA at mid time of exposure            
    #TARG_DEC=  -10.146173055555559 / / Target Dec at mid time of exposure           
    #SRCTYPE = 'POINT   '           / / Advised source type (point/extended)         
    #INSTRUME= 'NIRISS  '           / / Instrument used to acquire the data          
    #DETECTOR= 'NIS     '           / / Name of detector used to acquire the data    
    #FILTER  = 'CLEAR   '           / / Name of the filter element used              
    #PUPIL   = 'GR700XD '           / / Name of the pupil element used               
    #EXP_TYPE= 'NIS_SOSS'           / / Type of data in the exposure                 
    #READPATT= 'NISRAPID'           / / Readout pattern                              
    #NINTS   =                    1 / / Number of integrations in exposure           
    #NGROUPS =                   10 / / Number of groups in integration              
    #NFRAMES =                    1 / / Number of frames per group                   
    #GROUPGAP=                    0 / / Number of frames dropped between groups      
    #TFRAME  =                5.491 / / [s] Time between frames                      
    #TGROUP  =                5.491 / / [s] Time between groups                      
    #DURATION=                  1.0 / / [s] Total duration of exposure               
    #SUBARRAY= 'SUBSTRIP256'        / / Subarray used                                
    #SUBSTRT1=                    1 / / Starting pixel in axis 1 direction           
    #SUBSTRT2=                 1793 / / Starting pixel in axis 2 direction           
    #SUBSIZE1=                 2048 / / Number of pixels in axis 1 direction         
    #SUBSIZE2=                  256 / / Number of pixels in axis 2 direction         
    #FASTAXIS=                   -2 / / Fast readout axis direction                  
    #SLOWAXIS=                   -1 / / Slow readout axis direction                  
    #END 
    #
    #XTENSION= 'IMAGE   '           / IMAGE extension                                
    #BITPIX  =                  -32 / number of bits per data pixel                  
    #NAXIS   =                    4 / number of data axes                            
    #NAXIS1  =                 2048 / length of data axis 1                          
    #NAXIS2  =                  256 / length of data axis 2                          
    #NAXIS3  =                   10 / length of data axis 3                          
    #NAXIS4  =                    1 / length of data axis 4                          
    #PCOUNT  =                    0 / required keyword; must = 0                     
    #GCOUNT  =                    1 / required keyword; must = 1                     
    #EXTNAME = 'SCI     '                                                            
    #END

    # Create primary HDU
    prim_hdu = fits.PrimaryHDU()
    phdr = prim_hdu.header
    phdr.set('NRSTSTRT',1,'the number of resets at the start of the expo')
    phdr.set('NRESETS',1,'the number of resets between integrations')
    phdr.set('DATE','2019-12-05T11:09:28.097', '[yyyy-mm-ddThh:mm:ss.ss] UTC date file cre')
    phdr.set('FILENAME',filename,'Name of the file')
    phdr.set('DATAMODL','RampModel','Type of data model')
    phdr.set('TELESCOP', 'JWST','Telescope used to acquire the data')
    phdr.set('DATE-OBS','2020-02-05','[yyyy-mm-dd] Date of observation')
    phdr.set('TIME-OBS','11:08:45.000','[hh:mm:ss.sss] UTC time at start of exposure')
    phdr.set('TARG_RA',188.38685,'Target RA at mid time of exposure')
    phdr.set('TARG_DEC',-10.14617305555556,'Target Dec at mid time of exposure')
    phdr.set('SRCTYPE','POINT','Advised source type (point/extended)')
    phdr.set('INSTRUME','NIRISS','Instrument used to acquire the data')
    phdr.set('DETECTOR','NIS','Name of detector used to acquire the data')
    phdr.set('FILTER','CLEAR','Name of the filter element used')
    phdr.set('PUPIL','GR700XD','Name of the pupil element used')
    phdr.set('EXP_TYPE','NIS_SOSS','Type of data in the exposure')
    phdr.set('READPATT','NISRAPID','Readout pattern')
    # Check that the data is a valid subarray
    if (dimy != 2048) & (dimy !=256) & (dimy !=96):
        print('The array Y-axis needs to be 96, 256 or 2048 pixels.')
    if (dimx != 2048):
        print('The array X-axis needs to be 2048 pixels.')
    phdr.set('NINTS',nint,'Number of integrations in exposure')        
    phdr.set('NGROUPS',ngroup,'Number of groups in integration')
    phdr.set('NFRAMES',1,'Number of frames per group')
    phdr.set('GROUPGAP',0,'Number of frames dropped between groups')
    phdr.set('TFRAME',5.491,'[s] Time between frames')
    phdr.set('TGROUP',5.491,'[s] Time between groups')
    phdr.set('DURATION',1.0,'[s] Total duration of exposure')
    subarray = 'CUSTOM'
    if (dimy == 96) & (dimx == 2048):
        subarray = 'SUBSTRIP96'
        substrt1,subsize1 = 1,2048
        substrt2,subsize2 = 1803,96
    elif (dimy == 256) & (dimx == 2048):
        subarray = 'SUBSTRIP256'
        substrt1,subsize1 = 1,2048
        substrt2,subsize2 = 1793,256
    elif (dimy == 2048) & (dimx == 2048):
        subarray = 'FF'
        substrt1,subsize1 = 1,2048
        substrt2,subsize2 = 1,2048
    else:
        print('image size not correct.')
        subarray = 'CUSTOM'
        substrt1,subsize1 = 1,dimx
        substrt2,subsize2 = 1,dimy
    phdr.set('SUBARRAY',subarray,'Subarray used')
    phdr.set('SUBSTRT1',substrt1,'Starting pixel in axis 1 direction')
    phdr.set('SUBSTRT2',substrt2,'Starting pixel in axis 2 direction')
    phdr.set('SUBSIZE1',subsize1,'Number of pixels in axis 1 direction')
    phdr.set('SUBSIZE2',subsize2,'Number of pixels in axis 2 direction')
    phdr.set('FASTAXIS',-2,'Fast readout axis direction')
    phdr.set('SLOWAXIS',-1,'Slow readout axis direction')

    # Create extension HDU
    ext_hdu = fits.ImageHDU(data)
    # Add keywords to the extension header
    xhdr = ext_hdu.header
    xhdr.set('EXTNAME','SCI')
    # Create a list of HDU with primary and extension HDUs
    hdulist = fits.HDUList([prim_hdu,ext_hdu])
    # Save on disk
    hdulist.writeto(filename, overwrite=True)
    hdulist.close()
    print('{:} was written with dimensions nint={:}, ngroup={:}, dimy={:}, dimx={:}, subarray={:}'.format(filename,
        nint,ngroup,dimy,dimx,subarray))
    print('write_dmsready_fits completed successfuly.')
    

def test():
    #import write_dmsready_fits as yes
    import numpy as np
    im = np.ones((2048,96))
    im[300,2] = 10.0
    im[310,10] = 10.0
    im[320,20] = 10.0
    im[500,30] = 10.0
    im[2000,80] = 10.0
    #im[:,5] = 10.0
    write_dmsready_fits(im,'totoune.fits',os=1,input_frame='native')                                       
