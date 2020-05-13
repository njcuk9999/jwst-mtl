#!/usr/bin/env python
# original code by Thatte, Anand, Sahlmann, Greenbaum
# utility routines and constants for AMI image simulations reorganized by Sahlmann, Anand 3/2016
# anand@stsci.edu 18 Mar 2016

"""
"""

import numpy as np
import sys, time
from astropy.io import fits
import os

# global variables
global readnoise,background,darkcurrent


cdsreadnoise = 21.0                      # CDS read noise (e-)
readnoise = cdsreadnoise/np.sqrt(2)      # read noise for one frame
                                         # 0.012 e-/sec  value used until 09 2016
darkcurrent = 0.04                       # ~0.12 e-/sec 09/2016, 10x earlier, still 6e- in max 800 frames
                                         # Kevin Volk via Deepashri Thatte
background = 0.462*0.15*1.75             # 0.125 e-/sec 
ips_size = 256                           # holdover from before AMISUB became 80x80
flat_sigma = 0.001                       # flat field error
pixscl = 0.0656                           # arcsec/pixel WebbPSF 0.064 - DL 0.065
tframe = 0.0745                          # frame time for NISRAPID on AMI SUB80
amisubfov = 80
SAT_E = 72.0e3                           # Fullerton December 20, 2016 e-mail. Also consistent with STScI JWST ETC


#ither_stddev_as = 0.0015                 # 15 mas placement error one-axis
#itter_stddev_as = 0.007                 # 7 mas level 2 reqt on JWST, arcsec,

#ither_stddev_as = 0.0015                # Anand Alex detectionlimits
#itter_stddev_as = 0.0001                # Anand Alex centering  Also for Stefenie test reductiondetectionlimits

dither_stddev_as = 0.005                 # Goudfrooij Sep 2 2016 email to anand@ - good to SAMs of 30 arcsec
jitter_stddev_as = 0.004                 # NEA ~1mas jitter FGS, plus other slower error, Kevin 2016.09.16
                                         # Post-flight determination required for more realism in simulations...
                                         # In practise expert reduction should do rapid centroiding
                                         # (as in Holfeltz et al. TRs) through all integrations to 
                                         # determine the level of jitter, and calculate CPs in reasonable
                                         # subsets of these integrations.  

# Anand's email 2016-02-10 orginally from Volk
F277W, F380M, F430M, F480M = ("F277W", "F380M", "F430M", "F480M")
ZP = {F277W: 26.14,  
      F380M: 23.75,
      F430M: 23.32,
      F480M: 23.19} # replace w/Neil R.'s values consistent w/ STScI 


debug_utils = False
# debug_utils = True

def get_flatfield(detshape,pyamiDataDir,uniform=False,random_seed=None, overwrite=0):
    """
    Read in a flat field that possesses the requested flat field error standard deviation, 
    or if the file does not exist, create, write, and return it 
    """

#     # J. Sahlmann 2017-02-02: bug here, pyamiDataDir does not always exist because it is derived from the location where the driver script is stored/run
#     pathname = os.path.dirname(sys.argv[0])
#     fullPath = os.path.abspath(pathname)
#     pyamiDataDir = fullPath + '/pyami/etc/NIRISSami_apt_calcPSF/'

    ffe_file = os.path.join(pyamiDataDir ,'flat_%dx%d_sigma_%.4f.fits'%(detshape[0],detshape[1],flat_sigma))

    if (os.access(ffe_file, os.F_OK) == True) & (overwrite==0):
        #print "\tflat field file %s" % ffe_file
        pflat = fits.getdata(ffe_file)
    else:
        if uniform:
            pflat = np.ones(detshape)
        else:
            if random_seed is not None:
                np.random.seed(random_seed)
            pflat = np.random.normal(1.0, flat_sigma, size=detshape)
        print("creating flat field and saving it to  file %s" % ffe_file)

        (year, month, day, hour, minute, second, weekday, DOY, DST) =  time.gmtime()

        fitsobj = fits.HDUList()
        hdu = fits.PrimaryHDU()
        hdu.header['DATE'] = '%4d-%02d-%02dT%02d:%02d:%02d' % \
                  (year, month, day, hour, minute, second), 'Date of calculation'
        hdu.header['AUTHOR'] = '%s@%s' % (os.getenv('USER'), os.getenv('HOST')), \
                  'username@host for calculation'

        hdu.data = pflat

        fitsobj.append( hdu )
        fitsobj.writeto(ffe_file, clobber=True) 
        fitsobj.close()

    return pflat

# fast rebin Klaus Pontooppidan found on the web
def krebin(a, shape):
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
        return a.reshape(sh).sum(-1).sum(1)
# legacy slow rebin rewritten to use fast rebin
def rebin(a = None, rc=(2,2), verbose=None):
        r, c = rc
        R, C = a.shape
        sh = (int(R//r), int(C//c))
        return krebin(a, sh)


def jitter(no_of_jitters, osample, random_seed=None):
    """ returns in oversampled pixel units.  
        no_of_jitters is known as nint in STScI terminology
    """ 
    mean_j, sigma_j = 0, jitter_stddev_as * osample / pixscl

    if random_seed is not None:
        np.random.seed(random_seed)
    xjit = np.random.normal(mean_j,sigma_j,no_of_jitters)
    xjit_r = [int(round(n, 0)) for n in xjit]

    if random_seed is not None:
        np.random.seed(random_seed+1); # independent noise in X and Y, thus modify random_seed in a controlled way
    yjit = np.random.normal(mean_j,sigma_j,no_of_jitters)
    yjit_r = [int(round(n, 0)) for n in yjit]
    return xjit_r, yjit_r


def create_ramp(countspersec, _fov, ngroups, utr_,verbose=0, include_noise=1,random_seed=None):
    """ 
       input counts per second
       output: ramp has ngroups+1 slices, units are detected e- + noise
       create_ramp() called nint number of times to provide nint ramps
       Noise contributions can be switched off by setting include_noise = 0
   """
   
    global readnoise,background,darkcurrent
   
    #       JSA 2017-02-22: investigate effects of various noise contributions
    if include_noise == -1:
        # zero all noises except photon noise
        background = 0.
        readnoise = 1.e-16
        darkcurrent = 0.
        include_noise = 1
        
   
    if utr_ :
        nreadouts = ngroups + 1
        timestep = tframe
    else:
        if ngroups > 1:
            nreadouts = 3
            timestep = (ngroups-1) * tframe
        else:
            nreadouts = 2

    readnoise_cube                = np.zeros((nreadouts,int(_fov),int(_fov)), np.float64)
    background_cube               = np.zeros((nreadouts,int(_fov),int(_fov)), np.float64)
    dark_cube                     = np.zeros((nreadouts,int(_fov),int(_fov)), np.float64)
    poisson_noise_cube            = np.zeros((nreadouts,int(_fov),int(_fov)), np.float64)
    cumulative_poisson_noise_cube = np.zeros((nreadouts,int(_fov),int(_fov)), np.float64)
    ramp                          = np.zeros((nreadouts,int(_fov),int(_fov)), np.float64)

    if (debug_utils) | (verbose):
        print("\tcreate_ramp(): ngroups", ngroups, end=' ') 
        print("  countspersec.sum() = %.2e"%countspersec.sum(), end=' ') 
        print("  countsperframe = %.2e"%(countspersec.sum()*tframe))

    #calculate poisson noise for single reads, then calculate poisson noise for reads up-the-ramp
    for iread in range(nreadouts):

        if iread == 0:
            if include_noise == 0:
                ramp[iread,:,:] = np.zeros( (int(_fov),int(_fov)) )
            else:   
                if random_seed is not None:
                    np.random.seed(random_seed+111)
                readnoise_cube[iread,:,:] = np.random.normal(0, readnoise, (int(_fov),int(_fov))) 
                ramp[iread,:,:] = readnoise_cube[iread,:,:].mean()
            if (debug_utils) | (verbose):
                print("\t\tpoissoncube slice %2d:  %.2e"%(iread, poisson_noise_cube[iread,:,:].sum()), end=' ')
                print("poissoncube total %.2e"%poisson_noise_cube.sum())

        elif iread == 1:
            photonexpectation = countspersec * tframe
            photonexpectation[photonexpectation <0.0] = 0.0  # catch roundoff to e-13
            if include_noise == 0:
                ramp[iread,:,:] = photonexpectation
            else:
                if random_seed is not None:
                    # the noise in different frames should be independent, therefore modify random_seed between frames and between poisson and gaussian noise
                    np.random.seed(random_seed+iread+111)
                poisson_noise_cube[iread,:,:] = np.random.poisson(photonexpectation) # expose for tframe
                background_cube[iread,:,:] =  background * tframe
                dark_cube[iread,:,:] =  darkcurrent * tframe
                if random_seed is not None:
                    np.random.seed(random_seed+iread+111+10)
                readnoise_cube[iread,:,:] = np.random.normal(0, readnoise, (int(_fov),int(_fov))) 
                ramp[iread,:,:] = ramp[iread-1,:,:] + \
                              poisson_noise_cube[iread,:,:] + \
                              dark_cube[iread,:,:] + \
                              readnoise_cube[iread,:,:]
            if (debug_utils) | (verbose):
                print("\t\tpoissoncube slice %2d:  %.2e"%(iread, poisson_noise_cube[iread,:,:].sum()), end=' ')
                print("poissoncube total %.2e"%poisson_noise_cube.sum())

        else:
            photonexpectation = countspersec * timestep
            photonexpectation[photonexpectation <0.0] = 0.0
            if include_noise == 0:
                ramp[iread,:,:] = photonexpectation
            else:
                if random_seed is not None:
                    np.random.seed(random_seed + iread+111)
                poisson_noise_cube[iread,:,:] = np.random.poisson(photonexpectation) # expose for tframe or (ng-1)*tframe
                background_cube[iread,:,:] =  background * timestep
                dark_cube[iread,:,:] =  darkcurrent * timestep
                if random_seed is not None:
                    np.random.seed(random_seed + iread+111+10)
                readnoise_cube[iread,:,:] = np.random.normal(0, readnoise, (int(_fov),int(_fov))) 
                ramp[iread,:,:] = ramp[iread-1,:,:] + \
                              poisson_noise_cube[iread,:,:] + \
                              dark_cube[iread,:,:] + \
                              readnoise_cube[iread,:,:]
            if (debug_utils) | (verbose):
                print("\t\tpoissoncube slice %2d:  %.2e"%(iread, poisson_noise_cube[iread,:,:].sum()), end=' ')
                print("poissoncube total %.2e"%poisson_noise_cube.sum())


    
    
    if (debug_utils) | (verbose):
        s = "%.1e"
        print("\tpoissoncube total = %.1e" % poisson_noise_cube.sum()) # requested nphot / nint
        print("\tramp last slice total = %.1e" % ramp[-1,:,:].sum())   # approx same as above
        #print "\tramp last slice peak = %.1e" % ramp[-1,:,:].max() #should be ~sat_e typically
        for i in range(ramp.shape[0]):
            print("\t", s%ramp[i,:,:].sum(), ":", s%ramp[i,:,:].max(), end=' ')
        print("\n\tcreate_ramp: end")        
    return ramp


def create_integration(ramp): #????????
    """
    input: ramp in  e-, including 'zero read', ngroups+1 2D slices
    output: data in detected e-
    """

    if debug_utils:
        s = "%.1e"
        for i in range(ramp.shape[0]):
            print(" ", s%ramp[i,:,:].sum(), end=' ')
        print("\n\tcreate_integration: end")

    if ramp.shape[0] == 2:
        data = ramp[1,:,:] # no subtraction on readnoise+DC - ramp[0,:,:]
    if ramp.shape[0] > 2:
        data = ramp[-1,:,:] - ramp[1,:,:]
    return data


# old, now ...unused.. 09/2016
"""
def find_slope(utr, ngroups, fov):
    xval = np.zeros((ngroups+1,int(fov),int(fov)), np.float64)
    slope = np.zeros((int(fov),int(fov)), np.float64)
    for i in range(ngroups+1):
        xval[i]=i
    xm=float(ngroups)/2.0
    slope = (np.sum(xval*utr,axis=0)-xm*np.sum(utr,axis=0))/(np.sum(xval**2,axis=0)-ngroups*xm**2)
    return slope
"""


#origin is at bottom left of the image. (ds9?)
def apply_padding_image(a,e_x, e_y, fov, osample):

    err_x = int(e_x)
    err_y = int(e_y)

    if err_x <=  0 and err_y <=  0:
       b = np.pad(a, [(0,abs(err_y)),(0,abs(err_x))],mode='constant')
       c = b[abs(err_y):,abs(err_x):]

    elif err_x >=  0 and err_y <=  0:
       b = np.pad(a, [(0,abs(err_y)),(abs(err_x),0)],mode='constant')
       c = b[abs(err_y):,:(fov*osample)]

    elif err_x <=  0 and err_y >=  0:
       b = np.pad(a, [(abs(err_y),0),(0, abs(err_x))],mode='constant')
       c = b[:(fov*osample),abs(err_x):] 

    elif err_x >=  0 and err_y >=  0:
       b = np.pad(a, [(abs(err_y),0),(abs(err_x), 0)],mode='constant')
       c = b[:(fov*osample),:(fov*osample)]

    return c


#padding of 1 for IPS to avoid division by 0 when divided by IPS flat.
def apply_padding_ips(a,e_x,e_y, fov, osample):

    err_x, err_y = (int(e_x), int(e_y))

    if err_x <=  0 and err_y <=  0:
       b = np.pad(a, [(0,abs(err_y)),(0,abs(err_x))],mode='constant',constant_values=(1,1))
       c = b[abs(err_y):,abs(err_x):]
    elif err_x >=  0 and err_y <=  0:
       b = np.pad(a, [(0,abs(err_y)),(abs(err_x),0)],mode='constant',constant_values=(1,1))
       c = b[abs(err_y):,:(fov*osample)]
    elif err_x <=  0 and err_y >=  0:
       b = np.pad(a, [(abs(err_y),0),(0, abs(err_x))],mode='constant',constant_values=(1,1))
       c = b[:(fov*osample),abs(err_x):] 
    elif err_x >=  0 and err_y >=  0:
       b = np.pad(a, [(abs(err_y),0),(abs(err_x), 0)],mode='constant',constant_values=(1,1))
       c = b[:(fov*osample),:(fov*osample)]
    return c
