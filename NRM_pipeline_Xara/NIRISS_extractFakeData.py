#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:27:39 2019

@author: asoulain
"""

from NRM_extractor import NRM_extractor
#import datetime
#from astropy.io import fits
#from matplotlib import pyplot as plt
#import numpy as np
#from matplotlib.colors import PowerNorm

import NRM_fct

mask_name = '7holes_jwst' 

D = 6.5
pscale = 65.6 # plate scale of the image (in mas/pixel)
wl     = 4.2829e-6 # wavelength of the image  (in meters)
bandwidth   = 0.35e-6
ISZ    = 76    # image size

savedir = '__DatadpyDir'


NRM_fct.NRM_datacubeTodpy('DATA/t_myscene__myPSF__00.fits', exptime=50, mjd=50000, target='fakedisk', 
                          filesave='fakeobj.dpy', savedir=savedir, verbose=False)

NRM_fct.NRM_datacubeTodpy('DATA/c_myscene__myPSF__00.fits', exptime=50, mjd=50000, target='fakepsf', 
                          filesave='fakepsf.dpy', savedir=savedir, verbose=False)

data = {'star' : savedir + '/fakeobj.dpy',
        'calib' : savedir + '/fakepsf.dpy'}

NRM_extractor(data, mask_name, pscale, wl, bandwidth, ISZ = ISZ, 
              fakeerr = False, e_rel = 0.1,
              r1 = 30, dr = 5, icrop = 77, 
              Skycorr = True, Polyfit = False, apod = True,
              nmax = None, save = True, display = True)

      