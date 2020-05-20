#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:57:32 2019

@author: asoulain
"""

import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from termcolor import cprint

plt.close('all')


def Simulate_NIRISS_AMI(param, mag, filt, phot=1e8, fov=500, savedir='Simulated_data/', psfdir='PSF/', display=True):
    """ Simulate NRM data of NIRISS instrument using amisim et ami_etc package developped by stsci.

    Parameters
    ----------
    `param` : {dict}
        Input parameters of the astronomical scene (e.g.: {'type': 'disk',
              'diam': 200} to simulate an uniform disk with a diameter of
              200 mas),\n
    `mag` : {float}
        Magnitude of the target in the chosen filter (used for etc measurements),\n
    `filt` : {str}
        Name of the filter (e.g.: 'F277W', 'F380M', 'F430M', 'F480M'),\n
    `phot` : {float}, (optional)
        Number of photon received on the detector (default=1e8),\n
    `fov` : {float}
        Field of view of the displayed astronomical scene (display=True),\n
    `savedir` : {str}
        Name of the directory where to save the data cube,\n
    `psfdir` : {str}, (optional)
        Name of the directory containing the simulated PSF, by default 'PSF/'
    `display` : {bool}, (optional)
        If True, show all figures, by default True.

    Returns
    -------
    `image`: {array}
        Simulated astrophysical scene.
    """
    if not os.path.exists(savedir):
        print('### Create %s directory to save all requested simulation. ###' % savedir)
        os.system('mkdir %s' % savedir)

    size = 81
    oversampling = 11

    pscale = 65.6/oversampling

    N = oversampling*size
    obj = np.zeros([N, N])

    xx, yy = np.arange(obj.shape[0]), np.arange(obj.shape[1])
    xpos = obj.shape[1]/2.
    ypos = obj.shape[1]/2.
    xx2 = (xx-xpos)
    yy2 = (ypos-yy)

    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis]**2)

    if param['type'] == 'binary':
        tar = param['type']
        sep = param['sep']
        posang = param['posang']
        dm = param['dm']
        X_comp = sep * np.cos(np.deg2rad(posang + 90))
        Y_comp = sep * np.sin(np.deg2rad(posang + 90))
        obj[N//2, N//2] = 1
        obj[N//2 + int(Y_comp/pscale), N//2 + int(X_comp/pscale)] = 1/(2.5)**dm
    elif param['type'] == 'disk':
        tar = param['type']
        diam = param['diam']
        obj[distance <= diam/oversampling] = 1
    elif param['type'] == 'pointsource':
        tar = param['type']
        obj[N//2, N//2] = 1
    else:
        cprint("Error: target type not recognised (disk, binary or pointsource).", 'red')
        return None

    image = obj.copy()
    # Add little offset to determine the maximum position (software issue of driver_scene.py)
    offset = 0
    if tar == 'disk':
        image[N//2, N//2] = image.max()*1.00001
        offset = 0.5

    if display:
        extent = (
            np.array([N/2.+offset, -N/2.+offset, -N/2.-offset, N/2.-offset]))*pscale
        title = r'Simulated scene'
        if tar == 'binary':
            title += r" (%s): s = %2.1f mas, $\theta$ = %2.0f deg, dm = %2.1f mag" % (
                tar, sep, posang, dm)
        elif tar == 'disk':
            title += " (%s): diam = %2.1f mas" % (tar, diam)
        elif tar == 'pointsource':
            title += ' (%s)' % tar
        plt.figure(figsize=(6, 6))
        plt.title(title, fontsize=8)
        plt.imshow(image, cmap='afmhot', origin='lower',
                   vmin=0, norm=PowerNorm(.1), extent=extent)
        plt.vlines(0, extent[0], extent[1], 'r', lw=1, alpha=.5)
        plt.hlines(0, extent[0], extent[1], 'r', lw=1, alpha=.5)
        plt.axis([fov/2., -fov/2., -fov/2., fov/2.])
        plt.xlabel(r'E $\leftarrow\, \Delta \alpha$ (mas)')
        plt.ylabel(r'$\Delta \delta\, \rightarrow$ N (mas)')
        plt.tight_layout()
        plt.show(block=False)

    os.system('python ami_etc.py %s %2.1f %2.2e' % (filt, mag, phot))

    etc = np.loadtxt('save_etc.txt')

    ngrp = etc[0]
    nit = etc[1]
    cr = etc[3]
    o = 1
    utr = 0

    over = 'x%s' % oversampling
    psfname = psfdir + '%s_%i_flat_%s.fits' % (filt, size, over)

    objname = 'SCENES/'
    if tar == 'disk':
        objname += '%s_d=%2.0fmas_mag=%2.1f.fits' % (tar, diam, mag)
    elif tar == 'binary':
        objname += '%s_s=%2.0fmas_mag=%2.1f_dm=%2.1f_posang=%2.1f.fits' % (
            tar, sep, mag, dm, posang)
    elif tar == 'pointsource':
        objname += "%s_mag=%2.1f.fits" % (tar, mag)

    image_sum = image.sum()

    image_norm = (image/image_sum) * cr

    fits.writeto(objname, image_norm, overwrite=True)

    com = 'python driver_scene.py' + ' --output_absolute_path %s' % savedir +\
          ' -o %i' % o +\
          ' -utr %i' % utr +\
          ' -f %s' % filt +\
          ' -p %s' % psfname +\
          ' -s %s' % objname +\
          ' -os 11' +\
          ' -I %i' % nit +\
          ' -G %i' % ngrp +\
          ' -c 1' +\
          ' -cr %i' % cr +\
          ' -v 1' +\
          ' --random_seed 42'

    os.system(com)

    cprint('\nResults simulations:', 'cyan')
    cprint('--------------------', 'cyan')
    cprint('t_%s created,\n -> # frames = %i.' %
           (objname.split('/')[1], nit), 'cyan')
    return image


filt = 'F430M'

mag = 6
param_bin = {'type': 'binary',
             'sep': 200,  # [mas]
             'dm': 5,  # [mag]
             'posang': 45  # [degree]
             }

param_disk = {'type': 'disk',
              'diam': 100,  # [mas]
              }

param_calib = {'type': 'pointsource'}

Simulate_NIRISS_AMI(param_bin, mag, filt, phot=1e6)

plt.show()
