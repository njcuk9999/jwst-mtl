# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:49:19 2015

@author: asoulain
"""

import pickle, warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from scipy.signal import medfilt2d

warnings.filterwarnings("ignore", module='astropy.io.votable.tree')
warnings.filterwarnings("ignore", module='astropy.io.votable.xmlutil')

def mastorad(mas):
    """
    Short Summary
    -------------
    Convert angle in milli arc-sec to radians

    Parameters
    ----------
    mas: float
        angle in milli arc-sec

    Returns
    -------
    rad: float
        angle in radians
    """
    rad = mas * (10**(-3)) / (3600 * 180 / np.pi)
    return rad
    
def radtomas(rad):
    """
    Short Summary
    -------------
    Convert input angle in radians to milli arc sec

    Parameters
    ----------
    rad: float
        input angle in radians

    Returns
    -------
    mas: float
        input angle in milli arc sec
    """
    mas = rad * (3600. * 180 / np.pi) * 10.**3
    return mas

def crop_max(img, dim, filtmed = True, f = 3):
    """
    Short Summary
    -------------
    Resize an image on the brightest pixel.

    Parameters
    ----------
    img : numpy.array
        input image.
    dim : int
        resized dimension.
    filtmed : boolean
        True if perform a median filter on the image (to blur bad pixels).
    f : float
        if filtmed == True, kernel size of the median filter.()
        

    Returns
    -------
    cutout: numpy.array
        Resized image.
    """
    if filtmed:
        im_med = medfilt2d(img, f)
    else:
        im_med = img.copy()

    pos_max = np.where(im_med == im_med.max())
    X = pos_max[1][0]+1
    Y = pos_max[0][0]+1

    position = (X, Y)

    cutout = Cutout2D(img, position, dim)
    return cutout.data, position

def norm_max(tab):
    """
    Short Summary
    -------------
    Normalize an array or a list by the maximum.

    Parameters
    ----------
    tab : numpy.array, list
        input array or list.
    
    Returns
    -------
    tab_norm : numpy.array, list
        Normalized array.
    """
    tab_norm = tab/np.max(tab)
    return tab_norm

def crop_center(img, dim):
    """
    Short Summary
    -------------
    Resize an image on the center.

    Parameters
    ----------
    img : numpy.array
        input image.
    dim : int
        resized dimension.

    Returns
    -------
    cutout: numpy.array
        Resized image.
    """
    b = img.shape[0]
    position = (b//2,  b//2)
    cutout = Cutout2D(img, position, dim)
    return cutout.data

def crop_position(img, X, Y, dim):
    """
    Short Summary
    -------------
    Resize an image on a defined position.

    Parameters
    ----------
    img : numpy.array
        input image.
    X, Y : int
        Position to resize (new center of the image).
    dim : int
        resized dimension.

    Returns
    -------
    cutout: numpy.array
        Resized image.
    """
    position = (X,  Y)
    cutout = Cutout2D(img, position, dim)
    return cutout.data

def plot_JWST_ins_limit(inst):
        """
        Plot JWST instrument limits (sensitivity and saturation) for different filters.
        """
        file = open('/Users/asoulain/Documents/Add_Python_PATH/save_limit_JWST.dpy', 'rb')
        dic_JWST = pickle.load(file)
        file.close()
        
        if inst == 'NIRCAM':
            color = 'royalblue'
        elif inst == 'NIRISS':
            color = 'orange'
        elif inst == 'MIRI':
            color = 'crimson'
        
        i = 1
        l_filt = list(dic_JWST[inst].keys())
        for filt in l_filt:
            wl1 = dic_JWST[inst][filt]['wl0'] - dic_JWST[inst][filt]['bw']/2.
            wl2 = dic_JWST[inst][filt]['wl0'] + dic_JWST[inst][filt]['bw']/2.
            fmax = dic_JWST[inst][filt]['fmax']
            fmin = dic_JWST[inst][filt]['fmin']
            if i == 1:
                plt.fill_between([wl1, wl2], fmin, fmax, color = color, alpha = .2, label = 'JWST/'+inst)
            else:
                plt.fill_between([wl1, wl2], fmin, fmax, color = color, alpha = .2)
            i += 1
        return None
    
def plot_JWST_limit():
    plot_JWST_ins_limit('NIRCAM')
    plot_JWST_ins_limit('NIRISS')
    plot_JWST_ins_limit('MIRI')
    return None

def gauss_2d_asym(X, param):
    """
    Short Summary
    -------------
    Creates 2D oriented gaussian with an asymmetrical grid.
    
    Parameters
    ----------
    X : list. 
        Input values :
         - X[0] : x coordinates [pixels]
         - X[1] : y coordinates [pixels]
         - X[2] : pixels scale [mas]

    param : dict.
        Input parameters, with the present keys =
            - A : amplitude.
            - x0 : x offset from the center [mas].
            - y0 : y offset from the center [mas].
            - fwhm_x : width in x direction [mas].
            - fwhm_y : width in y direction [mas].
            - theta : orientation [deg].()
    Returns
    -------
    im: numpy.array
        image of a 2D gaussian function.
    """
    
    x_1d = X[0]
    y_1d = X[1]
    
    pixel_scale = X[2]
    
    dim = len(x_1d)
    
    x, y = np.meshgrid(x_1d, y_1d)
        
    fwhmx = param['fwhm_x']/pixel_scale 
    fwhmy = param['fwhm_y']/pixel_scale 
    
    sigma_x = (fwhmx / np.sqrt(8 * np.log(2)))
    sigma_y = (fwhmy / np.sqrt(8 * np.log(2)))
    
    amplitude = param['A']
    x0 = dim//2 + param['x0']/pixel_scale 
    y0 = dim//2 + param['y0']/pixel_scale 
    theta = np.deg2rad(param['theta'])
    size_x = len(x)
    size_y = len(y)
    im = np.zeros([size_y, size_x])
    x0 = float(x0)
    y0 = float(y0)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    im =  amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                            + c*((y-y0)**2)))
    return im
    
def conv_fft(image, psf):
    """
    Compute 2D convolution with the PSF, passing through Fourier space.
    """
    fft_im = np.fft.fft2(image)
    fft_psf = np.fft.fft2(psf)
    fft_conv = fft_im*fft_psf
    conv = abs(np.fft.fftshift(np.fft.ifft2(fft_conv)))
    return conv

class A(object):
       pass

class AllMyFields:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if type(v) == dict:
                a = A()
                for key in v.keys():
                    a.__dict__[key] = v[key]
                setattr(self, k, a)
            else:
                setattr(self, k, v)
