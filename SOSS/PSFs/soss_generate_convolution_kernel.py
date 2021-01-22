#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:06:06 2020

@author: albert
"""




import numpy as np
import matplotlib.pylab as plt
import subprocess as subprocess
from astropy.io import fits
from scipy.stats import binned_statistic
from scipy import ndimage
import os.path
import soss_get_tilt as sosstilt

def webbpsf_return_listof(verbose=None, psf_path=None):
    
    # Determine where the webbpsf PSFs reside
    if psf_path == None:
        # Try on Loic's laptop:
        psf_path = '/Users/albert/NIRISS/SOSSpipeline/webbpsf_psfs/'
        if os.path.exists(psf_path) is False:
            # Try guessing assuming github install
            psf_path = './psflibrary/'
    if os.path.exists(psf_path) is False:
        print('Pass the path where webbpsfs are to webbpsf_return_listof(psf_path=...)')
        return(False)

    # Spawn a shell ls command to get the list    
    commandstr = 'ls -1 '+psf_path+'SOSS_os10_128x128_*.fits'
    ls = subprocess.getoutput(commandstr)
    psfList = ls.split('\n')
    wavelength,psfName = [],[]
    for i in range(np.size(psfList)):
        psfname = psfList[i]
        hdu = fits.open(psfname)
        hdr = hdu[0].header
        wavelength.append(1e+6*hdr['WAVELEN'])
        psfName.append(psfname)
        if verbose is True:
            print('Filename: {:}, Wavelength={:}'.format(psfname, 1e+6*hdr['WAVELEN']))
    return(np.array(psfName),np.array(wavelength))

def calc_com(x,fx):
    # returns the x center of mass (or barycenter)
    # of a f(x) function.
    
    # put to zero the lowest fx value
    fx_c = fx - np.min(fx)
    com = np.sum(fx_c*x)/np.sum(fx_c)
    return(com)

def webbpsf_read_and_rotate(filename, angle, verbose=None):
    
    hdu = fits.open(filename)
    image = hdu[0].data*1
    dim = np.shape(image)[0]
    hdr = hdu[0].header
    oversampling = hdr['OVERSAMP']

    # Apply a rotation to the PSF image
    empirical_rotation = angle*1.0
    imagewarped = ndimage.rotate(image, empirical_rotation)
    
    if verbose is True:
        plt.figure(figsize=(10,10))
        plt.imshow(np.log10(image[500:-500,:]),origin='bottom')
        
        # Plot the PSF before/after rotation
        plt.figure(figsize=(10,10))
        plt.imshow(np.log10(imagewarped[500:-500]),origin='bottom')
        
        ## Save the rotated PSF on disk
        hdu = fits.PrimaryHDU()
        hdu.data = imagewarped
        hdu.writeto('/genesis/jwst/userland-soss/loic_review/test.fits',overwrite=True)

    return(imagewarped, oversampling, dim)


def cut_ker_box(kernels, width=10, n_os=10, fwhm=5):
    """
    Cut kernel with a gaussian smoothed box.
    width and fhwm in pixels
    """
    kernels = kernels.copy()
    ker_width, n_ker = kernels.shape
    # Define box around kernel center
    width = width * n_os
    ker_hwidth = ker_width // 2
    pixel_os = np.arange(-ker_hwidth, ker_hwidth + 1)
    box = np.abs(pixel_os) <= (width / 2)
    # Define gaussian kernel to smooth the box
    x0 = 0.0
    sigma = fwhm2sigma(fwhm * n_os)
    g_ker = gaussians(pixel_os, x0, sigma)
    g_ker = g_ker / np.sum(g_ker)
    # Convolve
    box = np.convolve(box, g_ker, mode='same')
    box = box / box.max()
    # Apply to kernels
    kernels *= box[:, None]
    # Re-norm
    kernels /= kernels.sum(axis=0)
    return kernels


def gaussians(x, x0, sig, amp=None):
    """
    Gaussian function
    """
    # Amplitude term
    if amp is None:
        amp = 1/np.sqrt(2 * np.pi * sig**2)
    return amp * np.exp(-0.5*((x - x0) / sig)**2)


def fwhm2sigma(fwhm):
    """
    Convert a full width half max to a standard deviation, assuming a gaussian
    """
    return fwhm / np.sqrt(8 * np.log(2))



def generate_kernel(output_path, verbose=None, psf_path=None, kernel_semi_width=None):
    #verbose=False

    # This is the function to call to generate the kernels.
    # Others above are utility functions.

    # Generate the spectral optics response of monochromatic light
    # based on all webbpsf oversampled PSFs. That will be used in the 
    # optimal extraction alogirthm by Antoine Darveau-Bernier.
    
    # This script will generate 10 matrices (10 images), each at a different
    # pixel oversampling. Its long axis is the number of monochromatic
    # wavelengths while its short axis is the spectral response kernel, an
    # odd number, flux normalized to 1.
    
    # For simplicity, each matrix is generated one by one, going through
    # all webbpsf PSFs. That may not be cpu optimal but it is conceptually
    # simpler.
    
    # Determine where the webbpsf PSFs reside
    if psf_path == None:
        # Try on Loic's laptop:
        psf_path = '/Users/albert/NIRISS/SOSSpipeline/webbpsf_psfs/'
        if os.path.exists(psf_path) is False:
            # Try guessing assuming github install
            psf_path = './psflibrary/'
    if os.path.exists(psf_path) is False:
        print('Pass the path where webbpsfs are to generate_kernel(psf_path=...)')
        return(False)
    else:
        print(psf_path)
    
    # Call the function that looks on a predefined path for webbpsf fits files
    # This will need editing when ran on a different machine or on a different
    # set of files.
    filename, lambdalist = webbpsf_return_listof(verbose=verbose,psf_path=psf_path)
    # Define how wide the kernel should be in terms of native pixels.
    # That is, this is the half-width of that kernel. The kernel will
    # have size of ksw*2 + one element. in units of native pixels.
    if kernel_semi_width == None: 
        ksw = kernel_semi_width = 7
    else:
        ksw = kernel_semi_width
    # The core_semi_width is a pixel distance along the spatial axis that 
    # defines what region of the trace to keep in the analysis.
    # core_semi_width is different than ksw. core_smi_width pertains to the
    # inital PSF image. If interested by the output size ofthe kernel matrix, then
    # ksw is the variable of interest.
    core_semi_width = 15


    
    osmax = 10 # maximum oversampling to make calculations for
    for i in range(osmax):
        osamp = i+1
        # Initialize the matrix at that oversampling
        kernel_matrix = np.zeros((osamp*2*ksw+1,np.size(lambdalist)))
        wavelength_matrix = np.zeros((osamp*2*ksw+1,np.size(lambdalist)))
        # The name of the output kernel matrix
        matrix_name = '{:}/spectral_kernel_matrix_os_{:}_width_{:}pixels.fits'.format(output_path,osamp,ksw*2+1)
        for n in range(np.size(lambdalist)):
            # Read and apply a rotation to the original oversampled webbpsf image.
            # angle = -3.0 # That is the tilt angle empirically seen in webbpsf's PSFs
            # Actually measure teh tilt on each PSF fits file
            angle = sosstilt.soss_get_tilt(filename[n])
            imagewarped, arg2, dim = webbpsf_read_and_rotate(filename[n], 
                                                           angle,
                                                           verbose=verbose)
            oversampling = arg2
            
            # Crunch along the spatial axis to get 1-d array
            # Reduce the image to the trace profil width (only use pixels where
            # the majority of the signal is.)

            xmin = int(dim/2-core_semi_width*osamp)
            xmax = int(dim/2+core_semi_width*osamp)
            # leave the first 50 and last 50 rows out of the sum as the image 
            # was rotated and they may contain bogus values.
            kernel_full = np.sum(np.array(imagewarped[50:-50,xmin:xmax]), axis=1)
            if verbose is True: plt.plot(kernel_full)
            
            # Determine the barycenter
            x = np.arange(np.size(kernel_full))
            
            cofm = calc_com(x,kernel_full)
            if verbose is True: print(cofm)
            if verbose is True: plt.axvline(x=cofm, color='red')
            # Iterate once on centroid
            cond1 = (x >= cofm - kernel_semi_width*oversampling - 0.5/oversampling)
            cond2 = (x <= cofm + kernel_semi_width*oversampling + 0.5/oversampling)
            ind = cond1 & cond2
            cofm2 = calc_com(x[ind],kernel_full[ind])
            if verbose is True: print(cofm2)
            if verbose is True: plt.axvline(x=cofm2, color='black')
            if verbose is True: plt.xlim((cofm2-kernel_semi_width*oversampling,cofm2+kernel_semi_width*oversampling))
            if verbose is True: plt.show()
            
            # On the PSF oversampled grid:
            x_psf_float = (x - cofm2)/oversampling # pixel position in units of native pixels (x_psf = 0 at center of mass)
            k_psf_float = kernel_full*1.0          # kernel value associated with x_psf, not normalized yet.
            # Sample extremely finely to minimize edge effect error in doing bins later
            x_bin_os = np.linspace(-ksw-1,ksw+1,1000000)
            k_bin_os = np.interp(x_bin_os,x_psf_float,k_psf_float)
            

            # Bin using binned_statistic
            nbins = (2*ksw*osamp)+1
            rangemin = -ksw-0.5/osamp
            rangemax = ksw+0.5/osamp
            k_bin, bin_edges, binnumber = binned_statistic(x_bin_os, k_bin_os, 
                                                           statistic='mean', 
                                                           bins=nbins, 
                                                           range=(rangemin,rangemax))
            x_bin, bin_edges, binnumber = binned_statistic(x_bin_os, x_bin_os, 
                                                           statistic='mean', 
                                                           bins=nbins, 
                                                           range=(rangemin,rangemax))
            # Normalize to one
            k_bin = k_bin-np.min(k_bin) # put lower point at zero
            k_bin = k_bin / (np.sum(k_bin))
            
            # Write to matrix
            kernel_matrix[:,n] = k_bin
            wavelength_matrix[:,n] = lambdalist[n]
            
            # recalculate center of mass for a check only
            com = np.sum(k_bin*x_bin)/np.sum(k_bin)
            if verbose is True: print('com=',com)
            
            # Show the compactness of the flux to see how well the tilt was guessed
            half = (np.size(k_bin)-1)//2
            #print('half = {:}, -os*1 = {:}'.format(half,half-os*1))
            core = np.sum(k_bin[half-osamp*2:half+osamp*2])
            if verbose is True: print('flux in core = {:}'.format(core))
            
            if verbose is True:
                # plot kernel
                plt.plot(x_bin,k_bin*osamp+0.04*n,marker='.',linestyle='-')
                plt.axvline(x=0.0,color='red',zorder=-1)
                plt.xlim((-ksw,ksw))
                plt.xlabel('Native pixels')
                plt.ylabel('Normalized Intensity (+ constant)')
            
            # print the kernel at each wavelength
            print(n, k_bin)

        # Antoine's contribution to taper down the kernel flux near the array edges
        # cut_ker_box will construct a box profile of width 'width' with tapered edges
        # over fwhm/2 on each side (with a gaussian shape) and multiply the kernel_matrix
        # with that box. So arrange such that fwhm + width = kernel_matrix size, i.e. ksw*2+1
        w = np.round(0.6666 * (2*ksw+1))
        fwhm = (2*ksw+1) - w
        print(w,fwhm)
        kernel_matrix = cut_ker_box(kernel_matrix, width=w, n_os=osamp, fwhm=fwhm)

        # Save the rotated PSF on disk
        if False:
            # Initial saving method that I provided Antoien with.
            #
            # Write the index of reddest end the monochromatic kernel
            # That determines which direction the kernel goes with
            # respect to your pixels.
            hdu = fits.PrimaryHDU()
            #hdu.header['REDINDEX'], hdu.header['BLUINDEX'] = 0, np.size(k_bin)-1
            hdu.header['REDINDEX'], hdu.header['BLUINDEX'] = np.size(k_bin)-1, 0
            hdu.data = [kernel_matrix,wavelength_matrix]
            hdu.writeto(matrix_name,overwrite=True)
        if True:
            # New saving method to adopt the new reference file definition
            #
            hdu = fits.PrimaryHDU(kernel_matrix)
            hdu.header['SPECOS'] = osamp
            hdu.header['HALFWIDT'] = ksw
            hdu.header['INDCENTR'] = osamp*ksw+1
            hdu.header['NWAVE'] = 95
            hdu.header['WAVE0'] = 0.5
            hdu.header['INDWAVE0'] = 1
            hdu.header['WAVEN'] = 5.2
            hdu.header['INDWAVEN'] = 95
            hdu.writeto(matrix_name, overwrite=True)

    return


# Call the scripts
generate_kernel('/genesis/jwst/userland-soss/loic_review/', verbose=False,
                psf_path='/genesis/jwst/jwst-ref-soss/monochromatic_PSFs/', kernel_semi_width=None)




