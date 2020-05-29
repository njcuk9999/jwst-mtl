#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:10:35 2018

@author: albert
"""

import numpy as np
from astropy.io import fits
import scipy.ndimage.measurements as measurements
import scipy.signal as signal
from scipy import interpolate
import matplotlib.pylab as plt
from astropy.stats import median_absolute_deviation
from astropy.stats import mad_std
from scipy import interpolate
import scipy.interpolate as interp
from scipy.optimize import curve_fit
from scipy import stats

def image_orient_from_cv(stack):
    # This function rotates and flip a NIRISS image obtained at cryo vacuum
    # test runs (CV1-CV3) to place it in the UdeM coordinate system.
    stack = np.flipud(np.fliplr(np.rot90(stack)))
    return(stack)

def image_orient_from_stsci(stack):
    # This function basically flips along the Y axis an image that is in
    # the STScI DMS pipeline reference coordinate to the UdeM coordinate system.
    stack = np.flipud(stack)
    return(stack)

def image_native_to_DMS(image):
    # This function converts from CV3 to DMS coordinates.
    # x_dms = y_native, y_dms = x_native
    # See: https://outerspace.stsci.edu/pages/viewpage.action?
    # spaceKey=NC&title=NIRISS+Commissioning&preview=/13498420/13499105/NIRISS_subarrays.pdf
    size = np.shape(image)
    # a single 2D image
    ndim = len(size)
    if ndim == 2:
        out = np.swapaxes(image, 0, 1)
    # a cube of images, or more than a cube (assume that last 2 are x and y)
    if ndim == 3:
        out = np.swapaxes(image, 1, 2)
    if ndim == 4:
        out = np.swapaxes(image, 2, 3)
    return(out)

def read_raw_ramp(filename):
    # Reads the fits file name, properly rotate it to the DRS coordinates

    thelist = np.array(filename)
    nfiles = np.size(thelist)

    if nfiles == 1:
        print('Opening the ramp named: {}'.format(filename))
        hdu = fits.open(filename)
        ramp = hdu[0].data + 32768.0
        hdr = hdu[0].header

        # Rotate and swap from native (CV3) to DRS coordinates
        ramp = image_native_to_DMS(ramp)

        return(ramp,hdr)
    else:
        print('Can only read one file, not a list. Stop.')
        return(False)


def get_order2_from_cv(atthesex=None):
    # This function generates a first-pass trace position solution (y vs x)
    # for the SOSS second order trace, based on a deep stack obtained at CV3.
    maskbadpix = True

    a = fits.open('LoicsWork/InputFiles/stack_256_ng3.fits')
    stack = a[0].data
    stack = image_orient_from_cv(stack)

    # Read the bad pixel mask
    a = fits.open('LoicsWork/InputFiles/badpix256.fits')
    badpix = a[0].data
    badpix = image_orient_from_cv(badpix)
    ind = np.where(badpix == 0)
    badpix[ind] = np.nan

    # Dimensions of the subarray
    dimx = np.shape(stack)[1]
    dimy = np.shape(stack)[0]

    # apply bad pixel mask or not
    if maskbadpix == True: stack = stack * badpix

    # Identify the floor level of all 2040 working pixels to subtract it first
    floorlevel = np.nanpercentile(stack,10,axis=0)
    backsubtracted = stack*1
    for i in range(dimx-8):
        backsubtracted[:,i] = stack[:,i] - floorlevel[i]
    floorlevel = np.nanpercentile(backsubtracted,10,axis=0)

    # mask the pixels found at the blue end of both orders defined
    # by two points: x1,y1 = 1985, 4 and x2,y2 = 1436,164
    m = (164.-4.)/(1436.-1985.)
    # b = y - mx
    b = 164. - m*1436.

    # Mask the first order trace from the stack
    rezz = get_order1_from_cv()
    x = np.linspace(0,2047,2048)
    y = np.polyval(rezz,x)
    w = 50
    # Put all pixels in the Order 1 trace to NaN
    mask = stack*1
    row = np.linspace(0,dimy-1,dimy)
    for i in range(np.size(x)):
        ymin = np.max([0,np.int(y[i]-w)])
        ymax = np.int(dimy-1) #np.min([dimy-1,np.round(y[i]+w)])
        mask[ymin:ymax,i] = np.nan

        # First. Maks pixels above the line defined above
        ycurr = b + m*x[i]
        ind = np.where(row >= ycurr)
        mask[ind,i] = np.nan

        # Second filtering of the image to get rid of the order 1 wings
        # sort each column for intensity and mask the brightest ~25 pixels.
        # That shoudl remove the seond order trace. Then fit the background
        # using a polynomial of low order.
        val = mask[:,i]
        ind = np.isfinite(val)
        valreal = val[ind]
        rowreal = row[ind]
        ind = np.argsort(valreal)
        ind = ind[::-1] #reverse array
        valmasked = valreal*1
        valmasked[ind[0:29]] = np.nan
        rowmasked = rowreal*1
        rowmasked[ind[0:29]] = np.nan
        ind = np.isfinite(valmasked)
        valmasked = valmasked[ind]
        rowmasked = rowmasked[ind]
        if np.size(valmasked) >= 20:
            coef = np.polyfit(rowmasked, valmasked, 2)
            back = np.polyval(coef,row)
            mask[:,i] = mask[:,i] - back

    hdu = fits.PrimaryHDU()
    hdu.data = mask
    hdu.writeto('toto.fits', overwrite=True)
    backsubtracted = mask*1

    # Find centroid - first pass, use all pixels in the column
    # Option to mask the bad pixel for CoM calculations
    if maskbadpix == True:
        tracex = []
        tracey = []
        row = np.arange(dimy)
        for i in range(dimx-8):
            val = backsubtracted[:,i+4]
            ind = np.where(np.isfinite(val) == True)
            #print(i, ind)
            thisrow = row[ind]
            thisval = val[ind]
            cx = np.sum(thisrow*thisval)/np.sum(thisval)
            #print(i,cx)
            tracex.append(i+4)
            tracey.append(cx)
    else:
        tracex = []
        tracey = []
        for i in range(dimx-8):
            cx = measurements.center_of_mass(backsubtracted[:,i+4])
            #print(i+4, cx[0])
            tracex.append(i+4)
            tracey.append(cx[0])
    # Adopt these trace values as best
    tracex_best = np.array(tracex)*1
    tracey_best = np.array(tracey)*1

    # Second pass, find centroid on a subset of pixels
    if maskbadpix == True:
        tracex = []
        tracey = []
        row = np.arange(dimy)
        w = 17
        for i in range(dimx-8):
            miny = np.int(np.nanmax( [np.around(tracey_best[i]-w),0] ))
            maxy = np.int(np.nanmin( [np.around(tracey_best[i]+w),dimy-1] ))
            val = backsubtracted[miny:maxy,i+4]
            ind = np.where(np.isfinite(val) == True)
            thisrow = (row[miny:maxy])[ind]
            thisval = val[ind]
            #print(i,miny, maxy, ind, np.size(thisval), np.size(thisrow))
            cx = np.sum(thisrow*thisval)/np.sum(thisval)
            #print(i,cx)
            tracex.append(i+4)
            tracey.append(cx)
    else:
        tracex = []
        tracey = []
        w = 17
        for i in range(dimx-8):
            miny = np.int(np.nanmax( [np.around(tracey_best[i]-w),0] ))
            maxy = np.int(np.nanmin( [np.around(tracey_best[i]+w),np.shape(stack)[0]-1] ))
            cx = measurements.center_of_mass(backsubtracted[miny:maxy,i+4])
            tracex.append(i+4)
            tracey.append(miny + cx[0])
    # Adopt these trace values as best
    tracex_best = np.array(tracex)
    tracey_best = np.array(tracey)


    # Third pass - Fit an analytical function instead
    # The above method is good for 1130 < x < 1775
    ind = np.where((tracex_best >=1130) & (tracex_best <= 1775))
    rezz = np.polyfit(tracex_best[ind], tracey_best[ind], 4)
    tracey_best = np.polyval(rezz,tracex_best)

    if atthesex is None:
        return(rezz)
    else:
        atthesey = np.polyval(rezz,atthesex)
        ind = np.where((atthesex < 1130) | (atthesex > 1775))
        atthesex[ind] = np.nan
        atthesey[ind] = np.nan
        return(atthesex,atthesey)


def get_order3_from_cv(atthesex=None):
    # This function generates a first-pass trace position solution (y vs x)
    # for the SOSS third order trace, based on a deep stack obtained at CV3.

    a = fits.open('LoicsWork/InputFiles/stack_256_ng3.fits')
    stack = a[0].data
    stack = image_orient_from_cv(stack)

    # Read the bad pixel mask
    a = fits.open('LoicsWork/InputFiles/badpix256.fits')
    badpix = a[0].data
    badpix = image_orient_from_cv(badpix)
    ind = np.where(badpix == 0)
    badpix[ind] = np.nan

    # Dimensions of the subarray
    dimx = np.shape(stack)[1]
    dimy = np.shape(stack)[0]

    # apply bad pixel mask
    stack = stack * badpix

    # Create the masks to put to nan the 1st and 2nd order traces
    mask_order1 = mask_trace(stack, order=1, returnmask=True, invertmask=True, nan=True, semiwidth=40)
    mask_order2 = mask_trace(stack, order=2, returnmask=True, invertmask=True, nan=True, semiwidth=25)
    # Apply
    stack = stack * mask_order1 * mask_order2

    # Further mask all pixels above y=128 (if 256 subarray)
    if dimy == 256:
        stack[128:,:] = np.nan

    if False:
        hdu = fits.PrimaryHDU()
        hdu.data = stack
        hdu.writeto('zzz.fits',overwrite=True)

    # By eye estimate of trace centroid, series of straight lines
    linex = [  4,242,462,577,700, 846, 1039] # last data point based on CV3 0.64um laser
    liney = [107,102, 93, 86, 76,  58, 30]
    # extrapolate with a straight line beyond the led data point
    slope = (30-58)/(1039-846)
    xintercept = 1039 + (0 - 30)/slope
    linex.append(xintercept)
    liney.append(0)
    # print(linex), last pixel is 1245.79
    # Interpolate at every native pixel
    cenestx = np.arange(1245)
    cenesty = np.interp(cenestx,linex,liney)
    #plt.scatter(estx,esty,marker='.')

    # Need to model the background very well
    #backimage
    tracex,tracey = [],[]
    datapixel = np.arange(dimy)
    #fig = plt.figure(figsize=(10,15))
    for x in cenestx:
        datavalue = stack[:,x]
        mask = (np.abs(datapixel-cenesty[x]) > 16) & np.isfinite(datavalue)
        if np.size(np.where(mask == True)) > 0:
            p = np.polyfit(datapixel[mask],datavalue[mask],3)
            backmodelvalue = np.polyval(p,datapixel)
            #if x <30:
            #    plt.scatter(datapixel,datavalue+50*x,marker='.',color='black')
            #    plt.scatter(datapixel[~mask],datavalue[~mask]+50*x,marker='.',color='yellow')
            #    plt.plot(datapixel,backmodelvalue+50*x,color='red')
            # do the background model subtraction
            datavalue = datavalue - backmodelvalue
            # Compute the center of mass (centroid) +/- 32 pixels around centroid estimate
            com_mask = np.isfinite(datavalue) & (np.abs(datapixel-cenesty[x]) < 32)
            cofmass = np.sum(datavalue[com_mask]*datapixel[com_mask])/np.sum(datavalue[com_mask])
            if np.abs(cofmass-cenesty[x]) < 5:
                # update trace solution
                tracex.append(x)
                tracey.append(cofmass)
            else:
                # keep the estimated position
                tracex.append(x)
                tracey.append(cenesty[x])
        else:
            # no data to compute centroid
            tracex.append(x)
            tracey.append(np.nan)

    #Complete the arrays to fill between 1245<x<2047
    for i in range(803):
        tracex.append(1245+i)
        tracey.append(np.nan)

    # Adopt these trace values as best
    tracex_best = np.array(tracex)
    tracey_best = np.array(tracey)

    #fig = plt.figure(figsize=(10,10))
    #plt.scatter(tracex_best,tracey_best,marker='.',color='black')

    # Third pass - Fit an analytical function instead
    mask = np.isfinite(tracey_best)
    rezz = np.polyfit(tracex_best[mask], tracey_best[mask], 10)
    tracey_best = np.polyval(rezz,tracex_best)
    nonvalid = tracex_best > 1245
    tracey_best[nonvalid] = np.nan


    #plt.plot(tracex_best,tracey_best,color='red')

    if atthesex is None:
        return(rezz)
    else:
        atthesey = np.polyval(rezz,atthesex)
        ind = np.where((atthesex < 0) | (atthesex > (dimx-1)) |
                       (atthesey < 0) | (atthesey > (dimy-1)) )
        atthesex[ind] = np.nan
        atthesey[ind] = np.nan
        nonvalid = atthesex > 1245
        atthesey[nonvalid] = np.nan
        return(atthesex,atthesey)

    return(tracex,tracey)






def get_order1_from_cv(atthesex=None):
    # This function generates a first-pass trace position solution (y vs x)
    # for the SOSS first order trace, based on a deep stack obtained at CV3.
    maskbadpix = True

    a = fits.open('LoicsWork/InputFiles/stack_256_ng3.fits')
    stack = a[0].data
    stack = image_orient_from_cv(stack)

    # Read the bad pixel mask
    a = fits.open('LoicsWork/InputFiles/badpix256.fits')
    badpix = a[0].data
    badpix = image_orient_from_cv(badpix)
    ind = np.where(badpix == 0)
    badpix[ind] = np.nan

    # Dimensions of the subarray
    dimx = np.shape(stack)[1]
    dimy = np.shape(stack)[0]

    # apply bad pixel mask or not
    if maskbadpix == True: stack = stack * badpix

    # Identify the floor level of all 2040 working pixels to subtract it first
    floorlevel = np.nanpercentile(stack, 10, axis=0)
    backsubtracted = stack*1
    for i in range(dimx-8):
        backsubtracted[:, i] = stack[:, i] - floorlevel[i]
    floorlevel = np.nanpercentile(backsubtracted, 10, axis=0)

    # Find centroid - first pass, use all pixels in the column
    # Option to mask the bad pixel for CoM calculations
    if maskbadpix == True:
        tracex = []
        tracey = []
        row = np.arange(dimy)
        for i in range(dimx - 8):
            val = backsubtracted[:, i + 4]
            ind = np.where(np.isfinite(val) == True)
            #print(i, ind)
            thisrow = row[ind]
            thisval = val[ind]
            cx = np.sum(thisrow * thisval) / np.sum(thisval)
            #print(i,cx)
            tracex.append(i + 4)
            tracey.append(cx)
    else:
        tracex = []
        tracey = []
        for i in range(dimx - 8):
            cx = measurements.center_of_mass(backsubtracted[:, i + 4])
            #print(i+4, cx[0])
            tracex.append(i + 4)
            tracey.append(cx[0])
    # Adopt these trace values as best
    tracex_best = np.array(tracex) * 1
    tracey_best = np.array(tracey) * 1

    # Second pass, find centroid on a subset of pixels
    # From an area around the centroid determined earlier
    if maskbadpix == True:
        tracex = []
        tracey = []
        row = np.arange(dimy)
        w = 17
        for i in range(dimx - 8):
            miny = np.int(np.nanmax( [np.around(tracey_best[i] - w),0] ))
            maxy = np.int(np.nanmin( [np.around(tracey_best[i] + w), dimy - 1] ))
            val = backsubtracted[miny:maxy, i + 4]
            ind = np.where(np.isfinite(val) == True)
            thisrow = (row[miny:maxy])[ind]
            thisval = val[ind]
            cx = np.sum(thisrow * thisval) / np.sum(thisval)
            tracex.append(i + 4)
            tracey.append(cx)

    else:
        tracex = []
        tracey = []
        w = 17
        for i in range(dimx - 8):
            miny = np.int(np.nanmax( [np.around(tracey_best[i] - w), 0] ))
            maxy = np.int(np.nanmin( [np.around(tracey_best[i] + w), np.shape(stack)[0] - 1] ))
            cx = measurements.center_of_mass(backsubtracted[miny:maxy, i + 4])
            tracex.append(i + 4)
            tracey.append(miny + cx[0])
    # Adopt these trace values as best
    tracex_best = np.array(tracex)
    tracey_best = np.array(tracey)

    # Third pass - Fit an analytical function
    # Such that y-values can be passed at the required x-values
    rezz = np.polyfit(tracex_best, tracey_best, 10)
    tracey_best = np.polyval(rezz, tracex_best)

    if atthesex is None:
        return(rezz)
    else:
        atthesey = np.polyval(rezz, atthesex)
        ind = np.where((atthesex < 0) | (atthesex > (dimx-1)) |
                       (atthesey < 0) | (atthesey > (dimy-1)) )
        atthesex[ind] = np.nan
        atthesey[ind] = np.nan
        return(atthesex, atthesey)

def old_get_order1_from_cv(polycoeff=None):
    # kept because it has sections to fit a median filter and spline

    # This function generates a first-pass trace position solution (y vs x)
    # for the SOSS first order trace, based on a deep stack obtained at CV3.
    a = fits.open('LoicsWork/InputFiles/stack_256_ng3.fits')
    stack = a[0].data
    stack = image_orient_from_cv(stack)

    # Identify the floor level of all 2040 working pixels to subtract it first
    floorlevel = np.percentile(stack,10,axis=0)
    backsubtracted = stack*1
    for i in range(2048):
        backsubtracted[:,i] = stack[:,i] - floorlevel[i]
        floorlevel = np.percentile(backsubtracted,10,axis=0)
    # Find centroid - first pass, use all pixels in the column
    tracex = []
    tracey = []
    for i in range(2040):
        cx = measurements.center_of_mass(backsubtracted[:,i+4])
        tracex.append(i+4)
        tracey.append(cx[0])
    # Second pass, apply a median filter to the first trace solution to remove outliers
    tracey2 = np.array(signal.medfilt(tracey, 9))
    tracex2 = tracex * 1
    # Third pass, find centroid on a subset of pixels
    tracex3 = []
    tracey3 = []
    for i in range(2040):
        w = 17
        miny = np.int(np.max( [np.around(tracey2[i]-w),0] ))
        maxy = np.int(np.min( [np.around(tracey2[i]+w),np.shape(stack)[0]-1] ))
        cx = measurements.center_of_mass(backsubtracted[miny:maxy,i+4])
        tracex3.append(i+4)
        tracey3.append(miny + cx[0])
    tracex3 = np.array(tracex3)
    tracey3 = np.array(tracey3)
    # Fourth pass, apply a median filter to smooth the curve
    tracey4 = signal.medfilt(tracey3, 11)
    tracex4 = tracex3*1
    # Fifth pass, interpolate using a spline to get a continuous and smooth curve, oversampled
    os = 10
    tracex5 = np.arange(1, 2048,1.0/os)
    s = interpolate.InterpolatedUnivariateSpline(tracex4, tracey4)
    tracey5 = s(tracex5)
    # Fit an analytical function instead
    rezz = np.polyfit(tracex4, tracey4, 10)
    print(rezz)
    tracex6 = tracex5*1.0
    tracey6 = np.polyval(rezz,tracex6)

    if polycoeff is True:
        return(rezz)
    else:
        return(tracex6,tracey6)


def wavelength_to_spectral_pixel(wavelength, order=None):
    # converts a wavelength (in nm) to a pixel position
    # Use the soltion obtained at CV3 from lasers at a few wavelengths

    if order == 1:
        rezz = [-2.03489e-5,1.09890,-905.219]
    if order == 2:
        rezz = [2.14318,-976.566]

    return(2048-np.polyval(rezz,wavelength))

def spectral_pixel_to_wavelength(pixelin, order=None):
    # Converts a pixel to a wavelength
    # Inverse of above

    pixel = 2048-pixelin

    if order == 1:
        rezz = [-2.03489e-5,1.09890,-905.219]
        wavelength = (-rezz[1] + np.sqrt(rezz[1]*rezz[1] - 4*(rezz[2]-pixel)*rezz[0]))/(2*rezz[0])
    if order == 2:
        rezz = [2.14318,-976.566]
        wavelength = (pixel - rezz[1])/rezz[0]

    return(wavelength)


def extrapolate_order2(atthesex=None, attheselambda=None):
    # If you want to get the trace position of order 2,
    # this is the equivalent of get_order1_from_cv for order 2.
    #
    # Here, the idea is to use the already calibrated x,y vs lambda
    # common to both orders, supplemented by the LED monochromatic PSFs
    # observed at CV3. At each monochromatic wavelength, the two traces
    # position form a vector characterized by its length and slope.
    # We use the vectors formed from the LED measurements to generate
    # an extrapolation of the slope vs. lambda and length vs. lambda.

    # The things we input to the function
    doPlot = True
    #doPlot = False

    lba_final = np.linspace(550,2850,3301)
    if atthesex is not None:
        print('Optional keyword atthesex invoked')
        lba_final = spectral_pixel_to_wavelength(atthesex, order=2)
    if attheselambda is not None:
        print('Optional keyword attheselambda invoked')
        lba_final = attheselambda*1.0

    # First generate the wavelengths that are common to both orders
    lba_nocont = np.linspace(550,2850,2301)
    x1_nocont = wavelength_to_spectral_pixel(lba_nocont,order=1)
    x2_nocont = wavelength_to_spectral_pixel(lba_nocont,order=2)
    x1_nocont,y1_nocont = get_order1_from_cv(atthesex=x1_nocont)
    x2_nocont,y2_nocont = get_order2_from_cv(atthesex=x2_nocont)
    # Remove the wavelengths that have no valid solution yet in order 2
    indcommon = np.isfinite(x2_nocont) & np.isfinite(x1_nocont)
    x2_common = x2_nocont[indcommon]
    y2_common = y2_nocont[indcommon]
    x1_common = x1_nocont[indcommon]
    y1_common = y1_nocont[indcommon]
    lba_common = lba_nocont[indcommon]

    # Compute the slope and length of the O1-->O2 vector for each wavelength
    length_nocont = np.sqrt((y2_nocont-y1_nocont)**2+(x2_nocont-x1_nocont)**2)
    slope_nocont = ((y2_nocont-y1_nocont)/(x2_nocont-x1_nocont))
    length_common = length_nocont[indcommon]
    slope_common = slope_nocont[indcommon]

    # From CV3 lamps, we got two slopes/lengths to use as anchor
    # Arguably, the slope may be off if there is non-repeatability
    # (by 0.16 degree max). But length should remain accurate.
    lba_led = np.array([1062.62, 1308.38])
    x1_led = np.array([1809,1549])
    y1_led = np.array([210,218])
    x2_led = np.array([740,225])
    y2_led = np.array([161,161])
    length_led = np.sqrt((y2_led-y1_led)**2+(x2_led-x1_led)**2)
    slope_led = ((y2_led-y1_led)/(x2_led-x1_led))

    if doPlot == True:
        fig = plt.figure(figsize=(14,14))
        grid = plt.GridSpec(5, 1, hspace=0.35, wspace=0.2)
        fig1 = fig.add_subplot(grid[0,0])
        fig2 = fig.add_subplot(grid[1,0])
        fig3 = fig.add_subplot(grid[2,0])
        fig4 = fig.add_subplot(grid[3,0])
        fig5 = fig.add_subplot(grid[4,0])

        fig1.plot(x1_nocont,y1_nocont,color='red', lw=5, ls='dotted',label='Order 1 - CV3 no contamination')
        fig1.plot(x2_nocont,y2_nocont,color='blue',lw=5, ls='dotted',label='Order 2 - CV3 no contamination')
        fig1.plot(x1_common,y1_common,color='red', lw=5, label='Order 1 - CV3 no contamination')
        fig1.plot(x2_common,y2_common,color='blue',lw=5, label='Order 2 - CV3 no contamination')
        fig1.scatter(x1_led,y1_led, color='red',marker='o', label='Order 1 - CV3 LED')
        fig1.scatter(x2_led,y2_led, color='blue',marker='o', label='Order 2 - CV3 LED')
        fig1.set_xlim((0,2247))
        fig1.set_ylim((0,255))
        fig1.set_xlabel('Pixel X')
        fig1.set_ylabel('Pixel Y')
        fig1.set_title('Trace position from CV3 - LED and uncontaminated parts')
        fig1.legend()


    # For the "length" component, the relation is pretty linear. So simply fit
    # the measured points.
    length_tmp = np.concatenate((length_common,length_led))
    lba_tmp = np.concatenate((lba_common,lba_led))
    c = np.polyfit(lba_tmp,length_tmp,1)
    length_final = np.polyval(c,lba_final)
    if doPlot == True:
        fig2.plot(lba_common, length_common,color='green',lw=5,label='Common uncontaminated CV3 trace')
        fig2.scatter(lba_led, length_led, marker='o', color='green',label='CV3 LED')
        fig2.plot(lba_final, length_final, color='black', linewidth=1,label='Fit')
        fig2.set_title('Length of matching pairs vs Wavelength')
        fig2.set_xlabel('Wavelength (nm)')
        fig2.set_ylabel('Pair Length (pixels)')
        fig2.set_xlim((500,1500))
        fig2.set_ylim((0,2000))
        fig2.legend()

    # For the "slope" component, the function has a shape requiring modelling.
    # Build tightly sampled arrays of the slope component. Assume a simple
    # linear trend between the last "nocont" measurement and both LED
    # measurements. Extrapolate that last fit to the end of the O2 trace.
    lba_lininterp = np.linspace(lba_common[-1],1500,100) # the number of samples control how tightly the fit will stick to this, so be gentle.
    # part 1 - between last good point and the first LED measurement
    m = (slope_led[0]-slope_common[-1])/(lba_led[0]-lba_common[-1])
    b = slope_common[-1] - m*lba_common[-1]
    ind = np.where((lba_lininterp >= lba_common[-1]) & (lba_lininterp <= lba_led[0]))
    slope_lininterp_1 = m * lba_lininterp[ind] + b
    # part 2 - from the first LED measurement to end
    m = (slope_led[1]-slope_led[0])/(lba_led[1]-lba_led[0])
    b = slope_led[0] - m*lba_led[0]
    ind = np.where(lba_lininterp > lba_led[0])
    slope_lininterp_2 = m * lba_lininterp[ind] + b
    # Stitch the two lines together
    slope_lininterp = np.concatenate((slope_lininterp_1,slope_lininterp_2))
    # Now stitch the "nocont" solution to the tightly linearly interpolated
    # solution of the second order trace extension.
    slope_tmp = np.concatenate((slope_common,slope_lininterp))
    lba_tmp = np.concatenate((lba_common,lba_lininterp))
    # Finally do the fit
    c = np.polyfit(lba_tmp,slope_tmp,6)
    slope_final = np.polyval(c,lba_final)
    if doPlot == True:
        fig3.plot(lba_common, slope_common, color='orange',lw=5,label='Uncontaminated CV3 trace')
        fig3.scatter(lba_led, slope_led, marker='o', color='orange',label='CV3 LED')
        fig3.plot(lba_lininterp, slope_lininterp, color='orange', linewidth=3,ls='dashed',label='Interpolation before final fit')
        fig3.plot(lba_final, slope_final, color='black', linewidth=1,label='Fit')
        fig3.set_title('Slope of matching pairs vs Wavelength')
        fig3.set_xlabel('Wavelength (nm)')
        fig3.set_ylabel('Pair Slope')
        fig3.set_xlim((500,1500))
        fig3.set_ylim((0.04,0.08))
        fig3.legend()

    # Finally compute the x2,y2 from the slope/length vector
    x2_final,y2_final = from_vector_to_x2y2(lba_final,slope_final,length_final)
    ind = np.where((x2_final >= 0) & (y2_final >= 0))
    x1_final = wavelength_to_spectral_pixel(lba_final,order=1)
    x1_final,y1_final = get_order1_from_cv(atthesex=x1_final)
    if doPlot == True:
        fig4.plot(x1_nocont,y1_nocont,color='red', lw=5,label='Order 1 - CV3 no contamination')
        fig4.plot(x2_nocont,y2_nocont,color='blue',lw=5,label='Order 2 - CV3 no contamination')
        fig4.scatter(x1_led,y1_led, color='red',marker='o', label='Order 1 - CV3 LED')
        fig4.scatter(x2_led,y2_led, color='blue',marker='o', label='Order 2 - CV3 LED')
        fig4.plot(x2_final[ind],y2_final[ind],color='black',label='Order 2 - Extended model')
        fig4.plot(x1_final,y1_final,color='black',label='Order 1 - CV3 no contamination')
        fig4.set_xlim((0,2047))
        fig4.set_ylim((0,255))
        fig4.set_xlabel('Pixel X')
        fig4.set_ylabel('Pixel Y')
        fig4.set_title('Trace position with extrapolation from matching O1/O2 pairs')
        fig4.legend()


    # The fit of slope/length and x2,y2 applies to only the longer than ~900 nm
    # domain. So, for x2,y2, return the initial values for lambda <900 nm
    # and return the fitted values for lambda > 900 nm.
    ind_1stguess = np.where(lba_final <= lba_common[-1])
    x2_tmp = wavelength_to_spectral_pixel(lba_final[ind_1stguess],order=2)
    x2_tmp,y2_tmp = get_order2_from_cv(atthesex=x2_tmp)
    ind_2ndguess = np.where(lba_final > lba_common[-1])
    x2_final = np.concatenate((x2_tmp,x2_final[ind_2ndguess]))
    y2_final = np.concatenate((y2_tmp,y2_final[ind_2ndguess]))

    # sort along x
    ind = np.argsort(x2_final)
    x2_final = x2_final[ind]
    y2_final = y2_final[ind]
    lba_final = lba_final[ind]

    # Further need to resample every column pixel (along x) because
    # as it stands, x are not linearly spaced integers.
    x2_integer = np.arange(2048)
    y2_integer = np.interp(x2_integer, x2_final, y2_final)
    lba_integer = np.interp(x2_integer, x2_final, lba_final)

    # On the blue end, past x=1775, y2 is all nans. Extrapolate using
    # a linear trend, instead.
    cond = np.isfinite(y2_integer) & (x2_integer >=1725)
    p = np.polyfit(x2_integer[cond],y2_integer[cond],1)
    indef = (np.isfinite(y2_integer) == False) & (x2_integer >=1725)
    y2_integer[indef] = np.polyval(p,x2_integer[indef])

    # OK, happy with the calibration, let's call it final, for real:
    x2_final = x2_integer*1
    y2_final = y2_integer*1
    lba_final = lba_integer*1

    print('check vals:',np.min(lba_final), np.max(lba_final))
    if doPlot is True:
        fig5.plot(x2_final,y2_final+10,color='black',label = 'Combined solution (offset)')
        fig5.set_xlim((0,2047))
        fig5.set_ylim((0,255))
        fig5.set_xlabel('Pixel X')
        fig5.set_ylabel('Pixel Y')
        fig5.set_title('Order 2 trace position - final')
        fig5.plot(x2_tmp,y2_tmp, color='pink',label='Uncontaminated trace solution')
        fig5.plot(x2_final[ind_2ndguess],y2_final[ind_2ndguess],color='grey',label='Extrapolated solution')
        fig5.legend()

    if doPlot is True:
        fig.savefig('WavelengthCalibration_Order2Extrapolation.pdf')
        #plt.show()
        plt.close()

    return(x2_final,y2_final,lba_final)

def from_vector_to_x2y2(lba,slope,length):
    # This function is usefull for building the order 2 trace solution.
    # Assuming we want the position x2,y2 of trace order 2
    # simply from the slope and length at a given lambda. Here
    # is what we need to do:

    print(np.size(lba),np.size(slope),np.size(length))

    x1 = wavelength_to_spectral_pixel(lba,order=1)
    x1,y1 = get_order1_from_cv(atthesex=x1)

    print(np.size(x1),np.size(y1))

    # for the trace solution using slopes and lengths (for order 2)
    #x1,y1 is the position of a given lambda in order 1
    #x2,y2 is the trace position for order 2 at the same lambda

    # Two solutions exist from merging these 2 equations:
    # slope = (y2-y1)/(x2-x1)
    # length^2 = (y2-y1)^2 + (x2-x1)^2
    #
    a = 1
    b = -2*x1
    c = x1**2 - length**2/(slope**2+1)

    #x2 = (-b + np.sqrt(b**2 - 4 * a * c))/(2*a)
    # or
    x2 = (-b - np.sqrt(b**2 - 4 * a * c))/(2*a)
    # then this follows:
    y2 = (x2-x1)*slope + y1

    ind = np.where(lba < 1600)

    #plt.scatter(x2[ind],y2[ind])
    #plt.show()

    return(x2,y2)

def show_trace_solution():

    fig = plt.figure(figsize=(15, 9))
    a = fits.open('LoicsWork/InputFiles/stack_256_ng3.fits')
    stack = a[0].data
    stack = image_orient_from_cv(stack)
    im = stack-10
    im = np.log10(im)
    plt.imshow(im,origin='lower',aspect=6, cmap='hsv')
    #plt.imshow(im,origin='lower',aspect=3,vmin=1.5,vmax=3.5, cmap='hsv')
    #plt.imshow(stack,origin='lower')
    plt.xlim((0,2047))
    plt.ylim((0,255))

    x2,y2,lba2 = extrapolate_order2(attheselambda=np.linspace(500,1500,1001))

    x1,y1 = get_order1_from_cv(atthesex=np.linspace(0,2047,2048))
    lba1 = spectral_pixel_to_wavelength(x1, order=1)

    plt.plot(x1,y1,color='black',lw=3,zorder=10)
    plt.plot(x2,y2,color='black',lw=3,zorder=10)
    fig.savefig('/Users/albert/NIRISS/simuSOSS/revival/WavelengthCalibration_After2ndOrderExtrapolation.pdf')
    #fig.show()
    #plt.close()


def oversample_slice(slice_native, osx, osy, scaleflux=None):
    # function called by extract_trace_order1_from_cv
    if np.size(np.shape(slice_native)) == 1 :
        #dimx = 1 # assume that it's a column, not a row
        dimy = np.shape(slice_native)
        slice_os = np.zeros(dimy*osy)
        #slice_os = np.zeros((dimy*osy, dimx*osx))
        for i in range(dimy):
            slice_os[i*osy:(i+1)*osy] = slice_native[i]

    if scaleflux is True:
        slice_os = slice_os / (osx*osy)

    return(slice_os)


def extract_trace_order1_from_cv(normalize=None):
    # This function uses the deep stack of the CV3 rehearsal to output the
    # rectified trace of order 1.

    # Define a model of the trace, 256 pixels wide, 2048 pixels high, one for each order
    a = fits.open('LoicsWork/InputFiles/stack_256_ng3.fits')
    deepstack = a[0].data
    deepstack = image_orient_from_cv(deepstack)
    deepstack = np.log10(deepstack)

    osx = 1
    osy = 1
    halfwidthy = 30
    halfcorey = 13
    straightdimy = 2*halfwidthy*osy+1
    straightdimx = 2048*osx
    resampling = True

    trace1 = np.zeros((straightdimy,straightdimx))
    trace2 = np.zeros((straightdimy,straightdimx))
    trace1n = np.zeros((straightdimy,straightdimx))
    trace2n = np.zeros((straightdimy,straightdimx))

    x = np.linspace(0,2047,2048)
    x1,y1 = get_order1_from_cv(atthesex=x)
    lba1 = spectral_pixel_to_wavelength(x1, order=1)
    x2,y2,lba2 = extrapolate_order2(atthesex=x)

    print('osx,osy,halfwidthy=',osx,osy,halfwidthy)

    fig = plt.figure(figsize=(15,6))
    for i in range(np.size(x1)):
        t = deepstack[:,i]
        # blow it by osx,osy
        tos = oversample_slice(t,osx,osy,scaleflux=True)
        # Project trace to a new straightened 2D map, either by resampling linearly along y, or not.
        if resampling == True:
            xnati,ynati,xshift,yshift = column_extract_and_interpolate(t,y1[i],halfwidthy,osy)
            for subpix in range(osx):
                trace1[:,i*osx+subpix] = yshift

        if resampling == False:
            yctr = int(np.round((y1[i])*osy+1.0))
            ymin = np.max([yctr-halfwidthy*osy,0])
            ymax = np.min([256*osy-1,yctr+halfwidthy*osy])
            #print(yctr,ymin,ymax)
            trace1[:,i*osx:(i+1)*osx] = tos[ymin:ymax+1,:]

        # now for the purpose of normalizing flux in the core
        yctr = halfwidthy*osy
        ymin_core = np.max([yctr-halfcorey*osy,0])
        ymax_core = np.min([straightdimy-1,yctr+halfcorey*osy])
        trace1n[:,i*osx:(i+1)*osx] = trace1[:,i*osx:(i+1)*osx] / np.nansum(trace1[ymin_core:ymax_core,i*osx:(i+1)*osx])

    if normalize == True:
        return(trace1n)
    else:
        return(trace1)

def column_extract_and_interpolate(columnvalue,pixelcentroid,halfwidth,oversampling):
    # input a native pixel full column of 256 pixels
    # input centroid of the trace (full decimal precision) in native pixels units
    # input pixel is the array from 0 to 255 in steps of 1
    # halfwidth is the number of pixels kept for the output in units of native pixels

    centroid_integer = round(pixelcentroid)
    centroid_fractional = pixelcentroid - centroid_integer

    x = np.linspace(0,len(columnvalue),len(columnvalue),endpoint=False)
    y = columnvalue*1.0

    # pixel values at which we want to resample
    num = 2*halfwidth*oversampling+1
    xi = np.linspace(-halfwidth,halfwidth,num) + pixelcentroid
    # perform resampling
    yi = np.interp(xi, x, y)

    return(x, y, xi, yi)




def profile_spatial_position(x, y, cenx=None, ceny=None, tilt=None):
    # x,y are the pixel position of the pixel for which we ask
    # what is its profile spatial position. It will be close to
    # the passed y minus centroid_y but with a correction that depends on the
    # monochromatic tilt and y centroid slope.

    # Assumed monochromatic tilt in degrees
    if tilt == None:
        MONOCHROMATIC_TILT = 0.4
    else:
        MONOCHROMATIC_TILT = tilt

    # Read the trace centroid position
    if cenx is None or ceny is None:
        cenx,ceny = get_order1_from_cv(atthesex=np.linspace(0,2047,2048))

    # Slope of the trace y-centroid
    alpha = np.arctan((ceny[x+1]-ceny[x])/(x+1-x))

    # Tilt of the monochromatic wavelength at that x,y position
    theta = np.deg2rad(MONOCHROMATIC_TILT) # should be passed as input of function instead

    # One side of the triangle
    side_a = y - ceny[x]

    # Corresponding opposite angle
    angle_a = np.pi/2 - theta + alpha

    # Angle opposite of the trace profile
    angle_c = np.pi/2 - alpha

    # Trace profile position (in units of pixels, close to y-ceny)
    # Use the sinus law
    side_c = side_a * np.sin(angle_c) / np.sin(angle_a)

    # Angle b (the centroid slope with respect to detector pixels)
    angle_b = theta

    # Trace spectral position (in units of pixels, close to x-cenx)
    # Use the sine law
    side_b = side_a * np.sin(angle_b) / np.sin(angle_a)

    return(side_b,side_c)


def trace_badpix_interpolation(traceref, badpix, normalizeref=None):
    # Goal is to interpolate the bad pixels on a 2D SOSS image
    # We assume: that we have a 2D map of all bad pixels, a
    # trace y centroid position and a tilt solution as a function of x.
    # We use the deep stack and properly interpolate it.

    return(1)

def rebin_resample(x,y,newx):
    # Assuming that the new x is regularly sampled array
    delta = np.median(newx[1:]-newx[0:-1])
    newy = np.zeros(len(newx))
    newy_meddev = np.zeros(len(newx))
    newy_n = np.zeros(len(newx))

    for i in range(len(newx)):
        ind = np.where((x > newx[i]-0.5*delta) & (x <= newx[i]+0.5*delta))
        # Do stats on data points within that bin
        xdata, ydata = x[ind], y[ind]
        newy_n[i] = len(xdata)
        newy[i] = np.nanmedian(ydata)
        #newy_meddev[i] = median_absolute_deviation(ydata,ignore_nan=True)
        newy_meddev[i] = mad_std(ydata,ignore_nan=True)

    return(newy,newy_meddev,newy_n)

def rebin_resample_spec(x,y,yweight,newx):
    # Assuming that the new x is regularly sampled array
    delta = np.median(newx[1:]-newx[0:-1])
    newy = np.zeros(len(newx))
    newy_meddev = np.zeros(len(newx))
    newy_n = np.zeros(len(newx))

    for i in range(len(newx)):
        ind = np.where((x > newx[i]-0.5*delta) & (x <= newx[i]+0.5*delta))
        # Do stats on data points within that bin
        xdata, ydata, ywdata = x[ind], y[ind], yweight[ind]
        notnan = ~np.isnan(ydata)
        #print('notnan:',notnan)
        newy_n[i] = len(ydata[notnan])
        newy[i] = np.average(ydata[notnan],weights=ywdata[notnan])
        #newy_meddev[i] = mad_std(ydata,ignore_nan=True)
        variance = np.average((ydata[notnan]-newy[i])**2, weights=ywdata[notnan])
        newy_meddev[i] = np.sqrt(variance)

    return(newy,newy_meddev,newy_n)

def uspec(u,v,flux,modelv,modelvflux):
    # Given the u,v positions of pixels and their flux, and given
    # a model of the trace profile along v, modelv and modelvflux,
    # construct the spectrum along the u direction.

    # Normalize the flux of the trace profile to 1
    # (eventually add bounds to the integration)
    modelvflux = modelvflux / np.nansum(modelvflux)

    # Divide the flux at v by the trace profile to produce a flat
    # flux at all v.
    #print('check-1',modelv, modelvflux)
    modelvflux_at_v = np.interp(v, modelv, modelvflux)
    #print('check',modelvflux_at_v)
    flux_flat = flux / modelvflux_at_v

    # Integrate the flux along the v direction, weighting according
    # to the model profile
    # This does not make sense: Int = np.sum(flux_flat*flux)

    # Compress along the v axis and rebin in u to get spectrum along u

    ubin_min = np.ceil(np.min(u))
    ubin_max = np.floor(np.max(u))
    ubin = np.arange(ubin_max-ubin_min+1)+ubin_min

    #fig = plt.figure(figsize=(10,6))
    #plt.scatter(u, flux_flat,marker='.')
    #plt.show()

    fluxbin,fluxbin_dev,fluxbin_n  = rebin_resample_spec(
            u,flux_flat,modelvflux_at_v,ubin)

    return(ubin,fluxbin,fluxbin_dev,fluxbin_n)



def interpolate_and_reject(xin, yin, xnew, nsig, kind=None):
    from scipy import interpolate
    from astropy.stats import median_absolute_deviation

    x, y = xin*1, yin*1
    sor = np.argsort(x) # Sort the array (required by interpolate.interp1d)
    f = interpolate.interp1d(np.array(x[sor]), np.array(y[sor]), kind=kind)
    xfit = x*1
    yfit = f(xfit)

    fig = plt.figure(figsize=(14,10))
    plt.plot(xfit, yfit, color='black')
    plt.scatter(x,y, marker='.')

    m = (yfit-yin)/yfit
    print(m)
    #dev = median_absolute_deviation(m)
    dev = np.std(m)
    print('iter 1, ndata, dev:',np.size(y), dev)

    for it in range(3):
        ind = np.where(m <=nsig*dev)
        x, y = x[ind]*1, y[ind]*1
        f = interpolate.interp1d(np.array(x), np.array(y), kind=kind)
        yfit = f(xfit)
        m = (yfit-yin)/yfit
        #dev = median_absolute_deviation(m)
        dev = np.std(m)
        print('iter n, ndata, dev:',np.size(y), dev)

    return(f(xnew))


def model_this_section(uall,vall,fluxall,urequest,vrequest,interpol=None, kind=None, reject=None):

    from scipy import interpolate
    # should have those passed in the first place
    #u = np.array(u)
    #v = np.array(v)
    #flux = np.array(flux)

    # Indices of valid data
    ind = np.where((np.isfinite(fluxall) == True) & (np.isfinite(vall) == True) &
                   (np.isfinite(uall) == True))

    # Start with a clean data sample
    u = uall[ind]
    v = vall[ind]
    flux = fluxall[ind]

    #print('urequest',urequest)
    #print('vrequest',vrequest)
    #print('u',u)
    #print('v',v)
    #print('flux',flux)

    # Rebin. Make a model by binning the points in pixel-size bins
    if interpol == True:
        if kind !=None:
            sor = np.argsort(v) # Sort the array (required by interpolate.interp1d)
            #print(v[sor],flux[sor])
            #print(np.shape(v[sor]))
            #print(vrequest)
            #print(np.shape(flux[sor]))
            if reject == True:
                print('reject')
                model_vf1 = interpolate_and_reject(np.array(v[sor]),np.array(flux[sor]),
                                                   vrequest, 3.0, kind=kind)
            else:
                print('interp1d higher')
                f = interpolate.interp1d(np.array(v[sor]), np.array(flux[sor]), kind=kind)
                #print('cubic')
                model_vf1 = f(vrequest)
        else:
            print('interpolate linear')
            model_vf1 = np.interp(vrequest, v, flux)
    else:
        print('integer bin')
        model_vf1,model_vf1_dev,model_vf1_n = rebin_resample(v, flux, vrequest)

    # Now use the model of the trace to assign a profile flux to each pixel
    # from its v position
    modeltrace_at_v = np.interp(v, vrequest, model_vf1)
    #print('modeltrace_at_v',modeltrace_at_v)


    # Normalize the trace profile flux
    modeltrace_at_v = modeltrace_at_v/np.nansum(modeltrace_at_v)

    # Normalize all pixels with that profile to generate a new spectrum, then
    # model that spectrum and keep it for next iteration
    # Divide each pixel by its spatial profile position flux
    model_flux = flux / modeltrace_at_v

    # Make an re-sampled model of the spectrum
    #print('u[ind]',u[ind])
    #print('v[ind]',v[ind])
    #print('flux[ind]',flux[ind])
    #print('modeltrace_at_v',modeltrace_at_v)
    #print('model_flux[ind]',model_flux)
    #print('model_vf1',model_vf1)

    model_u1, model_uf1, model_uf1_dev, model_uf1_n = uspec(u,v,model_flux,vrequest,model_vf1)

    # Go to iteration 2 - construct a better profile

    # Collapse to get the trace
    # Refit a spatial trace profile
    modelspec_at_u = np.interp(u, model_u1, model_uf1)
    model_flux = flux / modelspec_at_u
    if interpol == True:
        if kind != None:
            sor = np.argsort(v) # Sort the array (required by interpolate.interp1d)
            if reject == True:
                model_vf2 = interpolate_and_reject(np.array(v[sor]),np.array(model_flux[sor]),
                                                   vrequest, 3.0, kind=kind)
            else:
                f = interpolate.interp1d(np.array(v[sor]), np.array(model_flux[sor]), kind=kind)
                model_vf2 = f(vrequest)
        else:
            model_vf2 = np.interp(vrequest, v, model_flux)
    else:
        model_vf2,model_vf2_dev,model_vf2_n = rebin_resample(v, model_flux, vrequest)

    # Collapse in the other axis to get the spectrum
    model_u2, model_uf2, model_uf2_dev, model_uf2_n = uspec(u,v,flux,vrequest,model_vf2)

    if False:
        # Plot the spectrum
        fig = plt.figure(figsize=(14,7))
        x,y,dy = urequest,model_uf1/np.median(model_uf1),model_uf1_dev/np.median(model_uf1)
        plt.scatter(x,y,label='Spectrum - iteration 1')
        x,y,dy = urequest,model_uf2/np.median(model_uf2),model_uf2_dev/np.median(model_uf2)
        plt.errorbar(x,y,dy,marker='o',color='orange',label='Spectrum - iteration 2')
        plt.title('Spectrum')
        plt.legend()
        # Plot the Trace profile
        fig = plt.figure(figsize=(14,7))
        plt.scatter(vrequest,model_vf1/np.nanmax(model_vf1),label='Trace Profile - iteration 1')
        plt.scatter(vrequest,model_vf2/np.nanmax(model_vf2),label='Trace Profile - iteration 2')
        plt.title('Trace Profile')
        plt.legend()



    return(urequest, model_uf2, vrequest, model_vf2)



def model_trace_from_stack(image, order=None, cenx=None, ceny=None,
                           semiwidthx=None, semiwidthy=None, osdimy=None,
                           saveimage=None, setwingzero=None,
                           subtractbackground=None,bg_semiwidth=None,
                           bg_polyorder=None, debug=None):

    # Construct a model of the trace for all x positions in order 1, 2 or 3
    dimx, dimy = 2048, 256
    if semiwidthx == None: semiwidthx = 7
    if semiwidthy == None: semiwidthy = 32
    if osdimy == None: osdimy = 5

    # Read the trace centroid position
    if (cenx == None) or (ceny == None):
        # Read the best available trace centroid solution
        if (order == 1) or (order == None):
            pixels = np.linspace(0,dimx-1,dimx)
            cenx,ceny = get_order1_from_cv(atthesex=pixels)
        if order == 2:
            pixels = np.linspace(0,dimx-1,dimx)
            cenx,ceny,lba = extrapolate_order2(atthesex=pixels)
            #cenx,ceny = get_order2_from_cv(atthesex=pixels)
            #for ccc in range(len(pixels)):
            #    print(pixels[ccc],cenx[ccc],ceny[ccc])
        if order == 3:
            pixels = np.linspace(0,dimx-1,dimx)
            cenx,ceny = get_order3_from_cv(atthesex=pixels)
            #cenx,ceny = get_order2_from_cv(atthesex=pixels)

    model_trace = np.zeros((osdimy*2*semiwidthy+1,dimx))*np.nan

    # Only make a trace where the trace is defined
    # (cenx should not contain NaNs but who knows?)
    ind = np.isfinite(cenx) & np.isfinite(ceny)
    cenx,ceny = cenx[ind],ceny[ind]

    # Background subtraction section
    working_image = image
    if subtractbackground == True:
        # plot each column if requested
        if debug == True:
            fig = plt.figure(figsize=(8,15))
            adumax = np.nanpercentile(working_image,95)
            adumin = np.nanpercentile(working_image,2)
            adustep = (adumax-adumin)*1.0
            nstep = 0
            print('adumin={}, adumax={}, adustep={}'.format(adumin,adumax,adustep))

        # Set default values for:
        # 1) the width in native pixels beyond which pixels are considered
        #    background pixels,
        # 2) the polynomial function order for modelling the background.
        if bg_semiwidth == None: bg_semiwidth = 16
        if bg_polyorder == None: bg_polyorder = 3

        # Create a copy of the image
        image_bgsub = image*1

        # One native pixel column at a time, mask trace and model background.
        colpix = np.arange(dimy)
        for zzz in cenx:
            xo = int(zzz)
            yo = int(np.floor(ceny[xo]))
            #print('xo={}, yo={}'.format(xo,yo))
            colval = image[:,xo]
            # Keep pixels outside the trace and that are not already masked
            mask = (np.abs(colpix-yo) > bg_semiwidth) & (np.isfinite(colval))
            # Make sure at least 5 pixels are used in the background fit
            nvalid = np.size(np.where(mask == True))
            if nvalid < 5: # if less than 5 pixels are not masked
                continue # to next column
            # Fit a polynomial
            oo = bg_polyorder*1
            while oo>=1:
                p = np.polyfit(colpix[mask],colval[mask],oo)
                modelval = np.polyval(p,colpix)
                # Make sure that model makes sense (does not go below zero)
                if np.min(modelval) >= 0:
                    break
                else:
                    oo = oo - 1



            # Apply model subtraction
            image_bgsub[:,xo] = image_bgsub[:,xo] - modelval
            # Plot each column if requested
            if debug == True and (xo % 25) == 0:
                print('background subtracting column {}'.format(xo))
                plt.scatter(colpix,colval+nstep*adustep,marker='.',color='black')
                plt.scatter(colpix[~mask],colval[~mask]+nstep*adustep,marker='.',color='yellow')
                plt.plot(colpix,modelval+nstep*adustep,color='red')
                nstep = nstep + 1

        if debug == True:
            plt.ylim((adumin,(nstep+1)*adustep))

        working_image = image_bgsub


    thex = np.linspace(-semiwidthx,semiwidthx,2*semiwidthx+1)
    they = np.linspace(-semiwidthy,semiwidthy,2*semiwidthy+1)
    they2 = np.linspace(-semiwidthy,semiwidthy,osdimy*2*semiwidthy+1)

    for zzz in cenx:
        xo = int(zzz)
        yo = int(np.floor(ceny[xo]))
        dy = (ceny[xo] - yo)
        temp_trace = np.zeros((osdimy*2*semiwidthy+1,2*semiwidthx+1))
        xcount = -1
        if (xo <= 3) or (xo >=2044):
            # Don't build a model for the reference pixel position
            model_trace[:,xo] = np.nan
        else:
            # Model the trace for light-sensitive pixels
            for x in (xo+thex):
                xcount = xcount+1 # is the column count for this current matrix
                if (x <= 3) or (x>=2044):
                    # ref pixels, can't use them
                    #print('x <=3 or x>=2044')
                    temp_trace[:,xcount] = np.nan
                else:
                    if (order == 2) and (x >=2750):
                        temp_trace[:,xcount] = np.nan
                    else:
                        # Resample that column linearly
                        y1 = np.min(they+yo)
                        y2 = np.max(they+yo)
                        # check that min is at least 0. fill with nan if not
                        # but only if there are at least 3 valid pixels.
                        if (y1 < 0) & (y2 >3):
                            nnan = int(0 - y1)
                            prof = np.zeros(len(they))*np.nan
                            prof[nnan:] = working_image[0:int(y2)+1,int(x)]
                        elif (y1 >= 0):
                            prof = working_image[int(y1):int(y2)+1,int(x)]
                        else:
                            prof = np.zeros(len(they))*np.nan
                        #print('x={},y1={},y2={}'.format(x,y1,y2))
                        # nan
                        ind = np.isfinite(prof)
                        #print(x, np.size(np.where(ind == True)))
                        # Only compute a trace model when the number of valid
                        # pixels is more than 3. Optimally, it should be
                        # 2*semiwidthy+1 (last time I checked)
                        if np.size(np.where(ind == True)) >= 3:
                            prof = prof[ind]
                            y = they+yo
                            y = y[ind]
                            ynew = they2+yo
                            profresamp = np.interp(ynew, y-dy, prof)
                            # Normalize to each wavelength total flux
                            # But restrict to pixels in the core of the
                            # trace (not wings)
                            cond = np.abs(dy) < 12*osdimy
                            area = np.sum(profresamp[cond])
                            #area = np.sum(profresamp)
                            profresamp /= area
                            # fill the current column trace matrix
                            temp_trace[:,xcount] = profresamp
                        else:
                            temp_trace[:,xcount] = np.nan


            # Model the trace at the current xo position by medianing neighboring columns
            model_trace[:,xo] = np.nanmedian(temp_trace,axis=1)
        if setwingzero == True:
            # Force the last pixel on each wing to be zero by subtracting a
            # slope fitted from the first and last rows.
            # print('Option selected - Forcing wings to zero')
            yarr = np.arange(len(they2))
            m = (model_trace[-1,xo] - model_trace[0,xo])/np.max(yarr)
            b = model_trace[0,xo]
            #print(m,b)
            model_trace[:,xo] = model_trace[:,xo] - (m*yarr+b)
    # Save the trace
    if saveimage != None:
        hdu = fits.PrimaryHDU()
        hdu.data = model_trace
        hdu.header['OVRSAMPY'] = osdimy
        hdu.writeto(saveimage,overwrite=True)

    return(model_trace)




def model_trace_from_stack_nobackgroundsub(image, order=None, cenx=None, ceny=None,
                           semiwidthx=None, semiwidthy=None, osdimy=None,
                           saveimage=None, setwingzero=None):

    # Construct a model of the trace for all x positions in order 1 or 2
    dimx = 2048
    if semiwidthx == None: semiwidthx = 7
    if semiwidthy == None: semiwidthy = 32
    if osdimy == None: osdimy = 5

    # Read the trace centroid position
    if (cenx == None) or (ceny == None):
        # Read the best available trace centroid solution
        if (order == 1) or (order == None):
            pixels = np.linspace(0,dimx-1,dimx)
            cenx,ceny = get_order1_from_cv(atthesex=pixels)
        if order == 2:
            pixels = np.linspace(0,dimx-1,dimx)
            cenx,ceny,lba = extrapolate_order2(atthesex=pixels)
            #cenx,ceny = get_order2_from_cv(atthesex=pixels)
            #for ccc in range(len(pixels)):
            #    print(pixels[ccc],cenx[ccc],ceny[ccc])
        if order == 3:
            pixels = np.linspace(0,dimx-1,dimx)
            cenx,ceny = get_order3_from_cv(atthesex=pixels)
            #cenx,ceny = get_order2_from_cv(atthesex=pixels)

    model_trace = np.zeros((osdimy*2*semiwidthy+1,dimx))

    if True:
        # for order 2 only
        ind = np.isfinite(cenx)
        cenx = cenx[ind]


    thex = np.linspace(-semiwidthx,semiwidthx,2*semiwidthx+1)
    they = np.linspace(-semiwidthy,semiwidthy,2*semiwidthy+1)
    they2 = np.linspace(-semiwidthy,semiwidthy,osdimy*2*semiwidthy+1)

    for zzz in range(dimx):
        xo = int(zzz)
        yo = np.floor(ceny[xo])
        dy = (ceny[xo] - yo)
        temp_trace = np.zeros((osdimy*2*semiwidthy+1,2*semiwidthx+1))
        xcount = -1
        if (xo <= 3) or (xo >=2044):
            # Don't build a model for the reference pixel position
            model_trace[:,xo] = np.nan
        else:
            # Model the trace for light-sensitive pixels
            for x in (xo+thex):
                xcount = xcount+1 # is the column count for this current matrix
                if (x <= 3) or (x>=2044):
                    # ref pixels, can't use them
                    #print('x <=3 or x>=2044')
                    temp_trace[:,xcount] = np.nan
                else:
                    if (order == 2) and (x >=2750):
                        temp_trace[:,xcount] = np.nan
                    else:
                        # Resample that column linearly
                        y1 = np.min(they+yo)
                        y2 = np.max(they+yo)
                        # check that min is at least 0. fill with nan if not
                        # but only if there are at least 3 valid pixels.
                        if (y1 < 0) & (y2 >3):
                            nnan = int(0 - y1)
                            prof = np.zeros(len(they))*np.nan
                            prof[nnan:] = image[0:int(y2)+1,int(x)]
                        elif (y1 >= 0):
                            prof = image[int(y1):int(y2)+1,int(x)]
                        else:
                            prof = np.zeros(len(they))*np.nan
                        #print('x={},y1={},y2={}'.format(x,y1,y2))
                        # nan
                        ind = np.isfinite(prof)
                        #print(x, np.size(np.where(ind == True)))
                        # Only compute a trace model when the number of valid
                        # pixels is more than 3. Optimally, it should be
                        # 2*semiwidthy+1 (last time I checked)
                        if np.size(np.where(ind == True)) >= 3:
                            prof = prof[ind]
                            y = they+yo
                            y = y[ind]
                            ynew = they2+yo
                            profresamp = np.interp(ynew, y-dy, prof)
                            # Normalize to each wavelength total flux
                            area = np.sum(profresamp)
                            profresamp /= area
                            # fill the current column trace matrix
                            temp_trace[:,xcount] = profresamp
                        else:
                            temp_trace[:,xcount] = np.nan


            # Model the trace at the current xo position by medianing neighboring columns
            model_trace[:,xo] = np.nanmedian(temp_trace,axis=1)
        if setwingzero == True:
            # Force the last pixel on each wing to be zero by subtracting a
            # slope fitted from the first and last rows.
            # print('Option selected - Forcing wings to zero')
            yarr = np.arange(len(they2))
            m = (model_trace[-1,xo] - model_trace[0,xo])/np.max(yarr)
            b = model_trace[0,xo]
            #print(m,b)
            model_trace[:,xo] = model_trace[:,xo] - (m*yarr+b)
    # Save the trace
    if saveimage != None:
        hdu = fits.PrimaryHDU()
        hdu.data = model_trace
        hdu.header['OVRSAMPY'] = osdimy
        hdu.writeto(saveimage,overwrite=True)

    return(model_trace)



def model_trace_from_stack_order1only(image, cenx=None, ceny=None,
                                      semiwidthx=None, semiwidthy=None,
                                      osdimy=None, saveimage=None):
    # FUNCTION NOT USED ANYMORE, INSTEAD USE model_trace_from_stack
    #
    # Construct a model of the trace for all x positions in order 1
    dimx = 2048
    order = 1
    if semiwidthx == None: semiwidthx = 7
    if semiwidthy == None: semiwidthy = 32
    if osdimy == None: osdimy = 5

    # Read the trace centroid position
    if (cenx == None) or (ceny == None):
        # Read the best available trace centroid solution
        if (order == 1) or(order == None):
            cenx,ceny = get_order1_from_cv(atthesex=np.linspace(0,dimx-1,dimx))
        if order == 2:
            cenx,ceny = get_order2_from_cv(atthesex=np.linspace(0,dimx-1,dimx))

    model_trace = np.zeros((osdimy*2*semiwidthy+1,dimx))

    if True:
        # for order 2 only
        ind = np.isfinite(cenx)
        cenx = cenx[ind]


    thex = np.linspace(-semiwidthx,semiwidthx,2*semiwidthx+1)
    they = np.linspace(-semiwidthy,semiwidthy,2*semiwidthy+1)
    they2 = np.linspace(-semiwidthy,semiwidthy,osdimy*2*semiwidthy+1)

    for zzz in range(dimx):
        xo = int(zzz)
        yo = np.floor(ceny[xo])
        dy = (ceny[xo] - yo)
        temp_trace = np.zeros((osdimy*2*semiwidthy+1,2*semiwidthx+1))
        xcount = -1
        if (xo <= 3) or (xo >=2044):
            # Don't build a model for the reference pixel position
            model_trace[:,xo] = np.nan
        else:
            # Model the trace for light-sensitive pixels
            for x in (xo+thex):
                xcount = xcount+1 # is the column count for this current matrix
                if (x <= 3) or (x>=2044):
                    # ref pixels, can't use them
                    #print('x <=3 or x>=2044')
                    temp_trace[:,xcount] = np.nan
                else:
                    # Resample that column linearly
                    y1 = np.min(they+yo)
                    y2 = np.max(they+yo)
                    prof = image[int(y1):int(y2)+1,int(x)]
                    # nan
                    ind = np.isfinite(prof)
                    prof = prof[ind]
                    y = they+yo
                    y = y[ind]
                    ynew = they2+yo
                    profresamp = np.interp(ynew, y-dy, prof)
                    # Normalize to each wavelength total flux
                    area = np.sum(profresamp)
                    profresamp /= area
                    # fill the current column trace matrix
                    temp_trace[:,xcount] = profresamp
            # Model the trace at the current xo position by medianing neighboring columns
            model_trace[:,xo] = np.nanmedian(temp_trace,axis=1)

    # Save the trace
    if saveimage != None:
        hdu = fits.PrimaryHDU()
        hdu.data = model_trace
        hdu.header['OVRSAMPY'] = osdimy
        hdu.writeto(saveimage,overwrite=True)

    return(model_trace)


def rectify_trace(image, order=None, cenx=None, ceny=None, osdimy=None,
                  semiwidthy=None, saveimage=None):
    # Straightens the traces, assuming no tilt, so independent variable
    # is x pixels.

    # X dimension of the subarray (along the spectral axis)
    dimx = 2048
    dimy = np.shape(image)[0]

    # Rectify the trace to some width and some oversample for all x positions
    if semiwidthy == None: semiwidthy = 32
    if osdimy == None: osdimy = 5

    # Read the trace centroid position
    if (cenx == None) or (ceny == None):
        # Read the best available trace centroid solution
        if (order == 1) or(order == None):
            cenx,ceny = get_order1_from_cv(atthesex=np.linspace(0,dimx-1,dimx))
        if order == 2:
            #cenx,ceny = get_order2_from_cv(atthesex=np.linspace(0,dimx-1,dimx))
            cenx,ceny,tmp = extrapolate_order2(atthesex=np.linspace(0,dimx-1,dimx))

    # Rectify by resampling
    rectified_trace = np.zeros((osdimy*2*semiwidthy+1,dimx))

    # profile in native pixels, centered on center of mass
    they = np.linspace(-semiwidthy,semiwidthy,2*semiwidthy+1)
    # profile, oversampled, centered on CoM
    they2 = np.linspace(-semiwidthy,semiwidthy,osdimy*2*semiwidthy+1)
    for xo in range(dimx):
        yo = np.floor(ceny[xo])
        dy = (ceny[xo] - yo)
        if (xo <= 3) or (xo >=2044):
            # Don't build a model for the reference pixel position
            rectified_trace[:,xo] = np.nan
        else:
            # Resample that column linearly
            y1 = int(np.min(they+yo))
            y2 = int(np.max(they+yo))
            padlow, padupp = [], [] #pads of nan to put at each en dof the column
            if y1 < 0:
                padlow = np.zeros(np.abs(y1))*np.nan
                y1 = 0
            if y2 > (dimy-1):
                padupp = np.zeros(np.abs(y2-dimy+1))*np.nan
                y2 = dimy*1-1
            prof = image[y1:y2+1,xo]
            # Pad with NaNs
            prof = np.array(list(padlow) + list(prof) + list(padupp))
            # nan
            cond = np.isfinite(prof)
            y = they+yo
            ynew = they2+yo
            rectified_trace[:,xo] = np.interp(ynew, y[cond]-dy, prof[cond])
            # In padded regions, put NaNs in the oversampled rectified image
            cond = (ynew < y1) | (ynew > y2)
            rectified_trace[cond,xo] = np.nan

    if saveimage != None:
        # Save the trace
        hdu = fits.PrimaryHDU()
        hdu.data = rectified_trace
        hdu.header['OVRSAMPY'] = osdimy
        hdu.writeto(saveimage,overwrite=True)

    return(rectified_trace)


def dist(NAXIS,behave_as_idl=True):
    """Returns a rectangular array in which the value of each element is proportional to its frequency.
    >>> dist(3)
    array([[ 0.        ,  1.        ,  1.        ],
           [ 1.        ,  1.41421356,  1.41421356],
           [ 1.        ,  1.41421356,  1.41421356]])
    >>> dist(4)
    array([[ 0.        ,  1.        ,  2.        ,  1.        ],
           [ 1.        ,  1.41421356,  2.23606798,  1.41421356],
           [ 2.        ,  2.23606798,  2.82842712,  2.23606798],
           [ 1.        ,  1.41421356,  2.23606798,  1.41421356]])
    """

    if behave_as_idl == True:
        if np.mod(NAXIS,2) == 0: # if even
            axis = np.linspace(-NAXIS/2+1, NAXIS/2, NAXIS)
            result = np.sqrt(axis**2 + axis[:,np.newaxis]**2)
            return np.roll(result, int(NAXIS/2)+1, axis=(0,1))
        else: # if odd
            axis = np.linspace(-(NAXIS-1)/2, (NAXIS-1)/2, NAXIS)
            result = np.sqrt(axis**2 + axis[:,np.newaxis]**2)
            return np.roll(result, int((NAXIS-1)/2)+1, axis=(0,1))
    else:
        # Produces identical results to IDL dist.pro only if dim is even.
        # This version assumes the origin is half a pixel off
        axis = np.linspace(-NAXIS/2+1, NAXIS/2, NAXIS)
        result = np.sqrt(axis**2 + axis[:,np.newaxis]**2)
        return np.roll(result, int(NAXIS/2)+1, axis=(0,1))

def pixmtf(dim):
    # A conversion of David Lafreniere's IDL pixmtf.pro
    # Returns the MTF of a pixel
    # dim is the dimension of the image

    nu = dist(dim)
    nu[0,0] = 1.0
    arg = np.pi*nu/dim
    pmtf = np.sin(arg)/arg
    pmtf[0,0] = 1.0

    return(pmtf)


def psfth_bessel(teldiam, ps, lba, dim, cobs=None, xc=None, yc=None,
                 dlba=None, pmtf=None, saveimage=None):

    from scipy import special

    rdim = np.floor(dim/2)*2
    print(dim,rdim)
    if xc == None: xc = rdim/2
    if yc == None: yc = rdim/2
    print(xc,yc)

    # compute distance of the pixel to the PSF center (xc,yc)
    x = np.linspace(0, dim-1, dim)
    y = np.linspace(0, dim-1, dim)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt((xx-xc)**2+(yy-yc)**2)

    # Make sure to set the central pixel to non-zero but very small number
    r[r == 0] = 1e-20

    psf = np.zeros((dim,dim))

    # parameters for polychromatic PSF, or not
    if dlba == None:
        nlambda = 1
        l = np.zeros(1)
        l[0] = lba*1
    else:
        # For a shift of 0.1 pixel between lambdas in the corner
        nlambda = np.ceil(np.sqrt(2)*(dim/2)*dlba/lba/0.1)
        # all lambdas to consider
        l = lba*(1.0+dlba*(np.arange(nlambda)/(nlambda-1.0)-0.5))

    for n in range(nlambda):
        # Cutoff frequency in pixel^-1
        nuc = ps*teldiam/l[n]*(np.pi*1e+6)/(180*3600)
        arg1 = np.pi*r*nuc
        coeff=1.0
        term1=2.0*special.jn(1,arg1)/arg1
        term2=0.0

        if cobs != None:
            arg2 = cobs*arg1
            coeff = 1.0/(1.0-cobs**2)
            term2 = 2.0*cobs**2*special.jn(1,arg2)/arg2

        psfn = coeff*(term1-term2)**2
        psf = psf + psfn

    # Correct for the MTF of pixels, if desired
    if pmtf != None:
        # Go to the frequency space
        psffft = np.fft.fft2(np.roll(psf,(-int(rdim/2),-int(rdim/2)), axis=(0,1)))
        # Correct for the pixel MFT
        pmtf = pixmtf(dim)
        psffft = pmtf * psffft
        # Go back to the image space
        psf = np.roll(np.real(np.fft.ifft2(psffft)), (int(rdim/2), int(rdim/2)), axis=(0,1))


    psf = psf/np.sum(psf)

    if saveimage != None:
        # Save the trace
        hdu = fits.PrimaryHDU()
        hdu.data = psf
        hdu.writeto(saveimage,overwrite=True)

    return(psf)




def rectify_trace_old(image, order=None, cenx=None, ceny=None, osdimy=None,
                  semiwidthy=None, saveimage=None):
    # Straightens the traces, assuming no tilt, so independent variable
    # is x pixels.

    # X dimension of the subarray (along the spectral axis)
    dimx = 2048

    # Rectify the trace to some width and some oversample for all x positions
    if semiwidthy == None: semiwidthy = 32
    if osdimy == None: osdimy = 5

    # Read the trace centroid position
    if (cenx == None) or (ceny == None):
        # Read the best available trace centroid solution
        if (order == 1) or(order == None):
            cenx,ceny = get_order1_from_cv(atthesex=np.linspace(0,dimx-1,dimx))
        if order == 2:
            #cenx,ceny = get_order2_from_cv(atthesex=np.linspace(0,dimx-1,dimx))
            cenx,ceny,tmp = extrapolate_order2(atthesex=np.linspace(0,dimx-1,dimx))

    # Rectify by resampling
    rectified_trace = np.zeros((osdimy*2*semiwidthy+1,dimx))

    # profile in native pixels, centered on center of mass
    they = np.linspace(-semiwidthy,semiwidthy,2*semiwidthy+1)
    # profile, oversampled, centered on CoM
    they2 = np.linspace(-semiwidthy,semiwidthy,osdimy*2*semiwidthy+1)
    for xo in range(dimx):
        yo = np.floor(ceny[xo])
        dy = (ceny[xo] - yo)
        if (xo <= 3) or (xo >=2044):
            # Don't build a model for the reference pixel position
            rectified_trace[:,xo] = np.nan
        else:
            # Resample that column linearly
            y1 = np.min(they+yo)
            y2 = np.max(they+yo)
            prof = image[int(y1):int(y2)+1,xo]
            # nan
            ind = np.isfinite(prof)
            prof = prof[ind]
            y = they+yo
            y = y[ind]
            ynew = they2+yo
            rectified_trace[:,xo] = np.interp(ynew, y-dy, prof)

    if saveimage != None:
        # Save the trace
        hdu = fits.PrimaryHDU()
        hdu.data = rectified_trace
        hdu.header['OVRSAMPY'] = osdimy
        hdu.writeto(saveimage,overwrite=True)

    return(rectified_trace)




def replace_badpixels_using_trace_model(image, tracemodel=None, badpix=None,
                                        cenx=None, ceny=None, semiwidthy=None,
                                        osdimy=None, saveimage=None):
    # Purpose is to replace bad pixels on the 2D image by
    # values of the trace model, scaled appropriately.

    # tracemodel should be the rectified trace, could be (or not) oversampled y
    if (tracemodel == None) or (semiwidthy == None) or (osdimy == None):
        a = fits.open('LoicsWork/InputFiles/trace_order1.fits')
        tracemodel = a[0].data
        osdimy = a[0].header['OVRSAMPY']
        semiwidthy = int((np.shape(tracemodel)[0] - 1) / osdimy / 2)

    # Determine the image dimensions
    dim = np.shape(image)
    if np.size(dim) == 3:
        # cube
        dimz, dimy, dimx = dim
    if np.size(dim) == 2:
        # single image
        dimz = 1
        dimy, dimx = dim

    # Read the trace centroid position
    if (cenx == None) or (ceny == None):
        order = 1
        # Read the best available trace centroid solution
        if (order == 1) or(order == None):
            cenx, ceny = get_order1_from_cv(atthesex=np.linspace(0, dimx-1, dimx))
        if order == 2:
            cenx, ceny = get_order2_from_cv(atthesex=np.linspace(0, dimx-1, dimx))

    # Perform the thing on all images of the cube
    modely = np.linspace(-semiwidthy, semiwidthy, osdimy*2*semiwidthy+1)
    ypixel = np.linspace(0,255,256)
    nanslice = np.zeros(256)*np.nan
    cleanimage = image*1
    for zzz in range(dimz):
        for x in cenx:
            x = int(x)
            imageslice = nanslice*1
            modelslice = nanslice*1
            z = tracemodel[:, int(cenx[x])]
            y = ceny[x]+modely
            # Put to NaN, pixels outside the bounds of the model
            m1 = (ypixel > ceny[x]-semiwidthy) & (ypixel < ceny[x]+semiwidthy)
            ind = np.where(m1 == True)
            modelslice[ind] = np.interp(ypixel[ind], y, z)
            # extract slice of spatial pixels from current image
            if dimz == 1:
                imageslice[ind] = image[ind, int(cenx[x])]
            else:
                imageslice[ind] = image[zzz, ind, int(cenx[x])]
            # Find the fitting scale between model and stack
            scale = np.nanmedian(imageslice / modelslice)
            # identify bad pixels in stack and replace their value with scaled model
            m2 = np.isfinite(imageslice)
            m3 = m1 & ~m2
            ind = np.where(m3 == True)
            if dimz == 1:
                sss = cleanimage[:, int(cenx[x])]*1
            else:
                sss = cleanimage[zzz, :, int(cenx[x])]*1
            if x >= 4 and x <= 2043:
                sss[ind] = scale * modelslice[ind]
                if dimz == 1:
                    cleanimage[:, int(cenx[x])] = sss * 1
                else:
                    cleanimage[zzz, :, int(cenx[x])] = sss * 1

    if saveimage != None:
        print('Clean stack saved.')
        hdu = fits.PrimaryHDU()
        hdu.data = cleanimage
        hdu.writeto(saveimage, overwrite=True)

    return(cleanimage)


def project_modeltrace_on2Dmap(tracemodel=None, cenx=None, ceny=None,
                               semiwidthy=None, osdimy=None, subarray=None,
                               saveimage=None):

    # tracemodel should be the rectified trace, could be (or not) oversampled y
    if (tracemodel == None) or (semiwidthy == None) or (osdimy == None):
        a = fits.open('LoicsWork/InputFiles/trace_order1.fits')
        tracemodel = a[0].data
        osdimy = a[0].header['OVRSAMPY']
        semiwidthy = int((np.shape(tracemodel)[0] - 1) / osdimy / 2)

    dimx = 2048
    order = 1

    # Read the trace centroid position
    if (cenx == None) or (ceny == None):
        # Read the best available trace centroid solution
        if (order == 1) or(order == None):
            cenx,ceny = get_order1_from_cv(atthesex=np.linspace(0,dimx-1,dimx))
        if order == 2:
            cenx,ceny = get_order2_from_cv(atthesex=np.linspace(0,dimx-1,dimx))

    modely = np.linspace(-semiwidthy,semiwidthy,osdimy*2*semiwidthy+1)
    if subarray == None or subarray == 256:
        ymap = np.linspace(0,255,256)
        map2D = np.zeros((256,2048))*np.nan
    if subarray == 96:
        ymap = np.linspace(0,255,256)
        map2D = np.zeros((256,2048))*np.nan
    if subarray == 2048:
        ymap = np.linspace(0,2047,2048)
        map2D = np.zeros((2048,2048))*np.nan
    for x in cenx:
        x = int(x)
        z = tracemodel[:,int(cenx[x])]
        y = ceny[x]+modely
        slicemap = np.interp(ymap,y,z)
        ind = np.where((ymap > ceny[x]-semiwidthy) & (ymap < ceny[x]+semiwidthy))
        map2D[ind,x] = slicemap[ind]

    # Reformat to the requested subarray size
    if subarray == 96:
        map2D = map2D[150:246,:]

    # Save the trace
    if saveimage != None:
        print('Model trace projection on 2D map, saved.')
        hdu = fits.PrimaryHDU()
        hdu.data = map2D
        hdu.writeto(saveimage,overwrite=True)

    return(map2D)


def tilt_order1(x=None, unit=None, model=None, angle=None):
    # Function that describes the monochromatic tilt as a function
    # of x position.

    if x is None: x = np.linspace(0,2048,2048)

    if model == 'sinus':
        # For now, assumes a sinusoidal shape for experimentation
        #tilt = x*0 - 0.75 + 0.25*np.sin(100*np.pi*x/2048) # degrees
        tilt = x*0 - 0.75 + 2.0*np.sin(100*np.pi*x/2048) # degrees
        if unit == 'slope': tilt = np.tan(np.deg2rad(tilt))
        return(tilt,x)
    else:
        # Assumes that the tilt is constant with wavelength and
        # at an angle of -0.8 degrees.
        if angle != None:
            tilt = x*0 - angle # degrees
        else:
            tilt = x*0 - 0.8 # degrees
        if unit == 'slope': tilt = np.tan(np.deg2rad(tilt))
        return(tilt,x)


def mask_trace(image, osx=None, osy=None, order=None, cenx=None, ceny=None,
               invertmask=None, semiwidth=None, returnmask=None, nan=None):
    # invertmask==False? --> True in trace, nan out of trace
    # invertmask==True? --> True out of trace, nan in trace

    # Apply a mask to an image, setting to False all pixels **not** belonging to the
    # trace, i.e. not within the semiwidth parameters. If invertmask is set
    # then, it is the trace that is False while the rest of pixels remain
    # untouched at True.
    # osx and osy are the oversampling relative to native pixels of the input
    # image.

    mask = np.full(np.shape(image), True)

    # set the default values for optional input parameters
    if (osx == None): osx = 1
    if (osy == None): osy = 1
    if (semiwidth == None): semiwidth = 20
    if (order == None): order = 1

    dimx = int(np.shape(image)[1]/osx)
    dimy = int(np.shape(image)[0]/osy)

    thex = np.linspace(0,dimx-1,dimx*osx)
    they = np.linspace(0,dimy-1,dimy*osy)

    # Read the trace centroid position
    if (cenx is None) or (ceny is None):
        # Read the best available trace centroid solution
        if (order == 1) or (order == None):
            cenx,ceny = get_order1_from_cv(atthesex=thex)
        if order == 2:
            cenx,ceny,toto = extrapolate_order2(atthesex=thex)
        if order == 3:
            cenx,ceny = get_order3_from_cv(atthesex=thex)

    # Define lower/upper bound position at each x position
    for i in range(len(cenx)):
        yo = ceny[i]
        ymax = yo+semiwidth
        ymin = yo-semiwidth
        if ymax > dimy-1: ymax = dimy-1
        if ymin < 0: ymin = 0
        ind = (they >= ymin) & (they <= ymax)
        mask[:,i] = ind
        # True if within the trace, False otherwise

    # Return the boolean or nan float mask
    if returnmask == True:
        if invertmask == True:
            if nan == True:
                maskfloat = np.ones((dimy,dimx))
                maskfloat[mask] = np.nan
                return(maskfloat) # nan in trace
            else:
                return(~mask) # True out of trace
        else:
            if nan == True:
                maskfloat = np.ones((dimy,dimx))
                maskfloat[~mask] = np.nan
                return(maskfloat) # nan out of trace
            else:
                return(mask) # True in trace
    # Or return the masked array
    else: # (return masked image instead)
        if invertmask == True:
            if nan == True:
                image[mask] = np.nan
                return(image) # nan in trace
            else:
                return(image*~mask) # True out of trace
        else:
            if nan == True:
                image[~mask] = np.nan
                return(image) #
            else:
                return(image*mask)

# invertmask==False? --> True in trace, nan out of trace
# invertmask==True? --> True out of trace, nan in trace


def get_nice_rehearsal_stack(debug=None, return_highpass=None,
                             interpolate_nan=None, getonecds=False,
                             getf277stack=False, leave_background_step=False):

    # A facilitating function that reads a CV3 rehearsal deep stack, rotates,
    # applies a bad pixel mask, a flat fielding step and optionally, a
    # high-pass filter.

    if getonecds == True:
        a = fits.open('/Users/albert/NIRISS/simuSOSS/revival/cds_256_ng3.fits')
        #dimz = np.shape(a[0].data)[0]
        slicenbr = 200
        stack = a[0].data[slicenbr]
    elif getf277stack == True:
        a = fits.open('/Users/albert/NIRISS/simuSOSS/revival/stack_f277.fits')
        stack = a[0].data
    else:
        a = fits.open('LoicsWork/InputFiles/stack_256_ng3.fits')
        stack = a[0].data

    stack = image_orient_from_cv(stack)

    if debug == True:
        plt.figure(figsize=(15, 3))
        plt.imshow(stack,origin='lower')

    # Read the bad pixel mask
    a = fits.open('LoicsWork/InputFiles/badpix256.fits')
    badpix = a[0].data
    badpix = image_orient_from_cv(badpix)
    ind = np.where(badpix == 0)
    badpix[ind] = np.nan

    # Read the pixel flat
    a = fits.open('/Users/albert/NIRISS/simuSOSS/revival/niriss_ref_pxl_flt_GR150C_F200W.fits')
    flat = a[1].data
    flat = image_orient_from_stsci(flat)
    flat = flat[0:256,:]

    stack = stack * badpix / flat

    if leave_background_step != True:
        stack = remove_background_step(stack)

    if debug == True:
        plt.figure(figsize=(15, 3))
        plt.imshow(np.log10(stack),origin='lower',vmax=4.5)
        plt.show()

    if interpolate_nan == True:
        stacknan = stack*1
        cond = np.isfinite(stack) == False
        stack[cond] = np.nanmedian(stack)

    if debug == True:
        plt.figure(figsize=(15, 3))
        plt.imshow(np.log10(stack),origin='lower',vmax=4.5)
        plt.show()

    if return_highpass == True:
        if interpolate_nan != True:
            stacknan = stack*1
            cond = np.isfinite(stack) == False
            stack[cond] = np.nanmedian(stack)
        from scipy import ndimage
        lowpass = ndimage.gaussian_filter(stack, 9)
        stack_highpass = stacknan - lowpass
        stack = stack_highpass*1
        if debug == True:
            plt.figure(figsize=(15, 3))
            plt.imshow(np.log10(lowpass),origin='lower',vmax=4.5)

            plt.figure(figsize=(15, 3))
            #plt.imshow(np.log10(gauss_highpass),origin='lower',vmin=-2.0,vmax=4.0)
            plt.imshow(stack_highpass,origin='lower',vmin=-10,vmax=10)

            hdu = fits.PrimaryHDU()
            hdu.data = stack_highpass
            hdu.writeto('stack_highpass.fits', overwrite=True)

    return(stack)



def remove_background_step(image):
    # remove the background step at x=700 from the SOSS images
    low = [691,692,693,694,695,696]
    upp = [704,705,706,707,708,709]

    diff = image[:,upp] - image[:,low]
    step = np.nanmedian(diff[0:140,:])
    print(step)

    hdu = fits.PrimaryHDU()
    hdu.data = diff
    hdu.writeto('diff_background_step.fits', overwrite=True)

    #Can do a better subtraction at the interface but will do fo now.
    stack = image*1
    stack[:,700:-1] = stack[:,700:-1] - step

    return(stack)



def wingfit():
    # Fit the ripples found in the wings


    return()




def tracefit(imageForFit, modelTrace=None, order=None, cenx=None, ceny=None,
             fit_linearbackground=True, columnlist=None, debug=None,
             fit_iter2=False, modelResidu=None,
             return_residu=False, return_model=False, fit_radius=None,
             save_residu=None,save_model=None):
    # Function that fits a model of a trace to data, column by column.
    # Requires a model of the trace and its position. It will scale the
    # model to best match the observations.

    # THREE FUNCTIONS LOCAL TO THIS ARE DESCRIBED FIRST
    from scipy.integrate import trapz
    def trace_rebin_properly(x,fx,xbin,debug=None):
        # x is the pixel position for the model trace
        # fx is the intensity value for the model trace
        # xbin is the pixel position for the offsetted model
        # fxbin is the intensity value for the offsetted model

        # The concept: our trace model is over sampled and we want to
        # integrate it within the proper pixel bounds at the requested
        # trace position (where the data tells us). That means we'll first
        # interpolate the trace model at, say, 5X the native pixel resolution
        # anchored on the requested xbin positions so that we can then
        # integrate the intensities by simply summing 5 sub pixels to
        # produce the native pixel size at the requested position.

        # Now using scipy.integrate.trapz instead of sum

        # Assuming that xbin is regularly sampled array:
        # The median pixel difference between consecutive x position values in the array is
        # delta = np.median(xbin[1:]-xbin[0:-1]) # should be 1
        delta = np.min(xbin[1:]-xbin[0:-1]) # to address possible masked pixels, then min is more secure than median
        # Initialize the array of intensities for the requested xbin positions.
        # That is the output of the function.
        if debug == True: print('delta = {}'.format(delta))
        fxbin = np.zeros(len(xbin))
        # Also initialize an array 10X (if os=10) the requested pixel resolution, padding of 0.5
        # pixels is dialed in to ease later integration. So if xbin = [4,5,6,7,8] then
        # xos = [3.5,3.6,3.7, ...,8.3,8.4,8.5].
        os = 5
        nos = (np.max(xbin)+delta/2 - np.min(xbin)-delta/2 +1)*os + 1
        if debug == True: print('nos = {}'.format(nos))
        xos = np.linspace(np.min(xbin)-delta/2,np.max(xbin)+delta/2,nos)
        if debug == True: print('xos = {}'.format(xos))
        # Now interpolate 5X oversampled
        fxos = np.interp(xos, x, fx)
        if debug == True: print('fxos = {}'.format(fxos))
        # Then ready to sum up at the requested native pixel position
        for i in range(len(fxbin)):
            cond = np.abs(xos - xbin[i]) <= 0.501 #(add a 0.01 to make sure numeric errors do not exclude points)
            fxbin[i] = trapz(fxos[cond],xos[cond])
        return(fxbin)

    def tracemodel(pixeldata, pixeloffset, s, background):
        prof = trace_rebin_properly(pixelmodel, valuemodel, pixeldata-tracecenter)
        m = s * prof + background
        return(m)

    def tracemodel_lineartrend(pixeldata, pixeloffset, s, background, slope, yintercept):
        prof = trace_rebin_properly(pixelmodel, valuemodel, pixeldata-tracecenter)
        lineartrend = yintercept + pixeldata * slope
        m = s * prof + lineartrend
        return(m)

    def tracemodel_iter2(pixeldata, pixeloffset, s, background, k):
        prof = trace_rebin_properly(pixelmodel, valuemodel, pixeldata-tracecenter)
        resi = trace_rebin_properly(pixelmodel, valueresidu, pixeldata-tracecenter)
        m = s * prof + k * resi #+ background
        return(m)

    def integrate_tracemodel_iter2(pixeloffset, s, background, k):
        # Integrate the flux from the model, within the firtadius boundaries.
        # Do it in the oversampled regime of the models, rather than in native.
        cond = np.abs(pixelmodel) <= 13 # pixelmodel is in units of native pixels
        m = s * valuemodel + k * valueresidu + background
        return(np.sum(m[cond])/osdimy)

    def twotrace(x_data,s_o1,k_o1,s_o2,k_o2,background):
        # Fit with 2 iterations both traces simultaneously
        prof_o1 = trace_rebin_properly(x_model, fx_model_o1, x_data-xo_o1)
        resi_o1 = trace_rebin_properly(x_model, fx_residu_o1, x_data-xo_o1)
        prof_o2 = trace_rebin_properly(x_model, fx_model_o2, x_data-xo_o2)
        resi_o2 = trace_rebin_properly(x_model, fx_residu_o2, x_data-xo_o2)

        m = s_o1 * prof_o1 + k_o1 * resi_o1 + s_o2 * prof_o2 + k_o2 * resi_o2 + background
        return(m)

    # THREE FUNCTIONS LOCAL TO THIS - END
    #####################################



    dimx = 2048
    osdimy = 3
    if order == None: order = 1
    if modelTrace == None:
        # Do something along the line:
        # Call a function that returns a trace model at a monochromatic
        # wavelength. That function does not yet exist.
        # For now, just call the model trace function
        modelTrace = model_trace_from_stack(imageForFit, order=order,
                                            osdimy=osdimy, debug=debug,
                                            setwingzero=False, semiwidthy=32)
    else:
        # Read it from disk. Needs to have been generated with
        # model_trace_from_stack so that the semiwidthy and osdimy be known
        a = fits.open(modelTrace)
        modelTrace = a[0].data
        hdr = a[0].header
        osdimy = hdr['OVRSAMPY']
        semiwidthy = (np.shape(modelTrace)[0] - 1)/2/osdimy
        print('use disk trace',osdimy,semiwidthy)

    if modelResidu == None:
        modelResidu = 'model_trace_order1_residu.fits'
    if fit_iter2 == True:
        # Read the trace residu model from disk
        a = fits.open(modelResidu)
        modelResidu = a[0].data
        hdr = a[0].header
        osdimy2 = hdr['OVRSAMPY']
        semiwidthy2 = (np.shape(modelResidu)[0] - 1)/2/osdimy2
        if (osdimy2 != osdimy) or (semiwidthy2 != semiwidthy):
            print('fit_iter2 needs a residu model with same osdimy and semiwidthy as first iteration trace model.')
            print('osdimy=',osdimy)
            print('osdimy2=',osdimy2)
            print('semiwidthy=',semiwidthy)
            print('semiwidthy2=',semiwidthy2)
            stop

    if (cenx == None) or (ceny == None):
        # if the centroid x,y positions are not passed, then retrieve them
        if order == 1:
            cenx,ceny = get_order1_from_cv(atthesex=np.linspace(0,dimx-1,dimx))
        if order == 2:
            cenx,ceny,lba = extrapolate_order2(atthesex=np.linspace(0,dimx-1,dimx))
        if order == 3:
            cenx,ceny = get_order3_from_cv(atthesex=np.linspace(0,dimx-1,dimx))

    columnlist = list(columnlist)
    if list(columnlist) == None:
        columnlist = np.linspace(4,2040,2040)

    if fit_radius == None:
        fit_radius = 16

    # Initialize Residu and Model images
    imageResidu = imageForFit*1 # probably want to keep the NaN in there
    imageModel = np.zeros(np.shape(imageForFit)) # no NaN should be in there

    # Initialize the output spectrum with same length as columnlist
    spectrum = [] # not np.array(columnlist)

    # Initialize the x and f(x) of the model at the current column
    w = (np.shape(modelTrace)[0] - 1)/2
    pixelmodel = np.arange(2*w+1)/osdimy - w/osdimy
    pixeldata = np.linspace(0,255,256)

    # Loop over all columns
    for i in columnlist:
        x = int(i)
        tracecenter = ceny[x]
        valuemodel = modelTrace[:,x]
        if fit_iter2 == True: valueresidu = modelResidu[:,x]
        #ind = np.isfinite(valuemodel)
        if (order == 1) | ((order == 2) & (x <= 1800)):
            # The observed trace values
            valuedata = imageForFit[:,x]
            # Mask the data where we need to. Keep data
            # 1) where it is not a NaN;
            # 2) AND where it is within some radius from the trace center
            #    (do not fit the wings)
            mask = ~np.isnan(valuedata) & (np.abs(pixeldata - tracecenter) <= fit_radius)


            if fit_linearbackground == True:
                popt,pcov = curve_fit(tracemodel_lineartrend, pixeldata[mask], valuedata[mask],
                                  bounds=([-10,0,-1000,-10, -10000], [266, 1e+7, 1000,10,10000]),
                                  p0=[ceny[x],10,0,0,0])
                cond = np.abs(pixeldata-tracecenter) <= fit_radius
                imageModel[cond,x] = tracemodel_lineartrend(pixeldata,*popt)[cond]
                imageResidu[:,x] = imageForFit[:,x] - imageModel[:,x]
                spectrum.append(popt[1])
                if debug == True:
                    print('col={}, seed xo={:8.3f}, fit xo={:8.3f}, flux={:8f}'.format(
                            x, ceny[x], popt[0],popt[1]))
            elif fit_iter2 == True:
                popt,pcov = curve_fit(tracemodel_iter2, pixeldata[mask], valuedata[mask],
                                  bounds=([-10,0,-50,-10], [266, 1e+7, 50,1e+7]),
                                  p0=[ceny[x],10,0,0])
                cond = np.abs(pixeldata-tracecenter) <= fit_radius
                imageModel[cond,x] = tracemodel_iter2(pixeldata,*popt)[cond]
                imageResidu[:,x] = imageForFit[:,x] - imageModel[:,x]
                #spectrum.append(popt[1])
                spectrum.append(integrate_tracemodel_iter2(*popt))
                if debug == True:
                    print('col={}, seed xo={:8.3f}, fit xo={:8.3f}, flux={:8f}'.format(
                            x, ceny[x], popt[0],popt[1]))
            else:
                popt,pcov = curve_fit(tracemodel, pixeldata[mask], valuedata[mask],
                                  bounds=([-10,0,-500], [266, 1e+7, 500]),
                                  p0=[ceny[x],10,0])

                cond = np.abs(pixeldata-tracecenter) <= fit_radius
                imageModel[cond,x] = tracemodel(pixeldata,*popt)[cond]
                imageResidu[:,x] = imageForFit[:,x] - imageModel[:,x]
                spectrum.append(popt[1])
                if debug == True:
                    print('col={}, seed xo={:8.3f}, fit xo={:8.3f}, flux={:8f}'.format(
                            x, ceny[x], popt[0],popt[1]))
            # Plot the current column data
            if debug == True:
                if i == columnlist[0]: fig = plt.figure()
                plt.scatter(pixeldata[mask], valuedata[mask],marker='.',
                            color='black')
        else:
            spectrum.append(np.nan)

    spectrum = np.array(spectrum)

    if debug == True:
        hdu = fits.PrimaryHDU()
        hdu.data = imageResidu
        hdu.writeto('imageResidu.fits',overwrite=True)

        hdu = fits.PrimaryHDU()
        hdu.data = imageModel
        hdu.writeto('imageModel.fits',overwrite=True)

    if save_residu != None:
        hdu = fits.PrimaryHDU()
        hdu.data = imageResidu
        hdu.writeto(save_residu,overwrite=True)

    if save_model != None:
        hdu = fits.PrimaryHDU()
        hdu.data = imageModel
        hdu.writeto(save_model,overwrite=True)

    if (return_residu == True) and (return_model == True):
        return(spectrum,imageResidu,imageModel)
    if (return_residu == True) and (return_model == False):
        return(spectrum,imageResidu)
    if (return_residu == False) and (return_model == True):
        return(spectrum,imageModel)
    return(spectrum)



def return_monochromatic_trace_model(wavelength, interpolate=False,
                                     force_order2=False):
    # This function is called to build the monochromatic trace model and
    # return it for a given input wavelength.
    # input is the wavelength (in nm or microns)
    # output is the oversampled trace model

    a = fits.open('model_order1.fits')

    order1 = a[0].data
    hdr = a[0].header
    osdimy1 = hdr['OVRSAMPY']
    semiwidthy1 = (np.shape(order1)[0] - 1)/2/osdimy1
    column1 = np.linspace(4,2043,2040)
    wavelength1 = spectral_pixel_to_wavelength(column1, order=1)

    a = fits.open('model_order2.fits')
    order2 = a[0].data
    hdr = a[0].header
    osdimy2 = hdr['OVRSAMPY']
    semiwidthy2 = (np.shape(order2)[0] - 1)/2/osdimy2
    column2 = np.linspace(4,2043,2040)
    wavelength2 = spectral_pixel_to_wavelength(column2, order=2)

    if wavelength <= 6:
        # units requested are probably microns
        wavelength = wavelength*1000. # convert to nanometers

    if interpolate == True:
        # interpolate the model at the exact wavelength
        # do nothing for now.
        print('interpolation of trace model not implemented yet.')
        stop
    else:
        # return the trace model nearest in wavelength (no interpolation)
        deltawave = np.abs(wavelength1-wavelength)
        cond = deltawave == np.min(deltawave)
        ind = np.where(cond == True)[0][0]
        tracemodel = order1[:,ind]*1
        if (wavelength1[cond] == np.min(wavelength1)) or (force_order2 == True):
            # The requested wavelenght is at the blue end of first order.
            # Let's use the second order model instead
            deltawave = np.abs(wavelength2-wavelength)
            cond = deltawave == np.min(deltawave)
            ind = np.where(cond == True)[0][0]
            tracemodel = order2[:,ind]*1
            return(tracemodel,osdimy2,semiwidthy2)
        else:
            return(tracemodel,osdimy1,semiwidthy1)




def create_wavelength_map(dimx=None, dimy=None, order=None, osx=None, osy=None,
                          cenx=None, ceny=None, mask=True, semiwidth=None,
                          savefits=None, tiltmodel=None, angle=None):
    # See interp_variabletilt_1.ipynb
    # Assuming the trace position function
    # Assuming a pixel-dependent tilt function

    # set the default values for optional input parameters
    if (osx == None): osx = 1 # ovsesampling of output map
    if (osy == None): osy = 1
    if (semiwidth == None): semiwidth = 20
    if (order == None): order = 1
    if (dimx == None): dimx = 2048 #native pixel dimension
    if (dimy == None): dimy = 256

    subdim = 32    # dimension of the piece-by-piece interpolation
    subn = int(dimx/subdim) # needs to yield an integer
    pad = 5 # native pixels pad to build interpolation function
    speedit = 1 # factor by which we speed up the processing

    thex = np.linspace(0,dimx-1,dimx*osx)
    they = np.linspace(0,dimy-1,dimy*osy)

    # Read the trace centroid position
    if (cenx is None) or (ceny is None):
        # Read the best available trace centroid solution
        if (order == 1) or (order == None):
            cenx,ceny = get_order1_from_cv(atthesex=thex)
        if order == 2:
            cenx,ceny = get_order2_from_cv(atthesex=thex)

    # Read the trace tilt function
    if (order == 1) or (order == None):
        tilt,x = tilt_order1(x=cenx, unit='slope', model=tiltmodel, angle=angle)

    # Do a piece by piece map construction because:
    # The interp.Rbf breaks for input arrays of approximatively >5000 points
    # instead of doing a map of the wavelengths, instead make a map
    # of the trace center x position (cenx). We can later convert to wavelengths
    # using the cenx to wavelength calibration.
    mapx = np.zeros((dimy*osy,dimx*osx))
    for k in range(subn):
        # Now create arrays of x and of y and corresponding lambda from which
        # we can later interpolate.
        valx, valy, valz = [],[],[]
        they = np.linspace(-semiwidth*1.5,semiwidth*1.5,(semiwidth*1.5/speedit)*2+1)
        # Define the x where the interpolation will be establish. Add padding.
        if k > 0 and k < (subn-1):
            thex = np.linspace(k*subdim-pad,(k+1)*subdim+pad-1,subdim+2*pad)
        if k == 0:
            thex = np.linspace(k*subdim,(k+1)*subdim+pad-1,subdim+pad)
        if k == (subn-1):
            thex = np.linspace(k*subdim-pad,(k+1)*subdim-1,subdim+pad)

        for i in thex:
            ind = (i == cenx) #is a condition
            for j in range(len(they)):
                valx.append(cenx[ind][0]+they[j]*tilt[ind][0])
                valy.append(ceny[ind][0]+they[j])
                valz.append(cenx[ind][0])

        # Pixel grid of x and y - request the final x positions, pads don't
        # matter anymore.
        newx,newy = np.meshgrid(np.linspace(k*subdim,(k+1)*subdim-1,subdim*osx),
                                np.linspace(0,dimy-1,dimy*osy))
        # Interpolate
        print('interpolating for k=',k+1,' out of ',subn)#pixels=',thex)
        zfun_smooth_rbf = interp.Rbf(valx, valy, valz, function='linear', smooth=0)  # default smooth=0 for interpolation
        newz = zfun_smooth_rbf(newx, newy)  # not really a function, but a callable class instance
        mapx[:,k*subdim*osx:(k+1)*subdim*osx] = newz

    # Mask positions outside a reasonable width off the trace
    if mask == True:
        mapx = mask_trace(mapx, osx=osx, osy=osy, semiwidth=semiwidth,
                          order=order, cenx=cenx, ceny=ceny, nan=True)

    # Save on disk if requested
    if savefits != None:
        hdu = fits.PrimaryHDU()
        hdu.data = mapx
        hdu.writeto(savefits,overwrite=True)

    return(mapx)



def CCF_gaus(x,c,a,x0,sigma):
    return c+a*np.exp(-(x-x0)**2/(2*sigma**2))

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

    #fitfunc = 'parabola'
    #fitfunc = 'gauss'

    # Perform median filtering through the data to remove low-frequencies
    if False:
        ref = ref - signal.medfilt(ref, 25)
        cur = cur - signal.medfilt(cur, 25)

    # Perform the cross correlation function
    delta = np.linspace(-10, 10, 21, dtype='int')
    #print('delta = ',delta)
    delta_fine = np.linspace(-5, 5, 100)
    CCF = np.zeros(len(delta))
    for i in range(len(delta)):
        #print(i, 'delta=', delta)
        #a = np.correlate(cur,np.roll(ref,delta[i]))
        #print('CCF=',a)
        #if len(a) == 1:
        #    CCF[i] = a
        #else:
        #    CCF[i] = np.nan
        CCF[i] = np.correlate(cur, np.roll(ref, delta[i]), mode='valid')

    # Perform a fit of the CCF peak position
    # Either limit the fit to the data points very close to the peak
    if maxneighbors == True:
        # Select data points near the CCF peak only to fit
        indmax = np.where(CCF == np.max(CCF))
        indfit = np.linspace(indmax[0]-npix, indmax[0]+npix, npix*2+1, dtype='int')

        # Fit model function, either a gaussian or a parabola
        if fitfunc == 'parabola':
            popt, pcov = curve_fit(CCF_parabola, delta[indfit], CCF[indfit], p0=[0.0, np.max(CCF[indfit]), 4e-4])
            dx = popt[0]
        else:
            popt, pcov = curve_fit(CCF_gaus, delta[indfit], CCF[indfit], p0=[np.min(CCF), np.max(CCF)-np.min(CCF), 0, 1])
            dx = popt[2]
    # Or perform the fit on all available data points
    # (not recommended because of CCF asymmetry)
    else:
        popt, pcov = curve_fit(CCF_gaus, delta, CCF, p0=[np.min(CCF), np.max(CCF)-np.min(CCF), 0, 1])
        dx = popt[2]

    if makeplot is True:
        print('popt=',popt)

        fig = plt.figure(figsize=(15,4))
        plt.plot(delta,CCF,marker='s')
        plt.scatter(delta[indfit],CCF[indfit],marker='.',color='red',zorder=3)
        if fitfunc == 'parabola':
            plt.plot(delta_fine,CCF_parabola(delta_fine,popt[0],popt[1],popt[2]),color='orange')
            #plt.xlim((np.min(dy_fine),np.max(dy_fine)))
            #plt.ylim((np.min(CCF_parabola(dy_fine,popt[0],popt[1],popt[2])),np.max(CCF)))
        else:
            plt.plot(delta_fine,CCF_gaus(delta_fine,popt[0],popt[1],popt[2],popt[3]),color='orange')
            #plt.xlim((np.min(dy_fine),np.max(dy_fine)))
            #plt.ylim((np.min(CCF_gaus(dy_fine,popt[0],popt[1],popt[2],popt[3])),np.max(CCF)))
        #plt.xlim((-npix*2,npix*2))

    return(dx)


def patch_CV3_header(filename,outfilename=None):

    import gc
    from jwst.datamodels import RampModel
    import astropy.io.fits as fits
    import numpy as np

    # Patch the headers of a SOSS fits image taken at CV3 to enable
    # using the STScI pipeline to reduce it.

    if outfilename == None:
        outfilename = 'test_uncal.fits'

    filesoss = fits.open(filename)
    orig_data = filesoss[0].data
    tlscp = filesoss[0].header['TELESCOP']
    inst = filesoss[0].header['INSTRUME']
    filt = filesoss[0].header['FWCCRFIL']
    pup = filesoss[0].header['PWCCRPUP']
    nfrm = filesoss[0].header['NFRAME']
    nint = filesoss[0].header['NINT']
    ngrp = filesoss[0].header['NGROUP']
    grpgp = filesoss[0].header['GROUPGAP']
    tfrm = filesoss[0].header['TFRAME']
    tgrp = filesoss[0].header['TGROUP']
    inttme = filesoss[0].header['INTTIME']
    exptme = filesoss[0].header['EXPTIME']
    dobs = filesoss[0].header['DATE-OBS']
    dend = filesoss[0].header['DATE-END']
    tobs = filesoss[0].header['TIME-OBS']
    tend = filesoss[0].header['TIME-END']
    rdout = filesoss[0].header['READOUT']
    rdata = np.swapaxes(orig_data, 1, 2)[:, ::-1, ::-1]
    sh1 = rdata.shape
    orig_data = 0.
    filesoss.close()
    rampdq = np.zeros(sh1,dtype=np.int8)
    rdata = np.expand_dims(rdata,0)
    rampdq = np.expand_dims(rampdq,0)
    new_model = RampModel(data=rdata,groupdq=rampdq)
    new_model.meta.telescope = tlscp
    new_model.meta.instrument.name = inst
    new_model.meta.instrument.detector = 'NIS'
    new_model.meta.instrument.filter = filt
    new_model.meta.instrument.pupil = pup
    new_model.meta.exposure.type = 'NIS_SOSS'
    new_model.meta.exposure.nints = nint
    new_model.meta.exposure.ngroups = ngrp
    new_model.meta.exposure.nframes = nfrm
    new_model.meta.exposure.groupgap = grpgp
    new_model.meta.subarray.name = 'FULL'
    new_model.meta.subarray.xsize = rdata.shape[3]
    new_model.meta.subarray.ysize = rdata.shape[2]
    new_model.meta.subarray.xstart = 1
    new_model.meta.subarray.ystart = 1
    new_model.meta.subarray.fastaxis = -2
    new_model.meta.subarray.slowaxis = -1
    new_model.save(outfilename)
    fits.setval(outfilename, 'TFRAME', value=float(tfrm), ext=0, comment='Time in seconds between frames (sec)')
    fits.setval(outfilename, 'TGROUP', value=float(tgrp), ext=0, comment='Delta time between groups (sec)')
    fits.setval(outfilename, 'INTTIME', value=float(inttme), ext=0, comment='Total integration time for one MULTIACCUM (sec)')
    fits.setval(outfilename, 'EXPTIME', value=float(exptme), ext=0, comment='Exposure duration calculated (sec)')
    fits.setval(outfilename, 'DATE-OBS', value=str(dobs), ext=0, comment='UT date of observation (yyyy-mm-dd)')
    fits.setval(outfilename, 'DATE-END', value=str(dend), ext=0, comment='UT date end of observation (yyyy-mm-dd)')
    fits.setval(outfilename, 'TIME-OBS', value=str(tobs), ext=0, comment='Approximate UT start time of observation')
    fits.setval(outfilename, 'TIME-END', value=str(tend), ext=0, comment='UT time of end of observation')
    fits.setval(outfilename, 'READPATT', value=rdout, ext=0, comment='Readout pattern name')
    new_model = 0.
    gc.collect()

    return()



def build_superbias(case = None, returnmodulo = None):

    # List of integrations obtained at CV3 of darks in the SUBSTRIP256 and
    # SUBSTRIP96 subarrays.

    if case == None: case = '256NG3'

    if case == '256NG50':
        # Here, 256x2048, each of 3 integrations of ngroup=50
        list256_ngroup50 = [
                'NISNIS-SS256-160052217_11_496_SE_2016-01-06T01h33m21.fits',
                'NISNIS-SS256-160052217_28_496_SE_2016-01-06T05h48m36.fits',
                'NISNIS-SS256-160060813_11_496_SE_2016-01-06T11h29m49.fits',
                'NISNIS-SS256-160060813_28_496_SE_2016-01-06T15h57m39.fits',
                'NISNIS-SS256-160080524_11_496_SE_2016-01-08T08h41m00.fits',
                'NISNIS-SS256-160080524_28_496_SE_2016-01-08T12h53m58.fits']
        nexp = len(list256_ngroup50)
        for i in range(nexp):
            list256_ngroup50[i] = '/Users/albert/NIRISS/CV3/myanalysis/superbias/'+list256_ngroup50[i]

        # Read in the exposures
        cube = np.zeros((nexp*3,50,256,2048))
        for i in range(nexp):
            expo = fits.open(list256_ngroup50[i])
            cube[i*3,:,:,:] = image_native_to_DMS(expo[0].data[0:50,:,:])
            cube[i*3+1,:,:,:] = image_native_to_DMS(expo[0].data[50:100,:,:])
            cube[i*3+2,:,:,:] = image_native_to_DMS(expo[0].data[100:150,:,:])

        print(np.shape(cube))
        superb = np.median(cube,axis=0)
        hdu = fits.PrimaryHDU()
        hdu.data = superb
        hdu.writeto('/Users/albert/NIRISS/CV3/myanalysis/superbias/superbias256_ngroup50.fits',overwrite=True)

        return(superb, cube)


    if case == '256NG3':
        # Here, 256x2048, each of 50 integrations of ngroup=3
        list256_ngroup3 = [
            'NISNIS-SS256-160052217_10_496_SE_2016-01-06T01h21m42.fits',
            'NISNIS-SS256-160052217_25_496_SE_2016-01-06T04h57m33.fits',
            'NISNIS-SS256-160052217_26_496_SE_2016-01-06T05h15m10.fits',
            'NISNIS-SS256-160052217_27_496_SE_2016-01-06T05h34m20.fits',
            'NISNIS-SS256-160052217_8_496_SE_2016-01-06T00h44m23.fits',
            'NISNIS-SS256-160052217_9_496_SE_2016-01-06T00h59m52.fits',
            'NISNIS-SS256-160060813_10_496_SE_2016-01-06T11h15m31.fits',
            'NISNIS-SS256-160060813_25_496_SE_2016-01-06T14h53m22.fits',
            'NISNIS-SS256-160060813_26_496_SE_2016-01-06T15h19m32.fits',
            'NISNIS-SS256-160060813_27_496_SE_2016-01-06T15h31m51.fits',
            'NISNIS-SS256-160060813_8_496_SE_2016-01-06T10h37m50.fits',
            'NISNIS-SS256-160060813_9_496_SE_2016-01-06T10h56m41.fits',
            'NISNIS-SS256-160080524_10_496_SE_2016-01-08T08h28m22.fits',
            'NISNIS-SS256-160080524_25_496_SE_2016-01-08T12h02m27.fits',
            'NISNIS-SS256-160080524_26_496_SE_2016-01-08T12h20m48.fits',
            'NISNIS-SS256-160080524_27_496_SE_2016-01-08T12h39m38.fits',
            'NISNIS-SS256-160080524_8_496_SE_2016-01-08T07h48m32.fits',
            'NISNIS-SS256-160080524_9_496_SE_2016-01-08T08h06m53.fits']
        nexp = len(list256_ngroup3)
        for i in range(nexp):
            list256_ngroup3[i] = '/Users/albert/NIRISS/CV3/myanalysis/superbias/'+list256_ngroup3[i]

        # Read in the exposures
        cube = np.zeros((nexp*50,3,256,2048))
        for i in range(nexp):
            expo = fits.open(list256_ngroup3[i])
            for j in range(50):
                cube[i*50+j,:,:,:] = image_native_to_DMS(expo[0].data[j*3:(j+1)*3,:,:])

        if returnmodulo == True:
            intnbr = np.tile(range(50),nexp)
            ind = (intnbr % 4) == 0
            cubemodulo0 = cube[ind,:,:,:]
            ind = (intnbr % 4) == 1
            cubemodulo1 = cube[ind,:,:,:]
            ind = (intnbr % 4) == 2
            cubemodulo2 = cube[ind,:,:,:]
            ind = (intnbr % 4) == 3
            cubemodulo3 = cube[ind,:,:,:]
            superb0 = np.median(cubemodulo0,axis=0)
            hdu = fits.PrimaryHDU()
            hdu.data = superb0
            hdu.writeto('/Users/albert/NIRISS/CV3/myanalysis/superbias/superbias256_ngroup3_modulo0.fits',overwrite=True)
            superb1 = np.median(cubemodulo1,axis=0)
            hdu = fits.PrimaryHDU()
            hdu.data = superb1
            hdu.writeto('/Users/albert/NIRISS/CV3/myanalysis/superbias/superbias256_ngroup3_modulo1.fits',overwrite=True)
            superb2 = np.median(cubemodulo2,axis=0)
            hdu = fits.PrimaryHDU()
            hdu.data = superb2
            hdu.writeto('/Users/albert/NIRISS/CV3/myanalysis/superbias/superbias256_ngroup3_modulo2.fits',overwrite=True)
            superb3 = np.median(cubemodulo3,axis=0)
            hdu = fits.PrimaryHDU()
            hdu.data = superb3
            hdu.writeto('/Users/albert/NIRISS/CV3/myanalysis/superbias/superbias256_ngroup3_modulo3.fits',overwrite=True)

        print(np.shape(cube))
        superb = np.median(cube,axis=0)
        hdu = fits.PrimaryHDU()
        hdu.data = superb
        hdu.writeto('/Users/albert/NIRISS/CV3/myanalysis/superbias/superbias256_ngroup3.fits',overwrite=True)

        if returnmodulo == True:
            return(cubemodulo0,cubemodulo1,cubemodulo2,cubemodulo3)
        else:
            return(superb, cube)

def subtract_1overf_from_TSO(rawCDScube,badpix_in=None, weigh=False,
                             method='deepstack', return_mask=False,
                             verbose=False):
    # pass a raw cube of n CDS

    # Make sure of the orientation of the raw CDS cube
    ncds, naxis2, naxis1 = np.shape(rawCDScube)
    derot = False
    if naxis2 > naxis1:
        derot = True
        # vertical sub array - fast axis is along x
        if ncds > 1:
            # the raw image is a cube
            raw = np.rot90(rawCDScube,axes=(1,2),k=3)
        else:
            # the raw image is not a cube (only 1 slice)
            raw = np.rot90(rawCDScube,k=3)
    else:
        raw = rawCDScube*1.0
    ncds, nfast, nslow = np.shape(raw)
    if verbose is True:
        print('np.shape of the raw cube, after rotation is: ',np.shape(raw))

    # Make sur that if a bad pixel mask was passed, that its orientation is ok
    if badpix_in is None:
        badpix = np.zeros((nfast,nslow))+1.0 # all good pixels
    else:
        naxis2, naxis1 = np.shape(badpix_in)
        if naxis2 > naxis1:
            # vertical sub array - fast axis is along x
            badpix = np.rot90(badpix_in,k=3)
        else:
            badpix = badpix_in*1.0

    # GO with the stacking and subtracting

    # Create a deep stack (or running median deep stack)
    if method == 'deepstack':
        deepstack = np.median(raw*badpix,axis=0)

        # Create a mask of pixels that do not receive flux
        mask = mask_pixels_with_flux(deepstack, verbose=verbose)
        # Initialize the cube of 1/f measurements
        oneoverfcube = raw*0
        # The cube of images to be used to evaluate the i/f levels
        rawsubtracted = raw*badpix*mask - deepstack

        if verbose is True:
            plt.figure(figsize=(15,15))
            frame = plt.imshow(rawsubtracted[ncds-1,:,:],origin='lower')
            plt.title('Raw image - deep image, masked')
            plt.colorbar(frame,orientation='horizontal')
            plt.show()

        # Measure the level at each column
        if weigh is True: weightmap = 1.0/order1_distance()
        for i in range(ncds):
            ## Create a map of NaNs for use later in the numpy.ma.average
            #masknan = np.full(np.shape(rawsubtracted[i,:,:]),True)
            #masknan[np.isnan(rawsubtracted[i,:,:])] = False
            for j in range(nslow):
                col = rawsubtracted[i,:,j]
                ind = ~np.isnan(col)
                nind = np.size(np.where(ind == True))
                if nind >= 1:
                    if weigh is True:
                        # Weigh each pixel in the column according to its distance
                        # from the trace center.
                        colweight = weightmap[:,j]
                        med = np.average(col[ind],weights=colweight[ind])
                    else:
                        # Simply median all pixels in the column
                        med = np.average(col[ind])
                else:
                    med = np.nan

                oneoverfcube[i,:,j] = med

        # Apply the 1/f subtraction
        rawclean = raw - oneoverfcube

    if method == 'runningmedian':
        #TBW
        print('The runnningmedian method is not implemented yet. Stop.')

    if derot is True:
        if ncds > 1:
            rawclean = np.rot90(rawclean,axes=(1,2),k=1)
            oneoverfcube = np.rot90(oneoverfcube,axes=(1,2),k=1)
            rawsubtracted = np.rot90(rawsubtracted,axes=(1,2),k=1)
        else:
            rawclean = np.rot90(rawclean,k=1)
            oneoverfcube = np.rot90(oneoverfcube,k=1)
            rawsubtracted = np.rot90(rawsubtracted,k=1)

    if return_mask is True:
        return(rawclean,oneoverfcube,rawsubtracted,mask)
    else:
        return(rawclean,oneoverfcube,rawsubtracted)


def mask_pixels_with_flux(deepimage, verbose=False):

    nrow, ncol = np.shape(deepimage)
    if verbose is True:
        print('deepimage has {0:} col by {1:} row'.format(ncol,nrow))

    # make sure it's horizontal
    if nrow > ncol:
        # make it horizontal
        deepimage = np.rot90(deepimage,k=3)
        nrow, ncol = np.shape(deepimage)
        if verbose is True:
            print('Rotate deepimage by 90 degrees')
            print('deepimage has {0:} col by {1:} row'.format(ncol,nrow))

    mask = deepimage*0+1    # all ones unless deepimage has NaNs

    # Detect pixels that have flux with a rough threshloding
    # ensures that the image scaling is roughly similar to adus on raw images
    image = 50000.0 * deepimage / np.nanmax(deepimage)
    threshold = 500.0 # adu
    readoutnoise = 20 # adu

    mask[image >= threshold] = np.nan

    if verbose is True:
        plt.figure(figsize=(15,15))
        frame = plt.imshow(mask,origin='lower')
        plt.title('Simple thresholding of {0:}'.format(threshold))
        plt.colorbar(frame,orientation='horizontal')
        plt.show()

    if False:
        # column by column
        pixel = np.linspace(0,nrow-1,nrow)
        masked = image * mask
        for i in range(ncol):
            col = masked[:,i]
            sortedcol = np.sort(col[~np.isnan(col)])
            # identify the peak of the histogram
            histo,bin_edges = np.histogram(sortedcol, range=(-20,threshold), bins=10)
            bmin = (bin_edges[np.where(histo == np.max(histo))])[0] - 10
            bmax = bmin + 20
            med = np.median(sortedcol[(sortedcol >= bmin) & (sortedcol <= bmax)])
            # mask further the pixels in that column 1 readout noise sigma above
            # the median level just found above
            maskcol = mask[:,i]*1
            maskcol[col >= med+2*readoutnoise] = np.nan
            mask[:,i] = maskcol*1

            if (verbose is True) & (i == 100):
                plt.figure(figsize=(15,8))
                plt.title('Simple thresholding of {0:}'.format(threshold))
                plt.scatter(pixel,col)
                plt.plot([np.nanmin(pixel),np.nanmax(pixel)],[med,med],color='red')
                plt.show()
                # histogram
                plt.figure(figsize=(15,8))
                plt.hist(sortedcol)
                plt.plot([med,med],[0,np.nanmax(histo)],color='red')
                plt.show()
                print(histo)
                print(bmin,bmax)
                print(med)

        if verbose is True:
            plt.figure(figsize=(15,15))
            frame = plt.imshow(mask,origin='lower')
            plt.title('After line-by-line thresholding')
            plt.colorbar(frame,orientation='horizontal')
            plt.show()

    return(mask)

def order1_aperture(semiwidth=17,osx=1,osy=1):
    # Returns a mask defined to 1 on the first order trace with a semi width
    # of semiwidth pixels, zero everywhere else. The full width aperture will
    # be width = 2*semiwidth+1. All in the DMS coordinate reference frame.

    image = np.zeros((256*osy,2048*osx))
    mask = mask_trace(image, osx=osx, osy=osy, order=1, semiwidth=semiwidth,
                      returnmask=True)
    mask_zero_or_one = np.zeros(np.shape(mask))
    mask_zero_or_one[mask] = 1

    # Convert to the DMS reference frame
    mask_zero_or_one = np.fliplr(mask_zero_or_one)

    return(mask_zero_or_one)

def order1_trace_model(osx=1, osy=1, cenx=None, ceny=None,
                       subarray='SUBSTRIP256'):
    # Returns a 2D map of the trace model, by default to the native pixel
    # resolution and for the default trace position.

    # Warning, the oversampling in x and y are NOT IMPLEMENTED YET. Needs work.

    # Read the model trace for order 1
    a = fits.open('LoicsWork/InputFiles/trace_order1.fits')
    model = a[0].data
    # read from header the oversampling in y
    osdimy = a[0].header['OVRSAMPY']
    semiwidthy = int((np.shape(model)[0] - 1) / osdimy / 2)
    #print(semiwidthy)
    modely = np.linspace(-semiwidthy,semiwidthy,osdimy*2*semiwidthy+1)
    #print(modely)
    # Read the trace center (or pass a different one eventually)
    if (cenx is None) | (ceny is None):
        cenx, ceny = get_order1_from_cv(atthesex=np.linspace(0,2047,2048))

    # Project on a 2D map
    ymap = np.linspace(0,255,256)
    map2D = np.zeros((256,2048))*np.nan
    for x in cenx:
        x = int(x)
        #print(cenx[x],ceny[x])
        z = model[:,int(cenx[x])]
        y = ceny[x]+modely
        slicemap = np.interp(ymap,y,z)
        ind = np.where((ymap > ceny[x]-semiwidthy) & (ymap < ceny[x]+semiwidthy))
        map2D[ind,x] = slicemap[ind]

    if subarray == 'FULLFRAME':
        # Should really have a model that's larger so wings are ok for FF.
        # But for now, we simply copy and paste the SUBSTRIP256 into FF.
        tmp = np.zeros((2048,2048))*np.nan
        tmp[0:255,:] = map2D*1
        map2D = tmp*1

    if subarray == 'SUBSTRIP96':
        # The SUBSTRIP96 model is simply the ones generated for SUBSTRIP256
        # pasted with the proper offset in pixels of -10. The subarray starts
        # at position x=151 instead of x=1 for the SUBSTRIP256.
        tmp = np.zeros((96,2048))*np.nan
        tmp = map2D[150:245,:]
        map2D = tmp*1

    # Save the trace
    #print('Projection saved.')
    #hdu = fits.PrimaryHDU()
    #hdu.data = map2D
    #hdu.writeto('ModelTrace_2D_osy1.fits',overwrite=True)

    # Convert to DMS coordinate frame
    map2D = np.fliplr(map2D)

    return(map2D)



def order1_position(subarray='SUBSTRIP256'):
    # The pixel x, y central position (in the DMS coordinate reference frame)
    # of the first order trace.

    # Get the x,y pixel positions in the coordinate reference frame commonly
    # used at UdeM. Convert to DMS.
    x, y = get_order1_from_cv()
    x = 2047.0-x

    return(x,y)


def order1_distance(osx=1,osy=1):
    # Returns the distance of the pixel to the center of the first order trace
    # osx and osy are the oversampling relative to native pixels of the input
    # image.
    image = np.zeros((256*osy,2048*osx))
    mask = image * 0.0

    # set the default values for optional input parameters
    order = 1

    dimx = np.shape(image)[1]/osx
    dimy = np.shape(image)[0]/osy

    thex = np.linspace(0,dimx-1,dimx*osx)
    they = np.linspace(0,dimy-1,dimy*osy)

    # Read the trace centroid position
    # Read the best available trace centroid solution
    if (order == 1) or (order == None):
        cenx,ceny = get_order1_from_cv(atthesex=thex)
    if order == 2:
        cenx,ceny = get_order2_from_cv(atthesex=thex)

    # Find the y-axis distance from trace center, at each x position
    for i in range(len(cenx)):
        yo = ceny[i]
        d = np.absolute(they-yo)
        mask[:,i] = d

    mask = np.fliplr(mask)

    return(mask)

def order1_pixel_to_wavelength(pixel, order=1):
    # Converts a pixel to a wavelength

    if order == 1:
        rezz = [-2.03489e-5,1.09890,-905.219]
        wavelength = (-rezz[1] + np.sqrt(rezz[1]*rezz[1] - 4*(rezz[2]-pixel)*rezz[0]))/(2*rezz[0])
    if order == 2:
        rezz = [2.14318,-976.566]
        wavelength = (pixel - rezz[1])/rezz[0]

    return(wavelength)
