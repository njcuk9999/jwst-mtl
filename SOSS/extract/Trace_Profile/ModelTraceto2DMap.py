#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for creating a model SOSS trace profile, and placing
it on the detector.

Created on Fri May 29 11:58:29 2020

@author: MCR
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling import models, fitting
from tqdm import tqdm
import sys
sys.path.insert(1, '../../trace/')
import tracepol as tp


def calibrate_fit(trace, os=5, calslice=4, plot=False):
    '''Calibrate starting parameters for the trace fit by fitting the
    desired model to the first column in the rectified trace.
    _________________________________________________
    Inputs: trace - the rectified trace model to fit
            os - scale of spatial oversampling
            calslice - the detector column number to use for calibration
            plot - show a plot of the results?
    Ouputs: arrays of the best fitting amplitudes (amp), centers (pos), and
            widths (wid) of the four Gaussians in the trace model.
    '''

    # Set initial conditions for the first slice
    x = np.arange(64*os)
    data_init = trace[:, calslice]  # Get calibration frame from trace
    amp, pos, wid = [0.006, 0.006, 0.01, 0.002], [150, 190, 120, 150],\
        [8, 8, 8, 45]  # first guesses

    # Do 3 fits using the results of the previous fit as the inputs of the next
    for i in range(3):
        G_init = construct_model_4G(amp, pos, wid)

        # Fit model to data
        fit_init = fitting.LevMarLSQFitter()
        g_init = fit_init(G_init, x, data_init)

        # Determine Chi^2
        Csq = chi2(data_init, g_init(x))
        if i == 2 and np.abs((Csq - Csq_o)/Csq_o) > 0.1:
            print('Models have not converged on best fitting parameters in\
            three iterations')
        Csq_o = Csq

        # Update best fit parameters
        amp = [g_init.amplitude_0[0], g_init.amplitude_1[0],
               g_init.amplitude_2[0], g_init.amplitude_3[0]]
        pos = [g_init.mean_0[0], g_init.mean_1[0],
               g_init.mean_2[0], g_init.mean_3[0]]
        wid = [g_init.stddev_0[0], g_init.stddev_1[0],
               g_init.stddev_2[0], g_init.stddev_3[0]]

    if plot is True:
        # Show the fit results
        fig = plt.figure(figsize=(7, 5))

        ax_model = fig.add_axes([0, 0.22, 1, 0.78])
        ax_model.tick_params(direction='in', labelbottom=False)
        ax_resid = fig.add_axes([0, 0, 1, 0.2])

        ax_model.plot(x, g_init(x), c='black', label='Model', alpha=0.8)
        ax_model.scatter(x, data_init, s=2, c='grey', label='Trace Data')
        ax_model.set_ylabel('Trace Profile', fontsize=12)
        ax_model.legend(fontsize=10)

        ax_resid.plot(x, 100*(g_init(x) - data_init)/data_init)
        ax_resid.set_xlabel('Spatial Pixel', fontsize=12)
        ax_resid.set_ylabel('Residuals [%]', fontsize=12)

        ax_model.set_title('Calibration Fit: column=%s' % (calslice),
                           fontsize=14)

        return amp, pos, wid

    else:
        return amp, pos, wid


def chi2(data, model):
    '''Simple Chi^2 calculation.'''
    chi2 = np.sum((data - model)**2 / data)
    return chi2


def construct_model_4G(amp, mean, sig, bounds=None):
    '''Create model of 'horned' Gaussian trace profile
    _________________________________________________
    Inputs: amp - array of amplitudes of individual Gaussians
            mean - array of individual Gaussian means
            sig - array of individual Gaussian std deviations
            bounds - percent allowance for variation of parameters
                     from one pixel column to the next.
    Ouputs: astropy model of four summed Gaussians
    '''

    # If fitting bounds are not required
    if bounds is None:
        g1 = models.Gaussian1D(amp[0], mean[0], sig[0])
        g2 = models.Gaussian1D(amp[1], mean[1], sig[1])
        g3 = models.Gaussian1D(amp[2], mean[2], sig[2])
        g4 = models.Gaussian1D(amp[3], mean[3], sig[3])

    # Add percent fitting bounds
    else:
        ubound = 1 + bounds/100
        dbound = 1 - bounds/100
        g1 = models.Gaussian1D(amp[0], mean[0], sig[0], bounds={
            "amplitude": (dbound*amp[0], ubound*amp[0]),
            "mean": (dbound*mean[0], ubound*mean[0]),
            "stddev": (dbound*sig[0], ubound*sig[0])})
        g2 = models.Gaussian1D(amp[1], mean[1], sig[1], bounds={
            "amplitude": (dbound*amp[1], ubound*amp[1]),
            "mean": (dbound*mean[1], ubound*mean[1]),
            "stddev": (dbound*sig[1], ubound*sig[1])})
        g3 = models.Gaussian1D(amp[2], mean[2], sig[2], bounds={
            "amplitude": (dbound*amp[2], ubound*amp[2]),
            "mean": (dbound*mean[2], ubound*mean[2]),
            "stddev": (dbound*sig[2], ubound*sig[2])})
        g4 = models.Gaussian1D(amp[3], mean[3], sig[3], bounds={
            "amplitude": (dbound*amp[3], ubound*amp[3]),
            "mean": (dbound*mean[3], ubound*mean[3]),
            "stddev": (dbound*sig[3], ubound*sig[3])})

    return g1 + g2 + g3 + g4


def fit_rectrace(trace, Amp, Pos, Wid, os=5, plot=False):
    '''Fit desired model to each detector column
    _________________________________________________
    Inputs: trace - the rectified trace model to be fit
            Amp - starting guesses for the four Gaussian amplitudes
            Po - starting guesses for the four Gaussian positions
            Wid - starting guesses for the four Gaussian widths
            os - scale of spatial oversampling
            plot - show a plot of the results?
    Ouputs: dictionary of best fitting parameters for the four model
            Gaussians, as well as the fit Chi2 at each spectral column.
    '''

    x = np.arange(64*os)
    fit_4g = fitting.LevMarLSQFitter()

    # Output parameter dictionary
    params = dict()
    params['Amp_c'] = []
    params['Amp_r'] = []
    params['Amp_l'] = []
    params['Amp_w'] = []
    params['Mean_c'] = []
    params['Mean_r'] = []
    params['Mean_l'] = []
    params['Mean_w'] = []
    params['stddev_c'] = []
    params['stddev_r'] = []
    params['stddev_l'] = []
    params['stddev_w'] = []
    params['Chi2'] = []

    for i in tqdm(range(2040)):
        psf_slice = trace[:, i+4]  # First and last 4 cols of CV3 are nans

        # Do the fit
        mod4G = construct_model_4G(Amp, Pos, Wid, bounds=2)
        g = fit_4g(mod4G, x, psf_slice, acc=1e-9, maxiter=110)
        X2 = chi2(psf_slice, g(x))

        # Update best fit params
        Amp = [g.amplitude_0[0], g.amplitude_1[0], g.amplitude_2[0],
               g.amplitude_3[0]]
        Pos = [g.mean_0[0], g.mean_1[0], g.mean_2[0], g.mean_3[0]]
        Wid = [g.stddev_0[0], g.stddev_1[0], g.stddev_2[0], g.stddev_3[0]]

        # Store results
        params['Amp_c'].append(Amp[0])
        params['Amp_r'].append(Amp[1])
        params['Amp_l'].append(Amp[2])
        params['Amp_w'].append(Amp[3])
        params['Mean_c'].append(Pos[0])
        params['Mean_r'].append(Pos[1])
        params['Mean_l'].append(Pos[2])
        params['Mean_w'].append(Pos[3])
        params['stddev_c'].append(Wid[0])
        params['stddev_r'].append(Wid[1])
        params['stddev_l'].append(Wid[2])
        params['stddev_w'].append(Wid[3])
        params['Chi2'].append(X2)

    if plot is True:
        # Plot results
        fig = plt.figure(figsize=(12, 8))

        ax_X = fig.add_axes([0, 0.5, 0.45, 0.45])
        ax_X.tick_params(direction='in', labelbottom=False)
        ax_A = fig.add_axes([0, 0.025, 0.45, 0.45])
        ax_A.tick_params(direction='in')
        ax_M = fig.add_axes([0.5, 0.025, 0.45, 0.45])
        ax_M.tick_params(direction='in')
        ax_S = fig.add_axes([0.5, 0.5, 0.45, 0.45])
        ax_S.tick_params(direction='in', labelbottom=False)

        ax_A.plot(np.arange(2040)+4, params['Amp_c'], label='Center')
        ax_A.plot(np.arange(2040)+4, params['Amp_l'], label='Left')
        ax_A.plot(np.arange(2040)+4, params['Amp_r'], label='Right')
        ax_A.plot(np.arange(2040)+4, params['Amp_w'], label='Wings')
        ax_A.set_ylabel('Amplitude')
        ax_A.legend()

        ax_M.plot(np.arange(2040)+4, params['Mean_c'])
        ax_M.plot(np.arange(2040)+4, params['Mean_l'])
        ax_M.plot(np.arange(2040)+4, params['Mean_r'])
        ax_M.plot(np.arange(2040)+4, params['Mean_w'])
        ax_M.set_ylabel('Mean')

        ax_S.plot(np.arange(2040)+4, params['stddev_c'])
        ax_S.plot(np.arange(2040)+4, params['stddev_l'])
        ax_S.plot(np.arange(2040)+4, params['stddev_r'])
        ax_S.plot(np.arange(2040)+4, params['stddev_w'])
        ax_S.set_ylabel('Std Deviation')

        ax_X.plot(np.arange(2040)+4, params['Chi2'], c='black')
        ax_X.set_ylabel(r'$\chi^2$')

        ax_bord = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax_bord.set_xticks([])
        ax_bord.set_yticks([])
        ax_bord.set_xlabel('Spectral Pixel', fontsize=12)

        return params

    else:
        return params


def fit_specpoly(params_4G, os=1, method=None, plot=False):
    '''Fit polynomials to the spectral variation of the parameters of
    the Gaussian trace profile model.
    _________________________________________________
    Inputs: params_4G - dictionary of best fitting Gaussian parameters as
                        returned by fit_rectrace.
            os - oversampling scale of the trace model that was fit.
            method - if 'resweight', the best fitting polynomial model is
                     selected based on its BIC multiplied by model residuals.
                     Otherwise, just the BIC is used.
            plot - show a plot of the results?
    Ouputs: dictionary of the best fitting polynomial parameters.
    '''

    Outparams = dict()

    for par in params_4G:
        if par == 'Chi2':  # Don't fit the Chi^2
            continue

        # Running storage of best fit values and parameters
        Outparams[par] = []
        Model, BIC, res = [], [], []

        # Unsure why this is necessary. Amplitudes fit with 0 order otherwise
        if par in ['Amp_c', 'Amp_r', 'Amp_l', 'Amp_w']:
            order = 2
        else:
            order = 0

        # Try up to a 9th order polynomial
        for i in range(10):
            model_i = np.polyfit(np.arange(2040)+4, params_4G[par], order)
            Model.append(model_i)  # store polynomial params
            back = np.polyval(model_i, np.arange(2040)+4)

            res.append(np.mean(np.abs(back - params_4G[par])))  # calculate model residuals
            C = chi2(params_4G[par], back)  # calculate chi2

            # If best fit is chosen by BIC and residuals
            if method == 'resweight' or par in ['Amp_c', 'Amp_r', 'Amp_l',
                                                'Amp_w']:
                BIC.append((np.log(64*os)*order + C)*res[i])

            # If best fit is only chosen by BIC
            else:
                BIC.append((np.log(64*os)*order + C))

            order += 1

        # Find the mininum BIC (or res weighted BIC)
        minBIC = np.where((BIC == np.min(BIC)))[0][0]

        # If model residuals are less than half a native pixel
        if res[minBIC] < 0.5*os:
            Outparams[par].append(Model[minBIC])
        else:
            print('No parameters found for %s' % par)
            Outparams[par].append('NOPARAMS')

    if plot is True:
        fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, figsize=(15, 12))
        ax[0, 0].plot(np.arange(2040)+4, params_4G['Amp_c'], ls=':')
        ax[0, 0].plot(np.arange(2040)+4, np.polyval(Outparams['Amp_c'][0],
                      np.arange(2040)+4), c='black')
        ax[0, 0].set_ylabel('Amplitude', fontsize=12)
        ax[0, 1].plot(np.arange(2040)+4, params_4G['Amp_l'], ls=':')
        ax[0, 1].plot(np.arange(2040)+4, np.polyval(Outparams['Amp_l'][0],
                      np.arange(2040)+4), c='black')
        ax[0, 2].plot(np.arange(2040)+4, params_4G['Amp_r'], ls=':')
        ax[0, 2].plot(np.arange(2040)+4, np.polyval(Outparams['Amp_r'][0],
                      np.arange(2040)+4), c='black')
        ax[0, 3].plot(np.arange(2040)+4, params_4G['Amp_w'], ls=':')
        ax[0, 3].plot(np.arange(2040)+4, np.polyval(Outparams['Amp_w'][0],
                      np.arange(2040)+4), c='black')

        ax[1, 0].plot(np.arange(2040)+4, params_4G['Mean_c'], ls=':')
        ax[1, 0].plot(np.arange(2040)+4, np.polyval(Outparams['Mean_c'][0],
                      np.arange(2040)+4), c='black')
        ax[1, 0].set_ylabel('Position [pixels] \n (%sx Oversampled)' % os,
                            fontsize=12)
        ax[1, 1].plot(np.arange(2040)+4, params_4G['Mean_l'], ls=':')
        ax[1, 1].plot(np.arange(2040)+4, np.polyval(Outparams['Mean_l'][0],
                      np.arange(2040)+4), c='black')
        ax[1, 2].plot(np.arange(2040)+4, params_4G['Mean_r'], ls=':')
        ax[1, 2].plot(np.arange(2040)+4, np.polyval(Outparams['Mean_r'][0],
                      np.arange(2040)+4), c='black')
        ax[1, 3].plot(np.arange(2040)+4, params_4G['Mean_w'], ls=':')
        ax[1, 3].plot(np.arange(2040)+4, np.polyval(Outparams['Mean_w'][0],
                      np.arange(2040)+4), c='black')

        ax[2, 0].plot(np.arange(2040)+4, params_4G['stddev_c'], ls=':')
        ax[2, 0].plot(np.arange(2040)+4, np.polyval(Outparams['stddev_c'][0],
                      np.arange(2040)+4), c='black')
        ax[2, 0].set_ylabel('Width [pixels] \n (%sx Oversampled)' % os,
                            fontsize=12)
        ax[2, 0].set_xlabel('Center', fontsize=12)
        ax[2, 1].plot(np.arange(2040)+4, params_4G['stddev_l'], ls=':')
        ax[2, 1].plot(np.arange(2040)+4, np.polyval(Outparams['stddev_l'][0],
                      np.arange(2040)+4), c='black')
        ax[2, 1].set_xlabel('Left', fontsize=12)
        ax[2, 2].plot(np.arange(2040)+4, params_4G['stddev_r'], ls=':')
        ax[2, 2].plot(np.arange(2040)+4, np.polyval(Outparams['stddev_r'][0],
                      np.arange(2040)+4), c='black')
        ax[2, 2].set_xlabel('Right', fontsize=12)
        ax[2, 3].plot(np.arange(2040)+4, params_4G['stddev_w'], ls=':')
        ax[2, 3].plot(np.arange(2040)+4, np.polyval(Outparams['stddev_w'][0],
                      np.arange(2040)+4), c='black')
        ax[2, 3].set_xlabel('Wings', fontsize=12)

        return Outparams

    else:
        return Outparams


def make_rectified_trace(psf_1D, *args, specparams=None, os=1):
    ''' Take a 1D PSF, broadcast across 2048 spectral pixels.
    _________________________________________________
    Inputs: psf_1D - 1D PSF function
            *args - args for psf_1D (if no spectral dependence)
            specparams - dictionary of polynomials for the variation
                        of the Gaussian trace profile parameters,
                        as output by fit_specpoly.
            os - spectral oversampling scale
    Ouputs: rectified trace array
    '''

    tracemodel = np.zeros((int(64*os), 2048))  # Assume 64 pix native PSF width

    for i in tqdm(range(2048)):

        # Add spectral dependence and oversample if necessary
        if specparams is not None:
            theta = []
            for par in specparams:
                theta.append(np.polyval(specparams[par][0], i))
            G = psf_1D([theta[0], theta[1], theta[2], theta[3]], [theta[4],
                       theta[5], theta[6], theta[7]], [theta[8], theta[9],
                       theta[10], theta[11]])
            os_slice = G(np.arange(64*os))

        else:
            # Oversample slice if necessary
            os_slice = oversample_slice(psf_1D(*args), os)

        tracemodel[:, i] += os_slice

    return tracemodel


def make_2D_trace(tracemodel, orders=[1, 2], os=1, semiwidth=32, filename=None):
    ''' Take a 1D PSF and broadcast across 2048 spectral
    pixels. Returns a data cube with one frame for each order.
    _________________________________________________
    Inputs: tracemodel - rectified 2D trace of one single order
            orders - desired orders to include
            oversamp - oversampling factor
            semiwidth - semi-width of the 1D PSF
            filename - name of fits file to write detector image to
    Ouputs: 2D trace map with desired orders, directly returned or
            in a fits file
    '''

    naty = np.linspace(0, 255, 256)  # Native spectral axis
    modely = np.linspace(-32, 32, int(64*os))  # Model oversampled spectral axis
    dcube = []

    # Get trace solution coefficients
    tp2 = tp.get_tracepars()

    # Determine x and y positions of trace centroid
    for m in orders:
        map2D = np.zeros((256, 2048))
        cenx = np.arange(2048) + 0.5  # Centroid positions in X
        lmbd = tp.x2wavelength(cenx, tp2, m)[0]  # Wavelength at each centroid
        ceny = tp.wavelength2y(lmbd, tp2, m)[0][::-1]  # Y centroid at each X

        # Map trace model to 2D detector
        for x in cenx:
            x = int(x)
            z = tracemodel[:, int(cenx[x])]
            y = ceny[x] + modely  # Shift PSF to centroid position
            slicemap = np.interp(naty, y, z)  # Interpolate oversampled PSF onto native Y
            ind = np.where((naty > ceny[x] - 3*semiwidth) &
                           (naty < ceny[x] + 3*semiwidth))  # Trim wings
            map2D[ind, x] += slicemap[ind]

        dcube.append(map2D/np.nanmax(map2D))

    # Either write data to fits file
    if filename is not None:
        hdu = fits.PrimaryHDU()
        hdu.data = np.array(dcube)
        hdu.writeto(filename, overwrite=True)

    # Or return the detector image cube
    else:
        return np.array(dcube)


def oversample_slice(slice_native, os, scaleflux=True):
    ''' Oversample a 1D PSF to the desired scale.
    _________________________________________________
    Inputs: slice_native - 1D PSF slice (in spectral direction)
            os - oversampling scale
            scaleflux - option to scale down the oversampled flux
    Ouputs: oversampled PSF slice
    '''

    dimy = np.shape(slice_native)[0]
    slice_os = np.zeros(dimy*os)

    for i in range(dimy):
        slice_os[i*os:(i+1)*os] = slice_native[i]

    # Scale total flux in PSF
    if scaleflux is True:
        slice_os = slice_os / os

    return slice_os
