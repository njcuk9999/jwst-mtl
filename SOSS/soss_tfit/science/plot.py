#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-13

@author: cook
"""
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

import transitfit5 as transit_fit

from soss_tfit.core import base_classes
from soss_tfit.science import general
from soss_tfit.science import mcmc

# =============================================================================
# Define variables
# =============================================================================
# get input data class
InputData = general.InputData
# get mcmc transitfit class
TransitFit = mcmc.TransitFit
# get the parameter dictionary
ParamDict = base_classes.ParamDict
# must be a dict equal to the expected number of order
ORDER_COLORS = {0: 'black', 1: 'blue', 2: 'orange', 3: 'green'}


# =============================================================================
# Define functions
# =============================================================================
def plot_flux(data: InputData):
    """
    Plot the flux data for each wavelength

    :param data: InputData instance

    :return: None, plots graph
    """
    # Show a plot of the data. Each colour is a different wavelength.
    fig, frame = plt.subplots(ncols=1, nrows=1)
    frame.set(xlabel='Time (days)', ylabel='Flux')
    # get photometric data dictionary
    phot = data.phot
    # loop around photometric bandpasses
    for i_phot in range(data.n_phot):
        # set colour based on order
        order = int(np.mean(phot['ORDERS'][i_phot]))
        color = ORDER_COLORS[order]
        frame.plot(phot['TIME'][i_phot], phot['FLUX'][i_phot], color=color)
    # adjust limits
    plt.subplots_adjust(hspace=0, left=0.05, right=0.975, top=0.975,
                        bottom=0.05)
    # show and close
    plt.show()
    if not plt.isinteractive():
        plt.close()


def plot_transit_fit(tfit: TransitFit, bandpass: Optional[int] = None):
    """
    Plot the current parameter transit model fit along with the data

    :param tfit: TransFit data class, storage of transit params and data
    :param bandpass:

    :return: None, plots graph
    """
    # if we have a single bandpass sort out only using this bandpass
    if bandpass is not None:
        # check bandpass
        if bandpass < 0:
            emsg = f'Bandpass must be greater than 0'
            raise base_classes.TransitFitExcept(emsg)
        elif bandpass > tfit.n_phot:
            emsg = f'Bandpass must be less than {tfit.n_phot}'
            raise base_classes.TransitFitExcept(emsg)
        # get bandpasses
        bandpasses = [bandpass]
    # otherwise use all band passes
    else:
        bandpasses = np.arange(0, tfit.n_phot)
    # plot
    if len(bandpasses) == 1:
        fig, frame = plt.subplots(ncols=1, nrows=1, figsize=(12, 8))
        frames = [frame]
    else:
        fig, frames = plt.subplots(ncols=1, nrows=len(bandpasses), sharex='all',
                                   figsize=(12, 6*len(bandpasses)))

    # loop around bandpasses
    for b_it in range(len(bandpasses)):
        # get bandpass
        bpass = bandpasses[b_it]
        # get the frame for this bandpass
        frame = frames[b_it]
        # get transit for current parameters
        tkwargs = dict(sol=tfit.p0, time=tfit.time[bpass],
                       itime=tfit.itime[bpass],
                       ntt=tfit.pkwargs['NTT'], tobs=tfit.pkwargs['T_OBS'],
                       omc=tfit.pkwargs['OMC'])
        # set colour based on order
        order = int(np.mean(tfit.orders[bpass]))
        color = ORDER_COLORS[order]
        # get and plot the model
        model = transit_fit.transitmodel(**tkwargs)
        frame.plot(tfit.time[bpass], model, color=color)
        # plot data
        frame.errorbar(tfit.time[bpass], tfit.flux[bpass], color=color,
                       yerr=tfit.fluxerr[bpass], ls='None', marker='.')
        frame.set(xlabel='Time [days]', ylabel='Flux')
        # set title
        title = (f'{bpass} (order={order} '
                 f'wave={tfit.wavelength[bpass][0]:.3f} um)')
        frame.set_title(title, y=1.0, pad=-14)

    # adjust limits
    plt.subplots_adjust(hspace=0, left=0.05, right=0.975, top=0.975,
                        bottom=0.05)
    # show and close
    plt.show()
    if not plt.isinteractive():
        plt.close()


def plot_chain(chain: np.ndarray, chain_num: int):
    """
    Plot a single chain

    :param chain: np.ndarray, the chain [n_steps, x_n]
    :param chain_num: int, the position in chain to get (positive to count
                      from start, negative to count from end)

    :return: None, plots graph
    """
    plt.plot(chain[:, chain_num])
    # show and close
    plt.show()
    if not plt.isinteractive():
        plt.close()


def plot_chains(chain: np.ndarray, burnin: int, labels: np.ndarray):
    """
    Plot the full set of chains

    :param chain: np.ndarray, the chain [n_steps, x_n]
    :param burnin: int, the number of chains to burn (ignore) at start
    :param labels: np.ndarray, the array of names of fitted params [x_n]

    :return: None, plots graph
    """
    # get the number of parameters
    n_param = chain.shape[1]
    # setup figure and frames
    fig, frames = plt.subplots(nrows=n_param, ncols=1,
                               figsize=(12, 1.5 * n_param))
    # loop around parameters
    for param_it in range(n_param):
        # fig[i].subplot(npars, 1, i+1)
        frames[param_it].plot(chain[burnin:, param_it])  # ,c=colour[i])
        # set the tick parameters
        frames[param_it].tick_params(direction='in', length=10, width=2)
        # set the ylabel
        frames[param_it].set_ylabel(labels[param_it])
        # turn off the xtick labels for all but the last frame
        if param_it + 1 < n_param:
            frames[param_it].set_xticklabels([])
    # show and close
    plt.show()
    if not plt.isinteractive():
        plt.close()


def plot_hist(tfit: TransitFit, chain: np.ndarray,
              param_num: Optional[int] = None):
    """
    Plot the histogram for chains for one parameter or all parameters
    (if param_num is unset)

    :param tfit: Transit fit class
    :param chain: np.ndarray the chains [n_steps, n_param]
    :param param_num: int, either 0 to n_param or None - if set only plots
                      one parameter, otherwise plots them all

    :return: None, plots graph
    """
    # deal with one parameter vs all parameters
    #   then grid is NxM where NxM >= tfit.n_x
    if param_num is None:
        xnames = tfit.xnames
        nrows = int(np.sqrt(tfit.n_x))
        ncols = (tfit.n_x // nrows) + 1
        fig, frames = plt.subplots(ncols=ncols, nrows=nrows)

        ijarr = [(i, j)  for i in range(nrows) for j in range(ncols)]
    # else we have one plot - the grid is (1x1) and the plotting is easy
    else:
        xnames = tfit.xnames[param_num]
        fig, frame = plt.subplots(ncols=1, nrows=1)
        frames = np.array([[frame]])
        ijarr = [(0, 0)]
    # loop around rows and add hist
    for kt, ij in enumerate(ijarr):
        # get the ith and jth frame
        frame = frames[ij]
        # only plot those for which we have values (the grid may have some
        #   empty space)
        if kt < len(xnames) - 1:
            # plot the histogram
            frame.hist(chain[:, kt])
            # set title
            frame.set_title(xnames[kt], y=1.0, pad=-14)
        else:
            frame.axis('off')
    # show and close
    plt.show()
    if not plt.isinteractive():
        plt.close()


def plot_spectrum(data: InputData, results: Table, key: str = 'RD1',
                  model: Optional[Dict[int, Table]] = None,
                  binkey: str = 'RPRS', pkind: str = 'mode',
                  fullmodel: Optional[Table] = None):
    """
    Plots the parameter against wavelength

    :param data: InputData array, containing the wavelengths
    :param results: Table, the results table from Sampler.results()
    :param key: str, the y axis label (defaults to RD1)
    :param model: None or dictionary or orders each which is a Table
                  (loaded from general.load_model())
    :param binkey: str, the model key matching "key" (defaults to "RPRS")
    :param pkind: str, either mode or median (for the correct stats)
    :param fullmodel: Table or None, if given the full unbinned model for
                      comparison

    :return: None, plots graph
    """
    # set up figure
    fig, frame = plt.subplots(ncols=1, nrows=1, figsize=(12, 8))
    # get the results for binkey
    rmask = results['NAME'] == key
    # get the arrays
    wave = data.phot['WAVELENGTH'][:, 0]
    # get results
    if pkind == 'mode':
        yvalue = results['MODE'][rmask]
        yupper = results['MODE_UPPER'][rmask]
        ylower = results['MODE_LOWER'][rmask]
    else:
        yvalue = results['P50'][rmask]
        yupper = results['P50_UPPER'][rmask]
        ylower = results['P50_LOWER'][rmask]

    if fullmodel is not None:
        frame.plot(fullmodel['wave'], fullmodel[binkey], color='0.5',
                   alpha=0.2)

    # loop around orders and plot
    for order in data.orders:
        # get the order colour
        ordercolor = ORDER_COLORS[order]
        # get the order mask
        ordermask = data.phot['ORDERS'][:, 0] == order
        # plot the binned model
        if model is not None:
            frame.plot(model[order]['wave'], model[order][binkey], lw=1,
                       color=ordercolor, label=f'Binned model Order {order}',
                       zorder=1)
        # plot the results
        yerrord = np.array([ylower[ordermask], yupper[ordermask]])
        frame.errorbar(wave[ordermask], yvalue[ordermask], yerr=yerrord,
                       fmt='o', lw=1, color=ordercolor, mec='k', zorder=2,
                       label=f'Fit Order {order}')
    # set labels and limits
    frame.set(xlabel=r'Wavelength ($\mu$m)', ylabel=r'$R_{p}/R_{\star}$',
              xlim=[0.6, 3.0])
    frame.legend(loc=0)
    # show and close
    plt.show()
    if not plt.isinteractive():
        plt.close()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
