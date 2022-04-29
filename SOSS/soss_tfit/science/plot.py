#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-13

@author: cook
"""
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import os
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
# printer
cprint = base_classes.Printer()


# =============================================================================
# Define worker functions
# =============================================================================
def start_plot(params: ParamDict, name: str) -> bool:
    """
    For now just a switch to avoid plotting

    :param params: ParamDict, parameter dictionary of constants

    :return: bool, True if we want to exit (not plot)
    """
    if 0 in params['PLOTMODE']:
        return True
    else:
        # print that we are plotting
        cprint(f'PLOTTING: {name}')
        # return False
        return False


def end_plot(params: ParamDict, name: str):
    """
    End plotting (save or show)

    :param params: ParamDict, parameter dictionary of constants
    :param name: str, the short name for the filename

    :return: None, writes file to disk or shows on screen
    """
    # deal with saving to disk
    if 1 in params['PLOTMODE']:
        # get output path
        outpath = params['OUTDIR']
        outname = params['OUTNAME'] + '_' + name
        # deal with no output dir
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        # get full paths
        plot_file1 = os.path.join(outpath, outname + '.jpg')
        plot_file2 = os.path.join(outpath, outname + '.pdf')
        # save figures
        plt.savefig(plot_file1)
        plt.savefig(plot_file2)
    # else we show (PLOTMODE == 2)
    if 2 in params['PLOTMODE']:
        # show and close
        plt.show()
    # close plots (if not interactive)
    if not plt.isinteractive():
        plt.close()


# =============================================================================
# Define plot functions
# =============================================================================
def plot_flux(params: ParamDict, data: InputData):
    """
    Plot the flux data for each wavelength

    :param params: ParamDict, parameter dictionary of constants
    :param data: InputData instance

    :return: None, plots graph
    """
    # set plot name
    plot_name = 'plot_flux'
    # start plot
    if start_plot(params, name=plot_name):
        return
    # Show a plot of the data. Each colour is a different wavelength.
    fig, frame = plt.subplots(ncols=1, nrows=1, figsize=(16, 12))
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
    # end plot
    end_plot(params, name=plot_name)


def plot_transit_fit(params: ParamDict, tfit: TransitFit,
                     bandpass: Optional[int] = None):
    """
    Plot the current parameter transit model fit along with the data

    :param params: ParamDict, parameter dictionary of constants
    :param tfit: TransFit data class, storage of transit params and data
    :param bandpass:

    :return: None, plots graph
    """
    # set plot name
    plot_name = 'plot_transit_fit'
    # start plot
    if start_plot(params, name=plot_name):
        return
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
        fig, frame = plt.subplots(ncols=1, nrows=1, figsize=(16, 12))
        frames = [frame]
    else:
        fig, frames = plt.subplots(ncols=1, nrows=len(bandpasses), sharex='all',
                                   figsize=(16, 2*len(bandpasses)))

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
    # end plot
    end_plot(params, name=plot_name)


def plot_chain(params: ParamDict, chain: np.ndarray, chain_num: int):
    """
    Plot a single chain

    :param params: ParamDict, parameter dictionary of constants
    :param chain: np.ndarray, the chain [n_steps, x_n]
    :param chain_num: int, the position in chain to get (positive to count
                      from start, negative to count from end)

    :return: None, plots graph
    """
    # set plot name
    plot_name = 'plot_chain'
    # start plot
    if start_plot(params, name=plot_name):
        return
    # setup plot
    fig, frame = plt.subplots(ncols=1, nrows=1, figsize=(16, 12))
    # plot chain
    frame.plot(chain[:, chain_num])
    # end plot
    end_plot(params, name=plot_name)


def plot_chains(params: ParamDict, chain: np.ndarray, burnin: int,
                labels: np.ndarray):
    """
    Plot the full set of chains

    :param params: ParamDict, parameter dictionary of constants
    :param chain: np.ndarray, the chain [n_steps, x_n]
    :param burnin: int, the number of chains to burn (ignore) at start
    :param labels: np.ndarray, the array of names of fitted params [x_n]

    :return: None, plots graph
    """
    # set plot name
    plot_name = 'plot_chains'
    # start plot
    if start_plot(params, name=plot_name):
        return
    # get the number of parameters
    n_param = chain.shape[1]
    # setup figure and frames
    fig, frames = plt.subplots(nrows=n_param, ncols=1,
                               figsize=(16, 1 * n_param))
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
    # end plot
    end_plot(params, name=plot_name)


def plot_hist(params: ParamDict, tfit: TransitFit, chain: np.ndarray,
              param_num: Optional[int] = None):
    """
    Plot the histogram for chains for one parameter or all parameters
    (if param_num is unset)

    :param params: ParamDict, parameter dictionary of constants
    :param tfit: Transit fit class
    :param chain: np.ndarray the chains [n_steps, n_param]
    :param param_num: int, either 0 to n_param or None - if set only plots
                      one parameter, otherwise plots them all

    :return: None, plots graph
    """
    # set plot name
    plot_name = 'plot_hist'
    # start plot
    if start_plot(params, name=plot_name):
        return
    # deal with one parameter vs all parameters
    #   then grid is NxM where NxM >= tfit.n_x
    if param_num is None:
        xnames = tfit.xnames
        # get the number of rows and columns (try to make it a square grid)
        nrows = int(np.sqrt(tfit.n_x))
        ncols = (tfit.n_x // nrows) + 1
        # set up figure
        fig, frames = plt.subplots(ncols=ncols, nrows=nrows,
                                   figsize=(2*ncols, 2*nrows))
        # get all positions within the grid
        ijarr = [(i, j) for i in range(nrows) for j in range(ncols)]
    # else we have one plot - the grid is (1x1) and the plotting is easy
    else:
        xnames = tfit.xnames[param_num]
        # set up figure
        fig, frame = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        # we only have one frame but want to use a 1x1 grid
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
    # end plot
    end_plot(params, name=plot_name)


def plot_spectrum(params: ParamDict, data: InputData, results: Table,
                  key: str = 'RD1', model: Optional[Dict[int, Table]] = None,
                  binkey: str = 'RPRS', pkind: str = 'mode',
                  fullmodel: Optional[Table] = None):
    """
    Plots the parameter against wavelength

    :param params: ParamDict, parameter dictionary of constants
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
    # set plot name
    plot_name = f'plot_spectrum_{pkind}'
    # start plot
    if start_plot(params, name=plot_name):
        return
    # set up figure
    fig, frame = plt.subplots(ncols=1, nrows=1, figsize=(16, 12))
    # -------------------------------------------------------------------------
    # get the results for binkey
    rmask = results['NAME'] == key
    # get the arrays
    wave = data.phot['WAVELENGTH'][:, 0]
    # -------------------------------------------------------------------------
    # get results for mode / percentile
    if pkind == 'mode':
        yvalue = results['MODE'][rmask]
        yupper = results['MODE_UPPER'][rmask]
        ylower = results['MODE_LOWER'][rmask]
        title = 'Mode'
    else:
        yvalue = results['P50'][rmask]
        yupper = results['P50_UPPER'][rmask]
        ylower = results['P50_LOWER'][rmask]
        title = 'Percentile'
    # -------------------------------------------------------------------------
    # plot the full model
    if fullmodel is not None:
        frame.plot(fullmodel['wave'], fullmodel[binkey], color='0.5',
                   alpha=0.2, label='Full model')
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # set labels and limits
    frame.set(xlabel=r'Wavelength ($\mu$m)', ylabel=r'$R_{p}/R_{\star}$',
              xlim=[0.6, 3.0], title=title)
    frame.legend(loc=0)
    # end plot
    end_plot(params, name=plot_name)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
