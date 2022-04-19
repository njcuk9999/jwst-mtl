#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-13

@author: cook
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

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

# must be a dict equal to the expected number of order
ORDER_COLORS = {0: 'black', 1: 'blue', 2: 'orange', 3: 'green'}


# =============================================================================
# Define functions
# =============================================================================
def plot_flux(data: InputData):
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
    plt.close()


def plot_transit_fit(tfit: TransitFit, bandpass: Optional[int] = None):
    """
    Plot the current parameter transit model fit along with the data

    :param tfit: TransFit data class, storage of transit params and data
    :param bandpass:
    :return:
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
        tkwargs = dict(sol=tfit.ptmp, time=tfit.time[bpass],
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


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
