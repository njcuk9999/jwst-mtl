#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:03 2021

@author: MCR

File containing all diagnostic plotting functions for the APPLESOSS.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_badpix(clear, mask):
    """Plot the difference between the originanl dataframe, and the frame with
     bad pixels interpolated.
     """

    plt.imshow(clear - mask, origin='lower', aspect='auto',
               vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel('Spectral Pixel', fontsize=14)
    plt.ylabel('Spatial Pixel', fontsize=14)
    plt.show()


def plot_centroid(clear, centroid_dict):
    """Overplot the trace centroids extracted from the data over the data
    itself to verify accuracy.
    """

    plt.figure(figsize=(15, 3))
    for order in centroid_dict.keys():
        if order == 'order 1':
            plt.plot(centroid_dict[order]['X centroid'],
                     centroid_dict[order]['Y centroid'], c='black', ls='--',
                     label='trace centroids')
        else:
            plt.plot(centroid_dict[order]['X centroid'],
                     centroid_dict[order]['Y centroid'], c='black', ls='--')
    plt.imshow(np.log10(clear), origin='lower', cmap='jet')

    plt.xlabel('Spectral Pixel', fontsize=14)
    plt.ylabel('Spatial Pixel', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()


def plot_wing_simulation(stand, halfwidth, wing, wing2, ax, ystart, yend):
    """Do diagnostic plot for wing simulations.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(np.log10(stand), label='Simulated Profile')
    plt.axvline(400 // 2 + halfwidth, ls=':', c='black')
    plt.axvline(400 // 2 - halfwidth, ls=':', c='black')
    plt.plot(ax[yend:], np.log10(wing), c='red', label='Wing Model')
    plt.plot(ax[:ystart], np.log10(wing2), c='red')

    plt.legend(fontsize=12)
    plt.xlabel('Spatial Pixel', fontsize=14)
    plt.show()
