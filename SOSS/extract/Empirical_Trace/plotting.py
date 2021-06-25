#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:03 2021

@author: MCR

File containing all diagnostic plotting functions for the empirical trace
construction and centroiding.
"""

import matplotlib.pyplot as plt
import numpy as np


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


def plot_flux_deviations(dev_init, dev_fin, iteration):
    """Plot the diagnostic results of the order 2 smoothing iterations. The
    upper plot shows the initial column-wise flux deviations from the local
    mean, and the lower plot shows the same variations after the smoothing has
    been completed.
    """

    f, ax = plt.subplots(2, figsize=(15, 6), sharex=True)
    ax[0].plot(dev_init)
    ax[0].axhline(0.1, ls='--', c='black')
    ax[1].plot(dev_fin)
    ax[1].axhline(0.1, ls='--', c='black')

    max_val = np.max([len(dev_init), 2048])
    ax[1].set_xlim(0, max_val)
    ax[1].set_xlabel('Spectral Pixel', fontsize=14)
    ax[0].set_ylabel('Flux Variations (i=0)', fontsize=14)
    ax[1].set_ylabel('Flux Variations (i={})'.format(iteration-1), fontsize=14)
    plt.show()


def plot_interpmodel(waves, nw1, nw2, p1, p2):
    """Plot the diagnostic results of the derive_model function. Four plots
    are generated, showing the normalized interpolation coefficients for the
    blue and red anchors for each WFE realization, as well as the mean trend
    across WFE for each anchor profile, and the resulting polynomial fit to
    the mean trends.

    Parameters
    ----------
    waves : np.array of float
        Wavelengths at which WebbPSF monochromatic PSFs were created.
    nw1 : np.array of float
        Normalized interpolation coefficient for the blue anchor
        for each PSF profile.
    nw2 : np.array of float
        Normalized interpolation coefficient for the red anchor
        for each PSF profile.
    p1 : np.array of float
        Polynomial coefficients of the fit to the mean interpolation
        coefficients for the blue anchor.
    p2 : np.array of float
        Polynomial coefficients of the fit to the mean interpolation
        coefficients for the red anchor.
    """

    f, ax = plt.subplots(2, 2, figsize=(14, 6))
    for i in range(10):
        ax[0, 0].plot(waves, nw1[i])
        ax[1, 0].plot(waves, nw2[i])

    ax[0, 1].plot(waves, np.mean(nw1, axis=0))
    ax[0, 1].plot(waves, np.mean(nw2, axis=0))

    ax[-1, 0].set_xlabel('Wavelength [µm]', fontsize=14)
    ax[-1, 1].set_xlabel('Wavelength [µm]', fontsize=14)

    y1 = np.polyval(p1, waves)
    y2 = np.polyval(p2, waves)

    ax[1, 1].plot(waves, y1, c='r', ls=':')
    ax[1, 1].plot(waves, y2, c='b', ls=':')
    ax[1, 1].plot(waves, np.mean(nw1, axis=0), c='b', label='Blue Anchor')
    ax[1, 1].plot(waves, np.mean(nw2, axis=0), c='r', label='Red Anchor')
    ax[1, 1].set_xlim(np.min(waves), np.max(waves))
    ax[1, 1].legend(loc=1, fontsize=12)

    f.tight_layout()
    plt.show()


def plot_scaling_coefs(pixels, k_coefs, pp_k):
    """Do diagnostic plotting for the first-to-second order flux scaling
    relationship.
    """

    plt.figure(figsize=(8, 5))
    plt.scatter(pixels, k_coefs, s=4, c='blue', alpha=0.8,
                label='calculated')
    plt.plot(pixels, np.polyval(pp_k, pixels), ls='--', c='red', label='fit')

    plt.xlabel('Spectral Pixel', fontsize=14)
    plt.ylabel('O1-to-O2 Scaling', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()


def plot_trace_residuals(clear, order1, order2):
    """Plot the residuals after subtracting the first and second order spatial
    profile models from the original CLEAR dataframe.
    """

    plt.figure(figsize=(15, 3))
    plt.imshow((clear - (order1 + order2))/np.nansum(clear, axis=0),
               origin='lower', vmin=1e-7, vmax=1e-4)

    plt.xlabel('Spectral Pixel', fontsize=14)
    plt.ylabel('Spatial Pixel', fontsize=14)
    plt.colorbar()
    plt.show()


def plot_width_cal(fit_widths, fit_waves, width_poly):
    """Do the diagnostic plot for the trace width calibration relation.
    """

    plt.figure(figsize=(8, 5))
    plt.scatter(fit_waves[0][::10], fit_widths[0][::10], label='trace widths',
                c='blue', s=12,
                alpha=0.75)
    plt.scatter(fit_waves[1][::10], fit_widths[1][::10], c='blue', s=12,
                alpha=0.75)
    plt.plot(fit_waves[0], np.polyval(width_poly[0], fit_waves[0]), c='red',
             ls='--', label='width relation')
    plt.plot(fit_waves[1], np.polyval(width_poly[1], fit_waves[1]), c='red',
             ls='--')

    plt.xlabel('Wavelength [µm]', fontsize=14)
    plt.ylabel('Trace Spatial Width [pixels]', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()


def plot_wing_reconstruction(profile, ycens, axis_r, prof_r2, pp_r, newprof,
                             pad, text=None):
    """Do diagnostic plotting for wing reconstruction.
    """

    dimy = len(profile)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(dimy), np.log10(profile), ls=':', c='black',
             label='original profile')
    for ycen in ycens:
        plt.axvline(ycen, ls=':', c='grey')
    plt.scatter(axis_r, prof_r2, c='orange', s=15, label='unmasked points')
    ax_tot = np.arange(dimy+2*pad) - pad
    plt.plot(ax_tot, np.log10(newprof), c='blue', alpha=1,
             label='reconstructed profile',)
    plt.plot(ax_tot[(ycens[0]+18+pad):-(pad+4)],
             np.polyval(pp_r, ax_tot[(ycens[0]+18+pad):-(pad+4)]), c='red',
             lw=2, ls='--', label='right wing fit')
    if text is not None:
        plt.text(ax_tot[5], np.min(np.log10(newprof)), text, fontsize=14)

    plt.xlabel('Spatial Pixel', fontsize=12)
    plt.xlim(int(ax_tot[0]), int(ax_tot[-1]))
    plt.legend(fontsize=12)
    plt.show()


def plot_wing_reconstruction96(profile, newprof, text=None):
    """Do diagnostic plotting for the SUBSTRIP96 wing reconstruction.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(profile)), np.log10(profile), ls=':', c='black',
             label='original profile')
    add = len(newprof) - len(profile)
    plt.plot(np.arange(len(newprof))-add/2, np.log10(newprof), c='blue',
             label='reconstructed profile')
    if text is not None:
        plt.text(np.arange(len(newprof))[10], np.min(np.log10(newprof)), text,
                 fontsize=14)

    plt.xlabel('Spatial Pixel', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()
