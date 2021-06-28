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
import corner


def _plot_centroid(clear, centroid_dict):
    '''Utility function to overplot the trace centroids extracted from
    the data over the data itself to verify accuracy.
    '''
    plt.figure(figsize=(15, 3))
    for order in centroid_dict.keys():
        plt.plot(centroid_dict[order][0], centroid_dict[order][1], c='black')
    plt.imshow(np.log10(clear), origin='lower', cmap='jet')
    plt.show()

    return None


def _plot_corner(sampler):
    '''Utility function to produce the corner plot for results of _do_emcee.
    '''
    labels = [r"ang", "xshift", "yshift"]
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=labels)
    plt.show()

    return None


def _plot_interpmodel(waves, nw1, nw2, p1, p2):
    '''Plot the diagnostic results of the derive_model function. Four plots
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
    '''

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


def _plot_wing_reconstruction(profile, ycens, axis_l, axis_l_pad, axis_r_pad,
                              pp_l, pp_r, prof_l2, newprof):
    '''Do diagnositic plotting for wing reconstruction.
    '''

    plt.figure(figsize=(8, 5))
    plt.plot(np.log10(profile), ls=':', c='black', label='original profile')
    for ycen in ycens:
        plt.axvline(ycen, ls=':', c='grey')
    plt.scatter(axis_l, prof_l2, c='orange', s=15, label='unmasked points')
    ax_tot = np.linspace(axis_l_pad[0], axis_r_pad[-1], int((axis_r_pad[-1]-axis_l_pad[0])+1))
    plt.plot(ax_tot, np.log10(newprof), c='blue', alpha=1,
             label='reconstructed profile',)
    plt.plot(axis_l_pad, np.polyval(pp_l, axis_l_pad), c='red', lw=2, ls='--',
             label='left wing fit')
    plt.plot(axis_r_pad, np.polyval(pp_r, axis_r_pad), c='green', lw=2,
             ls='--', label='right wing fit')

    plt.xlabel('Spatial Pixel', fontsize=14)
    plt.xlim(int(axis_l_pad[0]), int(axis_r_pad[-1]))
    plt.legend(fontsize=12)
    plt.show()
