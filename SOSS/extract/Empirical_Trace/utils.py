#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Mar 11 14:35 2020

@author: MCR

Miscellaneous utility functions for the empirical trace construction.
"""

from astropy.io import fits
import numpy as np
from scipy.optimize import least_squares


def _lik(k, data, model):
    '''Utility likelihood function for flux rescaling. Esssentially a Chi^2
    multiplied by the data such that wing values don't carry too much weight.
    '''
    return np.nansum((data - k*model)**2)


def _local_mean(array, step):
    '''Calculate the mean of an array in chunks of 2*step.
    '''
    running_means = []
    for i in range(-step, step):
        if i == 0:
            continue
        running_means.append(np.roll(array, i))
    loc_mean = np.mean(running_means, axis=0)

    return loc_mean


def _robust_polyfit(x, y, p0):
    '''Wrapper around scipy's least_squares fitting routine implementing the
     Huber loss function - to be more resistant to outliers.

    Parameters
    ----------
    x : list
        Data describing dependant variable.
    y : list
        Data describing independant variable.
    p0 : tuple
        Initial guess straight line parameters. The length of p0 determines the
        polynomial order to be fit - i.e. a length 2 tuple will fit a 1st order
        polynomial, etc.

    Returns
    -------
    res.x : list
        Best fitting parameters of the desired polynomial order.
    '''

    def poly_res(p, x, y):
        '''Residuals from a polynomial'''
        return np.polyval(p, x) - y

    # Preform outlier resistant fitting.
    res = least_squares(poly_res, p0, loss='huber', f_scale=0.1, args=(x, y))
    return res.x


def _verbose_to_bool(verbose):
    '''Convert integer verbose to bool to disable or enable progress bars.
    '''

    if verbose in [2, 3]:
        verbose_bool = False
    else:
        verbose_bool = True

    return verbose_bool


def _write_to_file(o1frame, o2frame, filename):
    '''Write 2D trace spatial profiles to disk.
    '''

    hdu = fits.PrimaryHDU()
    hdu.data = np.dstack((o1frame, o2frame))
    hdu.writeto('{}.fits'.format(filename), overwrite=True)
