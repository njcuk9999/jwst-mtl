#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:17:45 2020

@author: caroline

Get forward coefficients to apply non-linearity on a linear ramp
based on the correction coefficients available on the CRDS website
"""

# Import modules
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import fits
import time


def get_forward(correc_coeffs, i_row=1380, i_col=55,
                range_calibration=None, npoints=100, poly_deg=4,
                plot=False):
    """
    Fit forward coefficients for a pixel
    - range_calibration: range of counts for which the forward coefficients
    are calculated
    - npoints: number of points to use for calibration
    - poly_deg: degree of the polynomial to fit for the forward coefficients
    - plot: plot the result of the polynomial fit
    """

    if range_calibration is None:
        range_calibration = [0., 100e3]
    fluxes_calibration = np.linspace(range_calibration[0],
                                     range_calibration[1], npoints)

    flux_with_nonlin = np.zeros_like(fluxes_calibration)

    # use root finding method to calculate the fluxes with added non-linearity
    for i in range(fluxes_calibration.size):
        coeffs = -correc_coeffs[:, i_col, i_row][::-1]
        coeffs[-1] = coeffs[-1] + fluxes_calibration[i]
        cr_nonlin = np.roots(coeffs)
        flux_with_nonlin[i] = np.real(cr_nonlin[np.isreal(cr_nonlin)])[0]

    # fit a polynomial to the fluxes with non-linearity
    fwd_coeffs = np.polyfit(fluxes_calibration, flux_with_nonlin, poly_deg)
    p = np.poly1d(fwd_coeffs)

    if plot:
        # make figure of the fluxes with non-linearity as a function of the
        # input "perfect" fluxes
        fig, ax = plt.subplots(1, 1)
        ax.set_title('Example: Col '+str(i_col)+', Row '+str(i_row))
        ax.plot(fluxes_calibration, flux_with_nonlin, marker='.', ls='',
                color='k', label='Root-finding results')
        ax.plot(fluxes_calibration, p(fluxes_calibration), color='r',
                label='Polynomial fit')
        ax.plot(fluxes_calibration, fluxes_calibration, zorder=-20,
                color='gray', ls='--', label='1:1')
        ax.set_xlabel('Ideal flux')
        ax.set_ylabel('Flux with non-linearity')
        ax.legend(loc=2)

    return fwd_coeffs


def calc_forward_coeffs_array(correc_coeffs, poly_deg=4, print_every_ncol=5):
    """
    Calculate the ndarray of (poly_deg+1)*ncols*nrows coefficients
    - correc_coeffs: correction coefficients from CRDS file
    - poly_deg: degree of the polynomial to fit for the forward coefficients
    - print_every_N_col: if None, does nothing. if = a number, will print to
    the screen every time this number of columns has been processed
    """

    ncols = 2048
    nrows = 2048
    forward_coeffs = np.zeros((poly_deg+1, ncols, nrows))

    start_time = time.time()

    for c in range(ncols):

        if print_every_ncol is not None:
            if (c % print_every_ncol) == 0:
                print('Column '+str(c+1)+'/'+str(ncols))

        for r in range(nrows):
            forward_coeffs[:, c, r] = get_forward(correc_coeffs, r, c)

            if c == 0 and r == 0:
                time_first = time.time() - start_time
                print('The first calculation took', time_first, 'seconds.')
                print('Estimated total run time:',
                      time_first * nrows * ncols, 'seconds.')

    return forward_coeffs


# Read in the CRDS file of correction polynomial coefficients
def main(argv):
    """
    Example call:

    python get_forward_coeffs.py path/to/CRDS/file.fits
    or
    python get_forward_coeffs.py path/to/CRDS/file.fits 0 100000 100 4
    """

    print('Input arguments: ', argv)

    lenargv = len(argv)
    # a minimum of 1 argument must be specified for a command-line call
    if lenargv > 1:
        crds_path = argv[1]  # path to the fits file containing the CRDS file
        if lenargv > 2:
            range_calibration = [float(argv[2]), float(argv[3])]
            npoints = int(argv[4])
            poly_deg = int(argv[5])
        else:
            # setup for calculation
            range_calibration = [0., 100e3]
            npoints = 100
            poly_deg = 4

    else:  # if not called from the command line
        path_soss_files = '/home/caroline/GitHub/SOSS/files/'
        filename = 'jwst_niriss_linearity_0011.fits'
        crds_path = path_soss_files + filename

        # setup for calculation
        range_calibration = [0., 100e3]
        npoints = 100
        poly_deg = 4

    # subsection for SOSS detector
    # correc_coeffs = fits.open(crdsPath)[1].data[:,1792:,:]
    correc_coeffs = fits.open(crds_path)[1].data

    fwd_coeffs = calc_forward_coeffs_array(correc_coeffs, poly_deg=poly_deg)

    np.save('files/'+crds_path[:-5].replace('/', '_') + '_range_'
            + str(int(range_calibration[0])) + '_'
            + str(int(range_calibration[0])) + '_npoints_' + str(npoints)
            + '_polydeg_'+str(poly_deg) + '_fullFrame.npy', fwd_coeffs)


if __name__ == "__main__":
    main(sys.argv)
