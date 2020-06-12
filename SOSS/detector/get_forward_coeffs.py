#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:17:45 2020

@author: caroline

Get inverse coefficients to apply non-linearity on a linear ramp
based on the correction coefficients available on the CRDS website
"""

# Import modules
import os

import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt


def visualize(row, col, coeffs, inv_coeffs, bounds, npoints):

    p1 = np.poly1d(coeffs[::-1, row, col])
    p2 = np.poly1d(inv_coeffs[::-1, row, col])

    fluxes_calibration = np.linspace(bounds[0], bounds[1], npoints)

    # Make figure comparing the forwards and backward polynomials.
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.set_title('Example: Row {}, Col {}'.format(row, col))
    ax.plot(fluxes_calibration, p1(fluxes_calibration), color='r', label='Original Polynomial')
    ax.plot(p2(fluxes_calibration), fluxes_calibration, color='g', label='Inverse Polynomial')
    ax.plot(fluxes_calibration, fluxes_calibration, zorder=-20, color='gray', ls='--', label='1:1')
    ax.set_xlabel('Non-linear Flux')
    ax.set_ylabel('Linear Flux')
    ax.set_aspect('equal')
    ax.set_ylim(0,)
    ax.set_xlim(0,)
    ax.legend(loc=2)

    return


def invert_polynomials(coeffs, bounds=None, npoints=100, deg=5):
    """
    Fit forward coefficients for a pixel
    - coeffs: the coefficients of the original polynomials.
    - bounds: upper and lower bound between which to evalaute the original polynomials.
    - npoints: number of points to evaluate the original polynomials at.
    - deg: degree of the inverse polynomial.
    """

    if bounds is None:
        bounds = [0., 6e4]

    # Prepare the coefficient arrays.
    ndeg, nrows, ncols = coeffs.shape
    coeffs = coeffs.reshape((ndeg, -1))
    coeffs = coeffs[::-1]  # numpy uses a highest degree first convention.
    inv_coeffs = np.zeros((deg + 1, nrows*ncols))
    inv_coeffs[-2] = 1.  # Fix slope and intercept.

    for i in range(nrows*ncols):

        # Evaluate the polynomials.
        flux_nonlinear = np.linspace(bounds[0], bounds[1], npoints)
        flux_linear = np.polyval(coeffs[:, i], flux_nonlinear)

        # Fit a polynomial to the fluxes with non-linearity
        x = flux_linear/bounds[1]  # Divide by the upper bound for a better conditioned problem.
        mat = np.vander(x, deg + 1)
        mat = mat[:, :-2]  # Fix slope and intercept.

        inv_coeffs[:-2, i] = np.linalg.lstsq(mat, flux_nonlinear - flux_linear, rcond=None)[0]

    # Manipulate array into standard form.
    inv_coeffs = inv_coeffs[::-1]  # numpy used a highest degree first convention.
    for i in range(2, deg + 1):  # Undo divide by upper bound.
        inv_coeffs[i] = inv_coeffs[i]/(bounds[1])**i
    inv_coeffs = inv_coeffs.reshape((deg + 1, nrows, ncols))

    # Deal with the reference pixels.
    inv_coeffs[:, :4, :] = 0
    inv_coeffs[:, :, :4] = 0
    inv_coeffs[:, -4:, :] = 0
    inv_coeffs[:, :, -4:] = 0
    inv_coeffs[1, :4, :] = 1
    inv_coeffs[1, :, :4] = 1
    inv_coeffs[1, -4:, :] = 1
    inv_coeffs[1, :, -4:] = 1

    return inv_coeffs


def main():
    """"""

    import argparse

    parser = argparse.ArgumentParser(description='Invert the coefficients of the non-linearity correction.')
    parser.add_argument('filename', type=str,
                        help='The CRDS non-linearity file to use.')
    parser.add_argument('--bounds', type=float, nargs=2, default=[0., 6e4],
                        help='The range of values over which to calculate the polynomials.')
    parser.add_argument('--npoints', type=int, default=100,
                        help='The number of sample point along the range.')
    parser.add_argument('--degree', type=int, default=5,
                        help='The degree of the polynomial used.')

    args = parser.parse_args()

    # Compute the coeffiecients for adding non-linearity to simulated observations.
    coeffs = fits.getdata(args.filename, ext=1)
    inv_coeffs = invert_polynomials(coeffs, bounds=args.bounds, npoints=args.npoints, deg=args.degree)

    # Save the coefficients to a .fits file.
    name = os.path.splitext(os.path.basename(args.filename))[0]
    namestr = 'files/{}_bounds_{:d}_{:d}_npoints_{:d}_deg_{:d}.fits'
    outfile = namestr.format(name, int(args.bounds[0]), int(args.bounds[1]), args.npoints, args.degree)

    hdu = fits.PrimaryHDU(inv_coeffs)
    hdu.writeto(outfile)

    return


if __name__ == "__main__":
    main()
