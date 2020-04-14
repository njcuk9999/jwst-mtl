"""
Created on Tue Jan 21 14:21:35 2020

@author: caroline

Introduce the detector response in the simulated images
"""

from __future__ import division, print_function
import numpy as np
import sys
import timeseries


def main(argv):
    """
    python detector.py path/to/fits/file.fits 1
    """
    print('Input arguments: ', argv)

    lenargv = len(argv)

    # a minimum of 2 arguments must be specified for a command-line call
    if lenargv > 2:
        # path to the fits file containing the image (Jason's output)
        ima_path = argv[1]

        # include effect of non-linearity in detector response if True
        add_non_linearity = bool(int(argv[2]))

    else:  # if not called from the command line
        path_soss = '/Users/caroline/Research/GitHub/SOSS/jwst-mtl/SOSS/'
        path_data = 'detector/data/'
        filename = 'jw00001001001_0110100001_NISRAPID_cal_c.fits'

        ima_path = path_soss + path_data + filename
        add_non_linearity = True

        print(ima_path, add_non_linearity)

    ts = timeseries.TimeSeries(ima_path)

    # adding Poisson noise to the images prior to non-linearity correction
    ts.add_poisson_noise()

    # modify time series for non-linearity effects
    if add_non_linearity:

        # File path containing the forward coefficients for the fit
        # forward coefficients calc using non-linearity data from CRDS website
        non_linearity = np.load('files/files_jwst_niriss_linearity_0011_range_0_100000_npoints_100_polydeg_4.npy')

        ts.add_non_linearity(non_linearity)

    ts.write_to_fits()  # write modified time series observations to a new file


if __name__ == "__main__":
    fit = main(sys.argv)
