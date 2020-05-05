"""
Created on Tue Jan 21 14:21:35 2020

@author: caroline

Introduce the detector response in the simulated images
"""

from __future__ import division, print_function
import numpy as np
from pkg_resources import resource_filename
import timeseries


def add_noise(filelist, nonlinearity=True, detector=True, flatfield=True, superbias=True, darkframe=True):
    """
    A function to add detector noise the the simulations.
    """

    for filename in filelist:

        tso = timeseries.TimeSeries(filename)

        tso.add_poisson_noise()

        # TODO add background signal.

        if flatfield:
            tso.apply_flatfield()

        if darkframe:
            tso.add_simple_dark()

        if nonlinearity:
            coef_file = 'files/files_jwst_niriss_linearity_0011_range_0_100000_npoints_100_polydeg_4.npy'
            non_linearity = np.load(resource_filename('detector', coef_file))
            tso.add_non_linearity(non_linearity)

        if superbias:
            tso.add_superbias()

        if detector:
            tso.add_detector_noise()

        tso.write_to_fits()

    return


def main():
    """
    A command line interface.
    """

    import argparse

    parser = argparse.ArgumentParser(description='Add sources of noise to the simulation.')
    parser.add_argument('filenames', type=str, nargs='+',
                        help='The simulation file(s) to process.')
    parser.add_argument('--nonlinearity', action='store_true',
                        help='Add the effect of non-linearity to the simulation.', dest='nonlinearity')
    parser.add_argument('--detector', action='store_true',
                        help='Add the effects of read-noise, 1/f noise, kTC noise, and ACN to the simulation.', dest='detector')
    parser.add_argument('--flatfield', action='store_true',
                        help='Apply the flat-field to the simulation.', dest='flatfield')
    parser.add_argument('--superbias', action='store_true',
                        help='Add the super bias to the simulation.', dest='superbias')
    parser.add_argument('--darkframe', action='store_true',
                        help='Add dark current to the simulation.', dest='darkframe')

    args = parser.parse_args()

    add_noise(args.filenames, args.nonlinearity, args.detector, args.flatfield, args.superbias)

    return


if __name__ == "__main__":
    main()
