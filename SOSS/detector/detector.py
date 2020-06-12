"""
Created on Tue Jan 21 14:21:35 2020

@author: caroline

Introduce the detector response in the simulated images
"""

from __future__ import division, print_function  # TODO do we need these? I'm assuming we're using python3?
import timeseries


def add_noise(filelist, normalize=False, zodibackg=True, flatfield=True, darkframe=True, nonlinearity=True,
              superbias=True, detector=True):
    """
    A function to add detector noise to the simulations.
    """

    normfactor = None
    for filename in filelist:

        tso = timeseries.TimeSeries(filename)

        if normalize:

            # If no normalization was provided use the first file to scale all files.
            if normfactor is None:
                normfactor = tso.get_normfactor()

            tso.apply_normfactor(normfactor)

        tso.add_poisson_noise()

        if zodibackg:
            tso.add_zodiacal_background()

        if flatfield:
            tso.apply_flatfield()

        if darkframe:
            tso.add_simple_dark()

        if nonlinearity:
            tso.add_non_linearity()

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
    parser.add_argument('--normalize', action='store_true', dest='normalize',
                        help='Applies a crude re-normalization to the simulation.')
    parser.add_argument('--no-zodibackg', action='store_false', dest='zodibackg',
                        help='Add the zodiacal light background to the simulation.')
    parser.add_argument('--no-flatfield', action='store_false', dest='flatfield',
                        help='Apply the flat-field to the simulation.')
    parser.add_argument('--no-darkframe', action='store_false', dest='darkframe',
                        help='Add dark current to the simulation.')
    parser.add_argument('--no-nonlinearity', action='store_false', dest='nonlinearity',
                        help='Add the effect of non-linearity to the simulation.')
    parser.add_argument('--no-superbias', action='store_false', dest='superbias',
                        help='Add the super bias to the simulation.')
    parser.add_argument('--no-detector', action='store_false', dest='detector',
                        help='Add the effects of read-noise, 1/f noise, kTC noise, and ACN to the simulation.')

    args = parser.parse_args()

    add_noise(args.filenames, args.normalize, args.zodibackg, args.flatfield, args.darkframe, args.nonlinearity,
              args.superbias, args.detector)

    return


if __name__ == "__main__":
    main()
