"""
Created on Tue Jan 21 14:21:35 2020

@author: caroline

Introduce the detector response in the simulated images
"""

import argparse

import timeseries


def add_noise(filelist, normalize=False, zodibackg=True, flatfield=True, darkframe=True, nonlinearity=True,
              superbias=True, detector=True, outputfilename=None):
    """
    A function to add detector noise to the simulations.

    :param filelist: list of string, the list of files to process
    :param normalize: bool, renormalize the simulation, default False.
    :param zodibackg: bool, include the effect of the zodiacal background, default True.
    :param flatfield: bool, include the effect of the flatfield, default True.
    :param darkframe: bool, include the effect of darkcurrent, default True.
    :param nonlinearity: bool, include the effect of non-linearity, default True.
    :param superbias: bool, include the effect of bias, default True.
    :param detector: bool, include the effects of detector noise, default True.

    :type filelist: list[str]
    :type normalize: bool
    :type zodibackg: bool
    :type flatfield: bool
    :type darkframe: bool
    :type nonlinearity: bool
    :type superbias: bool
    :type detector: bool

    :return:
    """

    # Check that the input is a string (a single filename) or a list (of filenames)
    if isinstance(filelist,str) == True:
        filelist_checked = [filelist]
    else:
        filelist_checked = filelist

    normfactor = None

    for filename in filelist_checked:

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

        tso.write_to_fits(outputfilename)

    return


def main():
    """A command line interface."""

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
