"""
Created on Tue Jan 21 14:21:35 2020

@author: caroline

Updates by Loic Albert

Introduce the detector response in the simulated images
"""

import argparse

#import timeseries
import timeseries





def add_noise(filelist, ralist, declist, APAlist, noisefiles_dir='/genesis/jwst/jwst-ref-soss/noise_files/', photon=True, zodibackg=True, flatfield=True, darkcurrent=True, nonlinearity=True,
              superbias=True, readout=True, oneoverf=True, cosmicray = False, field_stars = False, zodi_ref=None, flat_ref=None, dark_ref=None,
              nlcoeff_ref=None, superbias_ref=None, outputfilename=None, full_well=72000):

    """
    A function to add detector noise to the simulations.
    It is assumed that the input exposure is in ***** electrons, not adu *****.

    :param filelist: list of string, the list of files to process
    :param ralist: list of string, the list of right ascension of targets in order
    :param declist: list of string, the list of declinaison of targets in order
    :param APAlist: list of int, the list of the APA of the observations in order
    :param noisefiles_dir: where reference detector noise files can be found
    :param normalize: DEPRECATED bool, renormalize the simulation, default False. USE DEPRECATED
    :param photon: bool, turn on or off photon noise for the science target flux
    :param zodibackg: bool, include the effect of the zodiacal background, default True.
    :param flatfield: bool, include the effect of the flatfield, default True.
    :param darkcurrent: bool, include the effect of darkcurrent, default True.
    :param nonlinearity: bool, include the effect of non-linearity, default True.
    :param superbias: bool, include the effect of bias, default True.
    :param readout: bool, white readout noise, default is True.
    :param oneoverf: bool, 1 over f pink noise, default is True.
    :param detector: DEPRECATED bool, include the effects of detector noise, default True. USE DEPRECATED
    :param zodi_ref: string, CRDS reference file containing the zodibackground map
    :param flat_ref: string, CRDS reference file containing the flat field
    :param dark_ref: string, CRDS reference file containing the dark current map
    :param nlcoeff_ref: string, UdeM in-house reference file containing the coeff to delinearize the ramp
    :param superbias_ref: string, CRDS reference file containing the superbias

    :type filelist: list[str]
    :type photon: bool
    :type normalize: bool
    :type zodibackg: bool
    :type flatfield: bool
    :type darkcurrent: bool
    :type nonlinearity: bool
    :type superbias: bool
    :type detector: bool

    :return:
    """

    # Check that the input file is a string (a single filename) or a list (of filenames)
    if isinstance(filelist,str) == True:
        filelist_checked = [filelist]
    else:
        filelist_checked = filelist
        

    # Check that the input ra is a string (a single ra) or a list (of ra)
    if isinstance(ralist,str) == True:
        ralist_checked = [ralist]
    else:
        ralist_checked = ralist

    # Check that the input dec is a string (a single dec) or a list (of dec)
    if isinstance(declist,str) == True:
        declist_checked = [declist]
    else:
        declist_checked = declist

    # Check that the input APA is a string (a single APA) or a list (of APA)
    if isinstance(APAlist,int) == True:
        APAlist_checked = [APAlist]
    else:
        APAlist_checked = APAlist

    normfactor = None

    for target in range(len(filelist_checked)): 

        tso = timeseries.TimeSeries(filelist_checked[target], ralist_checked[target], declist_checked[target], APA = APAlist_checked[target],  noisefiles_dir=noisefiles_dir)

        #if normalize:
        #
        #    # If no normalization was provided use the first file to scale all files.
        #    if normfactor is None:
        #        normfactor = tso.get_normfactor()
        #
        #    tso.apply_normfactor

        # TODO: change frame time in write_dmsready_fits

        
        if tso.subarray == "FULL":  #TODO: Can be removed when the issue with FULL images is solved
            print('Fixing FULL image...')
            tso.FULL_fix()

        if field_stars:
            print('Add field stars')
            tso.add_field_stars()

        if cosmicray:
            print('Add cosmic rays')
            tso.add_cosmic_rays(sun_activity="SUNMIN")

        if zodibackg:
            print('Add zodiacal background')
            tso.add_zodiacal_background(zodifile=zodi_ref)

        if photon:
            print('Add Poisson noise')
            tso.add_poisson_noise()

        if flatfield:
            print('Add flat field response')
            tso.apply_flatfield(flatfile=flat_ref)

        if darkcurrent:
            #TODO: have better dark ref files. ref files for darks are very noisy with 1/f noise and rms of order 50% of the dark at read 50.
            print('Add dark current')
            tso.add_dark(darkfile=dark_ref)

        if nonlinearity:
            print('Add non linearity (delinearize)')
            tso.add_non_linearity(coef_file=nlcoeff_ref)

        if superbias:
            print('Add superbias')
            tso.add_superbias(biasfile=superbias_ref)

        if readout:
            print('Add readout noise')
            tso.add_readout_noise()

        if oneoverf:
            print('Add 1/f noise')
            tso.add_1overf_noise()

        #if detector:
        #    # We discourage the use of this until tested.
        #    tso.add_detector_noise()

        tso.write_to_fits(outputfilename)

    return


def main():
    """A command line interface."""

    #TODO: update this interface with latest input args of function above

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
