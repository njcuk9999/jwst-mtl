import numpy as np

import jwst

from astropy.io import fits

import matplotlib.pyplot as plt

from jwst.pipeline import calwebb_detector1

from jwst.pipeline import calwebb_spec2

import jwst.datamodels

import SOSS.commissioning.comm_utils as commutils

import SOSS.dms.soss_oneoverf as soss_oneoverf

import SOSS.dms.soss_outliers as soss_outliers

from SOSS.dms import soss_background

import sys


import os

CALIBRATION_DIR = '/Users/albert/NIRISS/Commissioning/analysis/pipelineprep/calibrations/'
FLAT = 'jwst_niriss_flat_0190.fits'
SUPERBIAS = 'jwst_niriss_superbias_0181.fits'
DARK = 'jwst_niriss_dark_0171.fits'
BADPIX = 'jwst_niriss_mask_0015.fits'
BACKGROUND = 'jwst_niriss_background_custom.fits'


def run_stage1(exposurename, outputname):

    # Define input/output
    calwebb_input = exposurename
    calwebb_output = outputname
    outdir = os.path.dirname(outputname)
    basename = os.path.basename(os.path.splitext(exposurename)[0])

    # Step by step DMS processing
    result = calwebb_detector1.dq_init_step.DQInitStep.call(calwebb_input, output_dir=outdir, save_results=True)#,
                                                            #override_mask=CALIBRATION_DIR+BADPIX)
    result = calwebb_detector1.saturation_step.SaturationStep.call(result, output_dir=outdir, save_results=True)
    # Custom 1/f correction
    result = soss_oneoverf.applycorrection(result, exposurename)
    result = calwebb_detector1.superbias_step.SuperBiasStep.call(result, output_dir=outdir, save_results=True)#,
                                                                 #override_superbias=CALIBRATION_DIR+SUPERBIAS)
    # Remove the DMS pipeline reference pixel correction
    #result = calwebb_detector1.refpix_step.RefPixStep.call(result, output_dir=outdir, save_results=True)
    result = calwebb_detector1.linearity_step.LinearityStep.call(result, output_dir=outdir, save_results=True)
    result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(result, output_dir=outdir, save_results=True)#,
                                                                      #override_dark=CALIBRATION_DIR+DARK)
    _, result = calwebb_detector1.ramp_fit_step.RampFitStep.call(result, output_dir=outdir, save_results=True)
    result = calwebb_detector1.gain_scale_step.GainScaleStep.call(result, output_dir=outdir, save_results=True)
    result.meta.filetype = 'countrate'
    result.write(calwebb_output)

    return

def run_stage2(rateints):

    calwebb_input = rateints
    outdir = os.path.dirname(rateints)
    basename = os.path.basename(os.path.splitext(rateints)[0])

    # Flat fielding
    result = calwebb_spec2.flat_field_step.FlatFieldStep.call(calwebb_input, output_dir=outdir, save_results=True)#,
                                                              #override_flat=CALIBRATION_DIR+FLAT)

    # Custom - Outlier flagging
    result = soss_outliers.flag_outliers(result, verbose=True)

    # Custom - Background subtraction step
    result = commutils.background_subtraction(result, aphalfwidth=[30,20,20], outdir=outdir, verbose=True,
                                              override_background=CALIBRATION_DIR+BACKGROUND, applyonintegrations=False)

    # Custom - Check that no NaNs in in the rateints data
    result = commutils.remove_nans(result)

    if False:
        # spectrum extraction - forcing no dx=0, dy=0, dtheta=0
        result = calwebb_spec2.extract_1d_step.Extract1dStep.call(result, output_dir=outdir, save_results=True,
                                                              #soss_transform=[0, 0, 0],
                                                              soss_modelname=outdir+'/'+basename+'_atoca_model.fits')

        # Conversion to SI units
        result = calwebb_spec2.photom_step.PhotomStep.call(result, output_dir=outdir, save_results=True)

    # Write results on disk
    result.close()

    return

def run_differentialextraction(rateints):
    ''' Doing the spectrum extraction on images - deepstack'''

    calwebb_input = rateints
    outdir = os.path.dirname(rateints)
    basename = os.path.basename(os.path.splitext(rateints)[0])



    # Flat fielding
    result = calwebb_spec2.flat_field_step.FlatFieldStep.call(calwebb_input, output_dir=outdir, save_results=True,
                                                              override_flat=CALIBRATION_DIR + FLAT)


    # Custom - Outlier flagging
    result = soss_outliers.flag_outliers(result, verbose=True)


    # Custom - Check that no NaNs in in the rateints data
    result = commutils.remove_nans(result)

    # spectrum extraction - forcing no dx=0, dy=0, dtheta=0
    result = calwebb_spec2.extract_1d_step.Extract1dStep.call(result, output_dir=outdir, save_results=True,
                                                              soss_transform=[0, 0, 0],
                                                              soss_modelname=outdir + '/' + basename + '_atoca_model.fits')


    # Perform the regular steps of stage 2, except on the high SNR deep stack
    # That is necessary to get absolute fluxes.
    #deepstackmodel = commutils.
    run_stage2()

    # Conversion to SI units
    result = calwebb_spec2.photom_step.PhotomStep.call(result, output_dir=outdir, save_results=True)

    # Write results on disk
    result.close()

    return






if __name__ == "__main__":
    ################ MAIN ###############
    dir = '/Users/albert/NIRISS/Commissioning/analysis/pipelineprep/'
    dataset = 'demo4ints_clear_noisy'
    dataset = 'twa33_substrip256_clear_noisy'

    dir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/'
    dataset = 'jw01092001001_02101_00001_nis_uncal' # TA
    dataset = 'jw01092001001_02101_00002_nis_uncal' # TA
    dataset = 'jw01092001001_02101_00003_nis_uncal' # TA
    dataset = 'jw01092001001_02101_00004_nis_uncal' # TA
    dataset = 'jw01092001001_03101_00001_nis_uncal' # 20 ints ss256 clear
    #dataset = 'jw01092001001_03102_00001_nis_uncal' # ss96
    #dataset = 'jw01092001001_03103_00001_nis_uncal' # ss96 ng=1 160 ints
    #dataset = 'jw01092001001_03104_00001_nis_uncal' # ss256 F277 nint=20 ng=5

    run_stage1(dir+dataset+'.fits', dir+dataset+'_custom_stage1.fits')
    run_stage2(dir+dataset+'_custom_stage1.fits')

    # Additional diagnostics
    spectrum_file = dir+dataset+'_custom_stage1_flatfieldstep_backsubtracted_extract1dstep.fits'
    a = commutils.plot_timeseries(spectrum_file, outdir=None, norder=3)

