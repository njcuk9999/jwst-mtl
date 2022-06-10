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

import socket


hostname = socket.gethostname()
if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
    CALIBRATION_DIR = '/Users/albert/NIRISS/Commissioning/analysis/pipelineprep/calibrations/'
    ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'
elif hostname == 'genesis':
    CALIBRATION_DIR = '/genesis/jwst/jwst-ref-soss/noise_files/'
    ATOCAREF_DIR = '/genesis/jwst/userland-soss/loic_review/commissioning/ref_files/'
else:
    print('Add your local computer name in the list.')
    sys.exit()

FLAT = 'jwst_niriss_flat_0190.fits'
SUPERBIAS = 'jwst_niriss_superbias_0181.fits'
DARK = 'jwst_niriss_dark_0171.fits'
BADPIX = 'jwst_niriss_mask_0015.fits'
BACKGROUND = 'jwst_niriss_background_custom.fits'
SPECTRACE = 'SOSS_ref_trace_table_SUBSTRIP256.fits'
WAVEMAP = 'SOSS_ref_2D_wave_SUBSTRIP256.fits'
SPECPROFILE = 'SOSS_ref_2D_profile_SUBSTRIP256.fits'





def run_stage1(exposurename, outlier_map=None):

    # Define input/output
    calwebb_input = exposurename
    outdir = os.path.dirname(exposurename)
    basename = os.path.basename(os.path.splitext(exposurename)[0])
    basename = basename.split('_nis')[0]+'_nis'

    # Step by step DMS processing
    result = calwebb_detector1.dq_init_step.DQInitStep.call(calwebb_input, output_dir=outdir, save_results=False)#,
                                                            #override_mask=CALIBRATION_DIR+BADPIX)

    result = calwebb_detector1.saturation_step.SaturationStep.call(result, output_dir=outdir, save_results=False)

    # Custom 1/f correction
    result = soss_oneoverf.applycorrection(result, output_dir=outdir, save_results=True, outlier_map=outlier_map)

    result = calwebb_detector1.superbias_step.SuperBiasStep.call(result, output_dir=outdir, save_results=False)#,
                                                                 #override_superbias=CALIBRATION_DIR+SUPERBIAS)

    # Remove the DMS pipeline reference pixel correction
    #result = calwebb_detector1.refpix_step.RefPixStep.call(result, output_dir=outdir, save_results=True)

    result = calwebb_detector1.linearity_step.LinearityStep.call(result, output_dir=outdir, save_results=False)

    #result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(result, output_dir=outdir, save_results=True)#,
                                                                      #override_dark=CALIBRATION_DIR+DARK)

    stackresult, result = calwebb_detector1.ramp_fit_step.RampFitStep.call(result, output_dir=outdir, save_results=False)

    result = calwebb_detector1.gain_scale_step.GainScaleStep.call(result, output_dir=outdir, save_results=False)
    result.meta.filetype = 'countrate'
    #result.meta.filename = outdir+'/'+basename+'_customrateints.fits'
    result.write(outdir+'/'+basename+'_customrateints.fits')

    # Process the stacked od all integrations as well
    stackresult = calwebb_detector1.gain_scale_step.GainScaleStep.call(stackresult, output_dir=outdir, save_results=False)
    stackresult.meta.filetype = 'countrate'
    #stackresult.meta.filename = outdir+'/'+basename+'_customrate.fits'
    stackresult.save(outdir+'/'+basename+'_customrate.fits')

    return


def run_stage2(rateints, contamination_mask=None, skip_atoca=False):

    calwebb_input = rateints
    outdir = os.path.dirname(rateints)
    basename = os.path.basename(os.path.splitext(rateints)[0])
    basename = basename.split('_nis')[0]+'_nis'
    stackbasename = basename+'_stack'

    # Flat fielding
    result = calwebb_spec2.flat_field_step.FlatFieldStep.call(calwebb_input, output_dir=outdir, save_results=False)#,
                                                              #override_flat=CALIBRATION_DIR+FLAT)

    # Custom - Outlier flagging
    result = soss_outliers.flag_outliers(result, window_size=(7,7), n_sig=9, verbose=True, outdir=outdir, save_diagnostic=True)

    # Custom - Background subtraction step
    result = commutils.background_subtraction(result, aphalfwidth=[30,20,20], outdir=outdir, verbose=False,
                                              override_background=CALIBRATION_DIR+BACKGROUND, applyonintegrations=False,
                                              contamination_mask=contamination_mask)

    # Custom - Check that no NaNs is in the rateints data
    result = commutils.remove_nans(result)
    print(result.meta.filename)
    #result.write('/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/custom_preextract1d.fits')

    #aaa = result.copy()
    #aaa.write(outdir+'/toto.fits')

    # Custom - Build a rate.fits equivalent (that profits from the outlier knowledge)
    #stackresult = commutils.stack_rateints(result, outdir=outdir)


    if False:
        # spectrum extraction on stack - forcing no dx=0, dy=0, dtheta=0
        stackresult = calwebb_spec2.extract_1d_step.Extract1dStep.call(stackresult, output_dir=outdir, save_results=True,
                                                                  soss_transform=[0, 0, 0],
                                                                  soss_bad_pix='model',
                                                                  soss_tikfac=1e-15,
                                                                  soss_modelname=outdir+'/'+stackbasename+'_atoca_model.fits',
                                                                  override_spectrace=ATOCAREF_DIR+SPECTRACE,
                                                                  override_wavemap=ATOCAREF_DIR+WAVEMAP,
                                                                  override_specprofile=ATOCAREF_DIR+SPECPROFILE)

        # Conversion to SI units
        stackresult = calwebb_spec2.photom_step.PhotomStep.call(stackresult, output_dir=outdir, save_results=True)

        # Write results on disk
        stackresult.close()

    if skip_atoca == True:
        # Write results on disk
        result.close()
        return
    else:
        if True:
            # spectrum extraction - forcing no dx=0, dy=0, dtheta=0
            print(result.meta.filename)
            result = calwebb_spec2.extract_1d_step.Extract1dStep.call(result, output_dir=outdir, save_results=True,
                                                                      soss_transform=[0, 0, 0],
                                                                      #soss_transform=[None, 0, 0],
                                                                      subtract_background=False,
                                                                      soss_bad_pix='model',
                                                                      #soss_tikfac=3.38e-15,
                                                                      soss_modelname=outdir+'/'+basename+'_atoca_model.fits',
                                                                      override_spectrace=ATOCAREF_DIR+SPECTRACE,
                                                                      override_wavemap=ATOCAREF_DIR+WAVEMAP,
                                                                      override_specprofile=ATOCAREF_DIR+SPECPROFILE)

            # Conversion to SI units
            result = calwebb_spec2.photom_step.PhotomStep.call(result, output_dir=outdir, save_results=True)

        # Write results on disk
        result.close()

    return






if __name__ == "__main__":
    ################ MAIN ###############

    if False:
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local') :
            dir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/'
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/SOSSwavecal/'
        else:
            sys.exit()
        datalist = ['jw01092001001_02101_00001_nis_uncal'] # TA
        datalist = ['jw01092001001_02101_00002_nis_uncal'] # TA
        datalist = ['jw01092001001_02101_00003_nis_uncal'] # TA
        datalist = ['jw01092001001_02101_00004_nis_uncal'] # TA
        datalist = ['jw01092001001_03101_00001_nis_uncal'] # 20 ints ss256 clear
        #datalist = ['jw01092001001_03102_00001_nis_uncal'] # ss96
        #datalist = ['jw01092001001_03103_00001_nis_uncal'] # ss96 ng=1 160 ints
        #datalist = ['jw01092001001_03104_00001_nis_uncal'] # ss256 F277 nint=20 ng=5
        #datalist = ['jw01081001001_0210d_00001_nis_uncal'] # ss256 dark
        #datalist = ['jw01093011001_03103_00002_nis_uncal'] # ami kpi 232 ints 80x80
        # re-observation
        datalist = ['jw01092010001_03101_00001_nis'] # SS256 CLEAR 20 ints

    # Flux Calibration
    if False:
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/'
            #contmask = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/mask_contamination.fits'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/SOSSfluxcal/'
        else:
            sys.exit()

        datalist = [
            'jw01091002001_03101_00001-seg001_nis',
            'jw01091002001_03101_00001-seg002_nis',
            'jw01091002001_03101_00001-seg003_nis',
            'jw01091002001_03101_00001-seg004_nis',
            'jw01091002001_03101_00001-seg005_nis'
        ]
        datalist = ['jw01091001001_03101_00001_nis'] # flux SS256 CLEAR short


    # HATP14b
    if True:
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/HATP14b/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01541001001_04101_00001-seg004_nis',
            'jw01541001001_04101_00001-seg001_nis',
            'jw01541001001_04101_00001-seg002_nis',
            'jw01541001001_04101_00001-seg003_nis'
        ]


    for dataset in datalist:
        run_stage1(dir+dataset+'_uncal.fits', outlier_map=dir+dataset+'_outliers.fits')
        run_stage2(dir+dataset+'_customrateints.fits', contamination_mask=contmask, skip_atoca=True)
        run_stage1(dir+dataset+'_uncal.fits', outlier_map=dir+dataset+'_outliers.fits')
        run_stage2(dir+dataset+'_customrateints.fits', contamination_mask=contmask)
    #for dataset in datalist:
        # Additional diagnostics
        commutils.check_atoca_residuals(dir+dataset+'_customrateints_backsubtracted.fits',
                                        dir+dataset+'_atoca_model_SossExtractModel.fits')
        spectrum_file = dir+dataset+'_customrateints_backsubtracted_extract1dstep.fits'
        a = commutils.plot_timeseries(spectrum_file, norder=3)

