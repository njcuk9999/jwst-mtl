import numpy as np

import jwst

from astropy.io import fits

import matplotlib.pyplot as plt

from jwst.pipeline import calwebb_detector1

from jwst.pipeline import calwebb_spec2

import jwst.datamodels

import SOSS.commissioning.comm_utils as commutils

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
#BACKGROUND = 'jwst_niriss_background_custom.fits'
BACKGROUND = 'model_background256.fits'  # Nestor's
SPECTRACE = 'SOSS_ref_trace_table_SUBSTRIP256.fits'
WAVEMAP = 'SOSS_ref_2D_wave_SUBSTRIP256.fits'
SPECPROFILE = 'SOSS_ref_2D_profile_SUBSTRIP256.fits'





def run_stage1(exposurename, outlier_map=None):

    '''
    These are the default DMS steps for stage 1.
    Default pipeline.calwebb_detector1 steps

            input = self.group_scale(input)
            input = self.dq_init(input)
            input = self.saturation(input)
            input = self.ipc(input)
            input = self.superbias(input)
            input = self.refpix(input)
            input = self.linearity(input)
            input = self.dark_current(input)
            input = self.jump(input)
            input = self.ramp_fit(input)
            input = self.gain_scale(input)
    '''


    # Define input/output
    calwebb_input = exposurename
    outdir = os.path.dirname(exposurename)
    basename = os.path.basename(os.path.splitext(exposurename)[0])
    basename = basename.split('_nis')[0]+'_nis'

    # Step by step DMS processing
    result = calwebb_detector1.group_scale_step.GroupScaleStep.call(calwebb_input, output_dir=outdir, save_results=True)

    result = calwebb_detector1.dq_init_step.DQInitStep.call(result, output_dir=outdir, save_results=True)

    result = calwebb_detector1.saturation_step.SaturationStep.call(result, output_dir=outdir, save_results=True)

    result = calwebb_detector1.superbias_step.SuperBiasStep.call(result, output_dir=outdir, save_results=True)

    result = calwebb_detector1.refpix_step.RefPixStep.call(result, output_dir=outdir, save_results=True)

    result = calwebb_detector1.linearity_step.LinearityStep.call(result, output_dir=outdir, save_results=True)

    result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(result, output_dir=outdir, save_results=True)

    result = calwebb_detector1.jump_step.JumpStep.call(result, output_dir=outdir, save_results=True)

    stackresult, result = calwebb_detector1.ramp_fit_step.RampFitStep.call(result, output_dir=outdir, save_results=True)

    result = calwebb_detector1.gain_scale_step.GainScaleStep.call(result, output_dir=outdir, save_results=True)

    result.meta.filetype = 'countrate'
    result.write(outdir+'/'+basename+'_rateints.fits')


    return


def run_stage2(rateints, contamination_mask=None, use_atoca=False, run_outliers=True):
    '''
    These are the default DMS steps for Stage 2.
        input = self.assign_wcs(input)
        input = self.flat_field(input)
        input = self.
        input = self.extract_1d(input)
        input = self.photom(input)

    '''

    calwebb_input = rateints
    outdir = os.path.dirname(rateints)
    basename = os.path.basename(os.path.splitext(rateints)[0])
    basename = basename.split('_nis')[0]+'_nis'
    stackbasename = basename+'_stack'

    # Flat fielding
    result = calwebb_spec2.flat_field_step.FlatFieldStep.call(calwebb_input, output_dir=outdir, save_results=True)

    if use_atoca:
        result = calwebb_spec2.extract_1d_step.Extract1dStep.call(result, output_dir=outdir, save_results=True,
                                                                  soss_transform=[0, 0, 0],
                                                                  soss_atoca = True,
                                                                  #soss_transform=[None, 0, None],
                                                                  subtract_background=False,
                                                                  soss_bad_pix='model',
                                                                  soss_width=25,
                                                                  #soss_tikfac=3.38e-15,
                                                                  soss_modelname=outdir+'/'+basename+'_atoca_model.fits',
                                                                  override_spectrace=ATOCAREF_DIR+SPECTRACE,
                                                                  override_wavemap=ATOCAREF_DIR+WAVEMAP,
                                                                  override_specprofile=ATOCAREF_DIR+SPECPROFILE)
    else:
        # soss_atoca=False --> box extraction only
        # carefull to not turn it on. Woul dif soss_bad_pix='model' or soss_modelname=set_to_something
        result = calwebb_spec2.extract_1d_step.Extract1dStep.call(result, output_dir=outdir, save_results=True,
                                                                  soss_transform=[0, 0, 0],
                                                                  soss_atoca=False,
                                                                  subtract_background=False,
                                                                  soss_bad_pix='masking',
                                                                  soss_width=25,
                                                                  # soss_tikfac=3.38e-15,
                                                                  soss_modelname=None,
                                                                  override_spectrace=ATOCAREF_DIR + SPECTRACE,
                                                                  override_wavemap=ATOCAREF_DIR + WAVEMAP,
                                                                  override_specprofile=ATOCAREF_DIR + SPECPROFILE)


    # Conversion to SI units
    result = calwebb_spec2.photom_step.PhotomStep.call(result, output_dir=outdir, save_results=True)

    # Write results on disk
    result.close()

    return






if __name__ == "__main__":
    ################ MAIN ###############

    datasetname = 'SOSSfluxcal'


    # Flux Calibration
    if datasetname == 'SOSSfluxcal':
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/'
            #contmask = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/mask_contamination.fits'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/SOSSfluxcal/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01091002001_03101_00001-seg001_nis',
            'jw01091002001_03101_00001-seg002_nis',
            'jw01091002001_03101_00001-seg003_nis',
            'jw01091002001_03101_00001-seg004_nis',
            'jw01091002001_03101_00001-seg005_nis'
        ]
        #datalist = [
        #    'jw01091002001_03101_00001-seg001_nis'
        #]
        dataset_string = 'jw01091002001_03101_00001'


    # Run the 2 iteration process
    for dataset in datalist:
        print('Running twice through the stage 1 + stage 2 steps.')
        print('Iteration 1 will produce an outlier map at stage 2.')
        custom_or_not = '_rateints'
        run_stage1(dir+dataset+'_uncal.fits')
        run_stage2(dir+dataset+custom_or_not+'.fits', use_atoca=False)
