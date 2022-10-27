import numpy as np

import jwst

from astropy.io import fits

import matplotlib.pyplot as plt

from jwst.pipeline import calwebb_detector1

from jwst.pipeline import calwebb_spec2

from jwst import datamodels

import SOSS.commissioning.comm_utils as commutils

import SOSS.dms.soss_oneoverf as soss_oneoverf

import SOSS.dms.soss_outliers as soss_outliers

import SOSS.dms.soss_boxextract as soss_boxextract

from SOSS.dms import oneoverf_step

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
#BACKGROUND = 'jwst_niriss_background_custom.fits'
BACKGROUND = 'model_background256.fits'  # Nestor's
SPECTRACE = 'SOSS_ref_trace_table_SUBSTRIP256.fits'
WAVEMAP = 'SOSS_ref_2D_wave_SUBSTRIP256.fits'
SPECPROFILE = 'SOSS_ref_2D_profile_SUBSTRIP256.fits'



def custom_loic(exposurelist, save_results=False):

    # Correct the 1/f noise at the full time-series level rather than
    # segment by segment (because 1/f residuals on the stack differ
    # between segments). That means that the DQinit and Saturation steps
    # need to occur also at the time-series wide level.

    nsegments = np.size(exposurelist)

    saturationstep_list = []
    for segment in range(nsegments):
        # Define input/output
        segmentname = exposurelist[segment]
        outdir = os.path.dirname(segmentname)
        #basename = os.path.basename(os.path.splitext(segmentname)[0])
        #basename = basename.split('_nis')[0] + '_nis'
        #basename_ts = basename.split('-seg')[0]

        if False:
            # DMS standard - GroupScaleStep
            result = calwebb_detector1.group_scale_step.GroupScaleStep.call(
                segmentname, output_dir=outdir, save_results=False)

            # DMS standard - DQInitStep
            result = calwebb_detector1.dq_init_step.DQInitStep.call(
                result, output_dir=outdir, save_results=False)

            # DMS standard - SaturationStep
            result = calwebb_detector1.saturation_step.SaturationStep.call(
                result, output_dir=outdir, save_results=True)

            # Keep a list of these files that were saved on disk for later usage with 1/f correction
            saturationstep_list.append(outdir+'/'+result.meta.filename)
        else:
            saturationstep_list = [
                '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/jw01091002001_03101_00001-seg001_nis_saturationstep.fits',
                '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/jw01091002001_03101_00001-seg002_nis_saturationstep.fits',
                '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/jw01091002001_03101_00001-seg003_nis_saturationstep.fits',
                '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/jw01091002001_03101_00001-seg004_nis_saturationstep.fits',
                '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/jw01091002001_03101_00001-seg005_nis_saturationstep.fits']
            print('saturation files list: ', saturationstep_list)


    # Custom - Proceed with construction of the deep stack for each group using
    # all segments available.
    deepstack, rms = soss_oneoverf.stack_multisegments(saturationstep_list, outdir=outdir, save_results=True)
    # TODO: Note that groupdq is mostly empty (only a few 3x3 NaNs squares are found in deepstack)

    # Proceed back on a segment by segment basis (rather than at the whole time-series level)
    for segment in range(nsegments):
        # Read back the file on disk
        result = datamodels.open(saturationstep_list[segment])

        # Custom - 1/f correction
        result = soss_oneoverf.applycorrection(
            result, output_dir=outdir, save_results=True, deepstack_custom=deepstack)

        # DMS standard - SuperBias subtraction
        # Skipped because the next step (dark subtraction) uses a custom made dark in which
        # the superbias is already included in. Warning, turn this back on if you revert the
        # dark subtraction to the default calibration file.
        # result = calwebb_detector1.superbias_step.SuperBiasStep.call(result, output_dir=outdir, save_results=True)#,
        # override_superbias=CALIBRATION_DIR+SUPERBIAS)

        # Custom - Dark + Superbias subtraction
        # The DMS dark subtraction is needed because it captures the hot pixels and their 4 neighbors
        # that otherwise can appear as uncorrected bad pixels in final products.
        # Here the custom dark includes the superbias level so we skipped the superbias step.
        # TODO: improve the current dark calibration file to remove the obvious 1/f noise residuals.
        result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(
            result, output_dir=outdir, save_results=False,
            override_dark=CALIBRATION_DIR+'jwst_niriss_dark_loiccustom.fits')

        # DMS standard - RefPix correction
        # Remove the DMS pipeline reference pixel correction
        #result = calwebb_detector1.refpix_step.RefPixStep.call(result, output_dir=outdir, save_results=True)

        # DMS standard - Non-linearity correction
        result = calwebb_detector1.linearity_step.LinearityStep.call(
            result, output_dir=outdir, save_results=False)

        # DMS standard - Jump detection
        result = calwebb_detector1.jump_step.JumpStep.call(
            result, output_dir=outdir, rejection_threshold=6, save_results=False)

        # DMS standard - Ramp fitting
        stackresult, result = calwebb_detector1.ramp_fit_step.RampFitStep.call(
            result, output_dir=outdir, save_results=False)

        # DMS standard - Gain step
        result = calwebb_detector1.gain_scale_step.GainScaleStep.call(
            result, output_dir=outdir, save_results=False)

        result.meta.filetype = 'countrate'
        #result.meta.filename = outdir+'/'+basename+'_customrateints.fits'
        result.write(outdir+'/'+basename+'_customrateints.fits')

        # Process the stacked od all integrations as well
        stackresult = calwebb_detector1.gain_scale_step.GainScaleStep.call(stackresult, output_dir=outdir, save_results=False)
        stackresult.meta.filetype = 'countrate'
        #stackresult.meta.filename = outdir+'/'+basename+'_customrate.fits'
        stackresult.save(outdir+'/'+basename+'_customrate.fits')




    return


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
    result = calwebb_detector1.group_scale_step.GroupScaleStep.call(calwebb_input, output_dir=outdir, save_results=False)

    result = calwebb_detector1.dq_init_step.DQInitStep.call(result, output_dir=outdir, save_results=False)#,
                                                            #override_mask=CALIBRATION_DIR+BADPIX)

    result = calwebb_detector1.saturation_step.SaturationStep.call(result, output_dir=outdir, save_results=False)

    # Add some bad pixels missed by the dq init stage but seen otherwise
    result = commutils.add_manual_badpix(result)

    # Custom 1/f correction
    result = soss_oneoverf.applycorrection(result, output_dir=outdir, save_results=True)

    #result = calwebb_detector1.superbias_step.SuperBiasStep.call(result, output_dir=outdir, save_results=True)#,
                                                                 #override_superbias=CALIBRATION_DIR+SUPERBIAS)

    # The DMS dark subtraction is needed because it captures the hot pixels and their 4 neighbors
    # that otherwise can appear as uncorrected bad pixels in final products.
    # TODO: improve the current dark calibration file to remove the obvious 1/f noise residuals.
    result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(result, output_dir=outdir, save_results=False,
                                                                      override_dark=CALIBRATION_DIR+'jwst_niriss_dark_loiccustom.fits')


    # Remove the DMS pipeline reference pixel correction
    #result = calwebb_detector1.refpix_step.RefPixStep.call(result, output_dir=outdir, save_results=True)

    #result = calwebb_detector1.linearity_step.LinearityStep.call(result, output_dir=outdir, save_results=False)
    # to test jump detection, save = True
    result = calwebb_detector1.linearity_step.LinearityStep.call(result, output_dir=outdir, save_results=False)


    result = calwebb_detector1.jump_step.JumpStep.call(result, output_dir=outdir, save_results=False,
                                                       rejection_threshold=6)
    #sys.exit()

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


def run_stage2(rateints, contamination_mask=None, use_atoca=False, run_outliers=True,
               optimal_extraction=False):
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
    result = calwebb_spec2.flat_field_step.FlatFieldStep.call(calwebb_input, output_dir=outdir, save_results=True)#,
                                                              #override_flat=CALIBRATION_DIR+FLAT)

    # Custom - Outlier flagging
    if run_outliers == True:
        result = soss_outliers.flag_outliers(result, window_size=(3,11), n_sig=9, verbose=True, outdir=outdir, save_diagnostic=True)

    # Still non-optimal
    # Custom - Background subtraction step
    #result = commutils.background_subtraction_v1(result, aphalfwidth=[40,20,20], outdir=outdir, verbose=False,
    #                                          override_background=CALIBRATION_DIR+BACKGROUND, applyonintegrations=True,
    #                                          contamination_mask=contamination_mask, trace_table_ref=ATOCAREF_DIR+SPECTRACE)
    #
    result = commutils.background_subtraction(result, aphalfwidth=[40,20,20], outdir=outdir, verbose=False,
                                              contamination_mask=contamination_mask, trace_table_ref=ATOCAREF_DIR+SPECTRACE)

    # Clean the outlier and bad pixels based on a deep stack
    result = commutils.soss_interp_badpix(result, outdir)

    # Subtract a local background below order 1 close to the trace
    # result = commutils.localbackground_subtraction(result, ATOCAREF_DIR+SPECTRACE, width=9, back_offset=-25)

    # Custom - Check that no NaNs is in the rateints data
    result = commutils.remove_nans(result, outdir=outdir, save_results=True)

    # Untested
    # Custom - Build a rate.fits equivalent (that profits from the outlier knowledge)
    #stackresult = commutils.stack_rateints(result, outdir=outdir)


    # spectrum extraction - forcing no dx=0, dy=0, dtheta=0
    print(result.meta.filename)

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
        if optimal_extraction:
            # soss_atoca=False --> box extraction only
            # carefull to not turn it on. Would if soss_bad_pix='model' or soss_modelname=set_to_something
            result = calwebb_spec2.extract_1d_step.Extract1dStep.call(result, output_dir=outdir, save_results=True,
                                                                      soss_transform=[0, 0, 0],
                                                                      soss_atoca=False,
                                                                      subtract_background=False,
                                                                      soss_bad_pix='masking',
                                                                      soss_extraction_type='optimal',
                                                                      soss_width=25,
                                                                      # soss_tikfac=3.38e-15,
                                                                      soss_modelname=None,
                                                                      override_spectrace=ATOCAREF_DIR + SPECTRACE,
                                                                      override_wavemap=ATOCAREF_DIR + WAVEMAP,
                                                                      override_specprofile=ATOCAREF_DIR + SPECPROFILE)

        else:
            # soss_atoca=False --> box extraction only
            # carefull to not turn it on. Would if soss_bad_pix='model' or soss_modelname=set_to_something
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

    # data set to process:
    #datasetname = 'SOSSwavecal'
    datasetname = 'SOSSfluxcal'
    #datasetname = 'HATP14b'
    #datasetname = 'T1'
    #datasetname = 'T1_2'

    # Wavelength calibration
    if datasetname == 'wavecal':
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local') :
            dir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/'
            contmask = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/mask_contamination.fits'
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/SOSSwavecal/'
            contmask = None
        else:
            sys.exit()
        datalist = ['jw01092010001_03101_00001_nis'] # SS256 CLEAR 20 ints

    # Flux Calibration
    if datasetname == 'SOSSfluxcal':
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'
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


    # HATP14b
    if datasetname == 'HATP14b':
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/ref_files/'
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
        dataset_string = 'jw01541001001_04101_00001'

    # T1
    if datasetname == 'T1':
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/T1/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1/ref_files/'
            contmask = '/Users/albert/NIRISS/Commissioning/analysis/T1/mask_contamination.fits'
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw02589001001_04101_00001-seg001_nis',
            'jw02589001001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw02589001001_04101_00001'

    # T1_2
    if datasetname == 'T1_2':
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/T1_2/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_2/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw02589002001_04101_00001-seg001_nis',
            'jw02589002001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw02589002001_04101_00001'

    # Run the 2 iteration process
    uncal_list = []
    for oneseg in datalist: uncal_list.append(dir+oneseg+'_uncal.fits')
    custom_loic(uncal_list, save_results=True)
    sys.exit()

    for dataset in datalist:
        print('Running twice through the stage 1 + stage 2 steps.')
        print('Iteration 1 will produce an outlier map at stage 2.')
        print('Iteration 2 uses that outlier map to better suppress the 1/f noise.')
        use_atoca = False
        optimal_extraction = False

        custom_or_not = '_customrateints' #'_rateints'
        #run_stage1(dir+dataset+'_uncal.fits')
        run_stage2(dir+dataset+custom_or_not+'.fits', contamination_mask=contmask, run_outliers=True,
                   use_atoca=use_atoca, optimal_extraction=optimal_extraction)

    # Post processing analysis
    for dataset in datalist:
        # Additional diagnostics - Subtracting the ATOCA model from the images
        if use_atoca:
            commutils.check_atoca_residuals(dir+dataset+custom_or_not+'.fits', dir+dataset+'_atoca_model_SossExtractModel.fits')
        spectrum_file = dir+dataset+custom_or_not+'_extract1dstep.fits'
        a = commutils.plot_timeseries(spectrum_file, norder=3)

    # Combining segments and creating timeseries greyscales
    outdir = '/Users/albert/NIRISS/Commissioning/analysis/'+datasetname+'/'
    wildcard = outdir+'supplemental_'+dataset_string+'-seg00?_nis/timeseries_greyscale_rawflux.fits'
    a = commutils.combine_timeseries(wildcard, outdir+'timeseries_greyscale.fits')
    a = commutils.greyscale_rms(outdir+'timeseries_greyscale.fits', title='')
    wildcard = outdir+dataset_string+'-seg00?_nis'+custom_or_not+'_extract1dstep.fits'
    print(wildcard)
    a = commutils.combine_multi_spec(wildcard, outdir+'extracted_spectrum.fits')
