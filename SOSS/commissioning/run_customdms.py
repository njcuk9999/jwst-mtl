import numpy as np

import jwst

from astropy.io import fits

import glob

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



def custom_loic(exposurelist, use_atoca=False, optimal_extraction=False,
                run_outliers=True, contamination_mask=None, extract_only=False,
                skip_stacking=False, erase_clean=False, satmap=None):

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
        basename = os.path.basename(os.path.splitext(segmentname)[0])
        basename = basename.split('_nis')[0] + '_nis'
        basename_ts = basename.split('-seg')[0]

        #if True:
        if (extract_only == False) & (skip_stacking == False):
            # DMS standard - GroupScaleStep
            result = calwebb_detector1.group_scale_step.GroupScaleStep.call(
                segmentname, output_dir=outdir, save_results=False)

            # DMS standard - DQInitStep
            result = calwebb_detector1.dq_init_step.DQInitStep.call(
                result, output_dir=outdir, save_results=False)

            # DMS standard - SaturationStep
            if satmap == None:
                result = calwebb_detector1.saturation_step.SaturationStep.call(
                    result, output_dir=outdir, save_results=True)
            else:
                satmap_filename = 'custom_saturation_' + str(satmap) + 'e.fits'
                result = calwebb_detector1.saturation_step.SaturationStep.call(
                    result, output_dir=outdir, save_results=True,
                    override_saturation=CALIBRATION_DIR + satmap_filename)

            # Keep a list of these files that were saved on disk for later usage with 1/f correction
            saturationstep_list.append(outdir+'/'+result.meta.filename)
        else:
            wildcard = outdir+'/'+basename_ts+'*_saturationstep.fits'
            saturationstep_list = sorted(glob.glob(wildcard))
            print('saturation files list: ', saturationstep_list)


    #if True:
    if (extract_only == False) & (skip_stacking == False):
        # Custom - Proceed with construction of the deep stack for each group using
        # all segments available.
        deepstack, rms = soss_oneoverf.stack_multisegments(saturationstep_list, outdir=outdir,
                                                           save_results=True)
        # TODO: Note that groupdq is mostly empty (only a few 3x3 NaNs squares are found in deepstack)
    else:
        oof_stackname = outdir+'/oof_deepstack_'+basename_ts+'.fits'
        deepstack = fits.getdata(oof_stackname)
        #deepstack = fits.getdata('/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/oof_deepstack_jw01091002001_03101_00001.fits')



    # Proceed back on a segment by segment basis (rather than at the whole time-series level)
    for segment in range(nsegments):
        # Read back the file on disk
        result = datamodels.open(saturationstep_list[segment])
        if erase_clean:
            # Erase the previous steps not longer used files from disk
            os.system('rm -rf '+saturationstep_list[segment])
        # Define input/output
        basename = result.meta.filename
        basename = os.path.basename(os.path.splitext(basename)[0])
        basename = basename.split('_nis')[0] + '_nis'

        if segment == 0: fn = open(outdir+'/filename.txt','w')

        #if True:
        if extract_only == False:
            # Custom - 1/f correction
            result = soss_oneoverf.applycorrection(
                result, output_dir=outdir, save_results=False, deepstack_custom=deepstack)
            if segment == 0: fn.write('{:} - After 1/f \n'.format(result.meta.filename))

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
            # TODO: improve the current dark calibration file by taking KTC noise into account.
            custom_darkname = 'jwst_niriss_dark_loiccustom.fits'
            if result.meta.subarray.name == 'SUBSTRIP96':
                custom_darkname = 'jwst_niriss_dark_loiccustom_substrip96.fits'
            result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(
                result, output_dir=outdir, save_results=False,
                override_dark=CALIBRATION_DIR+'/'+custom_darkname)
            if segment == 0: fn.write('{:} - After dark current \n'.format(result.meta.filename))

            # DMS standard - RefPix correction
            # Remove the DMS pipeline reference pixel correction
            # Replaced by our custom 1/f correction
            #result = calwebb_detector1.refpix_step.RefPixStep.call(result, output_dir=outdir, save_results=True)

            # DMS standard - Non-linearity correction
            result = calwebb_detector1.linearity_step.LinearityStep.call(
                result, output_dir=outdir, save_results=False)
            if segment == 0: fn.write('{:} - After linearity \n'.format(result.meta.filename))

            # DMS standard - Jump detection
            result = calwebb_detector1.jump_step.JumpStep.call(
                result, output_dir=outdir, rejection_threshold=6, save_results=False)
            if segment == 0: fn.write('{:} - After jump \n'.format(result.meta.filename))

            # DMS standard - Ramp fitting
            stackresult, result = calwebb_detector1.ramp_fit_step.RampFitStep.call(
                result, output_dir=outdir, save_results=False)
            if segment == 0: fn.write('{:} - After ramp fitting \n'.format(result.meta.filename))
        else:
            result = datamodels.open(outdir+'/'+basename+'_1_rampfitstep.fits')
            result.meta.filename = basename


        f = open(outdir+'/dq_trace.txt', 'w')
        f.write('DQ={:} - After ramp fit step \n'.format(result.dq[0,88,1361]))

        if extract_only == False:
            # DMS standard - Gain step
            result = calwebb_detector1.gain_scale_step.GainScaleStep.call(
                result, output_dir=outdir, save_results=False)
            f.write('DQ={:} - After gain step \n'.format(result.dq[0,88,1361]))
            if segment == 0: fn.write('{:} - After gain scale \n'.format(result.meta.filename))

            # Custom - Flag bad pixels found manually
            # Add some bad pixels missed by the dq init stage but seen otherwise
            result = commutils.add_manual_badpix(result)
            f.write('DQ={:} - After manual bad pix step \n'.format(result.dq[0,88,1361]))
            if segment == 0: fn.write('{:} - After manual bad pix \n'.format(result.meta.filename))

            hdu = fits.PrimaryHDU(result.dq)
            hdu.writeto(outdir+'/dq_postmanual.fits', overwrite=True)
            f.write('DQ={:} - After saving of manual bad pix step \n'.format(result.dq[0,88,1361]))
            if segment == 0: fn.write('{:} - After manual + fits.writeto \n'.format(result.meta.filename))

            # DMS standard - Save rateints on disk to end Stage 1
            result.meta.filetype = 'countrate'
            rateints_filename = outdir+'/'+basename+'_rateints.fits'
            result.write(rateints_filename)
            f.write('DQ={:} - After saving rateints \n'.format(result.dq[0,88,1361]))
            if segment == 0: fn.write('{:} - rateints save \n'.format(result.meta.filename))

            fn.close()

            # STAGE 2 starts here ------

            # DMS standard - Flat fielding
            result = calwebb_spec2.flat_field_step.FlatFieldStep.call(
                rateints_filename, output_dir=outdir, save_results=False)
                #override_flat=CALIBRATION_DIR+FLAT
            f.write('DQ={:} - After flat fielding step \n'.format(result.dq[0,88,1361]))

            if erase_clean:
                os.system('rm -rf '+rateints_filename)

            # Custom - Outlier flagging
            if run_outliers:
                result = soss_outliers.flag_outliers(
                    result, window_size=(3,11), n_sig=9, verbose=True, outdir=outdir,
                    save_diagnostic=~erase_clean, save_results=False)
            f.write('DQ={:} - After outlier flagging step \n'.format(result.dq[0,88,1361]))


            # Custom - Background subtraction
            result = commutils.background_subtraction(
                result, aphalfwidth=[40,20,20], outdir=outdir, verbose=False, save_results=~erase_clean,
                contamination_mask=contamination_mask, trace_table_ref=ATOCAREF_DIR+SPECTRACE)
            f.write('DQ={:} - After background subtraction step \n'.format(result.dq[0,88,1361]))


            # Custom - Bad pixel interpolation
            # Clean the outlier and bad pixels based on a deep stack
            hdu = fits.PrimaryHDU(result.data)
            hdu.writeto(outdir+'/prestack_data.fits', overwrite=True)
            hdu = fits.PrimaryHDU(result.dq)
            hdu.writeto(outdir+'/prestack_dq.fits', overwrite=True)
            # ici DQ est bon 1362,139 = 1
            result = commutils.soss_interp_badpix(result, outdir, save_results=True)
            f.write('DQ={:} - After bad pix interpolation step \n'.format(result.dq[0,88,1361]))
        else:
            result = datamodels.open(outdir+'/'+basename+'_badpixinterp.fits')

        # Subtract a local background below order 1 close to the trace
        # result = commutils.localbackground_subtraction(result, ATOCAREF_DIR+SPECTRACE, width=9, back_offset=-25)

        # Custom - Remove NaNs
        # Check that no NaNs is in the data before sending to extraction
        # (atoca can't handle Nans)
        result = commutils.remove_nans(result, outdir=outdir, save_results=True)
        f.write('DQ={:} - After remove nans step \n'.format(result.dq[0,88,1361]))
        f.close()

        # Spectrum extraction ----------------------------
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


        # DMS standard - Conversion to SI units
        result = calwebb_spec2.photom_step.PhotomStep.call(
            result, output_dir=outdir, save_results=False)

        # Write results on disk
        result.close()
        #sys.exit()

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
    #datasetname = 'LTT9779'
    #datasetname = 'SOSSwavecal'
    #datasetname = 'SOSSfluxcal'
    #datasetname = 'SOSSfluxcalss96ng3'
    datasetname = 'HATP14b'
    #datasetname = 'T1'
    #datasetname = 'T1_2'
    #datasetname = 'T1_3'
    #datasetname = 'T1_4'

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
        dataset_string = 'jw01091002001_03101_00001'

    # Flux Calibration -- SUBSTRIP96 NG=3
    if datasetname == 'SOSSfluxcalss96ng3':
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal_ss96_ng3/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'
            #contmask = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/mask_contamination.fits'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/SOSSfluxcal_ss96_ng3/'
            contmask = None
        else:
            sys.exit()

        datalist = ['jw01091001001_03102_00001_nis']

        dataset_string = 'jw01091001001_03102_00001'


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

    # T1_3
    if datasetname == 'T1_3':
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_3/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201101001_04101_00001-seg001_nis',
            'jw01201101001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw01201101001_04101_00001'

    # T1_4
    if datasetname == 'T1_4':
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/T1_4/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw02589003001_04101_00001-seg001_nis',
            'jw02589003001_04101_00001-seg002_nis',
            'jw02589003001_04101_00001-seg003_nis'
        ]
        dataset_string = 'jw02589003001_04101_00001'

    # LTT9779 - phase curve
    if datasetname == 'LTT9779':
        if (hostname == 'iiwi.sf.umontreal.ca') or (hostname == 'iiwi.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/LTT9779/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/LTT9779//'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201002001_04101_00001-seg001_nis',
            'jw01201002001_04101_00001-seg002_nis',
            'jw01201002001_04101_00001-seg003_nis',
            'jw01201002001_04101_00001-seg004_nis',
            'jw01201002001_04101_00001-seg005_nis',
            'jw01201002001_04101_00001-seg006_nis',
            'jw01201002001_04101_00001-seg007_nis',
            'jw01201002001_04101_00001-seg008_nis',
            'jw01201002001_04101_00001-seg009_nis',
            'jw01201002001_04101_00001-seg010_nis',
            'jw01201002001_04101_00001-seg011_nis',
            'jw01201002001_04101_00001-seg012_nis',
            'jw01201002001_04101_00001-seg013_nis',
            'jw01201002001_04101_00001-seg014_nis',
            'jw01201002001_04101_00001-seg015_nis',
            'jw01201002001_04101_00001-seg016_nis',
            'jw01201002001_04101_00001-seg017_nis',
            'jw01201002001_04101_00001-seg018_nis',
            'jw01201002001_04101_00001-seg019_nis',
            'jw01201002001_04101_00001-seg020_nis',
            'jw01201002001_04101_00001-seg021_nis',
            'jw01201002001_04101_00001-seg022_nis',
            'jw01201002001_04101_00001-seg023_nis',
            'jw01201002001_04101_00001-seg024_nis',
            'jw01201002001_04101_00001-seg025_nis',
            'jw01201002001_04101_00001-seg026_nis'
        ]
        dataset_string = 'jw01201002001_04101_00001'


    '''
    RUN THE PIPELINE--------------------------------------------------------------------
    '''
    custom_or_not = '_rateints'
    run_outliers = True
    use_atoca = False
    optimal_extraction = False
    extract_only = False
    skip_stacking = False
    erase_clean = False
    satmap = 35000 # or None

    if True:
        # Method 1 - streamlined
        uncal_list = []
        for oneseg in datalist: uncal_list.append(dir + oneseg + '_uncal.fits')
        custom_loic(uncal_list, use_atoca=use_atoca, run_outliers=run_outliers,
                    optimal_extraction=optimal_extraction, contamination_mask=contmask,
                    extract_only=extract_only, skip_stacking=skip_stacking,
                    erase_clean=erase_clean, satmap=satmap)
    else:
        # Method 2 - former way
        for dataset in datalist:
            use_atoca = False
            optimal_extraction = False

            custom_or_not = '_customrateints' #'_rateints'
            #run_stage1(dir+dataset+'_uncal.fits')
            run_stage2(dir+dataset+custom_or_not+'.fits', contamination_mask=contmask, run_outliers=True,
                       use_atoca=use_atoca, optimal_extraction=optimal_extraction)

    # Post processing analysis
    for oneseg in datalist:
        # Additional diagnostics - Subtracting the ATOCA model from the images
        if use_atoca:
            commutils.check_atoca_residuals(dir+oneseg+'_rateints.fits', dir+oneseg+'_atoca_model_SossExtractModel.fits')
        spectrum_file = dir+oneseg+'_extract1dstep.fits'
        a = commutils.plot_timeseries(spectrum_file, norder=3)

    # Combining segments and creating timeseries greyscales
    outdir = '/Users/albert/NIRISS/Commissioning/analysis/'+datasetname+'/'
    wildcard = outdir+'supplemental_'+dataset_string+'-seg???_nis/timeseries_greyscale_rawflux.fits'
    print(wildcard)
    a = commutils.combine_timeseries(wildcard, outdir+'timeseries_greyscale.fits')
    a = commutils.greyscale_rms(outdir+'timeseries_greyscale.fits', title='')
    wildcard = outdir+dataset_string+'-seg???_nis_extract1dstep.fits'
    print(wildcard)
    a = commutils.combine_multi_spec(wildcard, outdir+'extracted_spectrum.fits')
