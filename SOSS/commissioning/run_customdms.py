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

import SOSS.dms.soss_cosmicrays as soss_cosmicrays

import SOSS.dms.soss_boxextract as soss_boxextract

from SOSS.dms import oneoverf_step

from SOSS.dms import soss_background

import sys

import os

import socket


hostname = socket.gethostname()
if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
    CALIBRATION_DIR = '/Users/albert/NIRISS/Commissioning/analysis/pipelineprep/calibrations/'
    ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/SOSSwavecal/ref_files/'
elif hostname == 'genesis':
    CALIBRATION_DIR = '/genesis/jwst/jwst-ref-soss/noise_files/'
    ATOCAREF_DIR = '/genesis/jwst/userland-soss/loic_review/commissioning/ref_files/'
else:
    print('Add your local computer name in the list.')
    sys.exit()

CRDSDIR = '/Users/albert/NIRISS/CRDS_CACHE/references/jwst/niriss/'

FLAT = 'jwst_niriss_flat_0190.fits'
SUPERBIAS = 'jwst_niriss_superbias_0181.fits'
DARK = 'jwst_niriss_dark_0171.fits'
BADPIX = 'jwst_niriss_mask_0015.fits'
#BACKGROUND = 'jwst_niriss_background_custom.fits'
BACKGROUND = 'model_background256.fits'  # Nestor's
SPECTRACE = 'SOSS_ref_trace_table_SUBSTRIP256.fits'
WAVEMAP = 'SOSS_ref_2D_wave_SUBSTRIP256.fits'
SPECPROFILE = 'SOSS_ref_2D_profile_SUBSTRIP256.fits'
PHOTOM = 'jwst_niriss_photom_rev2.fits' #'jwst_niriss_photom_new.fits'


def custom_loic(exposurelist, use_atoca=False, optimal_extraction=False,
                run_outliers=True, contamination_mask=None, extract_only=False,
                skip_stacking=False, erase_clean=False, satmap=None,
                use_cds=False, box_width=25, cont_params=None):




    # Correct the 1/f noise at the full time-series level rather than
    # segment by segment (because 1/f residuals on the stack differ
    # between segments). That means that the DQinit and Saturation steps
    # need to occur also at the time-series wide level.

    nsegments = np.size(exposurelist)

    # Parse the exposurelist (uncal fits files list) to determine 3 important strings:
    # outdir - the full path where all files are
    # tso_basename - the string common to all segments
    # segment_basename_list - the list of each segment's basename common to all steps for that segment
    segment_basename_list = []
    lastsavedstep_list = []
    for segment in range(nsegments):
        currentsegmentname = exposurelist[segment]
        outdir = os.path.dirname(currentsegmentname)
        tmp = os.path.basename(os.path.splitext(currentsegmentname)[0])
        tmp = tmp.split('_nis')[0] + '_nis'
        tso_basename = tmp.split('-seg')[0]
        segment_basename_list.append(tmp)
        lastsavedstep_list.append(exposurelist[segment])

    # ==================================================================================================================
    for segment in range(nsegments):

        # Read in the uncal files (to make sure that the data models 'result'
        # exists in case groupstep is skipped
        result = datamodels.open(lastsavedstep_list[segment])

        # Check if a customized SPECTRACE was already generated for this data set
        # Use it. Otherwise, use one on the ATOCAREF_DIR
        subarray = result.meta.subarray.name
        if os.path.exists(outdir+'SOSS_ref_trace_table_'+subarray+'.fits'):
            spec_trace_ref_name = outdir+'SOSS_ref_trace_table_'+subarray+'.fits'
        else:
            spec_trace_ref_name = ATOCAREF_DIR+'SOSS_ref_trace_table_'+subarray+'.fits'

        # Generate a mask for the identified contaminants in the image.
        if cont_params is None and contamination_mask is None:
            contamination_mask = None
        if cont_params is not None:
            print('A list of order 0,1,2 contaminating sources are passed. Generate a mask.')
            print(result.meta.subarray.name)
            ncont = np.shape(cont_params)[0]
            # First contaminant initializes the image
            contamination_mask_fromlist = commutils.build_mask_contamination(
                cont_params[0][0], cont_params[0][1], cont_params[0][2],
                subarray=result.meta.subarray.name)
            for i in range(ncont - 1):
                print(i, ncont)
                contamination_mask_fromlist += commutils.build_mask_contamination(
                    cont_params[i + 1][0], cont_params[i + 1][1], cont_params[i + 1][2],
                subarray=result.meta.subarray.name)
            if contamination_mask is not None:
                contmask = fits.getdata(contamination_mask)
                contmask = contmask * contamination_mask_fromlist
            else:
                contmask = np.copy(contamination_mask_fromlist)
            # The mask should have NaNs where things need to be masked, 1 elsewhere.
            contmask[contmask == 1] = np.nan
            contmask[contmask == 0] = 1
            # Save the new contamination mask (union of passed mask + listed contaminants)
            outdir = os.path.dirname(exposurelist[0])
            contamination_mask = outdir + '/contamination_mask_unified.fits'
            hdu = fits.PrimaryHDU(contmask)
            hdu.writeto(contamination_mask, overwrite=True)

        # --------------------------------------------------------------------------------------------------------------
        save_groupstep = False
        if groupstep == True:
            # DMS standard - GroupScaleStep
            #result = calwebb_detector1.group_scale_step.GroupScaleStep.call(
            #    segmentname, output_dir=outdir, save_results=False)
            result = calwebb_detector1.group_scale_step.GroupScaleStep.call(
                result, output_dir=outdir, save_results=save_groupstep)
            if save_groupstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        else:
            print('groupstep = False, step skipped')

        # --------------------------------------------------------------------------------------------------------------
        save_dqinit = False
        if dqinitstep == True:
            # DMS standard - DQInitStep
            result = calwebb_detector1.dq_init_step.DQInitStep.call(
                result, output_dir=outdir, save_results=save_dqinit)
            if save_dqinit:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        else:
            print('dqinitstep = False, step skipped')

        # --------------------------------------------------------------------------------------------------------------
        save_saturationstep = True
        if saturationstep == True:
            # DMS standard - SaturationStep
            if satmap == None:
                result = calwebb_detector1.saturation_step.SaturationStep.call(
                    result, output_dir=outdir, save_results=save_saturationstep)
            else:
                satmap_filename = 'custom_saturation_' + str(satmap) + 'e.fits'
                result = calwebb_detector1.saturation_step.SaturationStep.call(
                    result, output_dir=outdir, save_results=save_saturationstep,
                    override_saturation=CALIBRATION_DIR + satmap_filename)

            if save_saturationstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        else:
            print('saturationstep = False, step skipped')

    # TODO: remove the next 6 lines once debug of SIMP0136 data set is complete
    lastsavedstep_list = [
        '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01209_001/loic_processing/jw01209001001_03101_00001-seg001_nis_saturationstep.fits',
        '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01209_001/loic_processing/jw01209001001_03101_00001-seg002_nis_saturationstep.fits',
        '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01209_001/loic_processing/jw01209001001_03101_00001-seg003_nis_saturationstep.fits',
        '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01209_001/loic_processing/jw01209001001_03101_00001-seg004_nis_saturationstep.fits'
    ]
    # ==================================================================================================================
    if oofdeepstackstep == True:
        # Custom - Proceed with construction of the deep stack for each group using
        # all segments available.
        oof_stackname  = soss_oneoverf.stack_multisegments(lastsavedstep_list, outdir=outdir,
                                                           save_results=True)
        # TODO: Note that groupdq is mostly empty (only a few 3x3 NaNs squares are found in deepstack)
    else:
        print('oofdeepstackstep = False, step skipped')
        oof_stackname = outdir+'/oof_deepstack_'+tso_basename+'.fits'
        #deepstack = fits.getdata(oof_stackname)

    # ==================================================================================================================
    # Proceed back on a segment by segment basis (rather than at the whole time-series level)
    for segment in range(nsegments):
        # Read back the file on disk
        result = datamodels.open(lastsavedstep_list[segment])
        #if erase_clean == True:
        #    # Erase the previous steps not longer used files from disk
        #    os.system('rm -rf '+lastsavedstep_list[segment])
        ## Define input/output
        #basename = result.meta.filename
        #basename = os.path.basename(os.path.splitext(basename)[0])
        #basename = basename.split('_nis')[0] + '_nis'

        # --------------------------------------------------------------------------------------------------------------
        save_oofstep = True
        if oofstep == True:
            # Custom - 1/f correction
            result = soss_oneoverf.applycorrection(
                result, output_dir=outdir, save_results=save_oofstep,
                deepstack_custom_name=oof_stackname, oddevenrows=True,
                outlier_map=outdir+'/outliers_'+segment_basename_list[segment]+'.fits',
                trace_table_ref=spec_trace_ref_name)
            if save_oofstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        else:
            result = calwebb_detector1.refpix_step.RefPixStep.call(
                    result, output_dir=outdir, save_results=save_oofstep)
            if save_oofstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename

        # --------------------------------------------------------------------------------------------------------------
        save_superbiasstep = False
        if superbiasstep == True:
            # DMS standard - SuperBias subtraction
            print()
            # Skipped because the next step (dark subtraction) uses a custom made dark in which
            # the superbias is already included in. Warning, turn this back on if you revert the
            # dark subtraction to the default calibration file.
            # result = calwebb_detector1.superbias_step.SuperBiasStep.call(result, output_dir=outdir, save_results=True)#,
            # override_superbias=CALIBRATION_DIR+SUPERBIAS)
            #if save_superbiasstep:
            #    superbiasstep_list.append(outdir + '/' + result.meta.filename)
            #    lastsavedstep_list = superbiasstep_list
        else:
            print('superbiasstep = False, step skipped')

        # --------------------------------------------------------------------------------------------------------------
        save_darkstep = False
        if darkstep == True:
            # Custom - Dark + Superbias subtraction
            # The DMS dark subtraction is needed because it captures the hot pixels and their 4 neighbors
            # that otherwise can appear as uncorrected bad pixels in final products.
            # Here the custom dark includes the superbias level so we skipped the superbias step.
            # TODO: improve the current dark calibration file by taking KTC noise into account.
            custom_darkname = 'jwst_niriss_dark_loiccustom.fits'
            if result.meta.subarray.name == 'SUBSTRIP96':
                custom_darkname = 'jwst_niriss_dark_loiccustom_substrip96.fits'
            result = calwebb_detector1.dark_current_step.DarkCurrentStep.call(
                result, output_dir=outdir, save_results=save_darkstep,
                override_dark=CALIBRATION_DIR+'/'+custom_darkname)
            if save_darkstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        else:
            print('darkstep = False, step skipped')

        # --------------------------------------------------------------------------------------------------------------
        save_nonlinearitystep = True
        if nonlinearitystep == True:
            # DMS standard - Non-linearity correction
            result = calwebb_detector1.linearity_step.LinearityStep.call(
                # For the A0 TSO for Etienne's BFE, turn save_results to True
                # result, output_dir=outdir, save_results=False)
                result, output_dir=outdir, save_results=save_nonlinearitystep)
            #if segment == 0: fn.write('{:} - After linearity \n'.format(result.meta.filename))
            if save_nonlinearitystep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        else:
            print('nonlinearitystep = False, step skipped')

    # ==================================================================================================================
    # Break out of the segment-by-segment to perform the cosmic rays detection/correction step
    if cosmicraystep == True:
        soss_cosmicrays.cosmicrays_step(lastsavedstep_list, outdir=outdir, save_results=True)
        wildcard = outdir + '/' + tso_basename + '*_cosmicraystep.fits'
        cosmicraystep_list = sorted(glob.glob(wildcard))
        lastsavedstep_list = cosmicraystep_list
    else:
        print('cosmicstep = False, step skipped')


    # ==================================================================================================================
    # Proceed back on a segment by segment basis (rather than at the whole time-series level)
    for segment in range(nsegments):
        # Read back the file on disk
        result = datamodels.open(lastsavedstep_list[segment])

        # --------------------------------------------------------------------------------------------------------------
        save_jumpstep = False
        if jumpstep == True:
            # DMS standard - Jump detection
            result = calwebb_detector1.jump_step.JumpStep.call(
                result, output_dir=outdir, rejection_threshold=6, save_results=save_jumpstep)
            if save_jumpstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        else:
            print('jumpstep = False, step skipped')
        print('after jumpstep',lastsavedstep_list)


        # --------------------------------------------------------------------------------------------------------------
        save_slopefitstep = False
        if use_cds == True:
            # TODO: test and debug this CDS option
            print('Instead of fitting a slope to the ramp, use last read - superbias.')
            stackresult, result = commutils.cds()
            if save_slopefitstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        elif rampfitstep == True:
            # DMS standard - Ramp fitting
            stackresult, result = calwebb_detector1.ramp_fit_step.RampFitStep.call(
                result, output_dir=outdir, save_results=save_slopefitstep)
            if save_slopefitstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        else:
            print('should either have use_cds True or rampfitstep = True. Abort.')
            sys.exit()
        print('after slopefitstep',lastsavedstep_list)


        # --------------------------------------------------------------------------------------------------------------
        save_gainstep = False
        if gainstep == True:
            # DMS standard - Gain step - only affect the ERR extension to properly handle photon noise statistics
            result = calwebb_detector1.gain_scale_step.GainScaleStep.call(
                result, output_dir=outdir, save_results=save_gainstep)
            if save_gainstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        print('after gainstep', lastsavedstep_list)

        # --------------------------------------------------------------------------------------------------------------
        save_flagbadpix = False
        if flagbadpix == True:
            # Custom - Flag bad pixels found manually
            # Add some bad pixels missed by the dq init stage but seen otherwise
            result = commutils.add_manual_badpix(result)
            hdu = fits.PrimaryHDU(result.dq)
            hdu.writeto(outdir+'/dq_postmanual.fits', overwrite=True)
            if save_flagbadpix:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        print('after flagbadpixstep',lastsavedstep_list)

        # --------------------------------------------------------------------------------------------------------------
        # DMS standard - Save rateints on disk to end Stage 1
        result.meta.filetype = 'countrate'
        rateints_filename = outdir+'/'+segment_basename_list[segment]+'_rateints.fits'
        result.write(rateints_filename)
        lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        print(lastsavedstep_list)

        # STAGE 2 starts here ------

        # Read in the rateints file here to make sure that the 'result'
        # datamodel exists
        result = datamodels.open(lastsavedstep_list[segment])

        # --------------------------------------------------------------------------------------------------------------
        save_flatfieldstep = True
        if flatfieldstep == True:
            # DMS standard - Flat fielding
            #result = calwebb_spec2.flat_field_step.FlatFieldStep.call(
            #    rateints_filename, output_dir=outdir, save_results=False)
            #    #override_flat=CALIBRATION_DIR+FLAT
            result = calwebb_spec2.flat_field_step.FlatFieldStep.call(
                result, output_dir=outdir, save_results=save_flatfieldstep)
                #override_flat=CALIBRATION_DIR+FLAT
            if save_flatfieldstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        print('after flatfieldstep', lastsavedstep_list)

        # --------------------------------------------------------------------------------------------------------------
        save_outlierstep = True
        if outlierstep == True:
            # Custom - Outlier flagging
            result = soss_outliers.flag_outliers(
                result, window_size=(5, 5), n_sig=3, verbose=True, outdir=outdir,
                kernel_enlarge='5x5', save_diagnostic=~erase_clean, save_results=save_outlierstep)
            if save_outlierstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        else:
            print('outlierstep = False, step skipped')
        print('after outlierstep',lastsavedstep_list)

    # ==================================================================================================================
    # Break the segment by segment to use all segments for background construction
    if stackbackgroundstep == True:
        # Custom - Background construction (needs to whole TSO, not individual segments)
        # This stpe does not update nor save the individual segments. It creates new fits files. So the
        # segment list is unaffected.
        bgd_stack, bgd_rms = commutils.stack_ramp_multisegments(lastsavedstep_list, outdir=outdir,
                                                           save_results=True)
    print('after stackbackgroundstep', lastsavedstep_list)

    # ==================================================================================================================
    # Proceed back on a segment by segment basis (rather than at the whole time-series level)
    for segment in range(nsegments):

        result = datamodels.open(lastsavedstep_list[segment])

        # --------------------------------------------------------------------------------------------------------------
        save_backsubstep = True
        if backgroundstep == True:
            result = commutils.background_subtraction(
                result, use_whole_exposure=True, whole_exposure_stack=bgd_stack,
                aphalfwidth=[40, 20, 20], outdir=outdir,
                verbose=False, save_results=save_backsubstep, contamination_mask=contamination_mask,
                trace_table_ref=spec_trace_ref_name)
            if save_backsubstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
        else:
            print('backgroundstep = False, step skipped.')
            ## Run the step to make sure to generate _backsubstep.fits but really do nothing
            ## by using skip_background=True
            #result = commutils.background_subtraction(
            #    result, use_whole_exposure=True, whole_exposure_stack=bgd_stack,
            #    aphalfwidth=[40, 20, 20], outdir=outdir,
            #    verbose=False, save_results=~erase_clean, contamination_mask=contamination_mask,
            #    trace_table_ref=spec_trace_ref_name,
            #    skip_background=True)
        print('after backgroundsubstep', lastsavedstep_list)
    # ==================================================================================================================
    # Whole exposure deep stack of background subtracted TSO - this will be used for
    # bad pixels interpolation.

    bgdsub_stack, bgdsub_rms = commutils.stack_ramp_multisegments(lastsavedstep_list, outdir=outdir,
                                                           save_results=True)
    print('after stacking of background subtracted segments', lastsavedstep_list)

    # ==================================================================================================================
    #outdir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01201_101/loic_processing'
    #tso_basename = 'jw01201101001_04101_00001'
    #lastsavedstep_list = [outdir+'/'+'jw01201101001_04101_00001-seg001_nis_backsubstep.fits',
    #                      outdir+'/'+'jw01201101001_04101_00001-seg002_nis_backsubstep.fits']
    if badpixinterpolationstep == True:
        commutils.soss_correct_badpix(lastsavedstep_list, outdir)
        wildcard = outdir + '/' + tso_basename + '*_badpixinterp.fits'
        lastsavedstep_list = sorted(glob.glob(wildcard))
    else:
        print('badpixinterpolationstep = False, step skipped')

    # ==================================================================================================================
    for segment in range(nsegments):

        result = datamodels.open(lastsavedstep_list[segment])
        # TODO: not sure why we have this line here
        result.meta.filename = np.copy(segment_basename_list[segment])

        if False:
            # --------------------------------------------------------------------------------------------------------------
            save_badpixinterp = True
            if badpixinterpolationstep == True:

                badpix_method = 'neighbors_interpolation'

                # Custom - Bad pixel interpolation
                # Clean the outlier and bad pixels based on a deep stack
                #hdu = fits.PrimaryHDU(result.data)
                #hdu.writeto(outdir+'/prestack_data.fits', overwrite=True)
                #hdu = fits.PrimaryHDU(result.dq)
                #hdu.writeto(outdir+'/prestack_dq.fits', overwrite=True)
                # ici DQ est bon 1362,139 = 1
                if badpix_method == 'stack_interpolation':
                    print('Pixel interpolation method: stack_interpolation')
                    result = commutils.soss_interp_badpix(result, outdir, save_results=save_badpixinterp,
                                                          use_whole_stack=True,
                                                          whole_exposure_stack=bgdsub_stack,
                                                          whole_exposure_stackrms=bgdsub_rms)
                    if save_badpixinterp:
                        lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
                elif badpix_method == 'neighbors_interpolation':
                    print('Pixel interpolation method: neighbors_interpolation')
                    result = commutils.soss_interp_badpix_8neighbors(result, outdir, save_results=save_badpixinterp)
                    if save_badpixinterp:
                        lastsavedstep_list[segment] = outdir + '/' + result.meta.filename
                else:
                    print('Choose bad pixel method!!!')
                    sys.exit()
                print('after badpixinterpstep', lastsavedstep_list)

    # ==================================================================================================================
    # Break the segment by segment to use all segments for building the trace position ref file
    if makespectraceref == True:
        # build a full TSO cube
        # Custom - (needs to whole TSO, not individual segments)
        clean_stack, clean_rms = commutils.stack_ramp_multisegments(lastsavedstep_list, outdir=outdir,
                                                           save_results=True)
        # extract spec trace position
        spec_trace_ref_name = commutils.soss_spectrace_reffile_maker(clean_stack, outdir=outdir,
                                                                     mask_params=cont_params,
                                                                     verbose=False)
    else:
        spec_trace_ref_name = spec_trace_ref_name
    print('The trace table reference file is '+spec_trace_ref_name)

    # Check the two versions of the trace position and wavelength: pastasoss and the ATOCA needed spec_trace_table
    pupilwheelposition = result.meta.instrument.pupil_position
    commutils.compare_tracetable_pastasoss(pupilwheelposition, spec_trace_ref_name, outdir)

    # ==================================================================================================================
    for segment in range(nsegments):

        result = datamodels.open(lastsavedstep_list[segment])
        # TODO: do we need this?
        result.meta.filename = str(segment_basename_list[segment])

        # --------------------------------------------------------------------------------------------------------------
        save_removenanstep = True
        if removenanstep == True:
            # Custom - Remove NaNs
            # Check that no NaNs is in the data before sending to extraction
            # (atoca can't handle Nans)
            result = commutils.remove_nans(result, outdir=outdir, save_results=save_removenanstep)
            result.meta.filename = np.copy(segment_basename_list[segment])
            print(segment_basename_list[segment])
            print(result.meta.filename)
            if save_removenanstep:
                lastsavedstep_list[segment] = outdir + '/' + result.meta.filename

        if extractstep == True:
            # Spectrum extraction ----------------------------
            if use_atoca:
                result = calwebb_spec2.extract_1d_step.Extract1dStep.call(result, output_dir=outdir, save_results=True,
                                                                          soss_transform=[0, 0, 0],
                                                                          soss_atoca = True,
                                                                          #soss_transform=[None, 0, None],
                                                                          subtract_background=False,
                                                                          soss_bad_pix='model',
                                                                          soss_width=box_width,
                                                                          #soss_tikfac=3.38e-15,
                                                                          soss_modelname=outdir+'/'+segment_basename_list[segment]+'_atoca_model.fits',
                                                                          override_spectrace=spec_trace_ref_name,
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
                                                                              soss_width=box_width,
                                                                              # soss_tikfac=3.38e-15,
                                                                              soss_modelname=None,
                                                                              override_spectrace=spec_trace_ref_name,
                                                                              override_wavemap=ATOCAREF_DIR + WAVEMAP,
                                                                              override_specprofile=ATOCAREF_DIR + SPECPROFILE)

                else:
                    # soss_atoca=False --> box extraction only
                    # carefull to not turn it on. Would if soss_bad_pix='model' or soss_modelname=set_to_something
                    print('Nom du datamodel filename avant extract1d = {:}'.format(result.meta.filename))
                    result = calwebb_spec2.extract_1d_step.Extract1dStep.call(result, output_dir=outdir, save_results=True,
                                                                              soss_transform=[0, 0, 0],
                                                                              soss_atoca=False,
                                                                              subtract_background=False,
                                                                              soss_bad_pix='masking',
                                                                              #soss_width=25,
                                                                              soss_width=box_width,
                                                                              # soss_tikfac=3.38e-15,
                                                                              soss_modelname=None,
                                                                              override_spectrace=spec_trace_ref_name,
                                                                              override_wavemap=ATOCAREF_DIR + WAVEMAP,
                                                                              override_specprofile=ATOCAREF_DIR + SPECPROFILE)


        if correctwavelengthstep == True:
            # Use the pastasoss to get a better wavelength calibration and update the extracted spectra
            result = commutils.correct_wavelengthsolution(result, outdir=outdir, save_results=True)

        if photomstep == True:

            # DMS standard - Conversion to SI units
            print('Nom du datamodel filename avant photom step= {:}'.format(result.meta.filename))
            result = calwebb_spec2.photom_step.PhotomStep.call(
                result, output_dir=outdir, save_results=True,
                override_photom= CRDSDIR + PHOTOM)



        # Write results on disk
        result.close()
        #sys.exit()

    return










if __name__ == "__main__":
    ################ MAIN ###############

    # data set to process:
    #datasetname = 'LTT9779'
    #datasetname = 'SOSSwavecal'
    #datasetname = 'SOSSfluxcal'
    #datasetname = 'SOSSfluxcalss96ng3'
    datasetname = 'HATP14b'
    #datasetname = 'darks'
    #datasetname = 'f277w'
    #datasetname = 'WASP52b'
    #datasetname = 'WASP107b'
    #datasetname = '01201101'
    #datasetname = '01201102'
    #datasetname = '01201103'
    #datasetname = '01201104'
    #datasetname = '01201105'
    #datasetname = '02589001'
    #datasetname = '02589002'
    #datasetname = '02589003'
    #datasetname = '02589004'
    #datasetname = 'thermalinstability' # aka K2-18b
    #datasetname = 'L9859d'
    #datasetname = 'WASP80b'
    #datasetname = 'HATP18b'
    #datasetname = 'LHS1140b'
    #datasetname = 'LHS1140b_2'
    datasetname = 'SIMP0136'
    #datasetname = 'salma'

    #custom_or_not = '_rateints'
    satmap = None  # 35000 # None
    use_cds = False
    extract_only = False
    use_atoca = False
    optimal_extraction = False
    skip_stacking = False
    erase_clean = False
    box_width = 40
    postproc_only = False
    direct_to_background = False
    direct_to_reffile = False


    # Default flow
    groupstep = True
    dqinitstep = True
    saturationstep = True
    oofdeepstackstep = True
    erase_clean = False
    oofstep = True
    superbiasstep = False
    darkstep = True
    nonlinearitystep = True
    cosmicraystep = True
    jumpstep = False # because cosmicraystep is True
    use_cds = False
    rampfitstep = True
    gainstep = True
    flagbadpix = True
    flatfieldstep = True
    outlierstep = False # because cosmicraystep is True
    stackbackgroundstep = True
    backgroundstep = True
    badpixinterpolationstep = True
    makespectraceref = True
    removenanstep = False # not needed with soss_correct_badpix removes them all
    extractstep = True
    correctwavelengthstep = True
    photomstep = True
    postprocstep = True



    if extract_only == True:
        groupstep = False
        dqinitstep = False
        saturationstep = False
        oofdeepstackstep = False
        oofstep = False
        darkstep = False
        nonlinearitystep = False
        cosmicraystep = False
        jumpstep = False
        rampfitstep = False
        gainstep = False
        flagbadpix = False
        flatfieldstep = False
        outlierstep = False
        stackbackgroundstep = False
        backgroundstep = False
        badpixinterpolationstep = False
        makespectraceref = False
        removenanstep = False

    if skip_stacking == True:
        oofdeepstackstep = False

    if postproc_only == True:
        groupstep = False
        dqinitstep = False
        saturationstep = False
        oofdeepstackstep = False
        oofstep = False
        darkstep = False
        nonlinearitystep = False
        cosmicraystep = False
        jumpstep = False
        rampfitstep = False
        gainstep = False
        flagbadpix = False
        flatfieldstep = False
        outlierstep = False
        stackbackgroundstep = False
        backgroundstep = False
        badpixinterpolationstep = False
        makespectraceref = False
        removenanstep = False
        extractstep = False
        photomstep = False
        postprocstep = True

    if direct_to_background == True:
        groupstep = False
        dqinitstep = False
        saturationstep = False
        oofdeepstackstep = False
        oofstep = False
        darkstep = False
        nonlinearitystep = False
        cosmicraystep = False
        jumpstep = False
        rampfitstep = False
        gainstep = False
        flagbadpix = False
        flatfieldstep = False
        outlierstep = False

    if direct_to_reffile == True:
        groupstep = False
        dqinitstep = False
        saturationstep = False
        oofdeepstackstep = False
        oofstep = False
        darkstep = False
        nonlinearitystep = False
        cosmicraystep = False
        jumpstep = False
        rampfitstep = False
        gainstep = False
        flagbadpix = False
        flatfieldstep = False
        outlierstep = False
        stackbackgroundstep = False
        backgroundstep = False




    # initialize for the default behavior
    cont_params = None

    # Wavelength calibration
    if datasetname == 'wavecal':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local') :
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
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
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
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
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
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01541_001/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/HATP14b/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01541001001_04101_00001-seg001_nis',
            'jw01541001001_04101_00001-seg002_nis',
            'jw01541001001_04101_00001-seg003_nis',
            'jw01541001001_04101_00001-seg004_nis'
        ]
        dataset_string = 'jw01541001001_04101_00001'

    # T1
    if datasetname == '02589001':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/02589_001/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = '/Users/albert/NIRISS/Commissioning/analysis/T1/mask_contamination.fits'
            contmask = None
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
    if datasetname == '02589002':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/02589_002/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
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
    if datasetname == '02589003':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/02589_003/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_3/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw02589003001_04101_00001-seg001_nis',
            'jw02589003001_04101_00001-seg002_nis',
            'jw02589003001_04101_00001-seg003_nis'
        ]
        dataset_string = 'jw02589003001_04101_00001'

    # T1_4
    if datasetname == '02589004':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/02589_004/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw02589004001_04101_00001-seg001_nis',
            'jw02589004001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw02589004001_04101_00001'

    if datasetname == '01201101':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01201_101/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            #dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201101001_04101_00001-seg001_nis',
            'jw01201101001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw01201101001_04101_00001'

    if datasetname == '01201102':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01201_102/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            #dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201102001_04101_00001-seg001_nis',
            'jw01201102001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw01201102001_04101_00001'

    if datasetname == '01201103':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01201_103/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            #dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201103001_04101_00001-seg001_nis',
            'jw01201103001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw01201103001_04101_00001'
        # contamination
        #cont_params = [[0,766,130],[0,863,149],[0,1244,214],[0,1288,185],
        #               [0,1483,82],[0,1415,27],[0,1760,94],[0,1024,167],
        #               [0,1128,148],[0,972,29]]


    if datasetname == '01201104':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01201_104/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            #dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201104001_04101_00001-seg001_nis',
            'jw01201104001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw01201104001_04101_00001'

    if datasetname == '01201105':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01201_105/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            #dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201105001_04101_00001-seg001_nis',
            'jw01201105001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw01201105001_04101_00001'

    # LTT9779 - phase curve
    if datasetname == 'LTT9779':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
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

    # dark
    if datasetname == 'darks':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/darks/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/SOSSfluxcal/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/darks/'
            contmask = None
        else:
            sys.exit()

        datalist = ['dark-seg001_nis']
        dataset_string = 'dark'

    # F277W
    if datasetname == 'f277w':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/Commissioning/analysis/f277w/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/HATP14b/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/darks/'
            contmask = None
        else:
            sys.exit()

        datalist = ['jw01541001001_04102_00001-seg001_nis']
        dataset_string = 'jw01541001001_04102_00001'

    # WASP52b
    if datasetname == 'WASP52b':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            dir = '/Volumes/T7/WASP52b/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201501001_04101_00001-seg001_nis',
            'jw01201501001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw01201501001_04101_00001'

    # WASP107b
    if datasetname == 'WASP107b':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            dir = '/Volumes/T7/WASP107b/'
            dir = '/Users/albert/NIRISS/Commissioning/analysis/WASP107b/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201008001_04101_00001-seg001_nis',
            'jw01201008001_04101_00001-seg002_nis',
            'jw01201008001_04101_00001-seg003_nis',
            'jw01201008001_04101_00001-seg004_nis'
        ]
        dataset_string = 'jw01201008001_04101_00001'

    # K2-18b Thermal instability analysis
    if datasetname == 'thermalinstability':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1b_1/ref_files/'
            dir = '/Volumes/T7/thermalinstability/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw02722003001_04101_00001-seg001_nis',
            'jw02722003001_04101_00001-seg002_nis',
            'jw02722003001_04101_00001-seg003_nis',
            'jw02722003001_04101_00001-seg004_nis'
        ]
        dataset_string = 'jw02722003001_04101_00001'

    # L98-59d
    if datasetname == 'L9859d':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1b_1/ref_files/'
            dir = '/Volumes/T7/L9859d/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201311001_04101_00001-seg001_nis']
        dataset_string = 'jw01201311001_04101_00001'

    # WASP80b
    if datasetname == 'WASP80b':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            dir = '/Volumes/T7/WASP80b/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201007001_04101_00001-seg001_nis',
            'jw01201007001_04101_00001-seg002_nis',
            'jw01201007001_04101_00001-seg003_nis',
            'jw01201007001_04101_00001-seg004_nis'
        ]
        dataset_string = 'jw01201007001_04101_00001'
        # contamination
        cont_params = [[0,1730,133],[0,1830,183],[0,1472,135],[0,1333,184],[0,1153,70],
                       [1,40,178]]

    if datasetname == 'HATP18b':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            dir = '/Volumes/T7/HATP18b/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw02734001001_04101_00001-seg001_nis',
            'jw02734001001_04101_00001-seg002_nis',
            'jw02734001001_04101_00001-seg003_nis',
            'jw02734001001_04101_00001-seg004_nis'
        ]
        dataset_string = 'jw02734001001_04101_00001'

    if datasetname == 'LHS1140b':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            dir = '/Volumes/T7/LHS1140b/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw06543001001_04101_00001-seg001_nis',
            'jw06543001001_04101_00001-seg002_nis',
            'jw06543001001_04101_00001-seg003_nis',
            'jw06543001001_04101_00001-seg004_nis',
            'jw06543001001_04101_00001-seg005_nis'
        ]
        dataset_string = 'jw06543001001_04101_00001'

    if datasetname == 'LHS1140b_2':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            dir = '/Volumes/T7/LHS1140b_2/'
            contmask = None
        elif hostname == 'genesis':
            dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw06543002001_04101_00001-seg001_nis',
            'jw06543002001_04101_00001-seg002_nis',
            'jw06543002001_04101_00001-seg003_nis',
            'jw06543002001_04101_00001-seg004_nis',
            'jw06543002001_04101_00001-seg005_nis'
        ]
        dataset_string = 'jw06543002001_04101_00001'

    if datasetname == 'SIMP0136':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01209_001/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            #dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        # SPECIAL SPECIAL SPECIAL SPECIAL!!!
        # the official uncal has no segments for this FULL TSO. But processing it made
        # my laptop memory crash. I wrote a split_tso script to split the TSO into segments
        # which use less memory at a time.
        commutils.split_tso(dir+'jw01209001001_03101_00001_nis_uncal.fits', nints_fullmax=21)

        datalist = [
            'jw01209001001_03101_00001-seg001_nis',
            'jw01209001001_03101_00001-seg002_nis',
            'jw01209001001_03101_00001-seg003_nis',
            'jw01209001001_03101_00001-seg004_nis'
        ]
        dataset_string = 'jw01209001001_03101_00001'

        # debugging SIMP0136
        groupstep = False
        dqinitstep = False
        saturationstep = False
        oofdeepstackstep = False


    if datasetname == 'salma':
        if (hostname == 'havelock.sf.umontreal.ca') or (hostname == 'havelock.local'):
            dir = '/Users/albert/NIRISS/sossisse/sossisse/sossiopath/JWST.NIRISS.SOSS/01201_101/loic_processing/'
            ATOCAREF_DIR = '/Users/albert/NIRISS/Commissioning/analysis/T1_3/ref_files/'
            contmask = None
        elif hostname == 'genesis':
            #dir = '/genesis/jwst/userland-soss/loic_review/Commissioning/T1_2/'
            contmask = None
        else:
            sys.exit()

        datalist = [
            'jw01201101001_04101_00001-seg001_nis',
            'jw01201101001_04101_00001-seg002_nis'
        ]
        dataset_string = 'jw01201101001_04101_00001'

        oofstep = False


    '''
    RUN THE PIPELINE--------------------------------------------------------------------
    '''

    if postproc_only == False:
        # Run the level 1 and 2 custom pipeline
        uncal_list = []
        for oneseg in datalist: uncal_list.append(dir + oneseg + '_uncal.fits')
        custom_loic(uncal_list, use_atoca=use_atoca,
                    optimal_extraction=optimal_extraction, contamination_mask=contmask,
                    extract_only=extract_only, skip_stacking=skip_stacking,
                    erase_clean=erase_clean, satmap=satmap, use_cds=use_cds, box_width=box_width,
                    cont_params=cont_params)

    if postprocstep == True:
        # Post processing analysis

        if False:
            extract1d_list = []
            for oneseg in datalist:
                # Additional diagnostics - Subtracting the ATOCA model from the images
                if use_atoca:
                    commutils.check_atoca_residuals(dir+oneseg+'_rateints.fits', dir+oneseg+'_atoca_model_SossExtractModel.fits')
                spectrum_file = dir+oneseg+'_extract1dstep.fits'
                extract1d_list.append(spectrum_file)
                a = commutils.plot_timeseries(spectrum_file, norder=3)

        outdir = dir
        if True:
            wildcard = outdir+dataset_string+'-seg???_nis_extract1dstep.fits'
            a = commutils.combine_multi_spec(wildcard, outdir+'extracted_spectrum_boxsize{:2.0f}.fits'.format(box_width))
            # produce the median and deviation spectrum
            a = commutils.median_absolute_spectrum(outdir+'extracted_spectrum_boxsize{:2.0f}.fits'.format(box_width),
                                               outdir+'extracted_oot_spectrum_boxsize{:2.0f}.fits'.format(box_width))
        if True:
            a = commutils.plot_timeseries(outdir+'extracted_spectrum_boxsize{:2.0f}.fits'.format(box_width),
                                          outdir = dir, norder=3)
            a = commutils.greyscale_rms(outdir+'timeseries_greyscale_extracted_spectrum_boxsize{:2.0f}_normalizedflux.fits'.format(box_width), title='From whole TSO')

        if True:
            # Combining flux calibrated segments and preparing a median spectrum
            outdir = '/Users/albert/NIRISS/Commissioning/analysis/'+datasetname+'/'
            outdir = dir
            wildcard = outdir+dataset_string+'-seg???_nis_photomstep.fits'
            a = commutils.combine_multi_spec(wildcard, outdir+'calibrated_spectrum_boxsize{:2.0f}.fits'.format(box_width))
            # produce the median and deviation spectrum
            a = commutils.median_absolute_spectrum(outdir+'calibrated_spectrum_boxsize{:2.0f}.fits'.format(box_width),
                                                   outdir+'calibrated_oot_spectrum_boxsize{:2.0f}.fits'.format(box_width))
