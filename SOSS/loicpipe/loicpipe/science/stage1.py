#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 1 wrapper functionality

This stage applies detector-level corrections to raw non-destructively
read "ramps" in order to produce count rate images per exposure, or per
integration for some modes.

Input: raw ramps for all integrations
Output: uncalibrated slope images for all integrations and exposures

Created on 2023-07-11

@author: cook
"""
import os
from typing import Any

from jwst import datamodels
from jwst.pipeline import calwebb_detector1
from jwst.pipeline import calwebb_spec2

from loicpipe.core import constants
from loicpipe.core import io

import SOSS.dms.soss_oneoverf as soss_oneoverf
import SOSS.commissioning.comm_utils as commutils

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'science.stage1.py'
# get Parameters class
Parameters = constants.Parameters

# =============================================================================
# Define functions
# =============================================================================
def main(params: Parameters) -> Parameters:
    # ----------------------------------------------------------------------
    # get a list of uncalibrated files and push it into data.uncal_list
    params = io.get_uncal_files(params)
    # ----------------------------------------------------------------------
    # run loicpipe stage 1
    if params['stage1.loicpipe.run']:
        params = stage1_loicpipe(params)
    # ----------------------------------------------------------------------
    return params


def stage1_loicpipe(params: Parameters) -> Parameters:
    # set func name
    funcname = f'{__NAME__}.stage1_loicpipe()'
    # get parameters used in this function (should be done at the start)
    uncal_list = params['data.uncal_list']
    # whether to skip stacking
    skip_stack = params['stage1.loicpipe.skip_stack']
    # get the saturation map file
    satmap = params['stage1.loicpipe.satmap']
    # erase clean saturation files
    erase_clean = params['stage1.loicpipe.erase_clean_sat']
    # rejection threshold for jump step
    jump_rej_thres = params['stage1.loicpipe.jump_rej_thres']
    # whether we fit the ramp (if False uses last read)
    fit_ramp = params['stage1.loicpipe.fit_ramp']
    # whether to only to extraction (no previous steps in stage 1 + 2)
    extract_only = params['stage2.loicpipe.extract_only']
    # ----------------------------------------------------------------------
    # get output directory
    params = io.get_output_directory(params)
    outdir = params['data.outdir']
    # if we have the extract only flag do not do any stage 1
    if extract_only:
        return params
    # ----------------------------------------------------------------------
    # Load or produce deepstack
    # ----------------------------------------------------------------------
    # loop around segments and run
    if skip_stack:
        # get the list of files from disk
        satlist = io.get_saturation_list_files(params)
        # display the list of saturation files
        print('Loaded saturation files:')
        for satfile in satlist:
            print('\t - ', satfile)
        # get the deepstack filename
        params = io.get_deepstack_file(params)
        # load the deepstack file
        deepstack = io.load_fits(params)
    else:
        # ----------------------------------------------------------------------
        # Run GroupScaleStep, DQInitStep, SaturationStep
        # ----------------------------------------------------------------------
        # storage for saturation file list
        satlist = []
        # loop around uncalibrated files
        for segment in uncal_list:
            # common arguments
            kwargs = dict(outputdir=outdir, save_results=False)
            # run the DMS standard - GroupScaleStep
            GroupScaleStep = calwebb_detector1.group_scale_step.GroupScaleStep
            result = GroupScaleStep.call(segment, **kwargs)
            # run the DMS standard - DQInitStep
            DQInitSte = calwebb_detector1.dq_init_step.DQInitSte
            result = DQInitSte.call(result, **kwargs)
            # common arguments
            kwargs = dict(outputdir=outdir, save_results=True,
                           overwrite_saturation=satmap)
            # run the DMS standard - SaturationStep
            SaturationStep = calwebb_detector1.saturation_step.SaturationStep
            result = SaturationStep.call(result, **kwargs)
            # get the saturation file from results3 meta data
            satfile = os.path.join(outdir, result.meta.filename)
            # append to filelist
            satlist.append(satfile)
        # ----------------------------------------------------------------------
        # Proceed with construction of the deep stack for each group using
        # all segments available.
        # ----------------------------------------------------------------------
        deepstack, _ = soss_oneoverf.stack_multisegments(satlist,
                                                         outdir=outdir,
                                                         save_results=True)
    # ----------------------------------------------------------------------
    # Proceed back on a segment by segment basis (rather than at the whole
    # time-series level)
    # ----------------------------------------------------------------------
    # storage of logs for output files
    # save filenames to filenames.txt
    filename_lines = []
    # save dq_trace to dq_trace.txt
    dq_trace_lines = []
    # loop around saturation files
    for it, satfile in enumerate(satlist):
        # Read back the file on disk
        result = datamodels.open(satfile)
        # get the filename from results.meta
        satfilename = str(result.meta.filename)
        # erase the previous steps (no longer used) from disk
        if erase_clean:
            os.remove(satfile)
        # construct the outlier map filename
        outliermap_file = io.get_outliermap_file(params, satfilename)
        # construct the trace table filename
        tracetable_file = io.get_tracetable_file(params)
        # set up kwargs
        kwargs = dict(outputdir=outdir, save_results=True,
                      deepstack_custom=deepstack, oddevenrows=True,
                      outlier_map=outliermap_file,
                      trace_table_ref=tracetable_file)
        # Custom - 1/f correction
        result = soss_oneoverf.applycorrection(result, **kwargs)
        # write to log (only for first iteration)
        if it == 0:
            filename_lines.append(f'{result.meta.filename} - After 1/f')
        # ----------------------------------------------------------------------
        # Custom Dark Current correction
        # ----------------------------------------------------------------------
        result = loic_dark_current_step(params, result)
        # write to log (only for first iteration)
        if it == 0:
            filename_lines.append(f'{result.meta.filename} '
                                  f'- After dark current')

        # ----------------------------------------------------------------------
        # DMS standard - RefPix correction
        # ----------------------------------------------------------------------
        # Remove the DMS pipeline reference pixel correction
        # Replaced by our custom 1/f correction
        # RefPixStep =  calwebb_detector1.refpix_step.RefPixStep
        # result = RefPixStep.call(result, output_dir=outdir, save_results=True)

        # ----------------------------------------------------------------------
        # DMS standard - Non-linearity correction
        # ----------------------------------------------------------------------
        LinearityStep = calwebb_detector1.linearity_step.LinearityStep
        result = LinearityStep.call(result, output_dir=outdir,
                                    save_results=False)
        # write to log (only for first iteration)
        if it == 0:
            filename_lines.append(f'{result.meta.filename} - After linearity')

        # ---------------------------------------------------------------------
        # DMS standard - Jump detection
        # ---------------------------------------------------------------------
        JumpStep = calwebb_detector1.jump_step.JumpStep
        result = JumpStep.call(result, output_dir=outdir,
                               rejection_threshold=jump_rej_thres,
                               save_results=False)
        # write to log (only for first iteration)
        if it == 0:
            filename_lines.append(f'{result.meta.filename} - After jump')
        # ---------------------------------------------------------------------
        # Ramp fitting
        # ---------------------------------------------------------------------
        if fit_ramp:
            RampFitStep = calwebb_detector1.ramp_fit_step.RampFitStep
            stackresult, result = RampFitStep.call(result, output_dir=outdir,
                                                   save_results=False)
        else:
            print('Instead of fitting a slope to the ramp, use last read '
                  '- superbias.')
            stackresult, result = commutils.cds(result, outdir=outdir)
        # write to log (only for first iteration)
        if it == 0:
            filename_lines.append(f'{result.meta.filename} '
                                  f'- After ramp fitting')

        # add dq trace
        dq_values = result.dq[params['stage1.loicpipe.dq_mask']]
        dq_trace_lines.append('DQ={0} - After ramp fit step'.format(dq_values))

        # ---------------------------------------------------------------------
        # DMS standard - Gain step
        # ---------------------------------------------------------------------
        # run the dms step
        GainScaleStep = calwebb_detector1.gain_scale_step.GainScaleStep
        result = GainScaleStep.call(result, output_dir=outdir,
                                    save_results=False)
        # add dq trace
        dq_values = result.dq[params['stage1.loicpipe.dq_mask']]
        dq_trace_lines.append('DQ={0} - After gain step'.format(dq_values))
        # write to log (only for first iteration)
        if it == 0:
            filename_lines.append(f'{result.meta.filename} - After gain scale')

        # ---------------------------------------------------------------------
        # Custom - Flag bad pixels found manually
        # ---------------------------------------------------------------------
        # Add some bad pixels missed by the dq init stage but seen otherwise
        result = commutils.add_manual_badpix(result)

        # add dq trace
        dq_values = result.dq[params['stage1.loicpipe.dq_mask']]
        dq_trace_lines.append('DQ={0} '
                              '- After manual bad pix step'.format(dq_values))
        # write to log (only for first iteration)
        if it == 0:
            filename_lines.append(f'{result.meta.filename} '
                                  f'- After manual bad pix ')

        # write to dq_postmanual.fits
        io.write_dqfile(params, result)

        # Question: This will always give the same results as last entry?
        # add dq trace
        dq_values = result.dq[params['stage1.loicpipe.dq_mask']]
        dq_trace_lines.append('DQ={0} - After saving of manual '
                              'bad pix step'.format(dq_values))
        # write to log (only for first iteration)
        if it == 0:
            filename_lines.append(f'{result.meta.filename} '
                                  f'- After saving of manual bad pix step')

        # ---------------------------------------------------------------------
        # DMS standard - Save rateints on disk to end Stage 1
        # ---------------------------------------------------------------------
        # write rateints to disk
        io.write_rateints(params, result, satfilename)

        # add dq trace
        dq_values = result.dq[params['stage1.loicpipe.dq_mask']]
        dq_trace_lines.append('DQ={0} - After saving _rateints.fits'
                              'bad pix step'.format(dq_values))
        # write to log (only for first iteration)
        if it == 0:
            filename_lines.append(f'{result.meta.filename} '
                                  f'- After saving _rateints.fits')
        # ---------------------------------------------------------------------
        # write dq log

        with open(os.path.join(outdir, 'dq_trace.txt'), 'w') as f:
            f.write('\n'.join(dq_trace_lines))
    # ----------------------------------------------------------------------
    return params



def loic_dark_current_step(params: Parameters, result0: Any) -> Any:
    # get the output directory
    outputdir = params['data.outdir']
    # get parameters used in this function (should be done at the start)
    custom_dark_dict = params['data.custom_dark']

    superbias_included = params['data.custom_dark_has_superbias_included']
    # get subarray name
    subarray = result0.meta.subarray.name

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


    # first check whether subarray is in the custom dark dictionary
    if subarray in custom_dark_dict:
        custom_dark = custom_dark_dict[subarray]
    else:
        custom_dark = custom_dark_dict['default']
    # -------------------------------------------------------------------------
    # SUPERBIAS STEP
    # -------------------------------------------------------------------------
    if not superbias_included:
        # get superbias file
        superbias_file = io.get_superbias_file(params)
        # run super bias step
        SuperBiasStep = calwebb_detector1.superbias_step.SuperBiasStep
        result1 = SuperBiasStep.call(result0, output_dir=outputdir,
                                    save_results=True,
                                    override_superbias=superbias_file)
    else:
        result1 = result0
    # -------------------------------------------------------------------------
    # DARK STEP
    # -------------------------------------------------------------------------
    # TODO: improve the current dark calibration file by taking KTC noise into account.
    # run the dark current step
    DarkCurrentStep = calwebb_detector1.dark_current_step.DarkCurrentStep
    result2 = DarkCurrentStep.call(result1, output_dir=outputdir,
                                   save_results=False,
                                   override_dark=custom_dark)
    # -------------------------------------------------------------------------
    # return dark result
    return result2





# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
