#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2 wrapper functionality

This stage applies physical corrections and calibrations to individual
exposures to produce fully calibrated (unrectified) exposures.

input: uncalibrated slope images for all integrations and exposures
output: calibrated slope images for all integrations and exposures

Created on 2023-07-11

@author: cook
"""
import os
from typing import List, Optional

from jwst import datamodels
from jwst.pipeline import calwebb_spec2

import SOSS.commissioning.comm_utils as soss_commutils
import SOSS.dms.soss_outliers as soss_outliers
from loicpipe.core import base
from loicpipe.core import constants
from loicpipe.core import io

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'science.stage2.py'
# get Parameters class
Parameters = constants.Parameters


# =============================================================================
# Define functions
# =============================================================================
def main(params: Parameters) -> Parameters:
    # ----------------------------------------------------------------------
    # run loicpipe stage 1
    if params['loicpipe.stage2.run']:
        stage2_loicpipe(params)
    # ----------------------------------------------------------------------
    return params


def stage2_loicpipe(params: Parameters) -> Parameters:
    # set func name
    func_name = f'{__NAME__}.stage2_loicpipe()'
    # whether to only to extraction (no previous steps in stage 1 + 2)
    extract_only: bool = params['loicpipe.extract_only']
    # get the dqmask
    dqmask: List[int] = params['loicpipe.dq_mask']
    # whether to run flat fielding
    flat_field: bool = params['loicpipe.stage2.flat_field']
    # erase clean rate int files
    erase_clean_rateint: bool = params['loicpipe.stage2.erase_clean_rateints']
    # whether to run custom outlier rejection
    custom_outlier: bool = params['loicpipe.stage2.custom_outlier_rejection']
    # define custom outlier window size
    custom_outlier_window: List[int] = params['loicpipe.stage2.custom_outlier_window']
    # define custom outlier n sigma for rejection
    custom_outlier_nsig: float = params['loicpipe.stage2.custom_outlier_nsig']
    # erase / clean custom outlier files
    erase_clean_custom_outlier: bool = params['loicpipe.stage2.erase_clean_custom_outlier']
    # define the aphalfwidth for background subtraction
    aphalfwidth: List[int] = params['loicpipe.stage2.aphalfwidth']
    # erase / clean background files
    erase_clean_background = params['loicpipe.stage2.erase_clean_background']
    # whether to do the background subtraction
    background_subtract: bool = params['loicpipe.stage2.background_subtract']
    # background contamination mask (None for no mask)
    contam_mask: Optional[str] = params['loicpipe.stage2.contam_mask']
    # whether to do the bad pixel interpolation
    bad_pix_interp: bool = params['loicpipe.stage2.bad_pix_interp']
    # whether to do a local background subtraction
    local_background_subtraction: bool = params['loicpipe.stage2.local_background_subtraction']
    # define the local background subtraction width
    local_sub_width: int = params['loicpipe.stage2.local_sub_width']
    # define the local background offset
    local_sub_back_offset: int = params['loicpipe.stage2.local_sub_back_offset']
    # whether to remove NaNs
    remove_nans: bool = params['loicpipe.stage2.remove_nans']
    # extraction method: atoca, optimal, box
    extraction_method: str = params['loicpipe.stage2.extraction_method']
    # extraction box width (used in all three methods)
    extraction_box_width: int = params['loicpipe.stage2.extraction_box_width']
    # extraction soss transform [three integers]
    soss_transform: List[int] = params['loicpipe.stage2.extraction_soss_transform']
    # ----------------------------------------------------------------------
    # get output directory (may not be in params if stage 1 skipped)
    params = io.get_output_directory(params)
    outdir = params['data.outdir']
    # get the list of files from disk
    satlist = io.get_saturation_list_files(params)
    # storage of logs for output files
    dqtracelog = io.LoicLog('dq_trace.txt')
    # loop around saturation files
    for it, satfile in enumerate(satlist):
        # deal with pre-extraction steps
        if not extract_only:
            # -----------------------------------------------------------------
            # write rateints to disk
            rateints_filename = io.get_rateints_file(params, satfile)
            result = datamodels.open(rateints_filename)
            # -----------------------------------------------------------------
            # DMS standard - Flat fielding
            # -----------------------------------------------------------------
            if flat_field:
                FlatFieldStep = calwebb_spec2.flat_field_step.FlatFieldStep
                result = FlatFieldStep.call(rateints_filename,
                                            output_dir=outdir,
                                            save_results=True)
                # add dq trace
                dqtracelog.write('DQ={0} - After flat fielding '
                                 'step'.format(result.dq[dqmask]))
                # erase the previous steps (no longer used) from disk
                if erase_clean_rateint:
                    os.remove(rateints_filename)

            # -----------------------------------------------------------------
            # Custom - Outlier flagging
            # -----------------------------------------------------------------
            if custom_outlier:
                # set up outlier kwargs
                outlier_kwargs = dict()
                outlier_kwargs['windows_size'] = custom_outlier_window
                outlier_kwargs['n_sig'] = custom_outlier_nsig
                outlier_kwargs['verbose'] = True
                outlier_kwargs['outdir'] = outdir
                outlier_kwargs['save_daignostic'] = not erase_clean_custom_outlier
                outlier_kwargs['save_results'] = False
                # run the outlier flagging
                result = soss_outliers.flag_outliers(result,
                                                     **outlier_kwargs)
                # add dq trace
                dqtracelog.write('DQ={0} - After outlier flagging '
                                 'step'.format(result.dq[dqmask]))

            # -----------------------------------------------------------------
            # Custom - background subtraction
            # -----------------------------------------------------------------
            if background_subtract:
                # get the trace table ref
                trace_table_ref = io.get_tracetable_file(params)
                # set up background subtraction kwargs
                back_kwargs = dict()
                back_kwargs['aphalfwidth'] = aphalfwidth
                back_kwargs['outdir'] = outdir
                back_kwargs['verbose'] = False
                back_kwargs['save_results'] = not erase_clean_background
                back_kwargs['contamination_mask'] = contam_mask
                back_kwargs['trace_table_ref'] = trace_table_ref
                # run the background subtraction
                result = soss_commutils.background_subtraction(result,
                                                               **back_kwargs)
                # add dq trace
                dqtracelog.write('DQ={0} - After background subtraction '
                                 'step'.format(result.dq[dqmask]))

            # -----------------------------------------------------------------
            # Custom - bad pixel interpolation
            # -----------------------------------------------------------------
            if bad_pix_interp:
                # Write prestack
                # Question, why do this it is overwritten by every segment?
                io.write_prestack(params, result)

                # interpret bad pixels
                result = soss_commutils.soss_interp_badpix(result,
                                                           outdir,
                                                           save_results=True)
                # add dq trace
                dqtracelog.write('DQ={0} - After bad pix interpolation '
                                 'step'.format(result.dq[dqmask]))
        # deal with just loading the result (with no pre-extraction steps done)
        else:
            # get the rate ints file after bad pixel interpolation
            rateints_filename = io.get_rateints_after_badpix_interp(params,
                                                                    satfile)
            # get the segement name
            seg_name = io.get_seg_name(satfile)
            # load the results
            result = datamodels.open(rateints_filename)
            # need to overwrite teh filename because otherwise we have the
            # '_badpixinterp' suffix present
            result.meta.filename = seg_name
            print('Filename of datamodel: {0}'.format(result.meta.filename))

        # -----------------------------------------------------------------
        # Custom - local background subtraction
        # -----------------------------------------------------------------
        if local_background_subtraction:
            # get the trace table ref
            trace_table_ref = io.get_tracetable_file(params)
            # set up background subtraction kwargs
            lback_kwargs = dict()
            lback_kwargs['trace_table_ref_file_name'] = trace_table_ref
            lback_kwargs['width'] = local_sub_width
            lback_kwargs['save_results'] = True
            lback_kwargs['back_offset'] = local_sub_back_offset
            # run the background subtraction
            result = soss_commutils.localbackground_subtraction(result,
                                                                **lback_kwargs)

        # -----------------------------------------------------------------
        # Custom - remove NaNs
        # -----------------------------------------------------------------
        if remove_nans:
            result = soss_commutils.remove_nans(result, outdir,
                                                save_results=True)
            # add dq trace
            dqtracelog.write('DQ={0} - After remove nans '
                             'step'.format(result.dq[dqmask]))

        # -----------------------------------------------------------------
        # Extraction
        # -----------------------------------------------------------------
        # get the trace table ref
        trace_table_ref = io.get_tracetable_file(params)
        # get the wavemap
        wavemap = io.get_wavemap(params)
        # get the spectrum profile
        spec_profile = io.get_spec_profile(params)
        # get the soss_modelname
        soss_modelname = io.get_soss_modelname(params, satfile)
        # shared kwargs
        extract_kwargs = dict()
        extract_kwargs['outdir'] = outdir
        extract_kwargs['save_results'] = True
        extract_kwargs['soss_transform'] = soss_transform
        extract_kwargs['subtract_background'] = False
        extract_kwargs['soss_width'] = extraction_box_width
        extract_kwargs['override_spectrace'] = trace_table_ref
        extract_kwargs['override_wavemap'] = wavemap
        extract_kwargs['override_specprofiel'] = spec_profile
        # print the name of the file before photom step
        print('Filename of datamodel before extraction step = {0}'
              ''.format(result.meta.filename))
        # add to extraction keyword arguments
        if extraction_method.upper() == 'ATOCA':
            # add ATOCA kwargs
            extract_kwargs['soss_atoca'] = True
            extract_kwargs['soss_bad_pix'] = 'model'
            extract_kwargs['soss_modelname'] = soss_modelname
        elif extraction_method.upper() == 'OPTIMAL':
            # add OPTIMAL kwargs
            extract_kwargs['soss_atoca'] = False
            extract_kwargs['soss_bad_pix'] = 'masking'
            extract_kwargs['soss_extraction_type'] = 'optimal'
            extract_kwargs['soss_modelname'] = None
        elif extraction_method.upper() == 'BOX':
            # add BOX kwargs
            extract_kwargs['soss_atoca'] = False
            extract_kwargs['soss_bad_pix'] = 'masking'
            extract_kwargs['soss_modelname'] = None
        else:
            emsg = ('Extraction method "{0}" not recognized. \n\t'
                    'Must be one of "atoca", "optimal", or "box". \n\t'
                    'Function = {1}')
            eargs = [extraction_method, func_name]
            raise base.LoicPipeError(emsg.format(*eargs))
        # run extraction
        Extract1dStep = calwebb_spec2.extract_1d_step.Extract1dStep
        result = Extract1dStep.call(result, **extract_kwargs)
        # -----------------------------------------------------------------
        #  DMS standard - Conversion to SI units
        # -----------------------------------------------------------------
        # print the name of the file before photom step
        print('Filename of datamodel before photom step = {0}'
              ''.format(result.meta.filename))
        # get the photom file
        photom_filename = io.get_photom_file(params)
        # convert to SI
        PhotomStep = calwebb_spec2.photom_step.PhotomStep
        result = PhotomStep.call(result, output_dir=outdir,
                                 save_results=True,
                                 override_photom=photom_filename)
        # -----------------------------------------------------------------
        # Write results on disk
        result.close()
    # ----------------------------------------------------------------------
    return params


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
