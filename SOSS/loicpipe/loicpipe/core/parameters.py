#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on {DATE}

@author: cook
"""

from loicpipe.core import base
from loicpipe.core import constants

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.parameters.py'
__VERSION__ = base.__VERSION__
__DATE__ = base.__DATE__
# get the constants class
Const = constants.Const

# =============================================================================
# Define parameters
# =============================================================================
# storage for all parameters
params = constants.Parameters()

# -----------------------------------------------------------------------------
# general parameters
# -----------------------------------------------------------------------------
__key__ = 'general.debug'
params[__key__] = Const(__key__, None, not_none=True, dtype=str,
                        path=__key__, source=__NAME__)


# -----------------------------------------------------------------------------
# data parameters
# -----------------------------------------------------------------------------
# set the raw data path
__key__ = 'data.raw'
params[__key__] = Const(__key__, None, not_none=True, dtype=str,
                        path=__key__, source=__NAME__)
# set the filelist (None will use all files)
__key__ = 'data.filelist'
params[__key__] = Const(__key__, None, dtype=str,
                        path=__key__, source=__NAME__)

# set the output directory
__key__ = 'data.outdir'
params[__key__] = Const(__key__, None, dtype=str,
                        path=__key__, source=__NAME__)

# set the data string (e.g. jw01201501001_04101_00001)
__key__ = 'data.data_string'
params[__key__] = Const(__key__, None, dtype=str,
                        path=__key__, source=__NAME__)

# set the crds directory
__key__ = 'data.crds-dir'
params[__key__] = Const(__key__, None, dtype=str,
                        path=__key__, source=__NAME__)

# set the calibration directory
__key__ = 'data.calib-dir'
params[__key__] = Const(__key__, None, dtype=str,
                        path=__key__, source=__NAME__)

# set the atoca reference directory
__key__ = 'data.atoca-dir'
params[__key__] = Const(__key__, None, dtype=str,
                        path=__key__, source=__NAME__)

# set the ref trace table
__key__ = 'data.ref_trace_table'
params[__key__] = Const(__key__, None, dtype=str,
                        path=__key__, source=__NAME__)

# custom dark - if default is None we use the default
#             - if subarray is named we use it, otherwise we use default
__key__ = 'data.custom_dark'
params[__key__] = Const(__key__, None, dtype=dict,
                        path=__key__, source=__NAME__)

# whether the custom dark has the super bias included
__key__ = 'data.custom_dark_has_superbias_included'
params[__key__] = Const(__key__, False, dtype=bool,
                        path=__key__, source=__NAME__)

# define superbias file
__key__ = 'data.superbias'
params[__key__] = Const(__key__, None, dtype=str,
                        path=__key__, source=__NAME__)

# set the list of uncal files
__key__ = 'output.uncal_list'
params[__key__] = Const(__key__, [], dtype=list, source=__NAME__)

# =============================================================================
# LOIC PIPE PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Loicpipe shared parameters
# -----------------------------------------------------------------------------
# define the dq mask values
__key__ = 'loicpipe.extract_only'
params[__key__] = Const(__key__, None, dtype=bool, path=__key__,
                        source=__NAME__)

# define the dq mask values
__key__ = 'loicpipe.dq_mask'
params[__key__] = Const(__key__, None, dtype=list, path=__key__,
                        source=__NAME__)

# -----------------------------------------------------------------------------
# Loicpipe stage 1 parameters
# -----------------------------------------------------------------------------
# whether to run loicpipe for stage 1
__key__ = 'loicpipe.stage1.run'
params[__key__] = Const(__key__, True, not_none=True, dtype=bool, path=__key__,
                        source=__NAME__)

# whether to skip stacking
__key__ = 'loicpipe.stage1.skip_stack'
params[__key__] = Const(__key__, True, not_none=True, dtype=bool, path=__key__,
                        source=__NAME__)

# the saturation map file
__key__ = 'loicpipe.stage1.satmap'
params[__key__] = Const(__key__, None, dtype=str, path=__key__,
                        source=__NAME__)

# erase clean saturation files
__key__ = 'loicpipe.stage1.erase_clean_sat'
params[__key__] = Const(__key__, False, dtype=str, path=__key__,
                        source=__NAME__)

# rejection threshold for jump step
__key__ = 'loicpipe.stage1.jump_rej_thres'
params[__key__] = Const(__key__, 6.0, dtype=float, path=__key__,
                        source=__NAME__)

# whether we fit the ramp (if False uses last read)
__key__ = 'loicpipe.stage1.fit_ramp'
params[__key__] = Const(__key__, None, dtype=bool, path=__key__,
                        source=__NAME__)

# -----------------------------------------------------------------------------
# Loicpipe stage 2 parameters
# -----------------------------------------------------------------------------
# whether to run loicpipe for stage 2
__key__ = 'loicpipe.stage2.run'
params[__key__] = Const(__key__, True, not_none=True, dtype=bool, path=__key__,
                        source=__NAME__)

# whether to run flat fielding
__key__ = 'loicpipe.stage2.flat_field'
params[__key__] = Const(__key__, True, dtype=bool, path=__key__,
                        source=__NAME__)

# erase clean saturation files
__key__ = 'loicpipe.stage1.erase_clean_rateint'
params[__key__] = Const(__key__, False, dtype=str, path=__key__,
                        source=__NAME__)

# whether to run custom outlier rejection
__key__ = 'loicpipe.stage2.custom_outlier_rejection'
params[__key__] = Const(__key__, False, dtype=bool, path=__key__,
                        source=__NAME__)

# define custom outlier window size (rows and columns)
__key__ = 'loicpipe.stage2.custom_outlier_window'
params[__key__] = Const(__key__, None, dtype=list, path=__key__,
                        source=__NAME__)

# define custom outlier n sigma for rejection
__key__ = 'loicpipe.stage2.custom_outlier_nsig'
params[__key__] = Const(__key__, None, dtype=float, path=__key__,
                        source=__NAME__)

# erase / clean custom outlier files
__key__ = 'loicpipe.stage2.erase_clean_custom_outlier'
params[__key__] = Const(__key__, False, dtype=str, path=__key__,
                        source=__NAME__)

# define the aphalfwidth for background subtraction
__key__ = 'loicpipe.stage2.aphalfwidth'
params[__key__] = Const(__key__, None, dtype=list, path=__key__,
                        source=__NAME__)

# erase / clean background files
__key__ = 'loicpipe.stage2.erase_clean_background'
params[__key__] = Const(__key__, False, dtype=str, path=__key__,
                        source=__NAME__)

# whether to do the background subtraction
__key__ = 'loicpipe.stage2.background_subtract'
params[__key__] = Const(__key__, True, dtype=bool, path=__key__,
                        source=__NAME__)

# background contamination mask (None for no mask)
__key__ = 'loicpipe.stage2.contam_mask'
params[__key__] = Const(__key__, None, dtype=bool,
                        path=__key__, source=__NAME__)

# whether to do the bad pixel interpolation
__key__ = 'loicpipe.stage2.bad_pix_interp'
params[__key__] = Const(__key__, True, dtype=bool,
                        path=__key__, source=__NAME__)

# whether to do a local background subtraction
__key__ = 'loicpipe.stage2.local_background_subtraction'
params[__key__] = Const(__key__, False, dtype=bool,
                        path=__key__, source=__NAME__)

# define the local background subtraction width
__key__ = 'loicpipe.stage2.local_sub_width'
params[__key__] = Const(__key__, None, dtype=int, path=__key__,
                        source=__NAME__)

# define the local background offset
__key__ = 'loicpipe.stage2.local_sub_back_offset'
params[__key__] = Const(__key__, None, dtype=int, path=__key__,
                        source=__NAME__)

# whether to remove NaNs
__key__ = 'loicpipe.stage2.remove_nans'
params[__key__] = Const(__key__, True, dtype=bool,
                        path=__key__, source=__NAME__)

# extraction method: atoca, optimal, box
__key__ = 'loicpipe.stage2.extraction_method'
params[__key__] = Const(__key__, 'atoca', dtype=str,
                        path=__key__, source=__NAME__)

# extraction box width (used in all three methods)
__key__ = 'loicpipe.stage2.extraction_box_width'
params[__key__] = Const(__key__, None, dtype=int, path=__key__,
                        source=__NAME__)

# extraction soss transform [three integers]
__key__ = 'loicpipe.stage2.extraction_soss_transform'
params[__key__] = Const(__key__, None, dtype=list, path=__key__,
                        source=__NAME__)

# -----------------------------------------------------------------------------
# Loicpipe stage 3 parameters
# -----------------------------------------------------------------------------
# whether to run loicpipe for stage 3
__key__ = 'loicpipe.stage3.run'
params[__key__] = Const(__key__, True, not_none=True, dtype=bool, path=__key__,
                        source=__NAME__)

# =============================================================================
# SUPREME SPOON PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# SupremeSpoon stage 1 parameters
# -----------------------------------------------------------------------------
# whether to run SupremeSpoon for stage 1
__key__ = 'supremespoon.stage1.run'
params[__key__] = Const(__key__, True, not_none=True, dtype=bool, path=__key__,
                        source=__NAME__)

# -----------------------------------------------------------------------------
# SupremeSpoon stage 2 parameters
# -----------------------------------------------------------------------------
# whether to run SupremeSpoon for stage 2
__key__ = 'supremespoon.stage2.run'
params[__key__] = Const(__key__, True, not_none=True, dtype=bool, path=__key__,
                        source=__NAME__)

# -----------------------------------------------------------------------------
# SupremeSpoon stage 3 parameters
# -----------------------------------------------------------------------------
# whether to run SupremeSpoon for stage 3
__key__ = 'supremespoon.stage3.run'
params[__key__] = Const(__key__, True, not_none=True, dtype=bool, path=__key__,
                        source=__NAME__)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
