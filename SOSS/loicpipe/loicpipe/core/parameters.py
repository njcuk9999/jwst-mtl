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

# -----------------------------------------------------------------------------
# stage 1 parameters
# -----------------------------------------------------------------------------
# whether to run loicpipe for stage 1
__key__ = 'stage1.loicpipe.run'
params[__key__] = Const(__key__, True, not_none=True, dtype=bool, path=__key__,
                        source=__NAME__)

# whether to skip stacking
__key__ = 'stage1.loicpipe.skip_stack'
params[__key__] = Const(__key__, True, not_none=True, dtype=bool, path=__key__,
                        source=__NAME__)

# the saturation map file
__key__ = 'stage1.loicpipe.satmap'
params[__key__] = Const(__key__, None, dtype=str, path=__key__,
                        source=__NAME__)

# erase clean saturation files
__key__ = 'stage1.loicpipe.erase_clean_sat'
params[__key__] = Const(__key__, False, dtype=str, path=__key__,
                        source=__NAME__)

# rejection threshold for jump step
__key__ = 'stage1.loicpipe.jump_rej_thres'
params[__key__] = Const(__key__, 6.0, dtype=float, path=__key__,
                        source=__NAME__)

# whether we fit the ramp (if False uses last read)
__key__ = 'stage1.loicpipe.fit_ramp'
params[__key__] = Const(__key__, None, dtype=bool, path=__key__,
                        source=__NAME__)


# -----------------------------------------------------------------------------
# stage 2 parameters
# -----------------------------------------------------------------------------
# whether to run loicpipe for stage 2
__key__ = 'stage2.loicpipe.run'
params[__key__] = Const(__key__, True, not_none=True, dtype=bool, path=__key__,
                        source=__NAME__)
# the contamination mask to use for loicpipe stage2
__key__ = 'stage2.loicpipe.contam_mask'
params[__key__] = Const(__key__, True,  not_none=True, dtype=bool,
                        path=__key__, source=__NAME__)

# -----------------------------------------------------------------------------
# stage 3 parameters
# -----------------------------------------------------------------------------


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
