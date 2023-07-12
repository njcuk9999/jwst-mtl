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
from loicpipe.core import constants

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
    if params['stage2.loicpipe.run']:
        stage2_loicpipe(params)
    # ----------------------------------------------------------------------
    return params

def stage2_loicpipe(params: Parameters) -> Parameters:
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
