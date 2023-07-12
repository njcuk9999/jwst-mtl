#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3 wrapper functionality

This stage combines the fully calibrated data from multiple exposures.

input: calibrated slope images for all integrations and exposures
output: combined, rectified level-3 data

Created on 2023-07-11

@author: cook
"""
from loicpipe.core import constants

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'science.stage3.py'
# get Parameters class
Parameters = constants.Parameters

# =============================================================================
# Define functions
# =============================================================================
def main(params: Parameters):
    pass

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
