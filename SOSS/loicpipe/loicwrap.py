#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2023-07-11

@author: cook
"""
from loicpipe.core import general
from loicpipe.science import stage1
from loicpipe.science import stage2
from loicpipe.science import stage3

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = ''

# =============================================================================
# Define functions
# =============================================================================
def main():
    """
    Main code

    :return:
    """
    # ----------------------------------------------------------------------
    # Main Code
    # ----------------------------------------------------------------------
    # load parameters
    params = general.load_params()
    # verify data
    general.verify_data(params)
    # ----------------------------------------------------------------------
    # run stage 1
    params = stage1.main(params)
    # ----------------------------------------------------------------------
    # run stage 2
    params = stage2.main(params)
    # ----------------------------------------------------------------------
    # run stage 3
    params = stage3.main(params)
    # ----------------------------------------------------------------------
    return params


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main with no arguments (get from command line - sys.argv)
    ll = main()

# =============================================================================
# End of code
# =============================================================================

