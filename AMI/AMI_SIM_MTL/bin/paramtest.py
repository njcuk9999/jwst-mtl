#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2020-05-21

@author: cook
"""
from ami_sim_mtl.core.instrument import constants
from ami_sim_mtl.core.core import param_functions

# =============================================================================
# Define variables
# =============================================================================
# set name
__NAME__ = 'bin/test.py'
__DESCRIPTION__ = 'test script for AMI_SIM_MTL'
# get default constants
consts = constants.Consts
# copy for update
lconsts = consts.copy(__NAME__)
# set very basic constants
__VERSION__ = lconsts.constants['PACKAGE_VERSION'].value
__DATE__ = lconsts.constants['PACKAGE_VERSION_DATE'].value

# add code specific arguments

# Define the scene fits file
lconsts.add_argument('SCENE', value=None, dtype=str,
                     source=__NAME__, user=True, argument=True,
                     group='code', description='Define the scene fits file',
                     command=['--scene'])


# =============================================================================
# Define functions
# =============================================================================
def main(**kwargs):
    # get params (run time + config file + constants file)
    params = param_functions.setup(lconsts, kwargs,
                                   description=__DESCRIPTION__)
    # run the __main__ to return products
    if not params['GENERATE_CONFIG_FILE']:
        __main__(params)


def __main__(params):
    # main code here
    for param in params:
        params.info(param)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    ll = main()

# =============================================================================
# End of code
# =============================================================================
