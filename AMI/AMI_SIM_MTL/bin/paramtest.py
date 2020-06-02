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
from ami_sim_mtl.core.core import log_functions

# =============================================================================
# Define variables
# =============================================================================
# set name
__NAME__ = 'paramtest.py'
__DESCRIPTION__ = 'test script for AMI_SIM_MTL'
# get default constants
consts = constants.Consts
# copy for update
lconsts = consts.copy(__NAME__)
# set very basic constants
__VERSION__ = lconsts.constants['PACKAGE_VERSION'].value
__DATE__ = lconsts.constants['PACKAGE_VERSION_DATE'].value
# set up the logger
log = log_functions.Log()
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
    """
    Main function: Do not add code here, add it in __main__
    :param kwargs:
    :return:
    """
    # for error handling set these
    params = consts.constants
    # wrap in error handler
    try:
        # get params (run time + config file + constants file)
        params = param_functions.setup(lconsts, kwargs, log=log,
                                       desc=__DESCRIPTION__, name=__NAME__)
        # run the __main__ to return products
        if not params['GENERATE_CONFIG_FILE']:
            return __main__(params)
    except Exception as e:
        if hasattr(e, '__log__'):
            log.exception(e.__log__())
            log.error(e.__log__())
        else:
            emsg = 'UNEXPECTED ERROR {0}: {1}'
            log.exception(emsg.format(type(e), str(e)))
            log.error(emsg.format(type(e), str(e)))
        return params
    except SystemExit as e:
        emsg = 'SYSTEM EXIT {0}: {1}'
        log.exception(emsg.format(type(e), str(e)))
        log.error(emsg.format(type(e), str(e)))
        return params


def __main__(params):

    params.log.ldebug('run __main__', 7)
    # main code here
    for param in params:
        params.info(param)

    # test logging
    params.log.info('This is a test of info')
    params.log.warning('This is a test of warning')
    params.log.error('This is a test of an error')

    return params

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

    # test
    import sys

    # sys.argv = 'test.py'.split()
    # run main code
    ll = main()

# =============================================================================
# End of code
# =============================================================================
