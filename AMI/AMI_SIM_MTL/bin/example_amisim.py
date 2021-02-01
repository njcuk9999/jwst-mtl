#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2020-05-21

@author: cook
"""
from astropy import units as uu

from ami_mtl.core.core import param_functions
from ami_mtl.core.core import log_functions
from ami_mtl.core.instrument import constants
from ami_mtl.science import etienne

# =============================================================================
# Define variables
# =============================================================================
# set name
__NAME__ = 'example_amisim.py'
__DESCRIPTION__ = 'test script for using ami-sim'
# get default constants
consts = constants.Consts
# copy for update
lconsts = consts.copy(__NAME__)
# set very basic constants
__VERSION__ = lconsts.constants['PACKAGE_VERSION'].value
__DATE__ = lconsts.constants['PACKAGE_VERSION_DATE'].value
# set up the logger
log = log_functions.Log()
# define group name
group = 'example_amisim'

# Define the target name
lconsts.add_argument('OBJ_TARGET_NAME', value=None, dtype=str,
                     source=__NAME__, user=True, argument=True,
                     group=group,
                     description='Define the target as it appears in the '
                                 'APT file',
                     command=['--target'])

# Define the separation of companion in arcsec
lconsts.add_argument('OBJ_COMP_SEP', value=None, dtype=float,
                     source=__NAME__, user=True, argument=True,
                     group=group, units=uu.arcsec,
                     description='Separation of companion in arcseconds',
                     command=['--separation'])

# Define the position angle of companion in arcsec
lconsts.add_argument('OBJ_COMP_PA', value=None, dtype=float,
                     source=__NAME__, user=True, argument=True,
                     group=group, units=uu.deg,
                     description='Position angle with respect to RA/Dec',
                     command=['--pa'])

# Define the fractional contrast of companion
lconsts.add_argument('OBJ_COMP_CONTRAST', value=None, dtype=float,
                     source=__NAME__, user=True, argument=True,
                     group=group,
                     description='Define the fractional contrast of companion',
                     command=['--contrast'])

# Define the number of integrations to use
lconsts.add_argument('NINT', value=None, dtype=int,
                     source=__NAME__, user=True, argument=True,
                     group=group,
                     description='Number of integrations to use',
                     command=['--nint'])

# Define the number of groups to use
lconsts.add_argument('NGROUPS', value=None, dtype=int,
                     source=__NAME__, user=True, argument=True,
                     group=group,
                     description='Number of groups to use',
                     command=['--ngroup'])

# Define one of the four AMI filters to use
lconsts.add_argument('FILTER', value=None, dtype=str,
                     source=__NAME__, user=True, argument=True,
                     group=group, options=['F380M', 'F430M', 'F480M', 'F277W'],
                     description='Define one of the four AMI filters to use',
                     command=['--filter'])

# Define the flux from the ETC expressed in e-/s
lconsts.add_argument('EXT_FLUX', value=None, dtype=float,
                     source=__NAME__, user=True, argument=True,
                     group=group, units=uu.electron / uu.s,
                     description='Define the flux from the ETC expressed '
                                 'in e-/s',
                     command=['--eflux'])

# Define the total exposure time in seconds
lconsts.add_argument('TOT_EXP', value=None, dtype=float,
                     source=__NAME__, user=True, argument=True,
                     group=group, units=uu.s,
                     description='Define the total exposure time in seconds',
                     command=['--texp'])


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
    except SystemExit as _:
        return params


def __main__(params):
    # debug where we have got to in the code
    params.log.ldebug('run __main__', 7)

    # -------------------------------------------------------------------------
    # step 1: compute sky scene
    # -------------------------------------------------------------------------
    sprops = etienne.simple_target_scene(params)

    # -------------------------------------------------------------------------
    # step 2: run ami-sim
    # -------------------------------------------------------------------------
    etienne.run_ami_sim(params, sky_scene=str(sprops['SKY_FILE']),
                        count_rate=float(sprops['COUNT_RATE']))

    return params


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # test
    import sys

    # sys.argv = 'test.py --getconfig=True --config=loictest.ini'.split()
    # run main code
    ll = main()

# =============================================================================
# End of code
# =============================================================================
