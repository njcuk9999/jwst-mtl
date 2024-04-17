#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on {DATE}

@author: cook
"""
import argparse
from typing import Any, Dict, Optional

from loicpipe.core import constants
from loicpipe.core import io
from loicpipe.core import parameters

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = ''
# get Parameters class
Parameters = constants.Parameters

# =============================================================================
# Define functions
# =============================================================================
def get_args() -> Dict[str, Any]:
    """
    Define the command line arguments

    :return: argparse namespace object containing arguments
    """
    parser = argparse.ArgumentParser(description='Run apero raw tests')
    # add obs dir
    parser.add_argument('yaml', type=str, default='None',
                        help='The profile yaml to use')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Run in debug mode')
    # load arguments with parser
    args = parser.parse_args()
    # return arguments
    return vars(args)


def add_cmd_arg(args, key, value, null=None):
    # define the null values
    null_values = {null, str(null), None}
    # if value is equal to the null value then we check "args"
    if value in null_values:
        # if the "args" is not in the null values then we use the "args" value
        if args[key] not in null_values:
            value = args[key]
    # force value to null if in any of the null values (i.e. a string version
    #   of the null)
    if value in null_values:
        return null
    # otherwise return our value
    else:
        return value


def load_params(yaml_file: Optional[str] = None,
                debug: Optional[bool] = None) -> Parameters:
    # set up the return dictionary
    params = parameters.params.copy()
    # -------------------------------------------------------------------------
    # read from command line
    args = get_args()
    # get yaml file from cmd args
    yaml_file = add_cmd_arg(args, 'yaml', yaml_file)
    # get debug from cmd args
    params['debug'] = add_cmd_arg(args, 'debug', debug)
    # set the debug source
    params('debug').source = 'command line'
    # -------------------------------------------------------------------------
    # load from yaml file
    yaml_params = io.read_yaml(yaml_file)
    # loop around params
    for key in params:
        # if key is in yaml_params
        if key in yaml_params:
            # set params[key] = yaml_params[key]
            params[key] = params(key).from_yaml(yaml_params)
            # set the source
            params(key).source = yaml_file
    # -------------------------------------------------------------------------
    # return params
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

