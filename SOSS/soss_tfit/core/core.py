#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-11

@author: cook
"""
import yaml

from soss_tfit.core import base_classes
from soss_tfit.core import parameters

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.core.py'
# Get the parameter dictionary class
ParamDict = base_classes.ParamDict
# Get the Fit Parameter class
FitParam = base_classes.FitParam


# =============================================================================
# Define functions
# =============================================================================
def load_params(config_file: str) -> ParamDict:
    # set function name
    func_name = __NAME__ + '.load_params()'
    # copy the parameters
    params = parameters.params.copy()
    # -------------------------------------------------------------------------
    # load yaml file
    with open(config_file, 'r') as yfile:
        yaml_inputs = yaml.load(yfile, yaml.FullLoader)
    # -------------------------------------------------------------------------
    # store planet keys
    planet_keys = []
    # loop around params and load those in a path
    for key in params:
        # skip keys without instance
        if key not in params.instances:
            continue
        # get the parameter instance
        instance = params.instances[key]
        # store planet keys and skip
        if '{N}' in instance.path:
            planet_keys.append(key)
            continue
        # check for yaml path
        if instance.path is not None:
            value = get_yaml_from_path(yaml_inputs, instance.path)
        # else we use default value
        else:
            continue
        # need to check for Null value --> use default value
        if value is None:
            # force error if non None parameter is still none
            if key in params.not_none:
                emsg = (f'ParamError: Parameter "{key}" ({instance.path}) '
                        f'cannot be None. Please set.')
                raise base_classes.TransitFitExcept(emsg)
            else:
                continue
        # deal with dtype = FitParam
        if issubclass(instance.dtype, FitParam):
            FitParam.check(name=key, **value)
            value = FitParam(key, label=instance.label, **value)
        # now update params
        params.set(key=key, value=value, source=func_name)
    # -------------------------------------------------------------------------
    # Deal with planets
    for nplanet in range(params['NPLANETS']):
        # set output key
        pkey = 'PLANET{N}'.format(N=nplanet+1)
        # set up a planet dictionary
        planet_dict = params.copy(mask=planet_keys)
        # loop around planet keys
        for key in planet_keys:
            # set the name
            name = f'{pkey}_{key}'
            # skip keys without instance
            if key not in params.instances:
                continue
            # get the parameter instance
            instance = params.instances[key]
            # get the path
            path = instance.path.format(N=nplanet+1)
            # check for yaml path
            if path is not None:
                value = get_yaml_from_path(yaml_inputs, path)
            # else we use default value
            else:
                emsg = f'ParamError: {pkey} must have "{key}" ({path}) set.'
                raise base_classes.TransitFitExcept(emsg)
            # need to check for Null value --> use default value
            if value is None:
                emsg = f'ParamError: {pkey} must have "{key}" ({path}) set.'
                raise base_classes.TransitFitExcept(emsg)
            # deal with dtype = FitParam
            if issubclass(instance.dtype, FitParam):
                FitParam.check(name=name, **value)
                value = FitParam(key, label=instance.label, **value)
            # now update params
            planet_dict.set(key=key, value=value, source=func_name)
        # push back to main parameters
        params.set(key=pkey, value=planet_dict, source=func_name)
    # -------------------------------------------------------------------------
    # remove planet keys from params
    for key in planet_keys:
        del params[key]
    # -------------------------------------------------------------------------
    return params


def get_yaml_from_path(ydict: dict, path: str):
    """
    Get a value from a yaml dictionary using a path key1.key2.key3.keyN
    for any number of keys

    :param ydict: dict, the yaml dictionary ydict[key1][key2][key3][..][keyN]
    :param path: str, the path in format key1.key2.key3.keyN

    :return:
    """
    # split the path into keys
    ykeys = path.split('.')
    # set the key level dict to top level
    ydict_tmp = ydict
    # loop around yaml keys
    for ykey in ykeys:
        # make sure key is valid
        if ykey not in ydict_tmp:
            return None
        # get the value
        value = ydict_tmp[ykey]
        # if we no longer have a dictionary then we have our value
        if not isinstance(ydict_tmp[ykey], dict):
            return value
        # else set the ydict_tmp to the value
        ydict_tmp = value
    # return the bottom level key described by path
    return ydict_tmp


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
