#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-11

@author: cook
"""
import argparse
from typing import Any
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
# printer
cprint = base_classes.Printer()


# =============================================================================
# Define functions
# =============================================================================
def get_args():
    """
    Get command line arguments
    :return:
    """
    # define the parser
    parser = argparse.ArgumentParser(description='Transit fit code')
    # add arguments here
    parser.add_argument(dest='config',
                        help='[STRING] The configuration yaml file',
                        type=str)
    # get the arguments
    args = parser.parse_args()
    # return the config file
    return str(args.config)


def load_params(config_file: str) -> ParamDict:
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
        if 'planet{N}' in instance.path: ###DL###
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
        elif instance.dtype is not None:
            value = _get_value(value, key, instance.path, instance.dtype)
        # now update params
        params.set(key=key, value=value, source=config_file)
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
            ###DL### put the check below in comment, to allow different models to use different params
#             if value is None:
#                 emsg = f'ParamError: {pkey} must have "{key}" ({path}) set.'
#                 raise base_classes.TransitFitExcept(emsg)
            # deal with dtype = FitParam
            if value is not None: ###DL### add this so it's added only if specified in yaml
                if issubclass(instance.dtype, FitParam):
                    FitParam.check(name=name, **value)
                    value = FitParam(key, label=instance.label, **value)
                # now update params
                planet_dict.set(key=key, value=value, source=config_file)
        # push back to main parameters
        params.set(key=pkey, value=planet_dict, source=config_file)
    # -------------------------------------------------------------------------
    # remove planet keys from params
    for key in planet_keys:
        del params[key]
    #---------
    #deal with star spots
    spot_keys=['SPOTX','SPOTY','SPOTR','SPOTC']
    for nspot in range(params['NSPOTS']):
        # set output key
        skey = 'SPOT{N}'.format(N=nspot+1)
        # set up a planet dictionary
        spot_dict = params.copy(mask=spot_keys)
        
        # loop around spot keys
        for key in spot_keys:
            # set the name
            name = f'{skey}_{key}'
            # skip keys without instance
            if key not in params.instances:
                continue
            # get the parameter instance
            instance = params.instances[key]
            # get the path
            path = instance.path.format(N=nspot+1)
            # check for yaml path
            if path is not None:
                value = get_yaml_from_path(yaml_inputs, path)
            # else we use default value
            else:
                emsg = f'ParamError: {skey} must have "{key}" ({path}) set.'
                raise base_classes.TransitFitExcept(emsg)
            if value is not None: ###DL### add this so it's added only if specified in yaml
                if issubclass(instance.dtype, FitParam):
                    FitParam.check(name=name, **value)
                    value = FitParam(key, label=instance.label, **value)
                # now update params
                spot_dict.set(key=key, value=value, source=config_file)
        # push back to main parameters
        params.set(key=skey, value=spot_dict, source=config_file)
    # remove spot keys from params
    for key in spot_keys:
        del params[key]
        
    #deal with trends ###DL###
    for ntrend in range(params['NTRENDS']):
        key='TREND_C'
        name='TREND_C{}'.format(ntrend+1)
        # get the parameter instance
        instance = params.instances[key]
        # get the path
        path = instance.path.format(N=ntrend+1)
        value = get_yaml_from_path(yaml_inputs, path)
        # deal with dtype = FitParam
        if issubclass(instance.dtype, FitParam):
            FitParam.check(name=name, **value)
            value = FitParam(key, label=instance.label, **value)
        elif instance.dtype is not None:
            value = _get_value(value, key, instance.path, instance.dtype)
        # now update params
        params.set(key=name, value=value, source=config_file)
    del params['TREND_C']
    # -------------------------------------------------------------------------
    # report loading
    cprint(f'\tLoaded: {config_file}')
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


def _get_value(value: Any, key: str, path: str, dtype: Any):
    """
    Get a value and check data type

    :param value:  Any, the value to test
    :param key: str, the name of the parameter we are testing
    :param path: str, the path in the yaml file
    :param dtype: List[type] or Any, the type to test

    :return:
    """
    # make dtype a list of dtypes
    if not isinstance(dtype, list):
        dtypes = [dtype]
    else:
        dtypes = list(dtype)
    # start a counter
    it = 0
    # loop around dtypes
    while it < len(dtypes):
        # try to get value
        try:
            # if we have a int or float try to convert it
            if dtype in [int, float, bool]:
                # if we have a string evaluate value
                if isinstance(value, str):
                    value = eval(value)
                # cast value
                return dtype(value)
            # any other type we don't try to cast
            else:
                return value
        # deal with exceptions (continue to next dtype or raise error)
        except Exception as _:
            # try next type
            if it < len(dtypes) - 1:
                it += 1
                continue
            # raise error once all dtypes have been tried
            emsg = (f'ParamError: Parameter "{key}" ({path} '
                    f'must be type {dtype}')
            raise base_classes.TransitFitExcept(emsg)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
