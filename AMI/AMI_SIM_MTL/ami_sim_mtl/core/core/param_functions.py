#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2020-05-21

@author: cook
"""
import argparse
from astropy.time import Time
from collections import OrderedDict
import copy
import numpy as np
import os
from pathlib import Path
import logging
import random
import string
import sys
import time
from typing import Union, List, Type

from ami_sim_mtl.core.core import general
from ami_sim_mtl.core.core import exceptions
from ami_sim_mtl.core.instrument import constants
from ami_sim_mtl.core.core import log_functions

# =============================================================================
# Define variables
# =============================================================================
# Get default constants
consts = constants.Consts
# define name
__NAME__ = 'core.core.constant_functions.py'
__version__ = consts.constants['PACKAGE_VERSION'].value
__date__ = consts.constants['PACKAGE_VERSION_DATE'].value
# define package name
PACKAGE = consts.constants['PACKAGE_NAME'].value
# get general functions
display_func = general.display_func
# get exceptions
ParamException = exceptions.ParamDictException
# get all chars
CHARS = string.ascii_uppercase + string.digits
# define true values
TRUE_VALUES = constants.constant_functions.TRUE_VALUES
# Define user config file header
CONFIG_FILE_HEADER = """
# =============================================================================
# User Config file
# =============================================================================
# Use this config file by defining it with argument --config
# 
#     Note: not all keys are used in each code

"""
# Define user config group header
CONFIG_GROUP_HEADER = """
# -----------------------------------------------------------------------------
# {0} constants
# -----------------------------------------------------------------------------
"""
# get the environmental variable for ami sim dir
AMIDIR = str(consts.constants['ENV_DIR'])
# get the default directory otherwise (put in the home dir)
AMIPATH = str(consts.constants['PACKAGE_DIRECTORY'])


# =============================================================================
# Define classes
# =============================================================================
# case insensitive dictionary
class CaseInsensitiveDict(dict):
    """
    Custom dictionary with string keys that are case insensitive
    """

    def __init__(self, *arg, **kw):
        """
        Construct the case insensitive dictionary class
        :param arg: arguments passed to dict
        :param kw: keyword arguments passed to dict
        """
        # set function name
        _ = display_func('__init__', __NAME__, 'CaseInsensitiveDict')
        # super from dict
        super(CaseInsensitiveDict, self).__init__(*arg, **kw)
        # force keys to be capitals (internally)
        self.__capitalise_keys__()

    def __getitem__(self, key: str) -> object:
        """
        Method used to get the value of an item using "key"
        used as x.__getitem__(y) <==> x[y]
        where key is case insensitive

        :param key: string, the key for the value returned (case insensitive)
        :type key: str

        :return value: object, the value stored at position "key"
        """
        # set function name
        _ = display_func('__getitem__', __NAME__, 'CaseInsensitiveDict')
        # make key capitals
        key = _capitalise_key(key)
        # return from supers dictionary storage
        return super(CaseInsensitiveDict, self).__getitem__(key)

    def __setitem__(self, key: str, value: object, source: str = None):
        """
        Sets an item wrapper for self[key] = value
        :param key: string, the key to set for the parameter
        :param value: object, the object to set (as in dictionary) for the
                      parameter
        :param source: string, the source for the parameter

        :type key: str
        :type value: object
        :type source: str

        :return: None
        """
        # set function name
        _ = display_func('__setitem__', __NAME__, 'CaseInsensitiveDict')
        # capitalise string keys
        key = _capitalise_key(key)
        # then do the normal dictionary setting
        super(CaseInsensitiveDict, self).__setitem__(key, value)

    def __contains__(self, key: str) -> bool:
        """
        Method to find whether CaseInsensitiveDict instance has key="key"
        used with the "in" operator
        if key exists in CaseInsensitiveDict True is returned else False
        is returned

        :param key: string, "key" to look for in CaseInsensitiveDict instance
        :type key: str

        :return bool: True if CaseInsensitiveDict instance has a key "key",
        else False
        :rtype: bool
        """
        # set function name
        _ = display_func('__contains__', __NAME__, 'CaseInsensitiveDict')
        # capitalize key first
        key = _capitalise_key(key)
        # return True if key in keys else return False
        return super(CaseInsensitiveDict, self).__contains__(key)

    def __delitem__(self, key: str):
        """
        Deletes the "key" from CaseInsensitiveDict instance, case insensitive

        :param key: string, the key to delete from ParamDict instance,
                    case insensitive
        :type key: str

        :return None:
        """
        # set function name
        _ = display_func('__delitem__', __NAME__, 'CaseInsensitiveDict')
        # capitalize key first
        key = _capitalise_key(key)
        # delete key from keys
        super(CaseInsensitiveDict, self).__delitem__(key)

    def get(self, key: str, default: Union[None, object] = None):
        """
        Overrides the dictionary get function
        If "key" is in CaseInsensitiveDict instance then returns this value,
        else returns "default" (if default returned source is set to None)
        key is case insensitive

        :param key: string, the key to search for in ParamDict instance
                    case insensitive
        :param default: object or None, if key not in ParamDict instance this
                        object is returned

        :type key: str
        :type default: Union[None, object]

        :return value: if key in ParamDict instance this value is returned else
                       the default value is returned (None if undefined)
        """
        # set function name
        _ = display_func('get', __NAME__, 'CaseInsensitiveDict')
        # capitalise string keys
        key = _capitalise_key(key)
        # if we have the key return the value
        if key in self.keys():
            return self.__getitem__(key)
        # else return the default key (None if not defined)
        else:
            return default

    def __capitalise_keys__(self):
        """
        Capitalizes all keys in ParamDict (used to make ParamDict case
        insensitive), only if keys entered are strings

        :return None:
        """
        # set function name
        _ = display_func('__capitalise_keys__', __NAME__, 'CaseInsensitiveDict')
        # make keys a list
        keys = list(self.keys())
        # loop around key in keys
        for key in keys:
            # check if key is a string
            if type(key) == str:
                # get value
                value = super(CaseInsensitiveDict, self).__getitem__(key)
                # delete old key
                super(CaseInsensitiveDict, self).__delitem__(key)
                # if it is a string set it to upper case
                key = key.upper()
                # set the new key
                super(CaseInsensitiveDict, self).__setitem__(key, value)


class ListCaseInsensitiveDict(CaseInsensitiveDict):
    def __getitem__(self, key: str) -> list:
        """
        Method used to get the value of an item using "key"
        used as x.__getitem__(y) <==> x[y]
        where key is case insensitive

        :param key: string, the key for the value returned (case insensitive)

        :type key: str

        :return value: list, the value stored at position "key"
        """
        # set function name
        _ = display_func('__getitem__', __NAME__, 'ListCaseInsensitiveDict')
        # return from supers dictionary storage
        # noinspection PyTypeChecker
        return list(super(ListCaseInsensitiveDict, self).__getitem__(key))

    def __setitem__(self, key: str, value: list, source: str = None):
        """
        Sets an item wrapper for self[key] = value
        :param key: string, the key to set for the parameter
        :param value: object, the object to set (as in dictionary) for the
                      parameter
        :param source: string, the source for the parameter

        :type key: str
        :type value: list
        :type source: str

        :return: None
        """
        # set function name
        _ = display_func('__setitem__', __NAME__, 'ListCaseInsensitiveDict')
        # then do the normal dictionary setting
        super(ListCaseInsensitiveDict, self).__setitem__(key, list(value))


class ParamDict(CaseInsensitiveDict):
    """
    Custom dictionary to retain source of a parameter (added via setSource,
    retreived via getSource). String keys are case insensitive.
    """
    def __init__(self, *arg, **kw):
        """
        Constructor for parameter dictionary, calls dict.__init__
        i.e. the same as running dict(*arg, *kw)

        :param arg: arguments passed to CaseInsensitiveDict
        :param kw: keyword arguments passed to CaseInsensitiveDict
        """
        # set function name
        _ = display_func('__init__', __NAME__, 'ParamDict')
        # storage for the sources
        self.sources = CaseInsensitiveDict()
        # storage for the source history
        self.source_history = ListCaseInsensitiveDict()
        # storage for the instances
        self.instances = CaseInsensitiveDict()
        # the print format
        self.pfmt = '\t{0:30s}{1:45s} # {2}'
        # the print format for list items
        self.pfmt_ns = '\t{1:45s}'
        # whether the parameter dictionary is locked for editing
        self.locked = False
        # define a log attachment
        self.log = None
        # run the super class (CaseInsensitiveDict <-- dict)
        super(ParamDict, self).__init__(*arg, **kw)

    def __getitem__(self, key: str) -> object:
        """
        Method used to get the value of an item using "key"
        used as x.__getitem__(y) <==> x[y]
        where key is case insensitive

        :param key: string, the key for the value returned (case insensitive)

        :type key: str
        :return value: object, the value stored at position "key"
        :raises ParamException: if key not found
        """
        # set function name
        func_name = display_func('__getitem__', __NAME__, 'ParamDict')
        # try to get item from super
        try:
            return super(ParamDict, self).__getitem__(key)
        except KeyError:
            # log that parameter was not found in parameter dictionary
            emsg = 'Parameter "{0}" not found in parameter dictionary'
            eargs = [key]
            raise ParamException(emsg.format(*eargs), 'get', self,
                                 funcname=func_name)

    def __setitem__(self, key: str, value: object,
                    source: Union[None, str] = None,
                    instance: Union[None, object] = None):
        """
        Sets an item wrapper for self[key] = value
        :param key: string, the key to set for the parameter
        :param value: object, the object to set (as in dictionary) for the
                      parameter
        :param source: string, the source for the parameter

        :type key: str
        :type source: Union[None, str]
        :type instance: Union[None, object]

        :return: None
        :raises ParamException: if parameter dictionary is locked
        """
        # set function name
        func_name = display_func('__setitem__', __NAME__, 'ParamDict')
        # deal with parameter dictionary being locked
        if self.locked:
            emsg = "ParamDict locked. \n\t Cannot add '{0}'='{1}'"
            eargs = [key, value]

            # log that parameter dictionary is locked so we cannot set key
            raise ParamException(emsg.format(*eargs), 'set', self,
                                 funcname=func_name)
        # if we dont have the key in sources set it regardless
        if key not in self.sources:
            self.sources[key] = source
            self.instances[key] = instance
        # if we do have the key only set it if source is not None
        elif source is not None:
            self.sources[key] = source
            self.instances[key] = instance
        # then do the normal dictionary setting
        super(ParamDict, self).__setitem__(key, value)

    def __contains__(self, key: str) -> bool:
        """
        Method to find whether ParamDict instance has key="key"
        used with the "in" operator
        if key exists in ParamDict True is returned else False is returned

        :param key: string, "key" to look for in ParamDict instance

        :return bool: True if ParamDict instance has a key "key", else False
        """
        # set function name
        _ = display_func('__contains__', __NAME__, 'ParamDict')
        # run contains command from super
        return super(ParamDict, self).__contains__(key)

    def __delitem__(self, key: str):
        """
        Deletes the "key" from ParamDict instance, case insensitive

        :param key: string, the key to delete from ParamDict instance,
                    case insensitive

        :return None:
        """
        # set function name
        _ = display_func('__delitem__', __NAME__, 'ParamDict')
        # delete item using super
        super(ParamDict, self).__delitem__(key)

    def __repr__(self):
        """
        Get the offical string representation for this instance
        :return: return the string representation

        :rtype: str
        """
        # set function name
        _ = display_func('__repr__', __NAME__, 'ParamDict')
        # get string from string print
        return self._string_print()

    def __str__(self) -> str:
        """
        Get the informal string representation for this instance
        :return: return the string representation

        :rtype: str
        """
        # set function name
        _ = display_func('__repr__', __NAME__, 'ParamDict')
        # get string from string print
        return self._string_print()

    def set(self, key: str, value: object,
            source: Union[None, str] = None,
            instance: Union[None, object] = None):
        """
        Set an item even if params is locked

        :param key: str, the key to set
        :param value: object, the value of the key to set
        :param source: str, the source of the value/key to set
        :param instance: object, the instance of the value/key to set

        :type key: str
        :type source: str
        :type instance: object

        :return: None
        """
        # set function name
        _ = display_func('set', __NAME__, 'ParamDict')
        # if we dont have the key in sources set it regardless
        if key not in self.sources:
            self.sources[key] = source
            self.instances[key] = instance
        # if we do have the key only set it if source is not None
        elif source is not None:
            self.sources[key] = source
            self.instances[key] = instance
        # then do the normal dictionary setting
        super(ParamDict, self).__setitem__(key, value)

    def lock(self):
        """
        Locks the parameter dictionary

        :return:
        """
        # set function name
        _ = display_func('lock', __NAME__, 'ParamDict')
        # set locked to True
        self.locked = True

    def unlock(self):
        """
        Unlocks the parameter dictionary

        :return:
        """
        # set function name
        _ = display_func('unlock', __NAME__, 'ParamDict')
        # set locked to False
        self.locked = False

    def get(self, key: str, default: Union[None, object] = None) -> object:
        """
        Overrides the dictionary get function
        If "key" is in ParamDict instance then returns this value, else
        returns "default" (if default returned source is set to None)
        key is case insensitive

        :param key: string, the key to search for in ParamDict instance
                    case insensitive
        :param default: object or None, if key not in ParamDict instance this
                        object is returned

        :type key: str

        :return value: if key in ParamDict instance this value is returned else
                       the default value is returned (None if undefined)
        """
        # set function name
        _ = display_func('get', __NAME__, 'ParamDict')
        # if we have the key return the value
        if key in self.keys():
            return self.__getitem__(key)
        # else return the default key (None if not defined)
        else:
            self.sources[key] = None
            return default

    def set_source(self, key: str, source: str):
        """
        Set a key to have sources[key] = source

        raises a ConfigError if key not found

        :param key: string, the main dictionary string
        :param source: string, the source to set

        :type key: str
        :type source: str

        :return None:
        :raises ConfigError: if key not found
        """
        # set function name
        func_name = display_func('set_source', __NAME__, 'ParamDict')
        # capitalise
        key = _capitalise_key(key)
        # don't put full path for sources in package
        source = _check_mod_source(source)
        # only add if key is in main dictionary
        if key in self.keys():
            self.sources[key] = source
            # add to history
            if key in self.source_history:
                self.source_history[key].append(source)
            else:
                self.source_history[key] = [source]
        else:
            # log error: source cannot be added for key
            emsg = (" Source cannot be added for key '{0}' "
                    "\n\t '{0}' is not in parameter dictionary")
            eargs = [key]
            raise ParamException(emsg.format(*eargs), 'set_source', self,
                                 funcname=func_name)

    def set_instance(self, key: str, instance: object):
        """
        Set a key to have instance[key] = instance

        raise a Config Error if key not found
        :param key: str, the key to add
        :param instance: object, the instance to store (normally Const/Keyword)

        :type key: str

        :return None:
        :raises ConfigError: if key not found
        """
        # set function name
        func_name = display_func('set_instance', __NAME__, 'ParamDict')
        # capitalise
        key = _capitalise_key(key)
        # only add if key is in main dictionary
        if key in self.keys():
            self.instances[key] = instance
        else:
            # log error: instance cannot be added for key
            emsg = ("Instance cannot be added for key '{0}' "
                    "\n\t '{0} is not in parameter dictionary")
            eargs = [key]
            raise ParamException(emsg.format(*eargs), 'set_instance', self,
                                 funcname=func_name)

    def append_source(self, key: str, source: str):
        """
        Adds source to the source of key (appends if exists)
        i.e. sources[key] = oldsource + source

        :param key: string, the main dictionary string
        :param source: string, the source to set

        :type key: str
        :type source: str

        :return None:
        """
        # set function name
        _ = display_func('append_source', __NAME__, 'ParamDict')
        # capitalise
        key = _capitalise_key(key)
        # if key exists append source to it
        if key in self.keys() and key in list(self.sources.keys()):
            self.sources[key] += ' {0}'.format(source)
        else:
            self.set_source(key, source)

    def set_sources(self, keys: List[str],
                    sources: Union[str, List[str], dict]):
        """
        Set a list of keys sources

        raises a ConfigError if key not found

        :param keys: list of strings, the list of keys to add sources for
        :param sources: string or list of strings or dictionary of strings,
                        the source or sources to add,
                        if a dictionary source = sources[key] for key = keys[i]
                        if list source = sources[i]  for keys[i]
                        if string all sources with these keys will = source

        :type keys: list
        :type sources: Union[str, list, dict]

        :return None:
        """
        # set function name
        _ = display_func('set_sources', __NAME__, 'ParamDict')
        # loop around each key in keys
        for k_it in range(len(keys)):
            # assign the key from k_it
            key = keys[k_it]
            # capitalise
            key = _capitalise_key(key)
            # Get source for this iteration
            if type(sources) == list:
                source = sources[k_it]
            elif type(sources) == dict:
                source = sources[key]
            else:
                source = str(sources)
            # set source
            self.set_source(key, source)

    def set_instances(self, keys: List[str],
                      instances: Union[object, list, dict]):
        """
        Set a list of keys sources

        raises a ConfigError if key not found

        :param keys: list of strings, the list of keys to add sources for
        :param instances: object or list of objects or dictionary of objects,
                        the source or sources to add,
                        if a dictionary source = sources[key] for key = keys[i]
                        if list source = sources[i]  for keys[i]
                        if object all sources with these keys will = source

        :type keys: list
        :type instances: Union[object, list, dict]

        :return None:
        """
        # set function name
        _ = display_func('set_instances', __NAME__, 'ParamDict')
        # loop around each key in keys
        for k_it in range(len(keys)):
            # assign the key from k_it
            key = keys[k_it]
            # capitalise
            key = _capitalise_key(key)
            # Get source for this iteration
            if type(instances) == list:
                instance = instances[k_it]
            elif type(instances) == dict:
                instance = instances[key]
            else:
                instance = instances
            # set source
            self.set_instance(key, instance)

    def append_sources(self, keys: List[str], sources: Union[str, List[str], dict]):
        """
        Adds list of keys sources (appends if exists)

        raises a ConfigError if key not found

        :param keys: list of strings, the list of keys to add sources for
        :param sources: string or list of strings or dictionary of strings,
                        the source or sources to add,
                        if a dictionary source = sources[key] for key = keys[i]
                        if list source = sources[i]  for keys[i]
                        if string all sources with these keys will = source

        :type keys: list
        :type sources: Union[str, List[str], dict]

        :return None:
        """
        # set function name
        _ = display_func('append_sources', __NAME__, 'ParamDict')
        # loop around each key in keys
        for k_it in range(len(keys)):
            # assign the key from k_it
            key = keys[k_it]
            # capitalise
            key = _capitalise_key(key)
            # Get source for this iteration
            if type(sources) == list:
                source = sources[k_it]
            elif type(sources) == dict:
                source = sources[key]
            else:
                source = str(sources)
            # append key
            self.append_source(key, source)

    def set_all_sources(self, source: str):
        """
        Set all keys in dictionary to this source

        :param source: string, all keys will be set to this source

        :type source: str

        :return None:
        """
        # set function name
        _ = display_func('set_all_sources', __NAME__, 'ParamDict')
        # loop around each key in keys
        for key in self.keys():
            # capitalise
            key = _capitalise_key(key)
            # set key
            self.sources[key] = source

    def append_all_sources(self, source: str):
        """
        Sets all sources to this "source" value

        :param source: string, the source to set

        :type source: str

        :return None:
        """
        # set function name
        _ = display_func('append_all_sources', __NAME__, 'ParamDict')
        # loop around each key in keys
        for key in self.keys():
            # capitalise
            key = _capitalise_key(key)
            # set key
            self.sources[key] += ' {0}'.format(source)

    def get_source(self, key: str) -> str:
        """
        Get a source from the parameter dictionary (must be set)

        raises a ConfigError if key not found

        :param key: string, the key to find (must be set)

        :return source: string, the source of the parameter
        """
        # set function name
        func_name = display_func('get_source', __NAME__, 'ParamDict')
        # capitalise
        key = _capitalise_key(key)
        # if key in keys and sources then return source
        if key in self.keys() and key in self.sources.keys():
            return str(self.sources[key])
        # else raise a Config Error
        else:
            # log error: no source set for key
            emsg = 'No source set for key={0}'
            eargs = [key]
            raise ParamException(emsg.format(*eargs), 'get_source', self,
                                 funcname=func_name)

    def get_instance(self, key: str) -> object:
        """
        Get a source from the parameter dictionary (must be set)

        raises a ConfigError if key not found

        :param key: string, the key to find (must be set)

        :return source: string, the source of the parameter
        """
        # set function name
        func_name = display_func('get_instance', __NAME__, 'ParamDict')
        # capitalise
        key = _capitalise_key(key)
        # if key in keys and sources then return source
        if key in self.keys() and key in self.instances.keys():
            return self.instances[key]
        # else raise a Config Error
        else:
            emsg = "No instance set for key={0}"
            eargs = [key]
            raise ParamException(emsg.format(*eargs), 'get_source', self,
                                 funcname=func_name)

    def source_keys(self) -> List[str]:
        """
        Get a dict_keys for the sources for this parameter dictionary
        order the same as self.keys()

        :return sources: values of sources dictionary
        """
        # set function name
        _ = display_func('source_keys', __NAME__, 'ParamDict')
        # return all keys in source dictionary
        return list(self.sources.keys())

    def source_values(self) -> List[object]:
        """
        Get a dict_values for the sources for this parameter dictionary
        order the same as self.keys()

        :return sources: values of sources dictionary
        """
        # set function name
        _ = display_func('source_values', __NAME__, 'ParamDict')
        # return all values in source dictionary
        return list(self.sources.values())

    def startswith(self, substring: str) -> List[str]:
        """
        Return all keys that start with this substring

        :param substring: string, the prefix that the keys start with

        :type substring: str

        :return keys: list of strings, the keys with this substring at the start
        """
        # set function name
        _ = display_func('startswith', __NAME__, 'ParamDict')
        # define return list
        return_keys = []
        # loop around keys
        for key in self.keys():
            # make sure key is string
            if type(key) != str:
                continue
            # if first
            if str(key).startswith(substring.upper()):
                return_keys.append(key)
        # return keys
        return return_keys

    def contains(self, substring: str) -> List[str]:
        """
        Return all keys that contain this substring

        :param substring: string, the sub-string to look for in all keys

        :type substring: str

        :return keys: list of strings, the keys which contain this substring
        """
        # set function name
        _ = display_func('contains', __NAME__, 'ParamDict')
        # define return list
        return_keys = []
        # loop around keys
        for key in self.keys():
            # make sure key is string
            if type(key) != str:
                continue
            # if first
            if substring.upper() in key:
                return_keys.append(key)
        # return keys
        return return_keys

    def endswith(self, substring: str) -> List[str]:
        """
        Return all keys that end with this substring

        :param substring: string, the suffix that the keys ends with

        :type substring: str

        :return keys: list of strings, the keys with this substring at the end
        """
        # set function name
        _ = display_func('endswith', __NAME__, 'ParamDict')
        # define return list
        return_keys = []
        # loop around keys
        for key in self.keys():
            # make sure key is string
            if type(key) != str:
                continue
            # if first
            if str(key).endswith(substring.upper()):
                return_keys.append(key)
        # return keys
        return return_keys

    def copy(self):
        """
        Copy a parameter dictionary (deep copy parameters)

        :return: the copy of the parameter dictionary
        :rtype: ParamDict
        """
        # set function name
        _ = display_func('copy', __NAME__, 'ParamDict')
        # make new copy of param dict
        pp = ParamDict()
        keys = list(self.keys())
        values = list(self.values())
        # loop around keys and add to new copy
        for k_it, key in enumerate(keys):
            value = values[k_it]
            # try to deep copy parameter
            if isinstance(value, ParamDict):
                pp[key] = value.copy()
            else:
                # noinspection PyBroadException
                try:
                    pp[key] = copy.deepcopy(value)
                except Exception as _:
                    pp[key] = type(value)(value)
            # copy source
            if key in self.sources:
                pp.set_source(key, str(self.sources[key]))
            else:
                pp.set_source(key, 'Unknown')
            # copy source history
            if key in self.source_history:
                pp.source_history[key] = list(self.source_history[key])
            else:
                pp.source_history[key] = []
            # copy instance
            if key in self.instances:
                pp.set_instance(key, self.instances[key])
            else:
                pp.set_instance(key, None)
        # return new param dict filled
        return pp

    def merge(self, paramdict, overwrite: bool = True):
        """
        Merge another parameter dictionary with this one

        :param paramdict: ParamDict, another parameter dictionary to merge
                          with this one
        :param overwrite: bool, if True (default) allows overwriting of
                          parameters, else skips ones already present

        :type paramdict: ParamDict
        :type overwrite: bool

        :return: None
        """
        # set function name
        _ = display_func('merge', __NAME__, 'ParamDict')
        # add param dict to self
        for key in paramdict:
            # deal with no overwriting
            if not overwrite and key in self.keys:
                continue
            # copy source
            if key in paramdict.sources:
                ksource = paramdict.sources[key]
            else:
                ksource = None
            # copy instance
            if key in paramdict.instances:
                kinst = paramdict.instances[key]
            else:
                kinst = None
            # add to self
            self.set(key, paramdict[key], ksource, kinst)

    def _string_print(self) -> str:
        """
        Constructs a string representation of the instance

        :return: a string representation of the instance
        :rtype: str
        """
        # set function name
        _ = display_func('_string_print', __NAME__, 'ParamDict')
        # get keys and values
        keys = list(self.keys())
        values = list(self.values())
        # string storage
        return_string = 'ParamDict:\n'
        strvalues = []
        # loop around each key in keys
        for k_it, key in enumerate(keys):
            # get this iterations values
            value = values[k_it]
            # deal with no source
            if key not in self.sources:
                self.sources[key] = 'None'
            # print value
            if type(value) in [list, np.ndarray]:
                sargs = [key, list(value), self.sources[key], self.pfmt]
                strvalues += _string_repr_list(*sargs)
            elif type(value) in [dict, OrderedDict, ParamDict]:
                strvalue = list(value.keys()).__repr__()[:40]
                sargs = [key + '[DICT]', strvalue, self.sources[key]]
                strvalues += [self.pfmt.format(*sargs)]
            else:
                strvalue = str(value)[:40]
                sargs = [key + ':', strvalue, self.sources[key]]
                strvalues += [self.pfmt.format(*sargs)]
        # combine list into single string
        for string_value in strvalues:
            return_string += '\n {0}'.format(string_value)
        # return string
        return return_string + '\n'

    def listp(self, key: str, separator: str = ',',
              dtype: Union[None, Type] = None) -> list:
        """
        Turn a string list parameter (separated with `separator`) into a list
        of objects (of data type `dtype`)

        i.e. ParamDict['MYPARAM'] = '1, 2, 3, 4'
        x = ParamDict.listp('my_parameter', dtype=int)
        gives:

        x = list([1, 2, 3, 4])

        :param key: str, the key that contains a string list
        :param separator: str, the character that separates
        :param dtype: type, the type to cast the list element to

        :return: the list of values extracted from the string for `key`
        :rtype: list
        """
        # set function name
        func_name = display_func('listp', __NAME__, 'ParamDict')
        # if key is present attempt str-->list
        if key in self.keys():
            item = self.__getitem__(key)
        else:
            # log error: parameter not found in parameter dict (via listp)
            emsg = ("parameter '{0}' not found in parameter dictionary "
                    "(via listp)")
            eargs = [key]
            raise ParamException(emsg.format(*eargs), 'listp', self,
                                 funcname=func_name)
        # convert string
        if key in self.keys() and isinstance(item, str):
            return _map_listparameter(str(item), separator=separator,
                                      dtype=dtype)
        elif isinstance(item, list):
            return item
        else:
            # log error: parameter not found in parameter dict (via listp)
            emsg = ("parameter '{0}' must be a string to convert "
                    "(via listp)")
            eargs = [key]
            raise ParamException(emsg.format(*eargs), 'listp', self,
                                 funcname=func_name)

    def dictp(self, key: str, dtype: Union[str, None] = None) -> dict:
        """
        Turn a string dictionary parameter into a python dictionary
        of objects (of data type `dtype`)

        i.e. ParamDict['MYPARAM'] = '{"varA":1, "varB":2}'
        x = ParamDict.listp('my_parameter', dtype=int)
        gives:

        x = dict(varA=1, varB=2)

        Note string dictionary must be in the {"key":value} format

        :param key: str, the key that contains a string list
        :param dtype: type, the type to cast the list element to

        :return: the list of values extracted from the string for `key`
        :rtype: dict
        """
        # set function name
        func_name = display_func('dictp', __NAME__, 'ParamDict')
        # if key is present attempt str-->dict
        if key in self.keys():
            item = self.__getitem__(key)
        else:
            # log error: parameter not found in parameter dict (via listp)
            emsg = ("parameter '{0}' not found in parameter dictionary "
                    "(via dictp)")
            eargs = [key]
            raise ParamException(emsg.format(*eargs), 'dictp', self,
                                 funcname=func_name)
        # convert string
        if isinstance(item, str):
            return _map_dictparameter(str(item), dtype=dtype)
        elif isinstance(item, dict):
            return item
        else:
            emsg = ("parameter '{0}' must be a string to convert "
                    "(via dictp)")
            eargs = [key]
            raise ParamException(emsg.format(*eargs), 'dictp', self,
                                 funcname=func_name)

    def get_instanceof(self, lookup: object, nameattr: str = 'name') -> dict:
        """
        Get all instances of object instance lookup

        i.e. perform isinstance(object, lookup)

        :param lookup: object, the instance to lookup
        :param nameattr: str, the attribute in instance that we will return
                         as the key

        :return: a dictionary of keys/value pairs where each value is an
                 instance that belongs to instance of `lookup`
        :rtype: dict
        """
        # set function name
        _ = display_func('get_instanceof', __NAME__, 'ParamDict')
        # output storage
        keydict = dict()
        # loop around all keys
        for key in list(self.instances.keys()):
            # get the instance for this key
            instance = self.instances[key]
            # skip None
            if instance is None:
                continue
            # else check instance type
            if isinstance(instance, type(lookup)):
                if hasattr(instance, nameattr):
                    name = getattr(instance, nameattr)
                    keydict[name] = instance
            else:
                continue
        # return keyworddict
        return keydict

    def info(self, key: str):
        """
        Display the information related to a specific key

        :param key: str, the key to display information about

        :type key: str

        :return: None
        """
        # set function name
        _ = display_func('info', __NAME__, 'ParamDict')
        # deal with key not existing
        if key not in self.keys():
            print("ParamDict Info: Key = '{0}' not found".format(key))
            return
        # print key title
        print("Information for key = '{0}'".format(key))
        # print value stats
        value = self.__getitem__(key)
        # print the data type
        print("\tData Type: \t\t {0}".format(type(value).__name__))
        # deal with lists and numpy array
        if isinstance(value, (list, np.ndarray)):
            sargs = [key, list(value), None, self.pfmt_ns]
            wargs = [np.nanmin(value), np.nanmax(value),
                     np.sum(np.isnan(value)) > 0, _string_repr_list(*sargs)]
            emsg = ("\tMin Value: \t\t {0} \n\tMax Value: \t\t {1} "
                    "\n\t Has NaNs: \t\t {2} \n\t Values: \t\t {3}")
            print(emsg.format(*wargs))
        # deal with dictionaries
        elif isinstance(value, (dict, OrderedDict, ParamDict)):
            strvalue = list(value.keys()).__repr__()[:40]
            sargs = [key + '[DICT]', strvalue, None]
            wargs = [len(list(value.keys())), self.pfmt_ns.format(*sargs)]
            print("\t Num Keys: \t\t {0} \n\t Values: \t\t {1}".format(*wargs))
        # deal with everything else
        else:
            strvalue = str(value)[:40]
            sargs = [key + ':', strvalue, None]
            wargs = [self.pfmt_ns.format(*sargs)]
            print("\tValue: \t\t {0}".format(*wargs))
        # add source info
        if key in self.sources:
            print("\tSource: \t\t {0}".format(self.sources[key]))
        # add instances info
        if key in self.instances:
            print("\tInstance: \t\t {0}".format(self.instances[key]))

    def history(self, key: str):
        """
        Display the history of where key was defined (using source)

        :param key: str, the key to print history of

        :type key: str

        :return: None
        """
        # set function name
        _ = display_func('history', __NAME__, 'ParamDict')
        # if history found then print it
        if key in self.source_history:
            # print title: History for key
            print("History for key = '{0}'".format(key))
            # loop around history and print row by row
            for it, entry in enumerate(self.source_history[key]):
                print('{0}: {1}'.format(it + 1, entry))
        # else display that there was not history found
        else:
            print("No history found for key='{0}'".format(key))


# =============================================================================
# Define functions
# =============================================================================
def setup(lconsts: constants.Consts, kwargs: dict, desc: str,
          log: Union[None, log_functions.Log] = None,
          quiet: bool = False, name: Union[None, str] = None) -> ParamDict:
    """
    Setup the code

    Order of priority (lowest to highest)
    - constants file (lconsts.constants)
    - config file (USER_CONFIG_FILE)
    - call to main (kwargs)
    - command line arguments (sys.argv)

    :param lconsts: Constants instance
    :param kwargs: dictionary of arguments from function call
    :param desc: str, the description of the input code
    :param log: None or log instance, the log instance to attach
    :param quiet: bool, if True does not print header

    :return:
    """
    # get parameters from constants
    params = ParamDict()
    # ----------------------------------------------------------------------
    # attach log
    if log is None:
        params.log = log_functions.Log()
    else:
        params.log = log
    # ----------------------------------------------------------------------
    # deal with name (remove .py from end)
    if name is not None:
        name = name.split('.py')[0]
    # ----------------------------------------------------------------------
    # Lowest priority: params from constants file (lconsts.constants)
    # ----------------------------------------------------------------------
    # get debug from the command line
    params = _quick_look_args(params, kwargs)
    # set function name (after param + log definition)
    func_name = display_func('setup', __NAME__, params=params)
    # ----------------------------------------------------------------------
    # Lowest priority: params from constants file (lconsts.constants)
    # ----------------------------------------------------------------------
    # debug print out
    params.log.ldebug('Lowest priority: params from constants', 7)

    # loop around constants
    for cname in lconsts.constants:
        # debug print out
        params.log.ldebug('Processing parameter: {0}'.format(cname), 8)
        # get constant
        constant = lconsts.constants[cname]
        # copy constant into parameters
        params[cname] = constant.value
        # set source and instance
        params.set_source(cname, constant.source)
        params.set_instance(cname, constant.copy())
    # ----------------------------------------------------------------------
    # read arguments from cmdline (need to update user_config_file)
    args = _read_from_cmdline(params, desc)
    # ----------------------------------------------------------------------
    # next priority: params from config file
    # ----------------------------------------------------------------------
    params = _read_from_config_file(params, args)
    # ----------------------------------------------------------------------
    # next priority: params from call to main (kwargs)
    # ----------------------------------------------------------------------
    params = _read_from_kwargs(params, kwargs)
    # ----------------------------------------------------------------------
    # next priority: params from command line arguments (sys.argv)
    # ----------------------------------------------------------------------
    params = _update_from_cmdline(params, args)
    # ----------------------------------------------------------------------
    # recheck the quick look args (this time just to update now params is
    # updated
    params = _quick_look_args(params, kwargs, check_only=True)
    # ----------------------------------------------------------------------
    # Set up a unique name (for logging etc)
    # ----------------------------------------------------------------------
    # set up process id
    pid, htime = _assign_pid()
    params.set('PID', value=pid, source=func_name)
    params.set('PIDTIME', value=htime, source=func_name)

    # ----------------------------------------------------------------------
    # Now we need to check/create some paths
    # ----------------------------------------------------------------------
    params = _setup_working_directory(params, name)

    # add a log file
    params.log.add_log_file(params['LOGFILE'])

    # ----------------------------------------------------------------------
    # print header
    # ----------------------------------------------------------------------
    _splash(params, quiet=quiet, name=name)

    # ----------------------------------------------------------------------
    # deal with config file generation
    # ----------------------------------------------------------------------
    if params['GENERATE_CONFIG_FILE']:
        _generate_config_file(params)

    # return the parameter dictionary
    return params


# =============================================================================
# Other private functions
# =============================================================================
# capitalisation function (for case insensitive keys)
def _capitalise_key(key: str) -> str:
    """
    Capitalizes "key" (used to make ParamDict case insensitive), only if
    key is a string

    :param key: string or object, if string then key is capitalized else
                nothing is done

    :return key: capitalized string (or unchanged object)
    """
    # set function name (cannot break here --> no access to inputs)
    _ = display_func('_capitalise_key', __NAME__)
    # capitalise string keys
    if type(key) == str:
        key = key.upper()
    return key


def _check_mod_source(source: str) -> str:
    """
    Check for a package path in a source and remove it

    :param source: str, the source name

    :return:
    """
    # set function name (cannot break here --> no access to inputs)
    _ = display_func('_check_mod_source', __NAME__)
    # if source doesn't exist also skip
    if not Path(source).exists():
        return source
    # get package path
    package_path = general.get_package_directory(PACKAGE, '')
    # if package path not in source then skip
    if str(package_path) not in source:
        return source
    # remove package path and replace with PACKAGE
    source = source.replace(str(package_path), PACKAGE.lower())
    # replace separators with .
    source = source.replace(os.sep, '.')
    # remove double dots
    while '..' in source:
        source = source.replace('..', '.')
    # return edited source
    return source


def _string_repr_list(key: str, values: Union[list, np.ndarray], source: str,
                      fmt: str) -> List[str]:
    """
    Represent a list (or array) as a string list but only the first
    40 charactersay

    :param key: str, the key the list (values) came from
    :param values: vector, the list or numpy array to print as a string
    :param source: str, the source where the values were defined
    :param fmt: str, the format for the printed list
    :return:
    """
    # set function name (cannot break here --> no access to inputs)
    _ = display_func('_load_from_file', __NAME__)
    # get the list as a string
    str_value = list(values).__repr__()
    # if the string is longer than 40 characters cut down and add ...
    if len(str_value) > 40:
        str_value = str_value[:40] + '...'
    # return the string as a list entry
    return [fmt.format(key, str_value, source)]


def _map_listparameter(value: Union[str, list], separator: str = ',',
                       dtype: Union[None, Type] = None) -> List[object]:
    """
    Map a string list into a python list

    :param value: str or list, if list returns if string tries to evaluate
    :param separator: str, where to split the str at to make a list
    :param dtype: type, if set forces elements of list to this data type
    :return:
    """
    # set function name (cannot break here --> no access to inputs)
    func_name = display_func('_map_listparameter', __NAME__)
    # return list if already a list
    if isinstance(value, (list, np.ndarray)):
        return list(value)
    # try evaluating is a list
    # noinspection PyBroadException
    try:
        # evulate value
        rawvalue = eval(value)
        # if it is a list return as a list
        if isinstance(rawvalue, list):
            return list(rawvalue)
    # if it is not pass
    except Exception as _:
        pass
    # deal with an empty value i.e. ''
    if value == '':
        return []
    # try to return dtyped data
    try:
        # first split by separator
        listparameter = value.split(separator)

        # return the stripped down values
        if dtype is not None and isinstance(dtype, type):
            return list(map(lambda x: dtype(x.strip()), listparameter))
        else:
            return list(map(lambda x: x.strip(), listparameter))
    except Exception as e:
        emsg = ("Parameter '{0}' can not be converted to a list. "
                "\n\t Error {1}: {2}. "
                "\n\t function = {3}")
        eargs = [value, type(e), e, func_name]
        raise ParamException(emsg.format(*eargs), 'listp', funcname=func_name,
                             exception=e)


def _map_dictparameter(value: str, dtype: Union[None, Type] = None) -> dict:
    """
    Map a string dictionary into a python dictionary

    :param value: str, tries to evaluate string into a dictionary
                  i.e. "dict(a=1, b=2)"  or {'a':1, 'b': 2}

    :param dtype: type, if set forces elements of list to this data type
    :return:
    """
    # set function name (cannot break here --> no access to inputs)
    func_name = display_func('_map_dictparameter', __NAME__)
    # deal with an empty value i.e. ''
    if value == '':
        return dict()
    # try evaulating as a dict
    try:
        rawvalue = eval(value)
        if isinstance(rawvalue, dict):
            returndict = dict()
            for key in rawvalue.keys():
                if dtype is not None and isinstance(dtype, type):
                    returndict[key] = dtype(rawvalue[key])
                else:
                    returndict[key] = rawvalue[key]
            return returndict
    except Exception as e:
        emsg = ("Parameter '{0}' can not be converted to a dictionary. "
                "\n\t Error {1}: {2}. \n\t function = {3}")
        eargs = [value, type(e), e, func_name]
        raise ParamException(emsg.format(*eargs), 'dictp', funcname=func_name,
                             exception=e)


def _read_from_config_file(params: ParamDict, args: argparse.Namespace,
                           configfile: Union[None, str] = None) -> ParamDict:
    """
    Read a config file

    :param params: ParamDict, the parameter dictionary for constants
    :param args: argparse.Namespace - the argparse namespace attribute holder
    :param configfile: str or None, if set, sets the config file

    :return:
    """
    # set function name
    func_name = display_func('_read_from_config_file', __NAME__)
    # get user config file and out dir from args
    cmd_userconfig = getattr(args, 'USER_CONFIG_FILE', None)
    cmd_outdir = getattr(args, 'DIRECTORY', None)
    # get from params
    package_dir = str(params['PACKAGE_DIRECTORY'])
    # ----------------------------------------------------------------------
    # if we have a config file defined use it
    if configfile is not None:
        user_config_file = str(configfile)
        params.set('USER_CONFIG_FILE', user_config_file, source=func_name)
    # if we have a command line argument use it
    elif cmd_userconfig is not None:
        user_config_file = cmd_userconfig
        params['USER_CONFIG_FILE'] = user_config_file
        params.set_source('USER_CONFIG_FILE', 'sys.argv')
        params.append_source('USER_CONFIG_FILE', func_name)
    # get the config file
    elif params['USER_CONFIG_FILE'] is not None:
        user_config_file = params['USER_CONFIG_FILE']
    else:
        user_config_file = None
    # ----------------------------------------------------------------------
    # get the output directory
    if cmd_outdir is not None:
        workingdir = Path(cmd_outdir)
        params['DIRECTORY'] = cmd_outdir
        params.set_source('DIRECTORY', 'sys.argv')
        params.append_source('DIRECTORY', func_name)
    # else if it is already in params
    elif params['DIRECTORY'] is not None:
        workingdir = Path(str(params['DIRECTORY']))
    elif AMIDIR in os.environ:
        workingdir = Path(os.environ[AMIDIR])
    else:
        workingdir = Path.home().joinpath(package_dir)
        params['DIRECTORY'] = str(workingdir)
        params.set_source('DIRECTORY', func_name)
    # ----------------------------------------------------------------------
    # at this stage we need to test directory and config directory exist
    if not workingdir.exists():
        try:
            os.makedirs(workingdir)
        except Exception as e:
            emsg = 'Cannot make working directory: {0}'.format(workingdir)
            emsg += '\nError {0}: {1}'.format(e.__name__, e)
            raise exceptions.ConstantException(emsg, kind='workingdir')
    # now make config sub-directory
    configdir = workingdir.joinpath('config')
    if not configdir.exists():
        configdir.mkdir()
    # set config directory
    params['CONFIGDIR'] = configdir
    params.set_source('CONFIGDIR', func_name)
    # ----------------------------------------------------------------------
    # deal with no user_config_file
    if user_config_file is None:
        return params
    # ----------------------------------------------------------------------
    # if user_config_file exists copy it to config dif
    if Path(user_config_file).exists():
        # construct new path for user config file
        outpath = Path(user_config_file).name
    else:
        # else the config is assumed to be in the config path
        outpath = configdir.joinpath(Path(user_config_file).name)

    # now test whether outpath exists (if it doesn't then we just return, no
    #   config is to be used)
    if not outpath.exists():
        return params
    # ----------------------------------------------------------------------
    # read constants file to directory
    keys, values = np.loadtxt(outpath, delimiter='=', unpack=True, comments='#',
                              dtype=str)
    keys = np.char.array(keys).strip().upper()
    values = np.char.array(values).strip()
    # push into dictionary
    configdict = dict(zip(keys, values))
    # loop around constants
    for cname in params:
        # get constant
        constant = params.instances[cname]
        # only update parameters with constants instance
        if isinstance(constant, constants.constant_functions.Constant):
            # only add constatns that have user=True
            if constant.user:
                # check if we have it in config dictionary
                if cname in configdict.keys():
                    # get config dictionary value
                    configvalue = configdict[cname]
                    # deal with a None value
                    if configvalue in [None, 'None', '']:
                        configvalue = None
                    # update value
                    constant.value = configvalue
                    # check values
                    constant.check_value()
                    # now add to params
                    params[cname] = constant.value
                    params.set_source(cname, str(outpath))
    # return params
    return params


def _read_from_kwargs(params: ParamDict, kwargs: dict) -> ParamDict:
    """
    Read arguments from kwargs (normally from call to main function) and
    push them into params

    :param params: ParamDict, the parameter dictionary of constants
    :param kwargs: dict, the keyword dictionary from call

    :return: the updated parameter dictionary
    """
    # set function name
    _ = display_func('_read_from_kwargs', __NAME__)
    # convert kwargs to case insensitive dictionary
    kwargs = CaseInsensitiveDict(kwargs)
    # loop around keyword arguments
    for cname in params:
        # get constant
        constant = params.instances[cname]
        # only update parameters with constants instance
        if isinstance(constant, constants.constant_functions.Constant):
            # only add constatns that have user=True
            if constant.user:
                # check if we have it in config dictionary
                if cname in kwargs:
                    # get config dictionary value
                    configvalue = kwargs[cname]
                    # deal with a None value
                    if configvalue in [None, 'None', '']:
                        configvalue = None
                    # update value
                    constant.value = configvalue
                    # check values
                    constant.check_value()
                    # now add to params
                    params[cname] = constant.value
                    params.set_source(cname, 'kwargs')
    # return params
    return params


def _quick_look_args(params: ParamDict, kwargs: dict,
                     check_only: bool = False) -> ParamDict:
    """
    Quick look arguments: Arguments required as soon as possible
    (i.e. debug mode) look in sys.argv and kwargs

    :param params:
    :param kwargs:
    :return:
    """
    # set function name (cannot use params here as debug is not set)
    func_name = display_func('_quick_look_args', __NAME__)
    # ----------------------------------------------------------------------
    # Debug
    # ----------------------------------------------------------------------
    # assume debug = 0 initially
    debug = params.get('DEBUG', 0)
    # only do this to check
    if not check_only:
        # check sys.argv
        for it, arg in enumerate(sys.argv):
            if arg.startswith('--debug='):
                try:
                    debug = int(arg.split('--debug=')[-1])
                except Exception as _:
                    pass
            elif arg.startswith('--debug') and it != len(sys.argv):
                try:
                    debug = int(sys.argv[it + 1])
                except Exception as _:
                    pass

        # check for debug in kwargs
        if 'DEBUG' in kwargs:
            try:
                debug = int(kwargs['DEBUG'])
            except Exception as _:
                pass
        # set debug in params
        params.set('DEBUG', debug, source=func_name)
    # set debug mode in log
    if debug > 0:
        params.log.set_lvl_debug(debug)
    # ----------------------------------------------------------------------
    # assume theme is dark initially
    theme = params.get('PACKAGE_THEME', 'DARK')
    # only do this to check
    if not check_only:
        # check sys.argv
        for it, arg in enumerate(sys.argv):
            if arg.startswith('--theme='):
                try:
                    theme = str(arg.split('--theme=')[-1])
                except Exception as _:
                    pass
            elif arg.startswith('--theme') and it != len(sys.argv):
                try:
                    theme = str(sys.argv[it + 1])
                except Exception as _:
                    pass
        # check for debug in kwargs
        if 'THEME' in kwargs:
            try:
                theme = int(kwargs['THEME'])
            except Exception as _:
                pass
    # update theme
    if params.log.theme != theme:
        params.log.update_theme(theme)
    # ----------------------------------------------------------------------
    # return params
    return params


def _read_from_cmdline(params: ParamDict,
                       description: str) -> argparse.Namespace:
    """
    Read the arguments from the command line (via argparse)

    :param params: ParamDict, the parameter dictionary of constants
    :param description: str, the description for the help page

    :return: the argparse argument holder
    """
    # set function name
    func_name = display_func('_read_from_cmdline', __NAME__, params=params)
    # basic check that sys.argv passed correctly
    if isinstance(sys.argv, str):
        emsg = 'sys.argv must be a list not a string (hint: use .split())'
        raise exceptions.ConstantException(emsg, kind='log', funcname=func_name)

    # print progress
    logging.debug('Reading args from command line:')
    logging.debug('\t sys.argv: {0}'.format(' '.join(sys.argv)))
    # set up parser
    parser = argparse.ArgumentParser(description=description)
    # loop around keyword arguments
    for cname in params:
        # get constant
        constant = params.instances[cname]
        # only update parameters with constants instance
        if isinstance(constant, constants.constant_functions.Constant):
            # only add constants that have argument=True
            if constant.argument:
                # deal with no command set
                if constant.command is None:
                    # generate error
                    emsg = ('Constant {0} must have a "command" defined (as '
                            '"argument=True")')
                    exceptions.ConstantException(emsg.format(cname),
                                                 kind='command')
                # deal with casting data type
                if constant.dtype in [int, float, str]:
                    dtype = constant.dtype
                else:
                    dtype = None
                # add argument
                parser.add_argument(*constant.command, type=dtype,
                                    action='store', default=None,
                                    dest=cname, help=constant.description)
    # parse arguments
    args = parser.parse_args()
    # return parser arguments
    return args


def _update_from_cmdline(params: ParamDict,
                         args: argparse.Namespace) -> ParamDict:
    """
    Update arguments set from command line (sys.argv)

    :param params:
    :param args:
    :return:
    """
    # set function name
    _ = display_func('_update_from_cmdline', __NAME__,
                             params=params)
    # ----------------------------------------------------------------------
    # now loop around and add to params
    # loop around keyword arguments
    for cname in params:
        # get constant
        constant = params.instances[cname]
        # only update parameters with constants instance
        if isinstance(constant, constants.constant_functions.Constant):
            # only add constant that have argument=True
            if constant.argument:
                # check if we have it in config dictionary
                if hasattr(args, cname):
                    # get config dictionary value
                    configvalue = getattr(args, cname, None)
                    # deal with a None value
                    if configvalue in [None, 'None', '']:
                        continue
                    # update value
                    constant.value = configvalue
                    # check values
                    constant.check_value()
                    # now add to params
                    params[cname] = constant.value
                    params.set_source(cname, 'sys.argv')
    # return params
    return params


def _setup_working_directory(params: ParamDict,
                             name: Union[None, str] = None) -> ParamDict:
    """
    Setup the working directory based on the state of params
    :param params:
    :return:
    """
    # set function name
    func_name = display_func('_setup_working_directory', __NAME__,
                             params=params)
    # get parameters from params
    package_name = str(params['PACKAGE_NAME'])
    package_dir = str(params['PACKAGE_DIRECTORY'])
    workingdir = str(params['DIRECTORY'])
    # default default working directory path
    if AMIDIR in os.environ:
        default_dir = Path(os.environ[AMIDIR])
    else:
        default_dir = Path.home().joinpath(package_name, package_dir)
    # ----------------------------------------------------------------------
    # at this point it should not be None but check again any way
    if workingdir in ['', 'None']:
        workingdir = Path(default_dir)
    # else we try to make this directroy
    else:
        try:
            workingdir = Path(workingdir)
        # if we cannot create it go with default
        except Exception as e:
            msg = 'Invalid working dir. Setting to default'
            msg += '\n\t{0}: {1}'.format(type(e), str(e))
            msg += '\n\tDefault = {0}'.format(workingdir)
            params.log.info(msg)
            workingdir = Path(default_dir)

    # ----------------------------------------------------------------------
    # now check if path exists
    while not workingdir.exists():
        try:
            workingdir.mkdir()
        except Exception as e:
            # if working directory is the default working dir we have to stop
            #   here
            if workingdir == default_dir:
                emsg = 'Cannot create default working dir: {0}'
                raise exceptions.ConstantException(emsg.format(default_dir),
                                                   kind='mkdir')
            # else set the working directory to default
            msg = 'Cannot create working dir. Setting to default'
            msg += '\n\t{0}: {1}'.format(type(e), str(e))
            msg += '\n\tDefault = {0}'.format(workingdir)
            params.log.info(msg)
            workingdir = Path(default_dir)

    # ----------------------------------------------------------------------
    # now we have the working dir we can set some paths
    inputdir = Path(workingdir).joinpath('inputs')
    outputdir = Path(workingdir).joinpath('outputs')
    logdir = Path(workingdir).joinpath('log')
    # construct log file name
    if name is not None:
        logfile = 'LOG-{0}-{1}.log'.format(name, str(params['PID']))
    else:
        logfile = '{0}.log'.format(str(params['PID']))
    # add to params
    params.set('INPUTDIR', inputdir, source=func_name)
    params.set('OUTPUTDIR', outputdir, source=func_name)
    params.set('LOGDIR', logdir, source=func_name)
    # set the log file name
    params.set('LOGFILE', logdir.joinpath(logfile), source=func_name)
    # ----------------------------------------------------------------------
    # now check that these paths exist and if not create them
    values = [inputdir, outputdir, logdir]
    names = ['input', 'output', 'log']
    # loop around values
    for it in range(len(values)):
        # get this iterations values
        value = values[it]
        name = names[it]
        # check input directory
        if not value.exists():
            try:
                value.mkdir()
            except Exception as e:
                emsg = 'Cannot create {0} dir: {1}'.format(name, value)
                emsg += '\nError {0}: {1}'.format(e.__name__, e)
                kind = '{0}dir'.format(name)
                raise exceptions.ConstantException(emsg, kind=kind)
    # ----------------------------------------------------------------------
    return params


def _assign_pid():
    """
    Assign a process id based on the time now and return it and the
    time now

    :return: the process id and the human time at creation
    :rtype: tuple[str, str]
    """
    # get unix char code
    unixtime, humantime, rval = unix_char_code()
    # write pid
    pid = 'PID-{0:020d}-{1}'.format(int(unixtime), rval)
    # return pid and human time
    return pid, humantime


def unix_char_code():
    # we need a random seed
    np.random.seed(random.randint(1, 2**30))
    # generate a random number (in case time is too similar)
    #  -- happens a lot in multiprocessing
    rint = np.random.randint(1000, 9999, 1) / 1e7
    # wait a fraction of time (between 1us and 1ms)
    time.sleep(rint)
    # get the time now from astropy
    timenow = Time.now()
    # get unix and human time from astropy time now
    unixtime = timenow.unix * 1e7
    humantime = timenow.iso
    # generate random four characters to make sure pid is unique
    rval = ''.join(np.random.choice(list(CHARS), size=4))
    return unixtime, humantime, rval


def _splash(params: ParamDict, quiet: bool = False,
            name: Union[str, None] = None):
    # if quiet do nothing
    if quiet:
        return
    # else print a header
    package_name = params['PACKAGE_NAME']
    package_version = params['PACKAGE_VERSION']
    # set up text
    headertext = '\t{0}\tVersion: {1}'
    # push to log
    params.log.header()
    params.log.empty(headertext.format(package_name, package_version), 'g')
    params.log.header()
    # deal with adding the code name (if set)
    if name is not None:
        nametext = '\tCode: {0}'
        params.log.empty(nametext.format(name), 'b')
        params.log.header()

def _generate_config_file(params: ParamDict):
    """
    Write a config file based on lconsts

    :param lconsts:
    :return:
    """
    # set function name
    func_name = display_func('_setup_working_directory', __NAME__,
                             params=params)
    # get parameters from params
    user_config_file = str(params['USER_CONFIG_FILE'])
    # deal with unset user config file
    if params['USER_CONFIG_FILE'] in ['', 'None']:
        user_config_file = 'user_config.ini'
    # get the output directory
    if Path(user_config_file).exists():
        outdir = Path(user_config_file)
    else:
        outdir = Path(params['CONFIGDIR']).joinpath(Path(user_config_file).name)
    # ----------------------------------------------------------------------
    # set up lines for adding to constants file
    lines = CONFIG_FILE_HEADER.split('\n')
    # used groups
    used_groups = []
    # loop around constants
    for cname in params:
        # get constant
        constant = params.instances[cname]
        # only update parameters with constants instance
        if isinstance(constant, constants.constant_functions.Constant):
            # only add constants that have user=True
            if constant.user:
                # --------------------------------------------------------------
                # deal with adding group section
                if constant.group not in used_groups:
                    line = CONFIG_GROUP_HEADER.format(constant.group)
                    lines += line.split('\n')
                    # add group to used groups
                    used_groups.append(constant.group)
                # --------------------------------------------------------------
                # get description lines
                dlines = _wraptext(constant.description)
                # loop around description lines and add to lines as comments
                for dline in dlines:
                    lines.append('# ' + dline)
                # --------------------------------------------------------------
                # then add the line NAME = VALUE
                lines.append('{0} = {1}'.format(constant.name, constant.value))
                lines.append('')
    # ----------------------------------------------------------------------
    # construct out path
    params.log.info('Writing constants file to {0}'.format(outdir))
    # write constants file to directory
    with open(outdir, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def _wraptext(text: str, length: int = 78) -> List[str]:
    """
    Wrap long text into several lines of text is smart enough to wrap
    words that are too long to a new line

    :param text: str, the long text to wrap
    :param length: int, the number of characters to wrap at

    :return: the list of strings each of less than "length" characters
    """
    lines = []
    buffer = ''
    # split all words in the text
    words = text.split(' ')
    # loop around words
    for word in words:
        # offload buffer to lines
        if len(buffer) + len(word) >= length:
            lines.append(buffer)
            buffer = ''
        # else add to the buffer
        else:
            buffer += '{0} '.format(word)
    # add the last buffer
    lines.append(buffer)
    # return the list of strings
    return lines

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
