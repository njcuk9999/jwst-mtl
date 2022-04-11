#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-06

@author: cook
"""
from astropy.table import Table
from collections import UserDict
from copy import deepcopy
from typing import Any, Dict, Type, Union

from soss_tfit.core import base

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'base_classes.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define classes
# =============================================================================
class TransitFitExcept(Exception):
    def __init__(self, message):
        """
        Constructor for LBL Exception
        :param message: str, the message to add to log + raise msg
        """
        self.message = message

    def __str__(self) -> str:
        """
        String representation of the Exception

        :return: return string representation
        """
        return self.message


class Const:
    def __init__(self, key: str, source: Union[str, None] = None,
                 desc: Union[str, None] = None,
                 arg: Union[str, None] = None,
                 dtype: Union[Type, None] = None,
                 options: Union[list, None] = None,
                 path: Union[str] = None, comment: Union[str, None] = None):
        """
        Constant class (for storing properties of constants)

        :param key: str, the key to set in dictionary
        :param source: str or None, if set the source of the parameter
        :param desc: str or None, if set the description of the parameter
        :param arg: str or None, the command argument
        :param dtype: Type or None, the type of object (for argparse) only
                      required if arg is set
        :param options: list or None, the options (choices) to allow for
                argparse
        :param comment: str or None, if set this is the comment to add to a
                        fits header
        """
        self.key = deepcopy(key)
        # set source
        self.source = deepcopy(source)
        # set description
        self.description = deepcopy(desc)
        # set arg (for run time argument)
        self.argument = deepcopy(arg)
        # set the dtype
        self.dtype = dtype
        # the allowed options for argparse (choices)
        self.options = deepcopy(options)
        # the path associated with a yaml file
        self.path = str(path)
        # the comment for a fits header
        self.comment = deepcopy(comment)

    def __str__(self) -> str:
        return 'Const[{0}]'.format(self.key)

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'Const':
        """
        Deep copy a Const

        :return: new Const instance
        """
        return Const(self.key, self.source, self.description,
                     self.argument, self.dtype, self.options,
                     self.path, self.comment)

    def update(self, key: str, source: Union[str, None] = None,
               desc: Union[str, None] = None,
               arg: Union[str, None] = None,
               dtype: Union[Type, None] = None,
               options: Union[list, None] = None,
               path: Union[str] = None,
               comment: Union[str, None] = None) -> 'Const':
        """
        Update a constant class - if value is None do not update

        :param key: str, the key to set in dictionary
        :param source: str or None, if set the source of the parameter
        :param desc: str or None, if set the description of the parameter
        :param arg: str or None, the command argument
        :param dtype: Type or None, the type of object (for argparse) only
                      required if arg is set
        :param options: list or None, the options (choices) to allow for
                argparse
        :param comment: str or None, if set this is the comment to add to a
                        fits header
        """
        if key is None:
            key = self.key
        # update source
        if source is None:
            source = self.source
        # set description
        if desc is None:
            desc = self.description
        # set arg (for run time argument)
        if arg is None:
            arg = self.argument
        # set the dtype
        if dtype is None:
            dtype = self.dtype
        # the allowed options for argparse (choices)
        if options is None:
            options = self.options
        # the path associated with a yaml file
        if path is None:
            path = self.path
        # the comment for a fits header
        if comment is None:
            comment = self.comment
        # return ne instance
        return Const(key, source, desc, arg, dtype, options, path, comment)


class ParamDict(UserDict):
    def __init__(self, *args, **kwargs):
        """
        Construct the parameter dictionary class

        :param args: args passed to dict constructor
        :param kwargs: kwargs passed to dict constructor
        """
        super().__init__(*args, **kwargs)
        # storage for constants
        self.instances = dict()
        # must be set (by instrument)
        self.not_none = []

    def set(self, key: str, value: Any, source: Union[str, None] = None,
            desc: Union[str, None] = None, arg: Union[str, None] = None,
            dtype: Union[Type, None] = None, not_none: bool = False,
            options: Union[list, None] = None,
            path: Union[str] = None, comment: Union[str, None] = None):
        """
        Set a parameter in the dictionary y[key] = value

        :param key: str, the key to set in dictionary
        :param value: Any, the value to give to the dictionary item
        :param source: str or None, if set the source of the parameter
        :param desc: str or None, if set the description of the parameter
        :param arg: str or None, the command argument
        :param dtype: Type or None, the type of object (for argparse) only
                      required if arg is set
        :param not_none: bool, if True and value is None error will be raised
                         when getting parameter (so devs don't forget to
                         have this parameter defined by instrument)
        :param options: list or None, the options (choices) to allow for
                        argparse
        :param path: list or str, the path associated with a yaml file
        :param comment: str or None, if set this is the comment to add to a
                        fits header

        :return: None - updates dict
        """
        # capitalize
        key = self._capitalize(key)
        # deal with storing not None
        if not_none:
            self.not_none.append(key)
        # set item
        self.__setitem__(key, value)
        # ---------------------------------------------------------------------
        # update / add instance
        # ---------------------------------------------------------------------
        # args for const
        cargs = [key, source, desc, arg, dtype, options, path, comment]
        # if instance already exists we just want to update keys that should
        #   be updated (i.e. not None)
        if key in self.instances and self.instances[key] is not None:
            self.instances[key] = self.instances[key].update(*cargs)
        # else we set instance from scratch
        else:
            self.instances[key] = Const(*cargs)

    def __setitem__(self, key: Any, value: Any):
        """
        Set an item from the dictionary using y[key] = value

        :param key: Any, the key for which to store its value
        :param value: Any, the value to store for this key

        :return: None - updates the dictionary
        """
        # capitalize
        key = self._capitalize(key)
        # then do the normal dictionary setting
        self.data[key] = value

    def __getitem__(self, key: Any) -> Any:
        """
        Get an item from the dictionary using y[key]

        :param key: Any, the key for which to return its value

        :return: Any, the value of the given key
        """
        # capitalize
        key = self._capitalize(key)
        # return from supers dictionary storage
        value = self.data[key]
        # deal with not none and value is None
        # if value is None:
        #     if key in self.not_none:
        #         emsg = ('Key {0} is None - it must be set by the instrument,'
        #                 'function inputs, command line or yaml file.')
        #         eargs = [key]
        #         raise LblException(emsg.format(*eargs))
        # return value
        return value

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
        # capitalize
        key = self._capitalize(key)
        # return True if key in keys else return False
        return key in self.data.keys()

    def __delitem__(self, key: str):
        """
        Deletes the "key" from CaseInsensitiveDict instance, case insensitive

        :param key: string, the key to delete from ParamDict instance,
                    case insensitive
        :type key: str

        :return None:
        """
        # capitalize
        key = self._capitalize(key)
        # delete key from keys
        del self.data[key]

    def copy(self) -> 'ParamDict':
        """
        Deep copy a parameter dictionary

        :return: new instance of ParamDict
        """
        new = ParamDict()
        keys, values = self.data.keys(), self.data.values()
        for key, value in zip(keys, values):
            # copy value
            new[key] = deepcopy(value)
            # copy instance
            if self.instances[key] is None:
                new.instances[key] = None
            else:
                new.instances[key] = self.instances[key].copy()
        # return parameter dictionary
        return new

    @staticmethod
    def _capitalize(key: str) -> str:
        """
        capitalize a key
        :param key: str, the key to capitalize
        :return: str, the capitalized key
        """
        if isinstance(key, str):
            return key.upper()
        else:
            return key

    def sources(self) -> Dict[str, str]:
        """
        Get the sources for this parameter dictionary (from instances)

        :return: dict, the source dictionary
        """
        source_dict = dict()
        for key in self.instances:
            source_dict[key] = self.instances[key].source
        return source_dict

    def __str__(self) -> str:
        """
        String representation of the parameter dictionary

        :return: str, the string representation of the parameter dictionary
        """
        # get keys, values, sources
        keys = list(self.keys())
        values = list(self.values())
        sources = self.sources()
        string = 'ParamDict:'
        for it, key in enumerate(keys):
            # get source
            if key not in sources:
                source = 'Not set'
            else:
                source = sources[key]
            sargs = [key + ':', str(values[it])[:40], source]
            string += '\n{0:30s}\t{1:40s}\t// {2}'.format(*sargs)
        return string

    def __repr__(self) -> str:
        """
        String representation of the parameter dictionary

        :return: str, the string representation of the parameter dictionary
        """
        return self.__str__()


class FitParam:

    name: str
    value: Any
    wfit: str
    ftype: str
    prior: Any
    label: str

    def __init__(self, name: str, value: Any, wfit: str, ftype: str,
                 prior: Any, label: str):
        # set the name
        self.name = name
        # set the value of
        self.value = value
        # deal with wave fit parameter
        if wfit in ['bolometric', 'chromatic']:
            self.wfit = wfit
        else:
            emsg = ('FitParamError: For constant {0} wfit must be '
                    '"bolometric" or "chromatic"' )
            raise TransitFitExcept(emsg.format(self.name))
        # deal with fit/fixed type parameter
        if ftype in ['fixed', 'fit']:
            self.ftype = ftype
        else:
            emsg = ('FitParamError: For constant {0} wfit must be '
                    '"fixed" or "fit"' )
            raise TransitFitExcept(emsg.format(self.name))
        # deal with prior
        # TODO: how do we want to handle prior?
        self.prior = prior
        # deal with label
        if label is None:
            self.label = str(self.name)
        else:
            self.label = label



# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
