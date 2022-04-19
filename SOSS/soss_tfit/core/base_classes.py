#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-06

@author: cook
"""
from collections import UserDict
from copy import deepcopy
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from soss_tfit.core import base


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.base_classes.py'
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
                 path: Union[str] = None, comment: Union[str, None] = None,
                 label: Union[str] = None):
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
        :param label: str, label for parameter
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
        # the label for the parameter
        self.label = str(label)

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
                     self.path, self.comment, self.label)

    def update(self, key: str, source: Union[str, None] = None,
               desc: Union[str, None] = None,
               arg: Union[str, None] = None,
               dtype: Union[Type, None] = None,
               options: Union[list, None] = None,
               path: Union[str] = None,
               comment: Union[str, None] = None,
               label: Union[str] = None) -> 'Const':
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
        :param label: str, label for parameter
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
        # the label for the parameter
        if label is None:
            label = self.label
        # return ne instance
        return Const(key, source, desc, arg, dtype, options, path, comment,
                     label)


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
            path: Union[str] = None, comment: Union[str, None] = None,
            label: Union[str] = None):
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
        :param label: str, label for parameter

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
        cargs = [key, source, desc, arg, dtype, options, path, comment,
                 label]
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

    def copy(self, mask: Optional[List[str]] = None) -> 'ParamDict':
        """
        Deep copy a parameter dictionary

        :param mask: if set these are the keys to keep

        :return: new instance of ParamDict
        """
        # deal with no mask
        if mask is None:
            mask = self.data.keys()
        # start a new parameter dictionary
        new = ParamDict()
        keys, values = self.data.keys(), self.data.values()
        for key, value in zip(keys, values):
            # skip if key in mask
            if key not in mask:
                continue
            # copy value
            new[key] = deepcopy(value)
            # copy instance
            if self.instances[key] is None:
                new.instances[key] = None
            else:
                new.instances[key] = self.instances[key].copy()
        # copy not_none keys
        new.not_none = list(self.not_none)
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
        keys = list(self.data.keys())
        values = list(self.data.values())
        sources = self.sources()
        string = 'ParamDict:'
        for it, key in enumerate(keys):
            # get source
            if key not in sources:
                source = 'Not set'
            else:
                source = sources[key]

            if isinstance(values[it], ParamDict):
                value = 'ParamDict[]'
            else:
                value = str(values[it])

            sargs = [key + ':', value[:40], source]
            string += '\n{0:30s}\t{1:40s}\t// {2}'.format(*sargs)
        return string

    def __repr__(self) -> str:
        """
        String representation of the parameter dictionary

        :return: str, the string representation of the parameter dictionary
        """
        return self.__str__()


class FitParam:
    """
    Fit parameter class, for parameters that can be fit (or fixed)
    """
    name: str
    value: Any
    wfit: str
    ftype: str
    prior: Dict[str, Any]
    label: str

    def __init__(self, name: str, value: Any, wfit: str, ftype: str,
                 prior: Dict[str, Any], label: Optional[str] = None):
        # set the name
        self.name = name
        # set the value of
        self.value = value
        # deal with wave fit parameter
        if wfit in ['bolometric', 'chromatic']:
            self.wfit = wfit
        else:
            emsg = (f'FitParamError: For constant {self.name} wfit must be '
                    '"bolometric" or "chromatic"' )
            raise TransitFitExcept(emsg)
        # deal with fit/fixed type parameter
        if ftype in ['fixed', 'fit']:
            self.ftype = ftype
        else:
            emsg = (f'FitParamError: For constant {self.name} wfit must be '
                    '"fixed" or "fit"' )
            raise TransitFitExcept(emsg)
        # deal with prior
        self.prior = dict(prior)
        # deal with label
        if label is None:
            self.label = str(self.name)
        else:
            self.label = label

    def print(self):
        """
        print all terms

        :return:
        """
        string = self.__str__()
        string += f'\n\tvalue: {self.value}'
        string += f'\n\twfit:  {self.wfit}'
        string += f'\n\tftype: {self.ftype}'
        string += f'\n\tprior: {self.prior}'
        string += f'\n\tlabel: {self.label}'
        print(string)

    def __str__(self) -> str:
        """
        String representation of FitParam
        :return:
        """
        return 'FitParam[{0}]'.format(self.name)

    def __repr__(self) -> str:
        """
        String representation of FitParam
        :return:
        """
        return self.__str__()

    @staticmethod
    def check(**kwargs):
        """
        Check we have keywords required arguments (for construction)
        :param kwargs:
        :return:
        """
        # define required keys
        keys = ['value', 'wfit', 'ftype', 'prior']
        # loop around keys
        for key in keys:
            # if we don't have this key raise an error
            if key not in kwargs:
                emsg = ('ParamError: {name} must have a "{key}" attribute. '
                        'Please set')
                ekwargs = dict(name=kwargs['name'], key=key)
                raise TransitFitExcept(emsg.format(**ekwargs))

    def get_value(self, n_phot: int
                  ) -> Tuple[np.ndarray, bool, bool, Dict[str, Any], str]:
        """
        Get the properties for the transit fitting

        :param n_phot:
        :return: tuple, 1. the array of values for each bandpass for this
                           fit variable
                        2. the array of True/False for each bandpass (if True
                           this is a fit parameter, if False this is fixed)
                        3. the name of this parameter
        """
        # get the values and push to full n_phot array

        if isinstance(self.value, (np.ndarray, list)):
            p0 = np.array(self.value)
            # if we do not have the right length break here
            if p0.shape[0] != n_phot:
                emsg = (f'Data Error: {self.name} is an array and must have '
                        f'length={n_phot}')
                raise TransitFitExcept(emsg)
        else:
            p0 = np.full(n_phot, self.value)
        # get whether fixed or fitted
        fitted = self.ftype == 'fit'
        # get whether chromatic
        chromatic = self.wfit == 'chromatic'
        # pass back the prior dictionary
        prior = self.prior
        # get name
        name = self.label
        # return values
        return p0, fitted, chromatic, prior, name


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
