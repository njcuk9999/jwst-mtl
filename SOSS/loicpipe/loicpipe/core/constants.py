#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on {DATE}

@author: cook
"""
import copy
from typing import Any, Optional
from collections import UserDict

from loicpipe.core import base

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.constants.py'

# =============================================================================
# Define classes
# =============================================================================
class Const:
    def __init__(self, key: str, value: Any, not_none: bool = False,
                 dtype: Any = None, path: str = None, source: str = None,
                 directory: Optional[str] = None):
        self.key = key
        self.value = value
        self.not_none = not_none
        self.dtype = dtype
        self.path = path
        self.source = source
        self.directory = directory

    def from_yaml(self, yaml_dict):
        """
        Get a value from a yaml dictionary using a path key1.key2.key3.keyN
        for any number of keys

        :param ydict: dict, the yaml dictionary ydict[key1][key2][key3][..][keyN]
        :param path: str, the path in format key1.key2.key3.keyN

        :return:
        """
        # split the path into keys
        ykeys = self.path.split('.')
        # set the key level dict to top level
        ydict_tmp = yaml_dict
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

    def check(self, value: Optional[Any] = None):
        # deal with internal check
        if value is None:
            value = self.value
        # check if not none
        if self.not_none and value is None:
            emsg = 'Const {0} must be set in yaml'
            eargs = [self.key]
            raise base.LoicPipeError(emsg.format(*eargs))
        # force dtype if set
        if self.dtype is not None:
            try:
                return self.dtype(value)
            except Exception as e:
                emsg = 'Const {0} must be of type {1} (error: {2})'
                eargs = [self.key, self.dtype, e]
                raise base.LoicPipeError(emsg.format(*eargs))
        # return value
        return value

    def __getitem__(self, key):
        return self.check(self.from_yaml(key))

    def __str__(self):
        return f'Const[{self.key}]={self.value}'

    def __repr__(self):
        return self.__str__()

    def copy(self):
        new = Const(str(self.key),
                    copy.deepcopy(self.value),
                    bool(self.not_none),
                    self.dtype,
                    copy.deepcopy(self.path),
                    copy.deepcopy(self.source),
                    copy.deepcopy(self.directory))
        return new


class Parameters(UserDict):
    def __init__(self):
        super().__init__()

    def __getitem__(self, key):
        value = self.data[key]
        if isinstance(value, Const):
            return value.value
        else:
            return value

    def __setitem__(self, key, value, not_none: bool = False,
                    dtype: Any = None, path: str = None, source: str = None,
                    directory: Optional[str] = None):
        # if we already have the value then just update it
        if key in self.data:
            if isinstance(self.data[key], Const):
                self.data[key].value = value
                return
        # if we are passing in a constant then just add it
        if isinstance(value, Const):
            self.data[key] = value
        else:
            # otherwise add it as a constant
            self.data[key] = Const(key, value, not_none, dtype, path, source,
                                   directory)

    def __call__(self, key):
        return self.data[key]

    def __str__(self) -> str:
        print_str = '=' * 50 + '\n'
        print_str += 'Parameters:\n'
        print_str += '=' * 50 + '\n'
        for key in self.data:
            if isinstance(self.data[key], Const):
                print_str += f'C[{key}]:\t{self.data[key].value}\n'
            else:
                print_str += f'{key}:\t{self.data[key]}\n'
        return print_str

    def __repr__(self):
        return self.__str__()

    def copy(self):
        new = Parameters()
        new.data = dict()
        # copy all keys
        for key in self.keys():
            if hasattr(self.data[key], 'copy'):
                new.data[key] = self.data[key].copy()
            else:
                new.data[key] = copy.deepcopy(self.data[key])

        return new

# =============================================================================
# Define functions
# =============================================================================


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
