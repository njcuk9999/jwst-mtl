#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.core.constant_functions.py

Functions related to constants are defined in here

Created on 2020-05-21

@author: cook

Rules:
- do not import anything from package
- except exceptions.py

"""
from collections import OrderedDict
from typing import Union, List
from pathlib import Path
from astropy import units as uu

from ami_sim_mtl.core.core import general
from ami_sim_mtl.core.core import exceptions

# =============================================================================
# Define variables
# =============================================================================
# define name
__NAME__ = 'core.core.constant_functions.py'
# get general functions
display_func = general.display_func
# get exceptions
ConstantException = exceptions.ConstantException
PathException = exceptions.PathException


# =============================================================================
# Define class
# =============================================================================
class Constant:
    def __init__(self, name: str, value: object = None,
                 dtype: Union[None, type] = None,
                 source: Union[None, str] = None,
                 user: bool = False,
                 argument: bool = False,
                 group: Union[None, str] = None,
                 description: Union[None, str] = None,
                 minimum: Union[None, int, float] = None,
                 maximum: Union[None, int, float] = None,
                 options: Union[None, list] = None,
                 check: bool = True,
                 command: Union[None, List[str], str] = None,
                 units: Union[None, str, uu.core.Unit] = None):
        """
        Construct a constant file

        :param name: the name of this constant (must be a string)
        :param value: the default value of this constant
                      (must be type: None or dtype)
        :param dtype: the data type (i.e. int, float, bool, list, path etc
        :param source: the source of this constant (e.g. __NAME__)
        :param user: whether this should be used in the user config file
        :param argument: whether this should be used as a command line argument
        :param group: the group this constant belongs to
        :param description: the descriptions to use for the help file /
                            user config file
        :param check: whether to check dtype/source/min/max etc
        :pararm command: string or list or strings, the commands to use as
                         arguments

        :type name: str
        :type value: object
        :type dtype: Union[None, type]
        :type source: Union[None, str]
        :type user: bool
        :type argument: bool
        :type group: Union[None, str]
        :type description: Union[None, str]
        :type check: bool
        :type command: Union[None, List[str], str]
        :type units: Union[None, str, uu.core.Unit]
        """
        self.name = name
        self.value = value
        self.dtype = dtype
        self.source = source
        self.user = user
        self.argument = argument
        self.group = group
        self.description = description
        self.minimum = minimum
        self.maximum = maximum
        self.options = options
        self.check = check
        # deal with commands (for command line arguments)
        self.command = command
        self.__list_commands()
        # check values
        self.__check_value()
        # check units
        self.units = units
        self.__check_units()

    def __check_value(self):
        """
        Check that the value satisfies the dtype/min/max/options

        :return:
        """
        # set function name
        func_name = display_func('__check_value', __NAME__, 'Constant')
        # if value is None then do not check it
        if self.value is None:
            return
        # ------------------------------------------------------------------
        # check dtype
        if self.dtype is not None:
            # need to check if maximum or minimum are set
            if self.minimum is not None or self.maximum is not None:
                check = True
            else:
                check = self.check
            # raise an exception if dtype of value is incorrect
            self.value = _check_type(self.name, 'value', self.value,
                                     self.dtype, self, check=check)
        # ------------------------------------------------------------------
        # check minimum value
        if self.minimum is not None:
            # raise an exception if dtype of minimum is incorrect
            self.minimum = _check_type(self.name, 'minimum', self.minimum,
                                       self.dtype, self)
            # now check limit
            if self.value < self.minimum:
                # construct error message
                emsg = 'Constant "{0}" value (={1}) is less than minimum (={2})'
                eargs = [self.name, self.value, self.minimum]
                # raise Constant Exception
                raise ConstantException(emsg.format(*eargs), 'minimum', self,
                                        funcname=func_name)
        # ------------------------------------------------------------------
        # check maximum value
        if self.maximum is not None:
            # raise an exception if dtype of minimum is incorrect
            self.maximum = _check_type(self.name, 'maximum', self.maximum,
                                       self.dtype, self)
            # now check limit
            if self.value > self.maximum:
                # construct error message
                emsg = ('Constant "{0}" value (={1}) is greater than '
                        'maximum (={2})')
                eargs = [self.name, self.value, self.maximum]
                # raise Constant Exception
                raise ConstantException(emsg.format(*eargs), 'maximum', self,
                                        funcname=func_name)
        # ------------------------------------------------------------------
        # check options
        if self.options is not None:
            # check that options is a list
            if not isinstance(self.options, list):
                # construct error message
                emsg = 'Constant "{0}" options argument must be a list.'
                eargs = [self.name]
                # raise Constant Exception
                raise ConstantException(emsg.format(*eargs), 'options', self,
                                        funcname=func_name)

            # check that all options are in dtype
            for it, option in enumerate(self.options):
                option = _check_type(self.name, 'options[{0}]'.format(it),
                                     option, self.dtype, self)
                # update option
                self.options[it] = option

            # check if value is in options
            if self.value not in self.options:
                # construct error message
                emsg = ('Constant "{0}" value (={1}) must be in options.'
                        '\n\tAllowed options: {2}')
                eargs = [self.name, self.value, '\n\t\t'.join(self.options)]
                # raise Constant Exception
                raise ConstantException(emsg.format(*eargs), 'options', self,
                                        funcname=func_name)

    def __list_commands(self):
        """
        Deal with listing commands
        :return:
        """
        # set function name
        func_name = display_func('__list_commands', __NAME__, 'Constant')
        # if argument is True we must have a command defined
        if self.argument and self.command is None:
            # construct error message
            emsg = ('Constant "{0}" argument=True, so must define at least '
                    'one command to use (command=X)')
            eargs = [self.name]
            # raise Constant Exception
            raise ConstantException(emsg.format(*eargs), 'command', self,
                                    funcname=func_name)

        # if commands is None do nothing
        if self.command is None:
            return
        # deal with command as string
        if isinstance(self.command, str):
            self.command = [self.command]
        # deal with command as list
        elif isinstance(self.command, list):
            self.command = self.command
        # else we have a problem with the command list
        else:
            # construct error message
            emsg = ('Constant "{0}" command value "{1}" is incorrect must be a '
                    'list of strings or a string')
            eargs = [self.name, self.command]
            # raise Constant Exception
            raise ConstantException(emsg.format(*eargs), 'command', self,
                                    funcname=func_name)
        # deal with commands being invalid
        for command in self.command:
            if not command.startswith('--'):
                # construct error message
                emsg = 'Constant "{0}" command "{1}" must start with "--"'
                eargs = [self.name, command]
                # raise Constant Exception
                raise ConstantException(emsg.format(*eargs), 'command', self,
                                        funcname=func_name)

    def __check_units(self):
        """
        Deal with constant having units
        :return:
        """
        # set function name
        func_name = display_func('__check_units', __NAME__, 'Constant')
        # if we do not have units then return
        if self.units is None:
            return
        # if units are astropy units already we are good
        if isinstance(self.units, uu.core.Unit):
            # value must be a float
            self.dtype = float
            # must then recheck values
            self.__check_value()
        # if units are a string we need to make sure they can be turned to
        #  into an astropy unit
        elif isinstance(self.units, str):
            try:
                self.units = uu.core.Unit(self.units)
            except Exception:
                # construct error message
                emsg = ('Constant "{0}" has "units" (="{1}") that are '
                        'invalid [str] (must be astropy valid units)')
                eargs = [self.name, self.units]
                # raise Constant Exception
                raise ConstantException(emsg.format(*eargs), 'units',
                                        self, funcname=func_name)
        # else units are invalid
        else:
            # construct error message
            emsg = ('Constant "{0}" has "units" (="{1}") that are '
                    'invalid [non str] (must be astropy valid units)')
            eargs = [self.name, self.units]
            # raise Constant Exception
            raise ConstantException(emsg.format(*eargs), 'units',
                                    self, funcname=func_name)

        # now add units to the value
        try:
            self.value = self.value * self.units
        except Exception:
            # construct error message
            emsg = ('Constant "{0}" has "units" (="{1}") that are '
                    'invalid [error] (must be astropy valid units)')
            eargs = [self.name, self.units]
            # raise Constant Exception
            raise ConstantException(emsg.format(*eargs), 'units',
                                    self, funcname=func_name)

    def __str__(self):
        """
        Return a string representation of Constant Class

        :return:
        """
        return 'Constant[{0}]'.format(self.name)

    def __repr__(self):
        """
        Return a string representation of Constant Class

        :return:
        """
        return self.__str__()


class Constants:
    def __init__(self, name=None):
        """
        Construct the Constants class

        :param source: The name of this Constants instance
        """
        if name is None:
            self.name = 'Unknown'
        else:
            self.name = name
        self.constants = OrderedDict()
        self.arguments = []
        self.commands = []

    def add(self, name, **kwargs):
        """
        Add a constant to the Constants list

        :param name: the name of this constant (must be a string)
        :param kwargs: see list below

        kwargs:
           - value: the default value of this constant (must be type: None or dtype)
           - dtype: the data type (i.e. int, float, bool, list, path etc
           - source: the source of this constant (e.g. __NAME__)
           - user: whether this should be used in the user config file
           - argument: whether this should be used as a command line argument
           - group: the group this constant belongs to
           - description: the descriptions to use for the help file / user
             config file
           - minimum:  if int/float set the minimum allowed value
           - maximum:  if int/float set the maximum allowed value
           - options:  a list of possible options (each option must be type dtype)

        :return:
        """
        # add to constants
        constant = Constant(name, **kwargs)
        # add to constants dictionary
        self.constants[name] = constant
        # check if constant needs to be an argument
        if constant.argument:
            # add arguments to list (for lookup)
            self.arguments.append(constant.name)
            # check that constant commands are unique
            self._check_unique_commands(constant)

    def _check_unique_commands(self, constant: Constant):
        """
        Check if a list of commands has no entries in self.commands
        :param constant:
        :return:
        """
        # set function name
        func_name = display_func('_check_unique_commands', __NAME__,
                                 'Constants')
        # get commands
        commands = constant.command
        # check whether each command is in list
        for command in commands:
            # if command is in commands
            if command in self.commands:
                # construct error message
                emsg = ('Command "{0}" is not unique '
                        '(used in more than just the constant "{1}")')
                eargs = [command, constant.name]
                # raise Constant Exception
                raise ConstantException(emsg.format(*eargs), 'command',
                                        constant, funcname=func_name)
            # else add it to the list for the next constant
            else:
                self.commands.append(command)

    def __str__(self):
        """
        Return a string representation of Constants Class

        :return:
        """
        return 'Constants[{0}]'.format(self.name)

    def __repr__(self):
        """
        Return a string representation of Constants Class

        :return:
        """
        return self.__str__()


# =============================================================================
# Define functions
# =============================================================================
def _check_type(cname, variable, value, dtype, instance=None, check=True):
    # set function name
    func_name = display_func('_check_type', __NAME__)
    # assume a value of None should not be tested
    if value is None:
        return
    # -------------------------------------------------------------------------
    # check path
    if dtype == 'path':
        # try to create path
        try:
            value = Path(value)
        except Exception as e:
            # construct error message
            emsg = 'Constant "{0}" {1} value (="{2}") is not a valid path.'
            eargs = [cname, variable, value]
            # raise a path exception
            raise PathException(value, e, funcname=func_name,
                                message=emsg.format(*eargs))
        # do we check?
        if not check:
            return value
        # now we have a path test whether it exists
        if not value.exists():
            # construct error message
            emsg = 'Constant "{0}" {1} value (="{2}") does not exist.'
            eargs = [cname, variable, value]
            # raise a path exception
            raise PathException(value, funcname=func_name,
                                message=emsg.format(*eargs))
    # -------------------------------------------------------------------------
    # do we check? if not just return
    if not check:
        return value
    # -------------------------------------------------------------------------
    # check if object is instance
    if not isinstance(value, dtype):
        try:
            value = dtype(value)
        except Exception:
            # construct error message
            emsg = 'Constant "{0}" {1} (="{2}") is not type "{3}"'
            # deal with dtype name
            if hasattr(dtype, '__name__'):
                eargs = [cname, variable, value, dtype.__name__]
            else:
                eargs = [cname, variable, value, str(dtype)]
            # raise Constant Exception
            raise ConstantException(emsg.format(*eargs), 'dtype', instance,
                                    funcname=func_name)
        # return value
        return value
    else:
        # finally return value
        return value


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
