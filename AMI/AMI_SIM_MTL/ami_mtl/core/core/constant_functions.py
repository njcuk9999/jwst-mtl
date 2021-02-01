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
import copy
from pathlib import Path
from typing import Any, List, Union

from ami_mtl.core.core import general
from ami_mtl.core.core import exceptions

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
# define true values
TRUE_VALUES = ['1', 1, True, 'True', 'T', 'Y', 'true', 'yes', 'YES']


# =============================================================================
# Define class
# =============================================================================
class Constant:
    def __init__(self, name: str, value: object = None,
                 dtype: Union[None, type] = None,
                 source: Union[None, str] = None,
                 user: bool = False,
                 argument: bool = False,
                 required: bool = False,
                 group: Union[None, str] = None,
                 description: Union[None, str] = None,
                 minimum: Union[None, int, float] = None,
                 maximum: Union[None, int, float] = None,
                 options: Union[None, list] = None,
                 check: bool = True,
                 command: Union[None, List[str], str] = None,
                 units: Union[None, Any] = None):
        """
        Construct a constant file

        :param name: the name of this constant (must be a string)
        :param value: the default value of this constant
                      (must be type: None or dtype)
        :param dtype: the data type (i.e. int, float, bool, list, path etc
        :param source: the source of this constant (e.g. __NAME__)
        :param user: whether this should be used in the user config file
        :param argument: whether this should be used as a command line argument
        :param required: whether this constant is required as a command
                         line argument
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
        :type required: bool
        :type group: Union[None, str]
        :type description: Union[None, str]
        :type check: bool
        :type command: Union[None, List[str], str]
        """
        self.name = name
        self.value = value
        self.dtype = dtype
        self.source = source
        self.user = user
        self.argument = argument
        self.required = required
        self.group = group
        self.description = description
        self.minimum = minimum
        self.maximum = maximum
        self.options = options
        self.check = check
        self.units = units
        # deal with commands (for command line arguments)
        self.command = command
        self.__list_commands()
        # check values
        self.check_value()

    def copy(self, **kwargs):
        # deal with getting kwargs from call or from self (as a deep copy)
        kwargs['name'] = kwargs.get('name', str(self.name))
        kwargs['value'] = kwargs.get('value', copy.deepcopy(self.value))
        kwargs['dtype'] = kwargs.get('dtype', copy.deepcopy(self.dtype))
        kwargs['source'] = kwargs.get('source', copy.deepcopy(self.source))
        kwargs['user'] = kwargs.get('user', bool(self.user))
        kwargs['argument'] = kwargs.get('argument', bool(self.argument))
        kwargs['required'] = kwargs.get('required', bool(self.required))
        kwargs['group'] = kwargs.get('group', copy.deepcopy(self.group))
        kwargs['description'] = kwargs.get('description', str(self.description))
        kwargs['minimum'] = kwargs.get('minimum', copy.deepcopy(self.minimum))
        kwargs['maximum'] = kwargs.get('maximum', copy.deepcopy(self.maximum))
        kwargs['options'] = kwargs.get('options', copy.deepcopy(self.options))
        kwargs['check'] = kwargs.get('check', bool(self.check))
        kwargs['command'] = kwargs.get('command', copy.deepcopy(self.command))
        kwargs['units'] = kwargs.get('units', copy.deepcopy(self.units))
        # return new instances of Constant
        return Constant(**kwargs)

    def check_value(self):
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
    def __init__(self, name: Union[str, None] = None):
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

    def add(self, name: str, **kwargs):
        """
        Add a constant to the Constants list

        :param name: the name of this constant (must be a string)
        :param kwargs: see list below

        :keyword value: the default value of this constant (must be type:
                        None or dtype)
        :keyword dtype: the data type (i.e. int, float, bool, list, path etc
        :keyword source: the source of this constant (e.g. __NAME__)
        :keyword user: whether this should be used in the user config file
        :keyword argument: whether this should be used as a command line
                           argument
        :keyword required: whether this constant is required as a command
                           line argument
        :keyword group: the group this constant belongs to
        :keyword description: the descriptions to use for the help file / user
                              config file
        :keyword minimum:  if int/float set the minimum allowed value
        :keyword maximum:  if int/float set the maximum allowed value
        :keyword options:  a list of possible options (each option must be
                           type dtype)

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

    def add_argument(self, name: str, group: Union[str, None] = None,
                     required: bool = False, **kwargs):
        """
        Add an argument to the Constants list

        :param name: the name of this constant (must be a string)
        :param group: the group this constant belongs to

        :param kwargs: see list below

        :keyword value: the default value of this constant (must be type:
                        None or dtype)
        :keyword dtype: the data type (i.e. int, float, bool, list, path etc
        :keyword source: the source of this constant (e.g. __NAME__)
        :keyword user: whether this should be used in the user config file
        :keyword description: the descriptions to use for the help file / user
                              config file
        :keyword minimum:  if int/float set the minimum allowed value
        :keyword maximum:  if int/float set the maximum allowed value
        :keyword options:  a list of possible options (each option must be
                           type dtype)

        :return:
        """
        # force argument to True
        kwargs['argument'] = True
        # add to constants
        constant = Constant(name, group=group, required=required, **kwargs)
        # add to constants
        self.constants[name] = constant
        # add arguments to list (for lookup)
        self.arguments.append(constant.name)
        # check that constant commands are unique
        self._check_unique_commands(constant)

    def copy(self, name: Union[str, None] = None):
        if name is None:
            name = 'Unknown'
        newinst = Constants(name)
        # copy over constants
        for key in self.constants:
            newinst.constants[key] = self.constants[key].copy()
        # copy over lists of strings
        newinst.arguments = list(self.arguments)
        newinst.commands = list(self.commands)
        # return new instance of constants
        return newinst

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
    if dtype in ['bool', bool]:
        if value in TRUE_VALUES:
            return True
        else:
            return False
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
