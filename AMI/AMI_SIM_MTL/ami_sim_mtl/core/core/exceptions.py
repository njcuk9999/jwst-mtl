#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.core.exceptions

Exceptions are defined in here for use through wihtout problem

Created on 2020-05-21

@author: cook

Rules: Cannot import from any package module
"""
from typing import Union
from pathlib import Path

# =============================================================================
# Define variables
# =============================================================================
# define name
__NAME__ = 'core.core.exceptions.py'

# =============================================================================
# Define classes
# =============================================================================
class ConstantException(Exception):
    def __init__(self, message: str, kind: str,
                 instance: Union[None, object] = None,
                 funcname: Union[None, str] = None):
        """
        Construct a Constant Exception instance

        for use with raise ConstantException(...)

        :param message: a string to explain the exception
        :param kind: the kind of error that occurred
        :param instance: object, a constants instance or None
        :param funcname: the function name that error was raise in
        """
        self.constant = instance
        self.kind = kind
        self.message = message
        self.funcname = funcname

    def __str__(self):
        """
        Return a string representation of the Constant Exception
        :return:
        """
        emsg = '[{0}] {1}'.format(self.kind, self.message)
        emsg += '\n\tFunc: {0}'.format(self.funcname)
        return emsg


class PathException(Exception):
    def __init__(self, path: Union[str, Path],
                 exception: Union[None, Exception] = None,
                 funcname: Union[str, None] = None,
                 message: Union[str, None] = None):
        """
        Create a Path Exception

        for use with raise PathException(...)

        :param path: str, the path to the file
        :param exception: the exception that was caused and then raised this
                          (can be None)
        :param funcname: the function name that error was raise in
        :param message: if set this is the error that is printed
        """
        self.exception = exception
        self.path = path
        self.funcname = funcname
        self.message = message

    def __str__(self):
        """
        Return a string representation of the Path Exception
        :return:
        """
        if self.message is not None:
            emsg = '{0}'.format(self.message)
        elif self.exception is not None:
            name = self.exception.__name__
            emsg = '{0}: {1}'.format(name, self.exception.__str__())
            emsg += '\n\tPath: {0}'.format(self.path)
            emsg += '\n\tFunc: {0}'.format(self.funcname)
        else:
            emsg = 'Path: {0}'.format(self.path)
            emsg += '\n\tFunc: {0}'.format(self.funcname)
        return emsg


class ParamDictException(Exception):
    def __init__(self, message: str, kind: str,
                 instance: Union[None, object] = None,
                 funcname: Union[None, str] = None,
                 exception: Union[None, Exception] = None):
        """
        Construct a Parameter Dictionary Exception instance

        for use with raise ParamDictException(...)

        :param message: a string to explain the exception
        :param kind: the kind of error that occurred
        :param instance: object, a constants instance or None
        :param funcname: the function name that error was raise in
        :param exception: store the exception if given
        """
        self.constant = instance
        self.kind = kind
        self.message = message
        self.funcname = funcname
        self.exception = exception

    def __str__(self):
        """
        Return a string representation of the Constant Exception
        :return:
        """
        emsg = '[{0}] {1}'.format(self.kind, self.message)
        emsg += '\n\tFunc: {0}'.format(self.funcname)
        return emsg


class ImportException(Exception):
    def __init__(self, message: str, kind: str,
                 instance: Union[None, object] = None,
                 funcname: Union[None, str] = None):
        """
        Construct a Import Exception instance

        for use with raise ImportException(...)

        :param message: a string to explain the exception
        :param kind: the kind of error that occurred
        :param instance: object, a constants instance or None
        :param funcname: the function name that error was raise in
        """
        self.constant = instance
        self.kind = kind
        self.message = message
        self.funcname = funcname

    def __str__(self):
        """
        Return a string representation of the Constant Exception
        :return:
        """
        emsg = '[{0}] {1}'.format(self.kind, self.message)
        emsg += '\n\tFunc: {0}'.format(self.funcname)
        return emsg


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
