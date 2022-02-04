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
class DrsException(Exception):
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
        self.__name__ = 'DrsException'
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

    def __log__(self):
        return '{0}: {1}'.format(self.__name__, self.__str__())


class LogException(DrsException):
    def __init__(self, message: str, kind: str,
                 instance: Union[None, object] = None,
                 funcname: Union[None, str] = None):
        """
        Construct an Exception instance

        for use with raise ConstantException(...)

        :param message: a string to explain the exception
        :param kind: the kind of error that occurred
        :param instance: object, a constants instance or None
        :param funcname: the function name that error was raise in
        """
        super().__init__(message, kind, instance, funcname)
        self.__name__ = 'LogException'


class ConstantException(DrsException):
    def __init__(self, message: str, kind: str,
                 instance: Union[None, object] = None,
                 funcname: Union[None, str] = None):
        """
        Construct an Exception instance

        for use with raise ConstantException(...)

        :param message: a string to explain the exception
        :param kind: the kind of error that occurred
        :param instance: object, a constants instance or None
        :param funcname: the function name that error was raise in
        """
        super().__init__(message, kind, instance, funcname)
        self.__name__ = 'ConstantException'


class PathException(DrsException):
    def __init__(self, path: Union[str, Path],
                 exception: Union[None, Exception] = None,
                 funcname: Union[str, None] = None,
                 message: Union[str, None] = None):
        """
        Construct an Exception instance

        for use with raise ConstantException(...)

        :param message: a string to explain the exception
        :param kind: the kind of error that occurred
        :param instance: object, a constants instance or None
        :param funcname: the function name that error was raise in
        """
        super().__init__(message, '', None, funcname)
        self.__name__ = 'PathException'
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


class ParamDictException(DrsException):
    def __init__(self, message: str, kind: str,
                 instance: Union[None, object] = None,
                 funcname: Union[None, str] = None):
        """
        Construct an Exception instance

        for use with raise ConstantException(...)

        :param message: a string to explain the exception
        :param kind: the kind of error that occurred
        :param instance: object, a constants instance or None
        :param funcname: the function name that error was raise in
        """
        super().__init__(message, kind, instance, funcname)
        self.__name__ = 'ParamDictException'


class ImportException(DrsException):
    def __init__(self, message: str, kind: str,
                 instance: Union[None, object] = None,
                 funcname: Union[None, str] = None):
        """
        Construct an Exception instance

        for use with raise ConstantException(...)

        :param message: a string to explain the exception
        :param kind: the kind of error that occurred
        :param instance: object, a constants instance or None
        :param funcname: the function name that error was raise in
        """
        super().__init__(message, kind, instance, funcname)
        self.__name__ = 'ImportException'


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
