#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.core.general.py

General functions that do not rely on other modules and can be used
throughout without problem

Created on 2020-05-21

@author: cook
"""
import os
from pathlib import Path
import pkg_resources
import string
import sys
import time
from typing import Union

from ami_mtl.core.base import base
from ami_mtl.core.core import exceptions


# =============================================================================
# Define variables
# =============================================================================
# define name
__NAME__ = 'core.core.general.py'
# get exceptions
ImportException = exceptions.ImportException
PathException = exceptions.PathException
# relative folder cache
REL_CACHE = dict()
# bad characters
BAD_CHARS = [' '] + list(string.punctuation.replace('_', ''))


# =============================================================================
# Define functions
# =============================================================================
class Unbuffered:
    """
    Modified from: https://stackoverflow.com/a/107717
    """
    def __init__(self, stream, text=None, log=None):
        """
        Construct stream
        :param stream:
        """
        self.stream = stream
        if text is None:
            self.text = ''
        else:
            self.text = text + ': '
        self.log = log

    def _check(self, data) -> bool:
        """
        Check if data is valid for printing / logging

        :param data: string, the stdout string

        :return: bool, True if we should print / log
        """
        data = data.strip()
        # if empty return False
        if len(data) == 0:
            return False
        # if just one bad character return False
        if data in BAD_CHARS:
            return False
        # else return True
        else:
            return True

    def _fmt(self, data: str) -> str:
        """
        Must remove new lines (otherwise this doesn't work)
        :param data:
        :return:
        """
        # first remove 'n
        data = data.replace('\n', '')
        # deal with prefix
        if len(self.text) > 0:
            if not data.startswith(self.text):
                data = self.text + data
        # return data
        return str(data)

    def _writelog(self, data: str):
        if self.log is not None:
            with open(self.log, 'a') as logfile:
                logfile.write(data + '\n')

    def write(self, data: str):
        """
        Write line of text and then flush the line
        :param data:
        :return:
        """
        if self._check(data):
            data = self._fmt(data)
            self.stream.write('\r' + data)
            self.stream.flush()
            self._writelog(data)
        if base.FLUSH_TEXT_TIME > 0:
            time.sleep(base.FLUSH_TEXT_TIME)

    def writelines(self, datas):
        """
        Write a set of lines (in a loop) then flush the line

        :param datas: list of strings
        :return:
        """
        for data in datas:
            if self._check(data):
                data = self._fmt(data)
                self.stream.write('\r' + data)
                self.stream.flush()
                self._writelog(data)
            if base.FLUSH_TEXT_TIME > 0:
                time.sleep(base.FLUSH_TEXT_TIME)
        # self.stream.writelines(datas)
        # self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


class ModifyPrintouts:
    """
    Class to hide printouts

    Use as follows:

    with HiddenPrints():
        # code to hide printouts

    Taken from https://stackoverflow.com/a/45669280
    """
    def __init__(self, text=None, flush=False, logfile=None,
                 debugmode: int = 0):
        """
        Construct the hidden printouts - if text is set this is displayed
        at the start

        :param text: str, the text displayed before the flushed text
        """
        self.text = text
        self.flush = flush
        # ---------------------------------------------------------------------
        # set debug mode (do not hide texst)
        if debugmode > 0:
            self.debugmode = True
        else:
            self.debugmode = False
        # ---------------------------------------------------------------------
        if logfile is not None:
            self.logfile = str(logfile)

    def __enter__(self):
        if self.debugmode:
            return
        self._original_stdout = sys.stdout
        if self.flush:
            sys.stdout = Unbuffered(sys.stdout, text=self.text,
                                    log=self.logfile)
        else:
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.debugmode:
            return
        if not self.flush:
            sys.stdout.close()
        # reset stdout
        sys.stdout = self._original_stdout


def display_func(func_name: str, program_name: str = None,
                 class_name: str = None,
                 params: Union[object, None] = None) -> str:
    """
    Constructs the displayed function name (may work with breakpoints later)

    :param func_name: the function name
    :param program_name: the code/program/recipe name (can be unset)
    :param class_name: the class name (if in a class)
    :return:
    """
    name = ''
    # add program name
    if program_name is not None:
        name += program_name + '.'
    # add class name
    if class_name is not None:
        name += class_name + '.'
    # add function name
    name += func_name + '()'
    # lowest debug level
    if params is not None:
        # only deal with ParamDict (not defined at this level unless passed)
        if hasattr(params, 'log'):
            params.log.ldebug('In: {0}'.format(name), 9)
    # return display name
    return name


def get_package_directory(package: str, directory: Union[str, Path]):
    """
    Get the absolute path of directory defined at relative path
    folder from package

    :param package: string, the python package name
    :param folder: string, the relative path of the config folder

    :return data: string, the absolute path and filename of the default config
                  file
    """
    global REL_CACHE
    # set function name (cannot break here --> no access to inputs)
    func_name = display_func('get_relative_folder', __NAME__)
    # ----------------------------------------------------------------------
    # deal with folder being string
    if isinstance(directory, str):
        directory = Path(directory)
    # ----------------------------------------------------------------------
    # check relative folder cache
    if package in REL_CACHE and directory in REL_CACHE[package]:
        return REL_CACHE[package][directory]
    # ----------------------------------------------------------------------
    # get the package.__init__ file path
    try:
        init = Path(pkg_resources.resource_filename(package, '__init__.py'))
    except ImportError:
        emsg = "Package name = '{0}' is invalid (function = {1})"
        eargs = [package, func_name]
        raise ImportException(emsg.format(*eargs), 'import', funcname=func_name)
    # Get the config_folder from relative path
    current = os.getcwd()
    # get directory name of folder
    dirname = init.parent
    # change to directory in init
    os.chdir(dirname)
    # get the absolute path of the folder
    data_folder = directory.absolute()
    # change back to working dir
    os.chdir(current)
    # test that folder exists
    if not data_folder.exists():
        # raise exception
        emsg = "Folder '{0}' does not exist in {1}"
        eargs = [data_folder.name, data_folder.parent]
        PathException(data_folder, funcname=func_name,
                      message=emsg.format(*eargs))
    # ----------------------------------------------------------------------
    # update REL_CACHE
    if package not in REL_CACHE:
        REL_CACHE[package] = dict()
    # update entry
    REL_CACHE[directory] = data_folder
    # ----------------------------------------------------------------------
    # return the absolute data_folder path
    return data_folder


def clean_name(name: str) -> str:
    """
    Clean a name for text file saving etc

    :param name: str, the name to clean

    :return: str, the cleaned name
    """
    # we can't do anything for non-strings
    if not isinstance(name, str):
        return name
    # remove bad characters
    for bad_char in BAD_CHARS:
        name = name.replace(bad_char, '_')
    # remove double underscores
    while '__' in name:
        name = name.replace('__', '_')
    # put name in upper case
    name = name.upper()
    # return clean name
    return name


def printtxt(message):
    print(message)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
