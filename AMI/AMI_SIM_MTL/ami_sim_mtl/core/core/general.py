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
from typing import Union

from ami_sim_mtl.core.core import exceptions


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

# =============================================================================
# Define functions
# =============================================================================
def display_func(func_name: str, program_name: str = None,
                 class_name: str = None) -> str:
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

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
