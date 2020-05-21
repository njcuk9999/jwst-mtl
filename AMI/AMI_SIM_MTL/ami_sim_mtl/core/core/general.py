#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.core.general.py

General functions that do not rely on other modules and can be used
throughout without problem

Created on 2020-05-21

@author: cook
"""

# =============================================================================
# Define variables
# =============================================================================


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




# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
