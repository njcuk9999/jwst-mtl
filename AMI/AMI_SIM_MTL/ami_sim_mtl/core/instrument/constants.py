#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.instruments.constants.py

Default constants are defined in here

Created on 2020-05-21

@author: cook
"""
from astropy.io import units as uu
from ami_sim_mtl.core.core import constant_functions
# set very basic constants
__NAME__ = 'core.instruments.constants.py'
__VERSION__ = '0.0.001'
__DATE__ = '2020-05-21'
# get constants class
Consts = constant_functions.Constants(__NAME__)


# =============================================================================
# Define variables added to Constants via Const.add

# Const.add Arguments:
#    - name: the name of this constant (must be a string)
#    - value: the default value of this constant (must be type: None or dtype)
#    - dtype: the data type (i.e. int, float, bool, list, path etc
#    - source: the source of this constant (e.g. __NAME__)
#    - user: whether this should be used in the user config file
#    - argument: whether this should be used as a command line argument
#    - group: the group this constant belongs to
#    - description: the descriptions to use for the help file / user
#      config file
#    - minimum:  if int/float set the minimum allowed value
#    - maximum:  if int/float set the maximum allowed value
#    - options:  a list of possible options (each option must be type dtype)

# =============================================================================

# =============================================================================
#   General constants
# =============================================================================
# define this groups name
group = 'General'

# Define the user config file
Consts.add('USER_CONFIG_FILE', value=None, dtype='path',
           source=__NAME__, user=True, argument=True, group=group,
           description='Define the user config file', command='--config')

# =============================================================================
#   Simulation constants
# =============================================================================
group = 'Simulation'
# Define whether to add the jitter
Consts.add('ADD_JITTER', value=True, dtype=bool,
           source=__NAME__, user=True, argument=True, group=group,
           description='Define whether to add the jitter',
           command='--add_jitter')

# Define whether to add the jitter
Consts.add('JITTER_RMS', value=7.0, dtype=float, units=uu.mas,
           source=__NAME__, user=True, argument=True, group=group,
           description='Define the jitter rms level',
           command='--jitter_rms')