#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-11

@author: cook
"""
from astropy.time import Time

# =============================================================================
# Define variables
# =============================================================================
__NAME__: str = 'base.py'
__version__: str = '0.0.11'
__date__: str = '2022-04-28'
__authors__: str = 'Neil Cook, Jason Rowe, David Lafreniere, Loic Albert'
__package__: str = 'soss_tfit'

# global variables
COLORLOG = True
# start the time once (can take extra time to load the first time)
t0 = Time.now()
time = Time

# =============================================================================
# End of code
# =============================================================================
