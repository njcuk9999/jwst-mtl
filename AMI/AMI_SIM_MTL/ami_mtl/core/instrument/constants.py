#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.instruments.constants.py

Default constants are defined in here

Created on 2020-05-21

@author: cook
"""
from astropy import units as uu

from ami_mtl.core.base import base
from ami_mtl.core.core import constant_functions

# set very basic constants
__NAME__ = 'core.instruments.constants.py'
__VERSION__ = base.VERSION
__DATE__ = base.DATE

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
Consts.add('USER_CONFIG_FILE', value=None, dtype=str,
           source=__NAME__, user=False, argument=True, group=group,
           description='Define the user config file', command='--uconfig')

# Define the package name
Consts.add('PACKAGE_NAME', value=base.PACKAGE, dtype=str,
           source=__NAME__, user=False, group=group,
           description='Define the package name')

# Define the package version
Consts.add('PACKAGE_VERSION', value=base.VERSION, dtype=str,
           source=__NAME__, user=False, group=group,
           description='Define the package version')

# Define the version date
Consts.add('PACKAGE_VERSION_DATE', value=base.DATE, dtype=str,
           source=__NAME__, user=False, group=group,
           description='Define the package version date')

# Define package log theme
Consts.add('PACKAGE_THEME', value='DARK', dtype=str,
           source=__NAME__, user=False, argument=True, group=group,
           description='Define package log theme',
           command=['--theme'], options=['DARK', 'LIGHT', 'OFF'])

# Define the package directory
Consts.add('PACKAGE_DIRECTORY', value='ami_sim_data', dtype=str, source=__NAME__,
           user=False, group=group,
           description='Define the default file directory name'
                       'contains input/output/log/config directories')

# Define whether we want to generate a config file
Consts.add('GENERATE_CONFIG_FILE', value=False, dtype=bool,
           source=__NAME__, user=False, argument=True, group=group,
           description='Define whether we want to generate a config file',
           command='--getconfig')

# Define an output directory
Consts.add('DIRECTORY', value=None, dtype=str, source=__NAME__, user=False,
           argument=True, group=group,
           description='Define the working directory '
                       '(note this can also be set by setting the AMIDIR '
                       'environmental variable)',
           command='--dir')

# set the environmental variable for ami sim dir
Consts.add('ENV_DIR', value=base.ENV_DIR, dtype=str, source=__NAME__,
           user=False, argument=False, group=group,
           description='set the environmental variable for ami sim directory'
                       '(If unset defaults to ~/{PACKAGE_NAME}/')

# set debug mode
Consts.add('DEBUG', value=0, dtype=int, source=__NAME__, user=True,
           argument=True, group=group,
           description='Set debug mode (1-9) the higher the number the '
                       'more verbose',
           command='--debug')

# =============================================================================
#   Instrument constants
# =============================================================================
group = 'instrument'

# Define the pixel scale (expected to be very close to 0.065 arcsec/pixel)
Consts.add('PIX_SCALE', value=0.065, dtype=float, units=uu.arcsec / uu.pixel,
           source=__NAME__, user=False, argument=False, group=group,
           description='Define the pixel scale (expected to be very close '
                       'to 0.065 arcsec/pixel)',
           path='instrument.pix_scale')

# Define all allowed filters
FILTERS = ['F277W', 'F380M', 'F430M', 'F480M']
Consts.add('ALL_FILTERS', value=FILTERS,
           dtype=list, source=__NAME__, user=False, argument=False,
           group=group,
           description='Define all allowed filters')

# =============================================================================
#   Simulation constants
# =============================================================================
group = 'Simulation'
# Define the pupil mask (for use when recomputing PSF)
Consts.add('PUPIL_MASK', value='MASK_NRM', dtype=str,
           source=__NAME__, user=True, argument=False,
           group=group,
           description='Define the pupil mask (for use when recomputing PSF)')

# -----------------------------------------------------------------------------
# psf filter constants (depends on which filters we want)
for _filter in FILTERS:
    # Define the psf {FILTER} location
    Consts.add('PSF_{0}_PATH'.format(_filter), value=None, dtype=str,
               source=__NAME__, user=False, argument=False, group=group,
               description='Define the {0} psf location'.format(_filter),
               path='simulation.psf.{0}.path'.format(_filter))
    # Define whether to recompute filter {FILTER}
    Consts.add('PSF_{0}_RECOMPUTE'.format(_filter), value=False, dtype=bool,
               source=__NAME__, user=False, argument=False, group=group,
               description='Define the {0} psf location'.format(_filter),
               path='simulation.psf.{0}.recompute_psf'.format(_filter))

# Define the native image size (FOV in pixels)
Consts.add('FOV_PIXELS', value=79, dtype=int, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the native image size (FOV in pixels)',
           path='simulation.fov_pixels')

# Define the oversampling factor
Consts.add('OVERSAMPLE_FACTOR', value=11, dtype=int, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the oversampling factor',
           path='simulation.oversample')

# =============================================================================
#   AMI-SIM constants
# =============================================================================
group = 'ami-smi'
# Define switch whether to use ami-sim
Consts.add('AMISIM-USE', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define switch whether to use ami-sim',
           path='ami-sim.use')

# Define whether to produce up-the-ramp images
Consts.add('AMISMI-UPTHERAMP', value=False, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define whether to produce up-the-ramp images',
           path='ami-sim.uptheramp')

# Define whether to create calibrator (passed to ami-sim)
Consts.add('AMISMI-CREATE_CALIBRATOR', value=0, dtype=int, source=__NAME__,
           options=[0, 1], user=False, argument=False, group=group,
           description='Define whether to create calibrator '
                       '(passed to ami-sim)',
           path='ami-sim.create_calibrator')

# Define whether to overwrite AMI-SIM outputs
Consts.add('AMISIM-OVERWRITE', value=1, dtype=int, source=__NAME__,
           options=[0, 1], user=False, argument=False, group=group,
           description='Define whether to overwrite AMI-SIM outputs',
           path='ami-sim.overwrite')

# Define whether to use uniform flat field (passed to ami-sim)
Consts.add('AMISIM-UNIFORM_FLATFIELD', value=0, dtype=int, source=__NAME__,
           options=[0, 1], user=False, argument=False, group=group,
           description='Define whether to use uniform flat field '
                       '(passed to ami-sim)',
           path='ami-sim.uniform_flatfield')

# Define whether to overwrite flat-field (passed to ami-sim)
Consts.add('AMISIM-OVERWRITE_FLATFIELD', value=0, dtype=int, source=__NAME__,
           options=[0, 1], user=False, argument=False, group=group,
           description='Define whether to overwrite flat-field '
                       '(passed to ami-sim)',
           path='ami-sim.overwrite_flatfield')

# Define the random seed ami-sim uses
Consts.add('AMISIM-RANDOM_SEED', value=1, dtype=int, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the random seed ami-sim uses',
           path='ami-sim.random-seed')

# Define whether ami-sim is verbose
Consts.add('AMISIM-VERBOSE', value=1, dtype=int, source=__NAME__,
           options=[0, 1], user=False, argument=False, group=group,
           description='Define whether ami-sim is verbose',
           path='ami-sim.verbose')

# Define whether ami-sim applies jitter
Consts.add('AMISIM-APPLY_JITTER', value=0, dtype=int, source=__NAME__,
           options=[0, 1], user=False, argument=False, group=group,
           description='Define whether ami-sim applies jitter',
           path='ami-sim.apply_jitter')

# Define whether ami-sim applies dither
Consts.add('AMISIM-APPLY_DITHER', value=0, dtype=int, source=__NAME__,
           options=[0, 1], user=False, argument=False, group=group,
           description='Define whether ami-sim applies dither',
           path='ami-sim.apply_dither')

# Define whether ami-sim includes detection noise
Consts.add('AMISIM-INCLUDE_DET_NOISE', value=1, dtype=int, source=__NAME__,
           options=[0, 1], user=False, argument=False, group=group,
           description='Define whether ami-sim includes detection noise',
           path='ami-sim.include-det-noise')

# Define where ami-sim is installed (None means already in python path)
Consts.add('AMISIM-INSTALL', value=None, dtype=str, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define where ami-sim is installed (None means '
                       'already in python path)',
           path='ami-sim.install-dir')

# Define the ami-sim package to run
Consts.add('AMISIM-PACKAGE', value='ami_sim.driver_scene', dtype=str,
           source=__NAME__, user=False, argument=False, group=group,
           description='Define the ami-sim package to run',
           path='ami-sim.package')

# Define any other ami-sim packages that need importing
Consts.add('AMISIM-MODULES', value='ami_sim.pyami', dtype=str,
           source=__NAME__, user=False, argument=False, group=group,
           description='Define any other ami-sim packages that need importing',
           path='ami-sim.modules')

# =============================================================================
#   Mirage constants
# =============================================================================

# =============================================================================
#   DMS constants
# =============================================================================

# =============================================================================
#   AMICAL constants
# =============================================================================

# =============================================================================
#   IMPLANEIA constants
# =============================================================================

