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
Consts.add('DIRECTORY', value='.', dtype=str, source=__NAME__, user=False,
           argument=True, group=group,
           description='Define the working directory '
                       '(note this can also be set by setting the AMIDIR '
                       'environmental variable)',
           command='--dir', path='general.out_dir')

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
#   APT constants
# =============================================================================
group = 'APT'

# Define the APT target dictionaries (to be filled in code)
Consts.add('APT-TARGETS', value=None, dtype=dict, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the APT target dictionaries '
                       '(to be filled in code)')

# Define the APT target name
Consts.add('APT-TARGET-NAME', value=None, dtype=list, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the APT target name',
           apt='TargetID')

# Define the APT-xml target Right Ascension
Consts.add('APT-TARGET-RA', value=None, dtype=list, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the APT target Right Ascension',
           apt='TargetRA')

# Define the APT-xml target Declination
Consts.add('APT-TARGET-DEC', value=None, dtype=list, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the APT target Declination',
           apt='TargetDec')

# Define the APT-xml target number of groups
Consts.add('APT-TARGET-NGROUP', value=None, dtype=list, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the APT-xml target number of groups',
           apt='Groups')

# Define the APT-xml target number of groups
Consts.add('APT-TARGET-NINT', value=None, dtype=list, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the APT-xml target number of integrations',
           apt='Integrations')

# Define the APT-xml sub array
Consts.add('APT-TARGET-SUBARRAYS', value=None, dtype=list, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the APT-xml sub array',
           apt='Subarray')

# Define the APT-xml filter list
Consts.add('APT-TARGET-FILTERS', value=None, dtype=list, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the APT-xml filters',
           apt='Filter')


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

# Define filter zero point (should have all filters from ALL_FILTERS)
ZEROPOINTS = dict(F277W=26.14, F380M=23.75, F430M=23.32, F480M=23.19)
Consts.add('ZEROPOINTS', value=ZEROPOINTS, dtype=dict, source=__NAME__,
           user=False, argument=False, group=group,
           description='Zero point for all filters')

# Define the valid sub array names
SUB_ARRAYS = ['FULL', 'SUB80']
Consts.add('SUBARRAYS', value=SUB_ARRAYS, dtype=list, source=__NAME__,
           user=False, argument=False, group=group,
           description='the valid sub array names')

# Define the default subarray (if APT file is set to None)
Consts.add('DEFAULT_SUBARRAY', value='SUB80', dtype=str, source=__NAME__,
           user=False, argument=False, group=group, options=SUB_ARRAYS,
           description='the default subarray (if APT file is set to None)')

# Define the frame time for different subarrays
T_FRAMES = dict(FULL=10.737, SUB80=0.07544)
Consts.add('T_FRAMES', value=T_FRAMES, dtype=dict, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the frame time for subarray FULL')


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

# Define path to save ami-sim files to / read ami-sim scenes from
Consts.add('AMISIM-PATH', value=None, dtype=str, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define path to save ami-sim files to / read ami-sim '
                       'scenes from',
           path='ami-sim.path')

# loop around filters
for _filter in FILTERS:
    # Define ami-sim scene file to save scenes (only use this if
    #     not creating scene)
    Consts.add('AMI-SIM-SCENE-{0}'.format(_filter), value=None, dtype=str,
               source=__NAME__, user=False, argument=False, group=group,
               description='Define ami-sim scene file to save scene '
                           '(filter {0}) (only use this if not creating '
                           'scene)'.format(_filter),
               path='ami-sim.scene.{0}'.format(_filter))
    # Define ami-sim output file (only use this if not running ami-sim)
    Consts.add('AMI-SIM-OUT_{0}'.format(_filter), value=None, dtype=str,
               source=__NAME__, user=False, argument=False, group=group,
               description='Define ami-sim output file (only use this if not '
                           'running ami-sim)',
               path='ami-sim.out.{0}'.format(_filter))

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
group = 'mirage'
# Define switch whether to use Mirage
Consts.add('MIRAGE-USE', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define switch whether to use Mirage',
           path='mirage.use')

# =============================================================================
#   DMS constants
# =============================================================================
group = 'dms'

# =============================================================================
#   AMICAL constants
# =============================================================================
group = 'amical'
# Define switch to use ami-cal extraction
Consts.add('AMICAL-EXT-USE', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define switch to use ami-cal extraction',
           path='amical.use.extraction')

# Define switch to use ami-cal analysis (requires ami-cal extraction done)
Consts.add('AMICAL-ANALYSIS-USE', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define switch to use ami-cal analysis '
                       '(requires ami-cal extraction done)',
           path='amical.use.analysis')

# Define whether object is fake (for amical extract save)
#    - i.e. observables are extracted from simulated data this means simbad
#    search is ignored
Consts.add('AMICAL_EXT_FAKE_OBJ', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define whether object is fake (for amical extract save)'
                       ' - i.e. observables are extracted from simulated data '
                       'this means simbad search is ignored',
           path='amical.extract.fake_obj')

# Define whether to plot amical extraction plots
Consts.add('AMICAL_EXT_DISPLAY_PLOT', value=False, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define whether to plot amical extraction plots',
           path='amical.extract.display_plots')

# Define the name of the mask used in amical extraction
Consts.add('AMICAL_EXT_MASK_NAME', value='g7', dtype=str, source=__NAME__,
           user=True, arguemnt=False, group=group,
           description='Define the name of the mask used in amical extraction',
           path='amical.extract.mask_name')

# Define whether to use the multiple triangle technique to compute the
#    bispectrum
Consts.add('AMICAL_EXT_BS_MULTI_TRI', value=False, dtype=bool, source=__NAME__,
           description='Define whether to use the multiple triangle technique '
                       'to compute the bispectrum',
           path='amical.extract.bs_multi_tri')

# Define which of the 3 methods are used to sample to u-v space
# - 'fft' uses fft between individual holes to compute the expected
#         splodge position;
# - 'square' compute the splodge in a square using the expected fraction
#            of pixel to determine its weight;
# - 'gauss' considers a gaussian splodge (with a gaussian weight) to get
#           the same splodge side for each n(n-1)/2 baselines
Consts.add('AMICAL_EXT_PEAK_METHOD', value='fft', dtype=str, source=__NAME__,
           description='Define which of the 3 methods are used to sample '
                       'to u-v space. "fft" uses fft between individual '
                       'holes to compute the expected splodge position '
                       '"square" compute the splodge in a square using the '
                       'expected fraction of pixel to determine its weight '
                       '"gauss" considers a gaussian splodge (with a gaussian '
                       'weight) to get the same splodge side for each n(n-1)/2 '
                       'baselines',
           path='amical.extract.peak_method')

# Define the hole diameter for amical extract
Consts.add('AMICAL_EXT_HOLE_DIAMETER', value=0.8, dtype=float, source=__NAME__,
           description='Define the hole diameter for amical extract',
           path='amical.extract.hole_diameter')

# Define the cut off for amical extract
Consts.add('AMICAL_EXT_CUTOFF', value=1e-4, dtype=float, source=__NAME__,
           description='Define the cut off for amical extract',
           path='amical.extract.cutoff')

# Define the relative size of the splodge used to compute multiple triangle
#    indices and the fwhm of the 'gauss' technique
Consts.add('AMICAL_EXT_FW_SPLODGE', value=0.7, dtype=float, source=__NAME__,
           description='Define the relative size of the splodge used to compute '
                       'multiple triangle indices and the fwhm of the "gauss" '
                       'technique',
           path='amical.extract.fw_splodge')

# Define switch, if True, the uncertainties are computed using the std of the
#     overall cvis or bs array. Otherwise, the uncertainties are computed using
#     covariance matrice
Consts.add('AMICAL_EXT_NATIVE_ERR', value=False, dtype=bool, source=__NAME__,
           description='Define switch, if True, the uncertainties are computed '
                       'using the std of the overall cvis or bs array. '
                       'Otherwise, the uncertainties are computed using '
                       'covariance matrice',
           path='amical.extract.native_err')

# Define the number of elements to sample the spectral filters
Consts.add('AMICAL_EXT_N_WL', value=3, dtype=int, source=__NAME__,
           description='Define the number of elements to sample the spectral '
                       'filters',
           path='amical.extract.n_wl')

# Define the number of separated blocks use to split the data cube and get
#     more accurate uncertainties (default: 0, n_blocks = n_ps)
Consts.add('AMICAL_EXT_N_BLOCKS', value=0, dtype=int, source=__NAME__,
           description='Define the number of separated blocks use to split the '
                       'data cube and get more accurate uncertainties '
                       '(default: 0, n_blocks = n_ps)',
           path='amical.extract.n_block')

# Define the angle in [deg] to rotate the mask compare to the detector
#      (if the mask is not perfectly aligned with the detector, e.g.: VLT/VISIR)
Consts.add('AMICAL_EXT_THETA_DET', value=0, dtype=float, source=__NAME__,
           description='Define the angle in [deg] to rotate the mask '
                       'compare to the detector (if the mask is not perfectly '
                       'aligned with the detector, e.g.: VLT/VISIR)',
           path='amical.extract.theta_detector')

# Define the scaling in the UV (Only used for IFU data)
Consts.add('AMICAL_EXT_SCALING_UV', value=1, dtype=float, source=__NAME__,
           description='Define the scaling in the UV (Only used for IFU data)',
           path='amical.extract.scaling_uv')

# Define i_wl
Consts.add('AMICAL_EXT_I_WL', value=None, dtype=int, source=__NAME__,
           description='Define i_wl', path='amical.extract.i_wl')

# Define the switch, if True the squared visibilities are unbiased using
# the Fourier base
Consts.add('AMICAL_EXT_UNBIAS_V2', value=True, dtype=bool, source=__NAME__,
           description='Define the switch, if True the squared visibilities '
                       'are unbiased using the Fourier base',
           path='amical.extract.unbias_v2')

# Define whether to compute CP cov
Consts.add('AMICAL_EXT_COMP_CP_COV', value=True, dtype=bool, source=__NAME__,
           description='Define whether to compute CP cov',
           path='amical.extract.comp_cp_cov')

# Define whether to do the expert plot
Consts.add('AMICAL_EXT_EXPERT_PLOT', value=False, dtype=bool, source=__NAME__,
           description='Define whether to do the expert plot',
           path='amical.extract.expert_plot')

# Define whether to print useful information during the amical extract process
Consts.add('AMICAL_EXT_VERBOSE', value=False, dtype=bool, source=__NAME__,
           description='Define whether to print useful information during the '
                       'amical extract process',
           path='amical.extract.verbose')

# =============================================================================
#   IMPLANEIA constants
# =============================================================================
group = 'implaneia'

# =============================================================================
#   End of constants file
# =============================================================================
