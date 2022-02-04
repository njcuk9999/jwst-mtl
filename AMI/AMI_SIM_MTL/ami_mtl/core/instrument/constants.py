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
               path='ami-sim.psf.{0}.path'.format(_filter))
    # Define whether to recompute filter {FILTER}
    Consts.add('PSF_{0}_RECOMPUTE'.format(_filter), value=False, dtype=bool,
               source=__NAME__, user=False, argument=False, group=group,
               description='Define the {0} psf location'.format(_filter),
               path='ami-sim.psf.{0}.recompute_psf'.format(_filter))

# Define the native image size (FOV in pixels)
Consts.add('FOV_PIXELS', value=79, dtype=int, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the native image size (FOV in pixels)',
           path='ami-sim.fov_pixels')

# Define the oversampling factor
Consts.add('OVERSAMPLE_FACTOR', value=11, dtype=int, source=__NAME__,
           user=False, argument=False, group=group,
           description='Define the oversampling factor',
           path='ami-sim.oversample')

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
           path='ami-sim.out_path')

# Define whether to create scene (if False will try to load from disk)
Consts.add('AMISIM-CREATE_SCENE', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define whether to create scene (if False will try to '
                       'load from disk)',
           path='ami-sim.create_scene')

# Define whether to create simulation (using AMI SIM)
Consts.add('AMISIM-CREATE_SIM', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define whether to create simulation (using AMI SIM)',
           path='ami-sim.create_simulation')

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

# Define the output directory to save mirage sim files to
Consts.add('MIRAGE-PATH', value=None, dtype=str, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define path to save ami-sim files to / read ami-sim '
                       'scenes from',
           path='mirage.out_path')

# Define the psf directory path to get psfs from (if None uses defaults)
Consts.add('MIRAGE-PSF-DIR', value=None, dtype=str, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the psf directory path to get psfs from '
                       '(if None uses defaults)',
           path='mirage.psf.path')

# Define the mirage psf wing threshold file
Consts.add('MIRAGE-PSF-WING-TFILE', value='config', dtype=str, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the mirage psf wing threshold file',
           path='mirage.psf.psf_wing_threshold_file')

# Define the reference file path (if None all default reference files are used)
Consts.add('MIRAGE-REFFILE-DIR', value=None, dtype=str, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the reference file path (if None all default '
                       'reference files are used)',
           path='mirage.reffiles.path')

# Define the astrometric reference file to use (None uses default)
Consts.add('MIRAGE-REFFILE-ASTROMETRIC', value=None, dtype=str,
           source=__NAME__, user=True, argument=False, group=group,
           description='Define the astrometric reference file to use '
                       '(None uses default)',
           path='mirage.reffiles.astrometric')

# Define the gain reference file to use (None uses default)
Consts.add('MIRAGE-REFFILE-GAIN', value=None, dtype=str,
           source=__NAME__, user=True, argument=False, group=group,
           description='Define the gain reference file to use '
                       '(None uses default)',
           path='mirage.reffiles.gain')

# Define the pixel flat reference file to use (None uses default)
Consts.add('MIRAGE-REFFILE-PIXELFLAT', value=None, dtype=str,
           source=__NAME__, user=True, argument=False, group=group,
           description='Define the pixel flat reference file to use '
                       '(None uses default)',
           path='mirage.reffiles.pixelflat')

# Define the super bias reference file to use (None uses default)
Consts.add('MIRAGE-REFFILE-SUPERBIAS', value=None, dtype=str,
           source=__NAME__, user=True, argument=False, group=group,
           description='Define the super bias reference file to use '
                       '(None uses default)',
           path='mirage.reffiles.superbias')

# =============================================================================
#   DMS constants
# =============================================================================
group = 'dms'

# =============================================================================
#   AMICAL general constants
# =============================================================================
group = 'amical-general'
# Define switch to use ami-sim input (requied as may not want to use the
#    ami-sim inputs)
Consts.add('AMICAL-INPUT-AMISIM', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define switch to use ami-sim input (requied as may '
                       'not want to use the ami-sim inputs)',
           path='amical.use.ami-sim_input')

# Define switch to use mirage input (requied as may not want to use the
#    mirage inputs)
Consts.add('AMICAL-INPUT-MIRAGE', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define switch to use mirage input (requied as may '
                       'not want to use the mirage inputs)',
           path='amical.use.mirage_input')

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

# =============================================================================
#   AMICAL extraction constants
# =============================================================================
group = 'amical-ext'
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
           user=True, argument=False, group=group,
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

# Define path to save amical extraction files to / read ami-cal extraction from
Consts.add('AMICAL_EXT_PATH', value=None, dtype=str, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define path to save ami-cal extraction files to / '
                       'read ami-cal extraction from',
           path='amical.extract.out_path')

# =============================================================================
#   AMICAL extraction constants
# =============================================================================
group = 'amical-ana'
# Define path to save amical analysis files to ami-cal.analysis.path
Consts.add('AMICAL_ANA_PATH', value=None, dtype=str, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define path to save ami-cal analysis files to',
           path='amical.analysis.out_path')

# Define whether to use candid in amical analysis
Consts.add('AMICAL_ANA_USE_CANDID', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define whether to use candid in amical analysis',
           path='amical.analysis.candid.use')

# Define whether to use pymask in amical analysis
Consts.add('AMICAL_ANA_USE_PYMASK', value=False, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define whether to use pymask in amical analysis',
           path='amical.analysis.pymask.pymask')

# Define candid inner radius of the grid [mas]
Consts.add('AMICAL_CANDID_RMIN', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define candid inner radius of the grid [mas]',
           path='amical.analysis.candid.rmin')

# Define candid outer radius of the grid [mas]
Consts.add('AMICAL_CANDID_RMAX', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define candid outer radius of the grid [mas]',
           path='amical.analysis.candid.rmax')

# Define candid grid sampling size
Consts.add('AMICAL_CANDID_STEP', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define candid grid sampling size',
           path='amical.analysis.candid.step')

# Define candid number of cores for multiprocessing
Consts.add('AMICAL_CANDID_NCORE', value=1, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define candid number of cores for multiprocessing',
           path='amical.analysis.candid.ncore')

# Define candid stellar diameter of the primary star [mas]
Consts.add('AMICAL_CANDID_DIAM', value=0, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define candid stellar diameter of the primary star [mas]',
           path='amical.analysis.candid.diam')

# Define the pymask prior on the separation
Consts.add('AMICAL_PYMASK_SEP_PRIOR', value=None, dtype=list, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask prior on the separation',
           path='amical.analysis.pymask.sep_prior')

# Define the pymask prior on the position angle
Consts.add('AMICAL_PYMASK_PA_PRIOR', value=None, dtype=list, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask prior on the position angle',
           path='amical.analysis.pymask.pa_prior')

# Define the pymask prior on the contrast ratio
Consts.add('AMICAL_PYMASK_CR_PRIOR', value=None, dtype=list, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask prior on the contrast ratio',
           path='amical.analysis.pymask.cr_prior')

# Define the pymask number of cores for multiprocessing
Consts.add('AMICAL_PYMASK_NCORE', value=1, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask number of cores for multiprocessing',
           path='amical.analysis.pymask.ncore')

# Define the extra error pymask parameter
Consts.add('AMICAL_PYMASK_EXTRA_ERR', value=0, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the extra error pymask parameter',
           path='amical.analysis.pymask.extra_error')

# Define the error scale pymask parameter
Consts.add('AMICAL_PYMASK_ERR_SCALE', value=0, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the extra error pymask parameter',
           path='amical.analysis.pymask.err_scale')

# Define the pymask mcmc number of iterations
Consts.add('AMICAL_PYMASK_MCMC_NITERS', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask mcmc number of iterations',
           path='amical.analysis.pymask.mcmc.niters')

# Define the pymask mcmc number of walkers
Consts.add('AMICAL_PYMASK_MCMC_NWALKERS', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask mcmc number of walkers',
           path='amical.analysis.pymask.mcmc.walkers')

# Define the pymask mcmc initial guess
Consts.add('AMICAL_PYMASK_MCMC_IGUESS', value=None, dtype=list, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask mcmc initial guess',
           path='amical.analysis.pymask.mcmc.initial_guess')

# Define the pymask mcmc burn in
Consts.add('AMICAL_PYMASK_MCMC_NBURN', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask mcmc burn in',
           path='amical.analysis.pymask.mcmc.burn_in')

# Define the pymask cr limit nsim parameter
Consts.add('AMICAL_PYMASK_CR_NSIM', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask cr limit nsim parameter',
           path='amical.analysis.pymask.crlimit.nsim')

# Define the pymask cr limit ncore parameter
Consts.add('AMICAL_PYMASK_CR_NCORE', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask cr limit ncore parameter',
           path='amical.analysis.pymask.crlimit.ncore')

# Define the pymask cr limit smax parameter
Consts.add('AMICAL_PYMASK_CR_SMAX', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask cr limit ncore parameter',
           path='amical.analysis.pymask.crlimit.smax')

# Define the pymask cr limit nsep parameter
Consts.add('AMICAL_PYMASK_CR_NSEP', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask cr limit nsep parameter',
           path='amical.analysis.pymask.crlimit.nsep')

# Define the pymask cr limit cmax parameter
Consts.add('AMICAL_PYMASK_CR_CMAX', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask cr limit cmax parameter',
           path='amical.analysis.pymask.crlimit.cmax')

# Define the pymask cr limit nth parameter
Consts.add('AMICAL_PYMASK_CR_NTH', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask cr limit nth parameter',
           path='amical.analysis.pymask.crlimit.nth')

# Define the pymask cr limit ncrat parameter
Consts.add('AMICAL_PYMASK_CR_NCRAT', value=None, dtype=int, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define the pymask cr limit ncrat parameter',
           path='amical.analysis.pymask.crlimit.ncrat')

# Define whether to plot candid and/or pymask results
Consts.add('AMICAL_ANA_PLOT', value=True, dtype=bool, source=__NAME__,
           user=True, argument=False, group=group,
           description='Define whether to plot candid and/or pymask results',
           path='amical.analysis.plot')


# =============================================================================
#   IMPLANEIA constants
# =============================================================================
group = 'implaneia'

# =============================================================================
#   End of constants file
# =============================================================================
