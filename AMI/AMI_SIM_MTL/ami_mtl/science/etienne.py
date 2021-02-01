#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-02-01

@author: cook
"""
import importlib
import numpy as np
import os
from pathlib import Path
import sys
from typing import Union
import webbpsf

from ami_mtl.io import drs_file
from ami_mtl.core.base import base
from ami_mtl.core.core import log_functions
from ami_mtl.core.core import param_functions
from ami_mtl.core.core import general

# =============================================================================
# Define variables
# =============================================================================
# set name
__NAME__ = 'blank.py'
__DESCRIPTION__ = 'description here'
# set very basic constants
__VERSION__ = base.VERSION
__DATE__ = base.DATE
# set up the logger
log = log_functions.Log()
# get checker
ParamDict = param_functions.ParamDict
get_param = param_functions.get_param
# get general functions
display_func = general.display_func


# =============================================================================
# Define functions
# =============================================================================
def simple_target_scene(params: ParamDict) -> ParamDict:
    """
    Create a simple target scene and save it to disk

    Construct a very simple sky scene on a oversampled pixel grid.
    The scene has 2 pixels that are non-zero, one for the primary, one for
    the secondary and the contrast between the two is defined by the
    'contrast' parameter  note that the separation/pa will not be exactly
    the same as the input values as we fall on integer pixels.

    :param params: ParamDict, the parameter dictionary of constants

    :return: ParamDict, the parameter dictionary of the simple scene
    """
    # set function name
    func_name = display_func('simple_scene', __NAME__)
    # -------------------------------------------------------------------------
    # Get parameters
    # -------------------------------------------------------------------------
    # get fov_pixels
    fov_pixels = get_param(params, 'FOV_PIXELS')
    # get the oversample function
    oversample = get_param(params, 'OVERSAMPLE_FACTOR')
    # get position angle
    position_angle = get_param(params, 'OBJ_COMP_PA')
    # get separation
    separation = get_param(params, 'OBJ_COMP_SEP')
    # get pixel scale
    pix_scale = get_param(params, 'PIX_SCALE')
    # get extracted flux
    ext_flux = get_param(params, 'EXT_FLUX')
    # get total exposure time in seconds
    tot_exp = get_param(params, 'TOT_EXP')
    # get companion contrast
    contrast = get_param(params, 'OBJ_COMP_CONTRAST')
    # get the target name
    target_name = get_param(params, 'OBJ_NAME')
    # get the filter
    filter = get_param(params, 'FILTER')
    # get output path
    output_dir = Path(str(params['OUTPUTDIR']))
    # -------------------------------------------------------------------------
    # set up
    # -------------------------------------------------------------------------
    # work out the over sample pixel width
    osample_pix_width = fov_pixels * oversample
    # create an image full of zeros
    image = np.zeros([osample_pix_width, osample_pix_width])
    # get the central pixel
    xcen = image.shape[1] // 2
    ycen = image.shape[0] // 2
    # calculate count rate
    count_rate = ext_flux * tot_exp
    # clean the target name
    target_name = general.clean_name(target_name)
    # construct filename for sky
    oargs = [target_name, filter]
    output_file = 'SKY_SCENE_{0}_{1}.fits'.format(*oargs)
    # construct path
    outpath = output_dir.joinpath(output_file)
    # -------------------------------------------------------------------------
    # Calculate pixel position for companion
    # -------------------------------------------------------------------------
    # get the dx and dy in pixels
    dx = np.cos(np.pi * position_angle / 180) * (separation / pix_scale)
    dy = np.sin(np.pi * position_angle / 180) * (separation / pix_scale)
    # get oversampling
    odx = dx * oversample
    ody = dy * oversample
    # get the x and y center for companion
    xcen2 = int(xcen + odx)
    ycen2 = int(ycen + ody)
    # -------------------------------------------------------------------------
    # Add primary + companion to image
    # -------------------------------------------------------------------------
    # add primary
    image[xcen, ycen] = count_rate
    # add companion
    image[xcen2, ycen2] = count_rate * contrast
    # -------------------------------------------------------------------------
    # Work out true separation and angle (given the rounding to the nearest
    #   pixel)
    # -------------------------------------------------------------------------
    # true separation
    tx2 = (xcen - int(xcen2)) ** 2
    ty2 = (ycen - int(ycen2)) ** 2
    true_sep = np.sqrt(tx2 + ty2) * pix_scale / oversample
    # true position angle
    ay2 = ycen - ycen2
    true_angle = 360 - (180 * np.arcsin(ay2 / np.sqrt(tx2 + ty2) / np.pi + 180))
    # log true separation and angle
    msg = 'True separation = {0:.4f} arcsec \nTrue angle = {0:.4f} deg'
    margs = [true_sep, true_angle]
    params.log.info(msg.format(*margs))
    # -------------------------------------------------------------------------
    # Write sky file
    # -------------------------------------------------------------------------
    header = drs_file.Header()
    header['IN_SEP'] = (separation, 'Input separation of scene [arcsec]')
    header['IN_ANG'] = (position_angle, 'Input position angle of scene [deg]')
    header['TRUE_SEP'] = (true_sep, 'True separation of scene [arcsec]')
    header['TRUE_ANG'] = (true_angle, 'True position angle of scene [deg]')
    header['COUNT0'] = (count_rate, 'Count rate of primary')
    header['COUNT1'] = (count_rate * contrast, 'Count rate of companion')
    # write file
    drs_file.write_fits(params, outpath, data=image, header=header,
                        overwrite=True)
    # -------------------------------------------------------------------------
    # Add outputs to out params
    # -------------------------------------------------------------------------
    props = ParamDict()
    props['COUNT_RATE'] = count_rate
    props['SKYFILE'] = outpath
    # add source
    keys = ['CONTRAST', 'SKY_FILE']
    props.set_sources(keys, func_name)
    # return properties
    return props


def recompute_psf(filter: str, filename: Union[str, Path],
                  fov_pixels: int, oversample: int,
                  pupil_mask: str):
    """
    Recompute the PSF using webbpsf

    :param filter: str, the filter to use
    :param filename: str, the output filename
    :param fov_pixels: int, the fov in pixels
    :param oversample: int, the oversampling factor
    :param pupil_mask: str, the pupil mask to use

    :return: None
    """
    # get niriss instance from webb psf
    niriss = webbpsf.NIRISS()
    # set the filter name
    niriss.filter = filter
    # set the pupil mask
    niriss.pupil_mask = pupil_mask
    # run the psf calculation
    niriss.calc_psf(filename, fov_pixels=fov_pixels, oversample=oversample)


def ami_sim(params: ParamDict):
    # get install parameter
    install_location = params['AMISIM-INSTALL']
    # get ami sim module
    install_package = params['AMISIM-PACKAGE']
    # get any other ami sim modules that need adding
    install_modules = params.listp('AMISIM-MODULES', dtype=str)
    # deal with install location being set
    if install_location is not None:
        # add install location (if not present)
        if install_location not in sys.path:
            sys.path.append(str(install_location))
        # add modules (if not present)
        if install_modules is not None:
            # get modules (split the string)
            for imod in install_modules:
                # construct path
                ipath = os.path.join(str(install_location), imod)
                idir = os.path.dirname(ipath)
                # add to sys.path if not present
                if idir not in sys.path:
                    sys.path.append(idir)
    # now try to import module
    try:
        return importlib.import_module(str(install_package))
    except Exception as e:
        # generate error message
        emsg = 'AMI-SIM Error: Cannot run {0}'.format(install_package)
        if install_location is not None:
            emsg += '\n\tLocation = {0}'.format(install_location)
        emsg += '\n\tError {0}: {1}'.format(type(e), str(e))
        params.log.error(emsg)


def run_ami_sim(params: ParamDict, sky_scene: Union[str, Path],
                count_rate: float):
    # set function name
    func_name = display_func('run_ami_sim', __NAME__)
    # -------------------------------------------------------------------------
    # Get parameters
    # -------------------------------------------------------------------------
    # get PSF file
    psf_filename = drs_file.get_param(params, 'PSF')
    # get fov_pixels
    fov_pixels = get_param(params, 'FOV_PIXELS')
    # get the oversample function
    oversample = get_param(params, 'OVERSAMPLE_FACTOR')
    # get the target name
    target_name = get_param(params, 'OBJ_TARGET_NAME')
    # clean the target name
    target_name = general.clean_name(target_name)
    # -------------------------------------------------------------------------
    # Check compatibility between PSF and SKY
    # -------------------------------------------------------------------------
    if psf_filename.exists():
        # get psf header
        psf_header = drs_file.read_fits(params, filename=psf_filename,
                                        get_data=False)
        # work out the over sample pixel width
        osample_pix_width = fov_pixels * oversample
        # compare NAXIS1 and NAXIS2
        if psf_header['NAXIS1'] != osample_pix_width:
            params['RECOMPUTE_PSF'] = True
            params.set_source('RECOMPUTE_PSF', func_name)
        if psf_header['NAXIS2'] != osample_pix_width:
            params['RECOMPUTE_PSF'] = True
            params.set_source('RECOMPUTE_PSF', func_name)
    else:
        params['RECOMPUTE_PSF'] = True
        params.set_source('RECOMPUTE_PSF', func_name)
    # -------------------------------------------------------------------------
    # Deal with recomputing PSF
    # -------------------------------------------------------------------------
    if params['RECOMPUTE_PSF']:
        # log that we are recomputing PSF
        params.log.info('Recomputing PSF')
        # recompute the AMI PSF
        recompute_psf(filter=str(params['FILTER']), filename=psf_filename,
                      pupil_mask=str(params['PUPIL_MASK']),
                      fov_pixels=fov_pixels, oversample=oversample)
    else:
        # log that we are not recomputing PSF
        params.log.info('Not recomputing PSF')
    # -------------------------------------------------------------------------
    # Get arguments for ami-sim
    # -------------------------------------------------------------------------
    try:
        # add arguments
        args = []
        # output directory path (relative to home directory)
        args += ['--target_dir', params['OUTPUTDIR']]
        # absolute output directory path, if specified it overrides --target_dir
        args += ['--output_absolute_path', params['OUTPUTDIR']]
        # overwrite yes/no, default 0 (no)
        args += ['--overwrite', params['AMISIM-OVERWRITE']]
        # generate up-the-ramp fits file? yes/no, default 0 (no)
        args += ['--uptheramp', params['AMISMI-UPTHERAMP']]
        # filter name (upper/lower case)
        args += ['--filter', params['FILTER']]
        # absolute path to oversampled PSF fits file. Spectral type set in this
        args += ['--psf', params['PSF']]
        # absolute path to oversampled sky scene fits file, normalized to sum
        #   to unity
        args += ['--sky', str(sky_scene)]
        # sky scene oversampling (must be odd integer number)
        args += ['--oversample', params['OVERSAMPLE_FACTOR']]
        # number of integrations (IR community calls these exposures sometimes)
        args += ['--nint', params['NINT']]
        # number of up-the-ramp readouts
        args += ['--ngroups', params['NGROUP']]
        # create calibrator observation yes/no default 1 (yes)
        args += ['--create_calibrator', params['AMISMI-CREATE_CALIBRATOR']]
        # Photon count rate on 25m^2 per sec in the bandpass
        args += ['--countrate', count_rate]
        # Tag to include in the names of the produced files
        args += ['--tag', target_name]
        # Generate random-noise flatfield (default) or uniform noiseless
        #     flatfield (if set to 1)
        args += ['--uniform_flatfield', params['AMISIM-UNIFORM_FLATFIELD']]
        # Random seed for all noise generations (seed is altered for every
        #    integration), allows for well-controlled simulations
        args += ['--random_seed', params['AMISIM-RANDOM_SEED']]
        # Directory for simulated flatfield. Defaults to targetDir.
        args += ['--flatfield_dir', params['OUTPUTDIR']]
        # Overwrite simulated flatfield. Defaults to No.
        args += ['--overwrite_flatfield', params['AMISIM-OVERWRITE_FLATFIELD']]
        # Verbose output to screen. Default is off
        args += ['--verbose', params['AMISIM-VERBOSE']]
        # Dither the observations. Default is on
        args += ['--apply_dither', params['AMISIM-APPLY_DITHER']]
        # Include pointing errors in the observations. Default is on
        args += ['--apply_jitter', params['AMISIM-APPLY_JITTER']]
        # Include photon noise, read noise, background noise, and dark
        #     current. Default is on
        args += ['--include_detection_noise',
                 params['AMISIM-INCLUDE_DET_NOISE']]
        # load module
        mod = ami_sim(params)
        # run module main function
        mod.main(args)

    except Exception as e:
        # log error
        emsg = 'AMI-SIM Error: {0}: {1}'
        eargs = [type(e), str(e)]
        params.log.error(emsg.format(*eargs))

    # -------------------------------------------------------------------------
    # Construct names of ami-sim outputs
    # -------------------------------------------------------------------------
    psf_filename = os.path.basename(params['PSF']).split('.fits')[0]

    oargs = [target_name, params['FILTER'], psf_filename]
    outfile = 't_{0}_{1}_{2}_Obs1_00.fits'


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================
