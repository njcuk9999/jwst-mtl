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
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
from typing import Tuple, Union, Optional
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
__NAME__ = 'science.etienne.py'
__DESCRIPTION__ = 'Functions adapted from Etienne Artigau'
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
def ami_sim_observation(fov_pixels: float, oversample: int,
                        pix_scale: float, ext_flux: float,
                        tot_exp: float) -> Tuple[np.ndarray, ParamDict]:
    """
    Create a simple target scene (one point source at the center)

    :param fov_pixels: the native image size (FOV in pixels)
    :param oversample: the oversampling factor
    :param pix_scale: pixel scale (expected to be very close to 0.065
                      arcsec/pixel)
    :param ext_flux: extracted flux in e-/s
    :param tot_exp: total exposure time in seconds

    :return: image and header
    """
    # set function name
    func_name = display_func('simple_scene', __NAME__)
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
    # -------------------------------------------------------------------------
    # Add primary
    # -------------------------------------------------------------------------
    # add primary
    image[xcen, ycen] = count_rate
    # -------------------------------------------------------------------------
    # Write sky file
    # -------------------------------------------------------------------------
    hdict = ParamDict()
    hdict['FOVPX'] = (fov_pixels, 'Input FOV pixels')
    hdict['OSAMPLE'] = (oversample, 'Input Oversample')
    hdict['PIXSCALE'] = (pix_scale, 'Input pixel scale')
    hdict['COUNT0'] = (count_rate, 'Count rate of primary')
    # set sources
    hdict.set_sources(['FOVPX', 'OSAMPLE', 'PIXSCALE', 'COUNT0'], func_name)
    # -------------------------------------------------------------------------
    return image, hdict


def ami_sim_add_companion(params: ParamDict, image: np.ndarray,
                          hdict: ParamDict, num: int,
                          position_angle: float, separation: float,
                          contrast: float, plot: bool = False
                          ) -> Tuple[np.ndarray, ParamDict]:
    """
    Add a companion to an image where the primary is assumed to be at the
    center of the image

    :param params: ParamDict, parameter dictionary of constants
    :param image: numpy array (2D) - the image before this companion is added
    :param hdict: ParamDict, the keys for header
    :param num: int, the companion number (must be unique for target)
    :param position_angle: float, the position angle of companion from
                           primary (ra-dec coord system) in degrees
    :param separation: float, separation between primary (assume to be at the
                       center or the detector) and companion in arcsec
    :param contrast: contrast of companion
    :param plot: bool, if True plots

    :return: updated  image and header
    """
    # set function name
    func_name = display_func('ami_sim_add_companion', __NAME__)
    # -------------------------------------------------------------------------
    # set up
    # -------------------------------------------------------------------------
    # get parameters
    pix_scale = float(hdict['PIXSCALE'][0])
    oversample = float(hdict['OSAMPLE'][0])
    count_rate = float(hdict['COUNT0'][0])
    # get the central pixel
    xcen = image.shape[1] // 2
    ycen = image.shape[0] // 2

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
    # Add companion to image
    # -------------------------------------------------------------------------
    # test bounds
    cond1 = (xcen2 > 0) and (xcen2 < image.shape[1])
    cond2 = (ycen2 > 0) and (ycen2 < image.shape[0])
    # add companion
    if cond1 and cond2:
        image[xcen2, ycen2] = count_rate * contrast
    # else log error
    else:
        emsg = ('AMI-SIM: Companion {0} out of bounds: PA: {1} deg '
                'Sep: {2} arcsec')
        eargs = [num, position_angle, separation]
        params.log.error(emsg.format(*eargs))
    # -------------------------------------------------------------------------
    # debug plot
    # -------------------------------------------------------------------------
    if plot:
        plt.imshow(image)
        plt.show()
        plt.close()
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
    true_angle = 180 * np.arcsin(ay2 / np.sqrt(tx2 + ty2)) / np.pi + 180
    # log true separation and angle
    msg = ('AMI-SIM: Companion {0} \n\tTrue separation = {1:.4f} arcsec '
           '\n\tTrue angle = {2:.4f} deg')
    margs = [num, true_sep, true_angle]
    params.log.info(msg.format(*margs))
    # -------------------------------------------------------------------------
    # add to hdict
    # -------------------------------------------------------------------------
    # text for comment
    ctxt = 'companion {0}'.format(num)
    # save input separation
    kw_in_sep = 'IN_SEP{0}'.format(num)
    hdict[kw_in_sep] = (separation,
                        'Input separation of scene [arcsec] {0}'.format(ctxt))
    # save input position angle
    kw_in_ang = 'IN_ANG{0}'.format(num)
    hdict[kw_in_ang] = (position_angle,
                        'Input position angle of scene [deg] {0}'.format(ctxt))
    # save true separtion
    kw_t_sep = 'T_SEP{0}'.format(num)
    hdict[kw_t_sep] = (true_sep,
                       'True separation of scene [arcsec] {0}'.format(ctxt))
    # save true position angle
    kw_t_ang = 'T_ANG{0}'.format(num)
    hdict[kw_t_ang] = (true_angle,
                       'True position angle of scene [deg] {0}'.format(ctxt))
    # save count rate of companion
    kw_count = 'COUNT{0}'.format(num)
    hdict[kw_count] = (count_rate * contrast, 'Count rate of {0}'.format(ctxt))
    # set sources
    hkeys = [kw_in_sep, kw_in_ang, kw_t_sep, kw_t_ang, kw_count]
    hdict.set_sources(hkeys, func_name)

    # -------------------------------------------------------------------------
    # return image and header
    return image, hdict


def ami_sim_add_disk(image: np.ndarray, hdict: ParamDict,
                     num: int, contrast: float, kind: str = 'disk',
                     roll: float = 0.0, inclination: Optional[float] = None,
                     width: float = 0.0, exponent: float = 2.0,
                     radius: float = 0.0, plot: bool = False):
    # set function name
    func_name = display_func('ami_sim_add_disk', __NAME__)
    # -------------------------------------------------------------------------
    # set up
    # -------------------------------------------------------------------------
    # get parameters
    pix_scale = float(hdict['PIXSCALE'][0])
    oversample = float(hdict['OSAMPLE'][0])
    count_rate = float(hdict['COUNT0'][0])
    # -------------------------------------------------------------------------
    # set up disk parameters
    # -------------------------------------------------------------------------
    # width of simulation iamge in pixels
    imagewid = image.shape[1]
    # get the x and y positions for each pixel and center it
    xpos, ypos = np.indices([imagewid, imagewid]) - imagewid / 2.0 + 0.5
    # scale these to the real size
    xpos = xpos * (pix_scale / oversample)
    ypos = ypos * (pix_scale / oversample)
    # -------------------------------------------------------------------------
    # rotate coords around "roll" degrees
    x1 = np.cos(roll * np.pi / 180) * xpos + np.sin(roll * np.pi / 180) * ypos
    y1 = -np.sin(roll * np.pi / 180) * xpos + np.cos(roll * np.pi / 180) * ypos
    # -------------------------------------------------------------------------
    # deal with a disk
    # -------------------------------------------------------------------------
    if kind == 'disk':
        # include inclination
        x2 = x1 / np.cos(inclination * np.pi / 180)
        y2 = np.array(y1)
        # get the radius
        rr = np.sqrt(x2**2 + y2**2)
        # work out the flux of the disk
        flux = np.exp(-0.5 * np.abs(rr-radius) / width) ** exponent
        # ---------------------------------------------------------------------
        # add keys to hdict
        # ---------------------------------------------------------------------
        # text for comment
        ctxt = 'companion {0}'.format(num)
        # add roll
        kw_in_roll = 'INROLL{0}'.format(num)
        hdict[kw_in_roll] = (roll,
                             'Input disk rotation on the sky plane [DEG]'
                             ' {0}'.format(ctxt))
        # add inclination
        kw_in_incl = 'ININCL{0}'.format(num)
        hdict[kw_in_incl] = (inclination,
                             'Input disk tilt towardr line of sight [DEG]'
                             ' {0}'.format(ctxt))
        # add width
        kw_in_wid = 'INWID{0}'.format(num)
        hdict[kw_in_wid] = (width,
                            'Input disk ewidth of annulus [ARCSEC]'
                            ' {0}'.format(ctxt))
        # add radius
        kw_in_rad = 'INRAD{0}'.format(num)
        hdict[kw_in_rad] = (radius,
                            'Input disk long axis radius [ARCSEC]'
                            ' {0}'.format(ctxt))
        # exponent
        kw_in_exp = 'INEXP{0}'.format(num)
        hdict[kw_in_exp] = (exponent,
                            'Input disk gaussian exponent')
        # set sources
        hkeys = [kw_in_roll, kw_in_incl, kw_in_wid, kw_in_rad, kw_in_exp]
        hdict.set_sources(hkeys, func_name)
    # -------------------------------------------------------------------------
    # deal with a bar
    # -------------------------------------------------------------------------
    # deal with a bar
    elif kind == 'bar':
        # add mask for inner gap (i.e. ring not disk)
        y1b = (np.abs(y1) - radius)/width
        # mask the region
        y1b[y1b < 0] = 0.0
        # work out the flux of the disk
        part1 = np.exp(-0.5 * np.abs((x1/width))**exponent)
        part2 = np.exp(-0.5 * y1b**exponent)
        # work out the flux of the bar
        flux = part1 * part2
        # ---------------------------------------------------------------------
        # add keys to hdict
        # ---------------------------------------------------------------------
        # text for comment
        ctxt = 'companion {0}'.format(num)
        # add roll
        kw_in_roll = 'INROLL{0}'.format(num)
        hdict[kw_in_roll] = (roll,
                             'Input bar rotation on the sky plane [DEG]'
                             ' {0}'.format(ctxt))
        # add width
        kw_in_wid = 'INWID{0}'.format(num)
        hdict[kw_in_wid] = (width,
                            'Input thickness of bar [ARCSEC]  {0}'.format(ctxt))
        # add radius
        kw_in_rad = 'INRAD{0}'.format(num)
        hdict[kw_in_rad] = (radius,
                            'Input long axis of the bar [ARCSEC]'
                            ' {0}'.format(ctxt))
        # exponent
        kw_in_exp = 'INEXP{0}'.format(num)
        hdict[kw_in_exp] = (exponent,
                            'Input bar gaussian exponent')
        # set sources
        hkeys = [kw_in_roll, kw_in_wid, kw_in_rad, kw_in_exp]
        hdict.set_sources(hkeys, func_name)
    else:
        flux = np.zeros_like(image)
    # -------------------------------------------------------------------------
    # add to the image
    image = image + (flux * count_rate * contrast)
    # -------------------------------------------------------------------------
    if plot:
        plt.imshow(image)
        plt.show()
    # -------------------------------------------------------------------------
    # return image and header
    return image, hdict

def ami_sim_save_scene(params: ParamDict, outpath: str,
                       image: np.ndarray, hdict: dict):
    """
    Save an ami sim scene to disk (using image and hdict)

    :param params: ParamDict, parameter dictionary of constants
    :param outpath: str, the path to save the file to
    :param image: numpy array 2D, the image to save
    :param hdict: dict, the keys/values/comments to add to header

    :return: None - write to fits file
    """
    # load hdict into header
    header = drs_file.Header()
    # loop around keys and add to header
    if hdict is not None:
        for key in hdict:
            if len(hdict[key]) == 1:
                header[key] = hdict[key]
            else:
                header[key] = (hdict[key][0], hdict[key][1])
    # log that we are recomputing PSF
    params.log.info('AMI-SIM: Writing file: {0}'.format(outpath))
    # write file to disk
    drs_file.write_fits(params, outpath, data=image, header=header,
                        overwrite=True)


def ami_sim_get_psf(params, path: Union[Path, str], fov_pixels: int,
                    oversample: int, _filter: str, recompute: bool = True):
    # -------------------------------------------------------------------------
    # Check compatibility between PSF and SKY
    # -------------------------------------------------------------------------
    # make psf_path a Path
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        # get psf header
        psf_header = drs_file.read_fits(params, filename=path, get_data=False)
        # work out the over sample pixel width
        osample_pix_width = fov_pixels * oversample
        # compare NAXIS1 and NAXIS2
        if psf_header['NAXIS1'] != osample_pix_width:
            recompute = True
        if psf_header['NAXIS2'] != osample_pix_width:
            recompute = True
    else:
        recompute = True
    # -------------------------------------------------------------------------
    # Deal with recomputing PSF
    # -------------------------------------------------------------------------
    if recompute:
        # log that we are recomputing PSF
        params.log.info('AMI-SIM: Recomputing PSF')
        # recompute the AMI PSF
        ami_sim_recompute_psf(_filter=_filter, filename=path,
                              pupil_mask=str(params['PUPIL_MASK']),
                              fov_pixels=fov_pixels, oversample=oversample)
    else:
        # log that we are not recomputing PSF
        params.log.info('AMI-SIM: Not recomputing PSF')
    # -------------------------------------------------------------------------
    # return psf filename
    return path


def ami_sim_recompute_psf(_filter: str, filename: Union[str, Path],
                          fov_pixels: int, oversample: int,
                          pupil_mask: str):
    """
    Recompute the PSF using webbpsf

    :param _filter: str, the filter to use
    :param filename: str, the output filename
    :param fov_pixels: int, the fov in pixels
    :param oversample: int, the oversampling factor
    :param pupil_mask: str, the pupil mask to use

    :return: None
    """
    # get niriss instance from webb psf
    niriss = webbpsf.NIRISS()
    # set the filter name
    niriss.filter = _filter
    # set the pupil mask
    niriss.pupil_mask = pupil_mask
    # TODO: This shouldn't be needed but without it calc_psf breaks?
    # TODO:  Error --> AttributeError: 'NIRISS' object has no attribute
    # TODO:                            '_extra_keywords'
    niriss._extra_keywords = []
    # run the psf calculation
    niriss.calc_psf(str(filename), fov_pixels=fov_pixels, oversample=oversample)


def ami_sim_run_code(params: ParamDict, path: str, _filter: str,
                     psf_filename: str, scene_file: str, count_rate: float,
                     simname: str, target_name: str, nint: int,
                     ngroups: int) -> str:
    """
    Run AMI-SIM code

    :param params: ParamDict, the parameter dictionary of constants
    :param path: str, the output path directory
    :param _filter: str, the filter to run AMI-SIM on
    :param psf_filename: str, the path to the PSF to use
    :param scene_file: str, the scene fits image file
    :param count_rate: float, the count rate
    :param simname: str, simulation name (used with target to name outputs)
    :param target_name: str, the target name
    :param nint: int, the number of integrations to use
    :param ngroups: int, the number of groups to use

    :return: the tag (for file name)
    """
    # construct tag
    tag = '{0}_{1}'.format(simname, target_name)
    # -------------------------------------------------------------------------
    # Get arguments for ami-sim
    # -------------------------------------------------------------------------
    try:
        # add arguments
        args = []
        # output directory path (relative to home directory)
        args += ['--target_dir', str(path)]
        # absolute output directory path, if specified it overrides --target_dir
        args += ['--output_absolute_path', str(path)]
        # overwrite yes/no, default 0 (no)
        args += ['--overwrite', params['AMISIM-OVERWRITE']]
        # generate up-the-ramp fits file? yes/no, default 0 (no)
        args += ['--uptheramp', params['AMISMI-UPTHERAMP']]
        # filter name (upper/lower case)
        args += ['--filter', _filter]
        # absolute path to oversampled PSF fits file. Spectral type set in this
        args += ['--psf', str(psf_filename)]
        # absolute path to oversampled sky scene fits file, normalized to sum
        #   to unity
        args += ['--sky', str(scene_file)]
        # sky scene oversampling (must be odd integer number)
        args += ['--oversample', params['OVERSAMPLE_FACTOR']]
        # number of integrations (IR community calls these exposures sometimes)
        args += ['--nint', nint]
        # number of up-the-ramp readouts
        args += ['--ngroups', ngroups]
        # create calibrator observation yes/no default 1 (yes)
        args += ['--create_calibrator', params['AMISMI-CREATE_CALIBRATOR']]
        # Photon count rate on 25m^2 per sec in the bandpass
        args += ['--countrate', count_rate]
        # Tag to include in the names of the produced files
        args += ['--tag', tag]
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
        # all args must be strings
        args = list(map(lambda x: str(x), args))
        # load module
        mod = ami_sim(params)
        # ---------------------------------------------------------------------
        # print that we are running amisim
        params.log.info('Running AMI SIM')
        argstring = ''
        for arg in args:
            if arg.startswith('--'):
                argstring += '\n\t' + arg
            else:
                argstring += ' ' + arg
        params.log.info(argstring)
        # ---------------------------------------------------------------------
        # run module main function (dealing with print outs)
        with general.ModifyPrintouts(text='AMI-SIM Output', flush=True,
                                     logfile=params['LOGFILE']):
            mod.main(args)
        # ---------------------------------------------------------------------
        # return the tag
        return tag

    except Exception as e:
        # log error
        emsg = 'AMI-SIM Error: {0}: {1}'
        eargs = [type(e), str(e)]
        params.log.error(emsg.format(*eargs))
    # ---------------------------------------------------------------------
    # return the tag
    return tag


def ami_sim(params: ParamDict):
    """
    Load the AMI-SIM module

    :param params: ParamDict, parameter dictionary of constants

    :return: AMI-SIM module (with a .main function)
    """
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


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================
