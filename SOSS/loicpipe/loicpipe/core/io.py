#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on {DATE}

@author: cook
"""
import glob
import os
from typing import Any, Dict, List, Union

import numpy as np
import yaml
from astropy.io import fits

from loicpipe.core import base
from loicpipe.core import constants

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.io.py'
# get Parameters class
Parameters = constants.Parameters


# =============================================================================
# Define classes
# =============================================================================
class LoicLog:
    def __init__(self, filename):
        self.filename = filename

    def write(self, message):
        with open(self.filename, 'a') as logfile:
            logfile.write(message + '\n')


# =============================================================================
# Define genearl functions
# =============================================================================
def read_yaml(yaml_filename: Union[str, None]) -> Dict[str, Any]:
    """
    Read the yaml file and add to settings

    :param yaml_filename: str, yaml file name

    :return: dict, updated settings dictionary
    """
    # deal with yaml_filename being None
    if yaml_filename is None:
        emsg = 'yaml_filename must be set to a valid file'
        raise base.LoicPipeError(emsg)
    # deal with yaml_filename not existing
    if not os.path.exists(yaml_filename):
        emsg = 'yaml_filename {0} does not exist'
        eargs = [yaml_filename]
        raise base.LoicPipeError(emsg.format(*eargs))
    # read the yaml file
    with open(yaml_filename, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    # add a profiles sub-dictionary
    settings = dict()
    # loop around yaml data
    for key, value in yaml_data.items():
        # if key is in settings
        settings[key] = value
    # return settings
    return settings


def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def load_fits(filename: str) -> np.ndarray:
    return fits.getdata(filename)


def write_dqfile(params: Parameters, results: Any):
    # get parameters used in this function (should be done at the start)
    outputdir = params['data.outdir']
    # define the dq base filename
    dq_basename = 'dq_postmanual.fits'
    # construct filename
    dq_filename = os.path.join(outputdir, dq_basename)
    # create a new hdu
    hdu = fits.PrimaryHDU(data=results.dq)
    # save to file
    hdu.writeto(dq_filename, overwrite=True)


def write_prestack(params: Parameters, results: Any):
    # get parameters used in this function (should be done at the start)
    outputdir = params['data.outdir']
    # define the dq base filename
    prestack_datafile = 'prestack_data.fits'
    prestack_dqfile = 'prestack_dq.fits'
    # construct paths
    prestack_datapath = os.path.join(outputdir, prestack_datafile)
    prestack_dqpath = os.path.join(outputdir, prestack_dqfile)
    # write prestack data file
    hdu1 = fits.PrimaryHDU(data=results.data)
    hdu1.writeto(prestack_datapath, overwrite=True)
    # write prestack dq file
    hdu2 = fits.PrimaryHDU(data=results.dq)
    hdu2.writeto(prestack_dqpath, overwrite=True)


# =============================================================================
# Define file naming / getting functions
# =============================================================================
def get_uncal_files(params: Parameters):
    # set func name
    funcname = f'{__NAME__}.get_uncal_files()'
    # get parameters used in this function (should be done at the start)
    rawpath = params['data.raw']
    filelist = params['data.filelist']
    uncal_suffix = '_uncal.fits'
    # -------------------------------------------------------------------------
    # deal with no raw data path set
    if rawpath is None:
        emsg = 'data.raw is not set in yaml file. Cannot get uncal files'
        raise base.LoicPipeError(emsg)
    # deal with data.filelist set
    if filelist is not None:
        # storage for uncal files
        uncal_list = []
        # loop around files in file list
        for filename in filelist:
            # get the uncalibrated files
            uncalfile = os.path.join(rawpath, filename + uncal_suffix)
            # check if file exists
            if os.path.exists(uncalfile):
                uncal_list.append(uncalfile)
    # if filelist is not set take all files with _uncal.fits
    else:
        # get all uncal files
        uncal_path = os.path.join(rawpath, '*' + uncal_suffix)
        uncal_list = glob.glob(uncal_path)
    # -------------------------------------------------------------------------
    # push into parameters
    params['output.uncal_list'] = uncal_list
    # set source
    params('output.uncal_list').source = funcname
    # return the parameters
    return params


def get_output_directory(params: Parameters) -> Parameters:
    # set func name
    funcname = f'{__NAME__}.get_output_directory()'
    # get parameters used in this function (should be done at the start)
    rawpath = params['data.raw']
    outputdir = params['data.outdir']
    # -------------------------------------------------------------------------
    # deal with output directory already set
    if outputdir is not None:
        return params
    # get the output directory
    output_dir = os.path.join(rawpath, 'output')
    # check if output directory exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # push into parameters
    params['data.outdir'] = output_dir
    # set source
    params('data.outdir').source = funcname
    # return the parameters
    return params


def get_data_string(filename: str) -> str:
    # remove extension from filename
    filename = os.path.splitext(filename)[0]
    # remove path
    basename = os.path.basename(filename)
    # only keep prefix before '_nis' (but add it back)
    data_string = basename.split('_nis')[0] + '_nis'
    # only keep prefix before '-seg'
    data_string = data_string.split('-seg')[0]
    # return the data string
    return data_string


def get_saturation_list_files(params: Parameters) -> List[str]:
    # get parameters used in this function (should be done at the start)
    outputdir = params['data.outdir']
    # get the data string
    data_string = params['data.data_string']
    # define the file suffix
    suffix = '_saturationstep.fits'
    # -------------------------------------------------------------------------
    # set up file wildcard
    wildcard = os.path.join(outputdir, f'{data_string}*{suffix}')
    # get all files matching wildcard
    filelist = glob.glob(wildcard)
    # deal with no saturation files
    if len(filelist) == 0:
        emsg = ('No saturation files found with "{0}". '
                'Please set skip_stack=False or update '
                'data.outdir or data.data_string')
        eargs = [wildcard]
        raise base.LoicPipeError(emsg.format(*eargs))
    # return the parameters
    return filelist


def get_deepstack_file(params: Parameters) -> str:
    # get parameters used in this function (should be done at the start)
    outputdir = params['data.outdir']
    # get the data string
    data_string = params['data.data_string']
    # define the file suffix
    prefix = 'oof_deepstack_'
    suffix = '.fits'
    # construct filename
    filename = os.path.join(outputdir, prefix + data_string + suffix)
    # return filename
    return filename


def get_seg_name(filename: str) -> str:
    # remove extension from filename
    filename = os.path.splitext(filename)[0]
    # remove path
    basename = os.path.basename(filename)
    # only keep prefix before '_nis' (but add it back)
    seg_name = basename.split('_nis')[0] + '_nis'
    # return the data string
    return seg_name


def get_outliermap_file(params: Parameters, filename: str) -> str:
    # get parameters used in this function (should be done at the start)
    outputdir = params['data.outdir']
    # get the seg name
    seg_name = get_seg_name(filename)
    # define prefix/suffix
    prefix = 'outliers_'
    suffix = '.fits'
    # construct filename
    outliermap_file = os.path.join(outputdir, prefix + seg_name + suffix)
    # return outlier map filename
    return outliermap_file


def get_tracetable_file(params: Parameters) -> str:
    # get parameters used in this function (should be done at the start)
    atoca_dir = params['data.atoca-dir']
    # get the ref trace table
    ref_trace_table = params['data.ref_trace_table']
    # construct filename
    tracetable_file = os.path.join(atoca_dir, ref_trace_table)
    # return trace table filename
    return tracetable_file


def get_superbias_file(params: Parameters) -> str:
    # get parameters used in this function (should be done at the start)
    calib_dir = params['data.calib-dir']
    # get the ref trace table
    ref_superbias = params['data.superbias']
    # construct filename
    superbias_file = os.path.join(calib_dir, ref_superbias)
    # return trace table filename
    return superbias_file


def get_rateints_file(params: Parameters, filename: str) -> str:
    # get parameters used in this function (should be done at the start)
    outputdir = params['data.outdir']
    # get the seg name
    seg_name = get_seg_name(filename)
    # suffix for rateints file
    suffix = '_rateints.fits'
    # construct filename
    rateints_filename = os.path.join(outputdir, seg_name + suffix)
    # return rateints filename
    return rateints_filename


def get_rateints_after_badpix_interp(params: Parameters, filename: str) -> str:
    # get parameters used in this function (should be done at the start)
    outputdir = params['data.outdir']
    # get the seg name
    seg_name = get_seg_name(filename)
    # suffix for rateints file
    suffix = '_badpixinterp.fits'
    # construct filename
    rateints_filename = os.path.join(outputdir, seg_name + suffix)
    # return rateints filename
    return rateints_filename


def get_wavemap(params: Parameters) -> str:
    # get parameters used in this function (should be done at the start)
    atoca_dir = params['data.atoca-dir']
    # get the ref trace table
    ref_wave_table = params['data.wave_map']
    # construct filename
    wavemap_file = os.path.join(atoca_dir, ref_wave_table)
    # return trace table filename
    return wavemap_file


def get_spec_profile(params: Parameters) -> str:
    # get parameters used in this function (should be done at the start)
    atoca_dir = params['data.atoca-dir']
    # get the ref trace table
    ref_spec_profile = params['data.spec_profile']
    # construct filename
    spec_profile_file = os.path.join(atoca_dir, ref_spec_profile)
    # return trace table filename
    return spec_profile_file


def get_soss_modelname(params: Parameters, filename: str) -> str:
    # get parameters used in this function (should be done at the start)
    outputdir = params['data.outdir']
    # get the seg name
    seg_name = get_seg_name(filename)
    # define prefix/suffix
    suffix = '_atoca_model.fits'
    # construct filename
    modelname_file = os.path.join(outputdir, seg_name + suffix)
    # return modelname filename
    return modelname_file


def get_photom_file(params: Parameters) -> str:
    # get parameters used in this function (should be done at the start)
    crds_dir = params['data.crds-dir']
    # get the ref trace table
    ref_photom = params['data.photom']
    # construct filename
    photom_file = os.path.join(crds_dir, ref_photom)
    # return trace table filename
    return photom_file


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
