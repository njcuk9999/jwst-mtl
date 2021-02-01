#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on {DATE}

@author: cook
"""
from astropy.io import fits
import numpy as np
from pathlib import Path
from typing import Tuple, Union

from ami_mtl.core.base import base
from ami_mtl.core.core import log_functions
from ami_mtl.core.core import param_functions

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
# Get parameter dictionary
ParamDict = param_functions.ParamDict
# get fits header
Header = fits.Header


# =============================================================================
# Define functions
# =============================================================================
# define complex read typing
ReadReturn = Union[np.ndarray, Tuple[np.ndarray, Header], Header]


def read_fits(params: ParamDict, filename: Union[str, Path],
              get_data: bool = True, get_header: bool = True,
              ext: Union[int, None] = None) -> ReadReturn:
    """
    Read a fits file from disk

    :param params: ParamDict, parameter dictionary of constants
    :param filename: str or path, the filename
    :param get_data: bool, if True reads data
    :param get_header: bool, if True reads header
    :param ext: int or None, if set forces the extension (else opens first
                readable extension)

    :return: if not get_header returns np.array of data, if not get_data
             returns header, if both returns data and header (as a tuple)
    """
    # wrap in try except
    try:
        # deal with no extension
        if ext is None:
            # deal with not getting data
            if not get_data:
                # return copied header
                return Header(fits.getheader(filename))
            # deal with not getting header
            elif not get_header:
                # return copied data
                return np.array(fits.getdata(filename))
            # deal with getting both data + header
            else:
                data, header = fits.getdata(filename, header=True)
                # return copied data + header
                return np.array(data), Header(header)
        # deal with having an extension
        else:
            # open in safe way
            with fits.open(filename) as hdu:
                # deal with not getting data
                if not get_data:
                    # return copied header
                    return Header(hdu[ext].header)
                # deal with not getting header
                elif not get_header:
                    # return copied data
                    return np.array(hdu[ext].data)
                # deal with getting both data + header
                else:
                    data = np.array(hdu[ext].data)
                    header = Header(hdu[ext].header)
                    # return copied data + header
                    return np.array(data), Header(header)
    except Exception as e:
        emsg = 'ReadFits Error {0}: {1}'
        eargs = [type(e), str(e)]
        params.log.error(emsg.format(*eargs))


def write_fits(params: ParamDict, filename: Union[str, Path],
               data: Union[np.ndarray, None] = None,
               header: Union[fits.Header, None] = None,
               overwrite: bool = True):
    """
    Write fits file to disk

    :param params: ParamDict, parameter dictionary of constants
    :param filename: str or path, the filename
    :param data: np.ndarray, the data (as an image)
    :param header: Header, the fits header
    :param overwrite: bool, if True allows overwriting files on disk
    :return:
    """
    # convert to string (if Path)
    filename = str(filename)
    # try to write to disk
    try:
        with fits.open(filename):
            primary = fits.PrimaryHDU(data=data, header=header)
            hdulist = fits.HDUList([primary])
            hdulist.writeto(filename, overwrite=overwrite)
    except Exception as e:
        emsg = 'WriteFits Error {0}: {1}'
        eargs = [type(e), str(e)]
        params.log.error(emsg.format(*eargs))


def get_param(params: ParamDict, key: str, check: bool = True):
    """
    Get file parameter key from ParamDict

    :param params: ParamDict, parameter dictionary of constants
    :param key: str, the key to get from params
    :param check: bool, if True checks that file exists on disk

    :return: value of the key (if it exists)
    """
    # check for key in params
    value = param_functions.get_param(params, key)
    # get as path
    path = Path(str(value))
    # check if exists
    if check:
        if not path.exists():
            emsg = 'ParamError: {0} does not exist \n\t Path = {1}'
            eargs = [key, str(path)]
            params.log.error(emsg.format(*eargs))
            return None
    # return params key
    return path


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run main code
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================
