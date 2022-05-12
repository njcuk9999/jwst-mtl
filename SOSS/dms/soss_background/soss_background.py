#
#  Module for subtracting the background from science data sets
#

import numpy as np
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def do_subtraction(input_model):
    return

def build_mask(input_model):

    # Open the basic trace mask
    if input_model.meta.subarray.name is 'SUBSTRIP256':
        trace_mask = fits.getdata('mask256.fits')
    elif input_model.meta.subarray.name is 'SUBSTRIP96':
        trace_mask = fits.getdata('mask96.fits')
    elif input_model.meta.subarray.name is 'FULL':
        trace_mask = fits.getdata('maskfull.fits')
    else:
        warnings.warn('CustomBackgroundSubtractionStep input datamodel none of the expected subarray.')
        sys.exit()

    return trace_mask

def read_ref_background(input_model):

    # Open the background reference file
    if input_model.meta.subarray.name is 'SUBSTRIP256':
        background_ref = fits.getdata('background256.fits')
    elif input_model.meta.subarray.name is 'SUBSTRIP96':
        background_ref = fits.getdata('background96.fits')
    elif input_model.meta.subarray.name is 'FULL':
        background_ref = fits.getdata('backgroundfull.fits')
    else:
        warnings.warn('CustomBackgroundSubtractionStep - read_ref_background - input datamodel none of the expected subarray.')
        sys.exit()

    return background_ref


def use_reference(input_model, background_ref):
    # scales the reference background to best fit the input image
    # and subtracts it.

