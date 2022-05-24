#
#  Module for subtracting the background from science data sets
#


import warnings

import sys

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
    print()



'''
Background subtracts a SOSS time-series
'''

def subtract_background(data, format='DMS', mask=True):
    '''
    Measures the background in a CDS or rate or rateints file.
    :param uncal:
    :param mask:
    :return:
    '''

    if format != 'DMS':
        data = fits.open(data)
        if data.ndim == 2:
            # Single image
            nints = 0
            dimy, dimx = np.shape(data)
        elif data.ndim == 3:
            # Multiple Integrations
            nints, dimy, dimx = np.shape(data)
        else:
            warnings.warn('Input data to subtract_background has neither 2 or 3 dimensions. It should.')
            sys.exit()

    else:
        # Assumes it is a DMS datamodel
        print()


    return







#def main():
#    """Placeholder for potential multiprocessing."""
#
#    return
#
#
#if __name__ == '__main__':
#    main()
