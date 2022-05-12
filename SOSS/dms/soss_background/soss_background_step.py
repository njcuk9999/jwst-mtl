from stpipe import Step
from jwst import datamodels
import sys

__all__ = ["CustomBackgroundSubtractionStep"]

class CustomBackgroundSubtractionStep(Step):
    """
    CustomBackgroundSubtractionStep: Performs custom background subtraction by
    subtracting levels devoid of astrophysical signal from the input science
    data model.
    """

    spec = """
    """

    reference_file_types = ['mask','background']

    def process(self, input, method='ref'):

        # Open the input data model
        with datamodels.RampModel(input) as input_model:

            if method is 'mask':
                # Do the background masking
                mask = soss_background.build_mask(input_model)

                # Do the background subtraction
                result = soss_background.do_subtraction(input_model, mask)
            else
                # Assume we use a background that we scale
                background_ref = soss_background.read_ref_background(input_model)
                result = soss_background.use_reference(input_model, background_ref)

            # Set the step to complete.
            # But would need a soss_background dict instead
            result.meta.cal_step.back_sub = 'COMPLETE'

        return result


'''
Background subtracts a SOSS time-series
'''


from jwst import datamodels

import numpy as np

import warnings

import sys


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










def main():
    """Placeholder for potential multiprocessing."""

    return


if __name__ == '__main__':
    main()