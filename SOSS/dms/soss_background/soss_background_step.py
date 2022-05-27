from jwst import datamodels

from jwst.stpipe import Step

from . import soss_background

#import sys


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
            elif method is 'scaleref':
                # Assume we use a background that we scale
                background_ref = soss_background.read_ref_background(input_model)
                #result = soss_background.use_reference(input_model, background_ref)
            elif method is 'ref':
                # We us e a background ref file straight without scaling, without nothing
                background_ref = soss_background.read_ref_background()
                self.log.info('Using CUSTOM BACKGROUND reference file')
                result = soss_background.do_subtraction(input_model, background_ref)
            else:
                self.log.warning('No background subtraction performed.')
                result = input_model.copy()
                return result

            # Set the step to complete.
            # But would need a soss_background dict instead
            result.meta.cal_step.back_sub = 'COMPLETE'

        return result