#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:46 2021

@author: MCR

Definition of the main class for the empirical trace construction module. This
class will be initalized and called by the user to create uncontaminated 2D
trace profiles for the first and second order.
"""

import os
import warnings
from SOSS.extract.empirical_trace import construct_trace as tm
from SOSS.extract.empirical_trace import utils


class Empirical_Trace():
    '''Class wrapper around the empirical trace construction module.

    Attributes
    ----------
    CLEAR : np.array of float (2D)
        SOSS CLEAR exposure data frame.
    F277W : np.array of float (2D)
        SOSS exposure data frame using the F277W filter. Pass None if no F277W
        exposure is available.
    badpix_mask : np.ndarray (2D) of bool
        Bad pixel mask, values of True represent bad pixels. Must be the same
        shape as the CLEAR dataframe.
    subarray : str
        NIRISS SOSS subarray identifier. One of 'SUBSTRIP96', 'SUBSTRIP256', or
        'FULL'.
    pad : tuple
        Amount of padding to include (in native pixels) in the spatial and
        spectral directions respectively.
    oversample : int
        Oversampling factor. Oversampling will be equal in the spectral and
        spatial directions.
    verbose : int
        Level of verbosity: either 3, 2, 1, or 0.
         3 - show all of progress prints, progress bars, and diagnostic plots.
         2 - show progress prints and bars.
         1 - show only progress prints.
         0 - show nothing.
    order1 : np.ndarray of float (2D)
        Uncontaminted first order trace profile.
    order2 : np.ndarray of float (2D)
        Uncontaminated second order trace profile.
    '''

    def __init__(self, clear, F277W, badpix_mask, pad=(0, 0), oversample=1,
                 verbose=0):

        # Initialize input attributes.
        self.CLEAR = clear
        self.F277W = F277W
        self.badpix_mask = badpix_mask
        self.pad = pad
        self.oversample = oversample
        self.verbose = verbose

        # Validate the parameters and determine the correct subarray.
        self.subarray = self.validate_inputs()
        self.order1 = None
        self.order2 = None

    def build_empirical_trace(self, normalize=True):
        '''Run the empirical trace construction module.

        Parameters
        ----------
        normalize : bool
            if True, column normalize the final spatial profiles such that the
            flux in each column sums to one.
        '''

        # Run the empirical trace construction.
        o1, o2 = tm.build_empirical_trace(self.CLEAR, self.F277W,
                                          self.badpix_mask, self.subarray,
                                          self.pad, self.oversample,
                                          normalize, self.verbose)
        # Store the uncontaminted profiles as attributes.
        self.order1, self.order2 = o1, o2

    def save_to_file(self, filename=None):
        '''Write the uncontaminated 2D trace profiles to a fits file.

        Parameters
        ----------
        filename : str (optional)
            Path to file to which to save the spatial profiles. Defaults to
            'SOSS_2D_profile_{subarray}.fits'.
        '''

        # Get default filename if none provided.
        if filename is None:
            filename = 'SOSS_2D_profile_{}.fits'.format(self.subarray)
        if self.verbose != 0:
            print('Saving trace profiles to file {}...'.format(filename))

        # Print overwrite warning if output file already exists.
        if os.path.exists(filename):
            msg = 'Output file {} already exists.'\
                  ' It will be overwritten'.format(filename)
            warnings.warn(msg)

        # Write trace profiles to disk.
        utils._write_to_file(self.order1, self.order2, self.subarray,
                             filename, self.pad, self.oversample)

    def validate_inputs(self):
        '''Validate the input parameters.
        '''

        return utils._validate_inputs(self)
