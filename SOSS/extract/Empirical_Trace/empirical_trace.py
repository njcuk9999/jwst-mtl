#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:46 2021

@author: MCR

Definition of the main class for the empirical trace construction module. This
class will be initialized and called by the user to create uncontaminated 2D
trace profiles for the first and second order.
"""

import os
import warnings
from SOSS.extract.empirical_trace import construct_trace as tm
from SOSS.extract.empirical_trace import utils


class EmpiricalTrace:
    """Class wrapper around the empirical trace construction module.

    Attributes
    ----------
    clear : np.array
        SOSS CLEAR exposure data frame.
    f277w : np.array
        SOSS exposure data frame using the F277W filter. Pass None if no F277W
        exposure is available.
    badpix_mask : np.array
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
    order1 : np.array
        Uncontaminated first order trace profile.
    order2 : np.array
        Uncontaminated second order trace profile.
    """

    def __init__(self, clear, f277w, badpix_mask, pad=(0, 0), oversample=1,
                 verbose=0):

        # Initialize input attributes.
        self.clear = clear
        self.f277w = f277w
        self.badpix_mask = badpix_mask
        self.pad = pad
        self.oversample = oversample
        self.verbose = verbose

        # Validate the parameters and determine the correct subarray.
        self.subarray = self.validate_inputs()
        self.order1 = None
        self.order2 = None

    def build_empirical_trace(self, normalize=True, max_iter=1):
        """Run the empirical trace construction module.

        Parameters
        ----------
        normalize : bool
            if True, column normalize the final spatial profiles such that the
            flux in each column sums to one.
        max_iter : int
            Number of refinement iterations to complete.
        """

        # Warn user that not iterating will yield inaccurate solutions.
        if max_iter < 1:
            warn_msg = 'max_iter has been set to 0 - not iterating is' \
                'unadvisable, and will lead to inaccurate solutions for the' \
                'trace profiles.'
            warnings.warn(warn_msg)
            print('', flush=True)

        # Run the empirical trace construction.
        o1, o2 = tm.build_empirical_trace(self.clear, self.f277w,
                                          self.badpix_mask, self.subarray,
                                          self.pad, self.oversample, normalize,
                                          max_iter, self.verbose)
        # Store the uncontaminated profiles as attributes.
        self.order1, self.order2 = o1, o2

    def save_to_file(self, filename=None):
        """Write the uncontaminated 2D trace profiles to a fits file.

        Parameters
        ----------
        filename : str (optional)
            Path to file to which to save the spatial profiles. Defaults to
            'SOSS_2D_profile_{subarray}.fits'.
        """

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
        utils.write_to_file(self.order1, self.order2, self.subarray, filename,
                            self.pad, self.oversample)

    def validate_inputs(self):
        """Validate the input parameters.
        """
        return utils.validate_inputs(self)
