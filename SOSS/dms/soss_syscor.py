#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def make_profile_mask(ref_2d_profile, threshold=1e-3):
    """Build a mask of the trace based on the 2D profile reference file.

    :param ref_2d_profile: the 2d trace profile reference.
    :param threshold: threshold value for excluding pixels based on
        ref_2d_profile.
    
    :type ref_2d_profile: array[float]
    :type threshold: float

    :returns: bkg_mask - Masks pixels in the trace based on the 2d profile
        reference file.
    :rtype: array[bool]
    """

    bkg_mask = (ref_2d_profile > threshold)

    return bkg_mask


def soss_background(scidata, scimask, bkg_mask=None):
    """Compute a columnwise background for a SOSS observation.

    :param scidata: the image of the SOSS trace.
    :param scimask: a boolean mask of pixls to be excluded.
    :param bkg_mask: a boolean mask of pixels to be excluded because they are in
        the trace, use for example make_profile_mask to construct such a mask.

    :type scidata: array[float]
    :type scimask: array[bool]
    :type bkg_mask: array[bool]

    :returns: scidata_bkg, col_bkg, npix_bkg - The background subtracted image,
        columnwise background values, and number of pixels used in each column.
    :rtype: Tuple(array[float], array[float], array[float])
    """

    # Check the validity of the input.
    data_shape = scidata.shape

    if scimask.shape != data_shape:
        msg = 'scidata and scimask must have the same shape.'
        raise ValueError(msg)

    if bkg_mask is not None:

        if bkg_mask.shape != data_shape:
            msg = 'scidata and bkg_mask must have the same shape.'
            raise ValueError(msg)

    # Combine the masks and create a masked array.
    if bkg_mask is not None:
        mask = scimask | bkg_mask
    else:
        mask = scimask

    scidata_masked = np.ma.array(scidata, mask=mask)

    # Compute the mean for each column and record the number of pixels used.
    col_bkg = np.ma.mean(scidata_masked, axis=0)
    npix_bkg = np.sum(~mask, axis=0)

    # Background subtract the science data.
    scidata_bkg = scidata - col_bkg

    return scidata_bkg, col_bkg, npix_bkg


def main():

    return


if __name__ == '__main__':
    main()
