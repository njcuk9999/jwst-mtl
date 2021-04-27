#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def soss_background(scidata, scimask, ref_2d_profile, threshold=1e-3):
    """Compute a columnwise background for a SOSS observation.

    :param scidata: the image of the SOSS trace.
    :param scimask: a boolean mask of pixls to be excluded.
    :param ref_2d_profile: the 2d trace profile reference.
    :param threshold: threshold value for excluding pixels based on ref_2d_profile.

    :type scidata: array[float]
    :type scimask: array[bool]
    :type ref_2d_profile: array[float]
    :type threshold: float

    :returns: scidata_bkg, col_bkg, npix_bkg - The background subtracted image, columnwise background values,
    and number of pixels used in each column
    :rtype: Tuple(array[float], array[float], array[float])
    """

    # Mask the trace and bad pixels.
    mask = scimask | (ref_2d_profile > threshold)
    scidata_masked = np.ma.array(scidata, mask=mask)

    # Compute a background value for each column and record the number of pixels used.
    col_bkg = np.ma.mean(scidata_masked, axis=0)
    npix_bkg = np.sum(~mask, axis=0)

    # Background subtract the science data.
    scidata_bkg = scidata - col_bkg

    return scidata_bkg, col_bkg, npix_bkg


def main():

    return


if __name__ == '__main__':
    main()
