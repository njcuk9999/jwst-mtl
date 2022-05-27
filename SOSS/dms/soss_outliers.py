import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from jwst.datamodels.dqflags import pixel

if False:
    import matplotlib

    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from astropy.io import fits

    import sys
    import pdb
    import time

    # from jwst.pipeline import Detector1Pipeline
    from jwst.pipeline import calwebb_detector1
    from jwst.pipeline import calwebb_spec2

    from jwst import datamodels

    from astropy.nddata.bitmask import bitfield_to_boolean_mask


def unfold_frame(image, window_size):
    '''
    Function that takes an image and a window, and slides the window
    across the image, returning for each pixel, the view of the window
    centered on the pixel

    Inputs:
    =======
    image       : (array) Detector 2D image (usually the medianCombined frame)
    window_size : (tuple) The size of the box to slide across the image. (rows, cols)
                        For row scanning, use something like (1,33) which
                        compares with neighbouring 32 row pixels

    Returns:
    ========
    image_unfolded : (array) A view of the window centered on each pixel
                            Shape: (image.shape, window_size) ex: (256, 2048, 1, 33)

    '''

    # pad the image to speed up the window scanning process around edges
    # and avoid having to handle it
    # use zero-padding for now (could also try with mirror padding)

    # Define the padding dimensions do ensure we output the image shape
    pad_rows = (window_size[0] - 1) // 2
    pad_cols = (window_size[1] - 1) // 2

    # pad the image with empty values for now (could be zero)
    padding = ((pad_rows, pad_rows), (pad_cols, pad_cols))
    image_pad = np.pad(image, padding, mode='empty')

    # Now obtain the unfolded view with a fast numpy function
    image_unfolded = sliding_window_view(image_pad, window_size)

    return image_unfolded


def find_outliers(image, window_size, n_sig=5):
    '''
    Finds the outliers in a single medianCombined frame
    Identifies the outliers in two ways
    - if a pixel is n_sig std dev (of the row) away from the median of the entire row
    - if a pixel is n_sig std dev (of the window) away from the median of a window around the pixel

    Inputs:
    =======
    image       : (array) Detector 2D image (usually the medianCombined frame)
    window_size : (tuple) The size of the box to slide across the image. (rows, cols)
    n_sig       : (int)   Number of standard deviations away from the median to be called outlier

    Returns:
    ========
    outliers : (array) Boolean array with same dimensions as the image.
                        True where outliers were identified

    '''

    # Row outliers-----------------------------------------------------------
    # first, find full row outliers, i.e. pixels that are 5sigma away from the
    # median of the entire rows
    # Compute the median and standard dev. for the full rows
    row_median = np.nanmedian(image, axis=-1)
    row_std = np.nanstd(image, axis=-1)

    # outliers are pixels with higher values than (median + 5 sigmas)
    row_threshold = row_median + n_sig * row_std

    # get the outliers mask (broadcast threshold to image shape)
    row_outliers = image > row_threshold[:, None]
    # Row outliers-----------------------------------------------------------

    # Window outliers--------------------------------------------------------
    # unfold the image to obtain all the views of the windows around each pixel
    # reshape to have these windows flattened
    unfolded = unfold_frame(image, window_size).reshape(image.shape[0], image.shape[1], -1)

    # obtain median and std. dev. of each window
    median_map = np.nanmedian(unfolded, axis=-1)
    std_map = np.nanstd(unfolded, axis=-1)

    # threshold is the same as before, except with different values at each pixels now
    threshold_map = median_map + n_sig * std_map

    # get the window outliers mask
    window_outliers = image > threshold_map
    # Window outliers--------------------------------------------------------

    # Combine both masks to obtain all the outliers
    outliers = row_outliers + window_outliers

    return outliers


def flag_outliers(result, nn=2, window_size=(1, 33), n_sig=5, verbose=False):
    '''
    Function that takes a timeseries of integrations and for each, finds the
    outlier pixels and flags them as such in the data quality (dq) object

    The outlier identification routine is based on Nikolov et al. 2014

    Inputs:
    =======
    result      : (jwst object) Stage 2 jwst pipeline object
    nn          : (int) number of integrations to consider before and after each integration
                        when looking for outliers
    window_size : (tuple) The size of the box to slide across the image when
                            scanning for outliers (rows, cols), should keep odd so there is a clear center pixel
    n_sig       : (int) Number of standard deviations away from the median to be called outlier
    verbose     : (bool) If True, activates print statements

    Returns:
    ========
    result : (jwst object) Stage 2 jwst pipeline object but the data quality (dq) map has been updated to flag outliers
    '''

    # load shape of the data set
    nb_int, h, w = result.data.shape

    if nb_int < (nn*2+1):
        print('Warning: Outlier flagging was skipped - not enough integrations.')
        return result

    # We will process each integration one after the other
    for i in range(nb_int):

        # The first step is to compute the difference between our integration we want
        # to analyze and the neighbouring integrations in time
        # We compute the difference with the preceding/following nn integrations

        # Near the start of the integration, we make sure we still choose the 2*nn nearest integrations
        # but can't choose nn on each side
        if i < nn:
            if verbose: print('Processing integration {}'.format(i))
            # define the target integration
            target = result.data[i]

            # define the list of the indices of the neighbouring frames
            # we need to handle them differently since we are at the start of the timeseries
            neighbours = np.arange(0, 2 * nn + 1)
            # remove the target frame from the list
            neighbours = np.delete(neighbours, np.argwhere(neighbours == i))

            # create an empty list that will hold the difference images
            # differenceImages = Target - neighbouringImages
            differenceImages = []
            for ii in neighbours:
                # Compute target-neighbour for preceding/next nn images
                differenceImages.append(target - result.data[ii])


        # Near the end of the timeseries, we again find the nearest 2*nn integrations, but
        # we can't choose nn on each side
        elif nb_int - i <= nn:
            if verbose: print('Processing integration {}'.format(i))
            # define the target integration
            target = result.data[i]

            # define the list of the indices of the neighbouring frames
            # we need to handle them differently since we are at the start of the time series
            neighbours = np.arange(nb_int - 2 * nn - 1, nb_int)
            # remove the target frame from the list
            neighbours = np.delete(neighbours, np.argwhere(neighbours == i))

            # create an empty list that will hold the difference images
            # differenceImages = Target - neighbouringImages
            differenceImages = []
            for ii in neighbours:
                # Compute target-neighbour for preceding/next nn images
                differenceImages.append(target - result.data[ii])

        # In the middle of the timeseries, we simply choose nn frames before
        # and nn frames after
        else:
            if verbose: print('Processing integration {}'.format(i))
            # define the target integration
            target = result.data[i]

            # create an empty list that will hold the difference images
            # differenceImages = Target - neighbouringImages
            differenceImages = []
            for ii in range(1, nn + 1):
                # Compute target-neighbour for preceding/next nn images
                differenceImages.append(target - result.data[i - ii])
                differenceImages.append(target - result.data[i + ii])

        # Compute the median of the set of differenceImages for our given integration
        medianCombined = np.nanmedian(differenceImages, axis=0)
        # Now, to ensure we do not overwrite the dq map of already flagged bad pixels,
        # we set these already flagged bad pixels to 0
        medianCombined[result.dq[i] != pixel['GOOD']] = 0

        # From here, we identify the outliers in the medianCombined image
        outliers = find_outliers(medianCombined, window_size, n_sig)

        # update the dq map with the new outliers
        result.dq[i][outliers] = pixel['OUTLIER']

        if verbose: print(
            'Processing integration {} : Identified {} outlier pixels\n'.format(i, np.count_nonzero(outliers)))

    return result

# example
# result = flag_outliers(result, verbose=True)