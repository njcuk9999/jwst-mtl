import numpy as np
from astropy.io import fits
from scipy import ndimage
from numpy.lib.stride_tricks import sliding_window_view
from jwst.datamodels.dqflags import pixel
import os




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


def flag_outliers(result, nn=2, window_size=(1, 33), n_sig=5, verbose=False, outdir=None, save_diagnostic=False,
                  kernel_enlarge=True):
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
    kernel_enlarge : (bool) If True, convolve the outlier map with a kernel to enlarge the flagginf into wings

    Returns:
    ========
    result : (jwst object) Stage 2 jwst pipeline object but the data quality (dq) map has been updated to flag outliers
    '''

    # load shape of the data set
    nb_int, dimy, dimx = result.data.shape

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

        # Add option to "enlarge" a pixel flagged as outlier with a kernel in the hope
        # of catching the wings of cosmic rays
        if kernel_enlarge:
            kernel = [[0,1,0],[1,1,1],[0,1,0]] # cross
            pad = np.max(np.shape(kernel)) # roughly
            im = np.zeros((dimy, dimx))
            im[outliers] = 1
            impadded = np.zeros((dimy+2*pad, dimx+2*pad))
            impadded[pad:-pad,pad:-pad] = np.copy(im)
            imconvolved = ndimage.convolve(impadded, kernel, mode='constant', cval=0.0)
            im = imconvolved[pad:-pad,pad:-pad]
            outliers = im > 0


        # update the dq map with the new outliers
        result.dq[i][outliers] = pixel['OUTLIER']

        if verbose: print(
            'Processing integration {} : Identified {} outlier pixels\n'.format(i, np.count_nonzero(outliers)))

    if save_diagnostic == True:
        # Save fits file of all integrations where the cosmic ray detections are set to NaN whihc
        # will allow to use ds9 to flash through and inspect that all went fine
        cube = np.copy(result.data)
        ind = result.dq == pixel['OUTLIER']
        cube[ind] = np.nan

        basename = os.path.splitext(result.meta.filename)[0]
        if outdir == None:
            outdir = './'
            outliersdir = './outliers_' + basename + '/'
        else:
            outliersdir = outdir + '/outliers_' + basename + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not os.path.exists(outliersdir):
            os.makedirs(outliersdir)

        hdu = fits.PrimaryHDU(cube)
        hdu.writeto(outliersdir+'flagged_outliers.fits', overwrite=True)

    return result

# example
# result = flag_outliers(result, verbose=True)