import numpy as np
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def get_box_weights(centroid, n_pix, shape, cols=None):
    """ Return the weights of a box aperture given the centroid and the width of
    the box in pixels. All pixels will have the same weights except at the ends
    of the box aperture.
    Parameters
    ----------
    centroid : array[float]
        Position of the centroid (in rows). Same shape as `cols`
    n_pix : float
        Width of the extraction box in pixels.
    shape : Tuple(int, int)
        Shape of the output image. (n_row, n_column)
    cols : array[int]
        Column indices of good columns. Used if the centroid is defined
        for specific columns or a sub-range of columns.
    Returns
    -------
    weights : array[float]
        An array of pixel weights to use with the box extraction.
    """

    nrows, ncols = shape

    # Use all columns if not specified
    if cols is None:
        cols = np.arange(ncols)

    # Row centers of all pixels.
    rows = np.indices((nrows, len(cols)))[0]

    # Pixels that are entierly inside the box are set to one.
    cond = (rows <= (centroid - 0.5 + n_pix / 2))
    cond &= ((centroid + 0.5 - n_pix / 2) <= rows)
    weights = cond.astype(float)

    # Fractional weights at the upper bound.
    cond = (centroid - 0.5 + n_pix / 2) < rows
    cond &= (rows < (centroid + 0.5 + n_pix / 2))
    weights[cond] = (centroid + n_pix / 2 - (rows - 0.5))[cond]

    # Fractional weights at the lower bound.
    cond = (rows < (centroid + 0.5 - n_pix / 2))
    cond &= ((centroid - 0.5 - n_pix / 2) < rows)
    weights[cond] = (rows + 0.5 - (centroid - n_pix / 2))[cond]

    # Return with the specified shape with zeros where the box is not defined.
    out = np.zeros(shape, dtype=float)
    out[:, cols] = weights

    return out


def box_extract(scidata, scierr, scimask, box_weights, cols=None):
    """ Perform a box extraction.
    Parameters
    ----------
    scidata : array[float]
        2d array of science data with shape (n_row, n_columns)
    scierr : array[float]
        2d array of uncertainty map with same shape as scidata
    scimask : array[bool]
        2d boolean array of masked pixels with same shape as scidata
    box_weights : array[float]
        2d array of pre-computed weights for box extraction,
        with same shape as scidata
    cols : array[int]
        1d integer array of column numbers to extract
    Returns
    -------
    cols : array[int]
        Indices of extracted columns
    flux : array[float]
        The flux in each column
    flux_var : array[float]
        The variance of the flux in each column
    """

    nrows, ncols = scidata.shape

    # Use all columns if not specified
    if cols is None:
        cols = np.arange(ncols)

    # Keep only required columns and make a copy.
    data = scidata[:, cols].copy()
    error = scierr[:, cols].copy()
    mask = scimask[:, cols].copy()
    box_weights = box_weights[:, cols].copy()

    # Check that all invalid values are masked.
    if not np.isfinite(data[~mask]).all():
        message = 'scidata contains un-masked invalid values.'
        log.critical(message)
        raise ValueError(message)

    if not np.isfinite(error[~mask]).all():
        message = 'scierr contains un-masked invalid values.'
        log.critical(message)
        raise ValueError(message)

    # Set the weights of masked pixels to zero.
    box_weights[mask] = 0.

    # Extract total flux (sum over columns).
    flux = np.nansum(box_weights * data, axis=0)
    npix = np.nansum(box_weights, axis=0)

    # Extract flux error (sum of variances).
    flux_var = np.nansum(box_weights * error**2, axis=0)
    flux_err = np.sqrt(flux_var)

    # Set empty columns to NaN.
    flux = np.where(npix > 0, flux, np.nan)
    flux_err = np.where(npix > 0, flux_err, np.nan)

    return cols, flux, flux_err, npix


def estim_error_nearest_data(err, data, pix_to_estim, valid_pix):
    """
    Function to estimate pixel error empirically using the corresponding error
    of the nearest pixel value (`data`). Intended to be used in a box extraction
    when the bad pixels are modeled.
    Parameters
    ----------
    err : 2d array[float]
        Uncertainty map of the pixels.
    data : 2d array[float]
        Pixel values.
    pix_to_estim : 2d array[bool]
        Map of the pixels where the uncertainty needs to be estimated.
    valid_pix : 2d array[bool]
        Map of valid pixels to be used to find the error empirically.
    Returns
    -------
    err_filled : 2d array[float]
        same as `err`, but the pixels to be estimated are filled with the estimated values.
    """
    # Tranform to 1d arrays
    data_to_estim = data[pix_to_estim]
    err_valid = err[valid_pix]
    data_valid = data[valid_pix]

    #
    # Use np.searchsorted for efficiency
    #
    # Need to sort the arrays used to find similar values
    idx_sort = np.argsort(data_valid)
    err_valid = err_valid[idx_sort]
    data_valid = data_valid[idx_sort]

    # Searchsorted: gives the position of the nearest higher value,
    # not necessarily the closest value
    idx_higher = np.searchsorted(data_valid, data_to_estim)
    idx_higher = np.clip(idx_higher, 0, err_valid.size - 1)
    # The nearest lower value is given by the preceding index
    idx_lower = np.clip(idx_higher - 1, 0, err_valid.size - 1)

    # Find the best between index around the value (lower and higher index) ...
    idx_around = np.vstack([idx_lower, idx_higher])
    # ... using the one with the smallest error
    distance = np.abs(data_valid[idx_around] - data_to_estim[None, :])
    idx_best_of_2 = np.argmin(distance, axis=0)
    idx_closest = idx_around[idx_best_of_2, np.arange(idx_best_of_2.size)]

    # Get the corresponding error (that's what we want to find!)
    err_estimate = err_valid[idx_closest]

    # Replace estimated values in the output error 2d image
    err_out = err.copy()
    err_out[pix_to_estim] = err_estimate

    return err_out


def extract_image(scidata_bkg, scierr, scimask, tracemodels, ref_files,
                  transform, subarray, width=40., bad_pix='masking'):
    """Perform the box-extraction on the image, while using the trace model to
    correct for contamination.
    Parameters
    ----------
    scidata_bkg : array[float]
        A single backround subtracted NIRISS SOSS detector image.
    scierr : array[float]
        The uncertainties corresponding to the detector image.
    scimask : array[float]
        Pixel mask to apply to the detector image.
    tracemodels : dict
        Dictionary of the modeled detector images for each order.
    ref_files : dict
        A dictionary of the reference file DataModels.
    transform : array_like
        A 3-element list or array describing the rotation and translation to
        apply to the reference files in order to match the observation.
    subarray : str
        Subarray on which the data were recorded; one of 'SUBSTRIPT96',
        'SUBSTRIP256' or 'FULL'.
    width : float
        The width of the aperture used to extract the uncontaminated spectrum.
    bad_pix : str
        How to handle the bad pixels. Options are 'masking' and 'model'.
        'masking' will simply mask the bad pixels, such that the number of pixels
        in each column in the box extraction will not be constant, while the
        'model' option uses `tracemodels` to replace the bad pixels.
    Returns
    -------
    wavelengths, fluxes, fluxerrs, npixels, box_weights : dict
        Each output is a dictionary, with each extracted order as a key.
    """
    # Which orders to extract.
    if subarray == 'SUBSTRIP96':
        order_list = [1]
    else:
        order_list = [1, 2, 3]

    order_str = {order: f'Order {order}' for order in order_list}

    # List of modeled orders
    mod_order_list = tracemodels.keys()

    # Create dictionaries for the output spectra.
    wavelengths = dict()
    fluxes = dict()
    fluxerrs = dict()
    npixels = dict()
    box_weights = dict()

    log.info('Performing the decontaminated box extraction.')

    # Extract each order from order list
    for order_integer in order_list:
        # Order string-name is used more often than integer-name
        order = order_str[order_integer]

        log.debug(f'Extracting {order}.')

        # Define the box aperture
        xtrace, ytrace, wavelengths[order] = get_trace_1d(ref_files, transform, order_integer)
        box_w_ord = get_box_weights(ytrace, width, scidata_bkg.shape, cols=xtrace)

        # Decontaminate using all other modeled orders
        decont = scidata_bkg
        if False:
            for mod_order in mod_order_list:
                if mod_order != order:
                    log.debug(f'Decontaminating {order} from {mod_order} using model.')
                    decont = decont - tracemodels[mod_order]

        # Deal with bad pixels if required.
        if False:
            if bad_pix == 'model':
                # Model the bad pixels decontaminated image when available
                try:
                    # Replace bad pixels
                    decont = np.where(scimask, tracemodels[order], decont)
                    # Update the mask for the modeled order, so all the pixels are usable.
                    scimask_ord = np.zeros_like(scimask)

                    log.debug(f'Bad pixels in {order} are replaced with trace model.')

                    # Replace error estimate of the bad pixels using other valid pixels of similar value.
                    # The pixel to be estimate are the masked pixels in the region of extraction
                    extraction_region = (box_w_ord > 0)
                    pix_to_estim = (extraction_region & scimask)
                    # Use only valid pixels (not masked) in the extraction region for the empirical estimation
                    valid_pix = (extraction_region & ~scimask)
                    scierr_ord = estim_error_nearest_data(scierr, decont, pix_to_estim, valid_pix)

                except KeyError:
                    # Keep same mask and error
                    scimask_ord = scimask
                    scierr_ord = scierr
                    log.warning(f'Bad pixels in {order} will be masked instead of modeled: trace model unavailable.')
            else:
        if True:
            # Mask pixels
            scimask_ord = scimask
            scierr_ord = scierr
            log.info(f'Bad pixels in {order} will be masked.')

        # Save box weights
        box_weights[order] = box_w_ord
        # Perform the box extraction and save
        out = box_extract(decont, scierr_ord, scimask_ord, box_w_ord, cols=xtrace)
        _, fluxes[order], fluxerrs[order], npixels[order] = out

    return wavelengths, fluxes, fluxerrs, npixels, box_weights
