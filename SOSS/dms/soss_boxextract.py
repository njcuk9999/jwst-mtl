import numpy as np


def get_box_weights(cols, centroid, n_pix, shape):
    """
    Return the weights of a box aperture given the centroid
    and the width of the box in pixels.
    All pixels will have the same weights except at the
    ends of the box aperture.
    Parameters
    ----------
    cols: 1d array, integer
        Columns index positions. Useful if the centroid is defined for
        specific columns or a subrange of columns.
    centroid: 1d array
        Position of the centroid (in rows). Same shape as `cols`
    n_pix: float
        full width of the extraction box in pixels.
    shape: 2 integers tuple
        shape of the output image. (n_row, n_column)
    Ouput
    -----
    2d image of the box weights
    """
    # Row centers of all pixels
    rows = np.indices((shape[0], len(cols)))[0]

    # Pixels that are entierly inside the box are set to one
    cond = (rows <= (centroid - 0.5 + n_pix / 2))
    cond &= ((centroid + 0.5 - n_pix / 2) <= rows)
    weights = cond.astype(float)

    # Upper bound
    cond = (centroid - 0.5 + n_pix / 2) < rows
    cond &= (rows < (centroid + 0.5 + n_pix / 2))
    weights[cond] = (centroid + n_pix / 2 - (rows - 0.5))[cond]

    # Lower bound
    cond = (rows < (centroid + 0.5 - n_pix / 2))
    cond &= ((centroid - 0.5 - n_pix / 2) < rows)
    weights[cond] = (rows + 0.5 - (centroid - n_pix / 2))[cond]

    # Return with the specified shape
    # with zeros where the box is not define
    out = np.zeros(shape, dtype=float)
    out[:, cols] = weights

    return out


def box_extract(data, uncert, box_weights, cols=None, mask=None):
    '''
    Make a box extraction
    Parameters
    ----------
    data: 2d array of shape (n_row, n_columns)
        scidata
    uncert: 2d array of shape (n_row, n_columns)
        uncertainty map
    box_weights: 2d array, same shape as data
        pre-computed weights for box extraction.
    lam_col: 1d array of shape (n_columns)
        wavelength associated with each columns. If not given,
        the column position is taken as ordinates.
    cols: 1d-array, integer
        Which columns to extract
    mask: 2d array, boolean, same shape as data
        masked pixels
    Output
    ------
    (column position, spectrum, spectrum_variance)

    Example
    -------
    Assume you have an array of column position `good_cols`
    and the `centroid` associated with these columnns.
    It can be define for a subset of some columns of the detector.
    Then we can first compute a weight map for a box aperture
    of 20 pixels on the :

    >>> box_weights = get_box_weights(good_cols, centroid, 20, data.shape)

    The output map will have the same shape as `data`. Then we can extract:

    >>> x_col, spectrum, var = box_extract(data, box_weights, mask=~np.isfinite(data))

    '''
    # Use all columns if not specified
    if cols is None:
        cols = np.arange(data.shape[1])

    # Define mask if not given
    if mask is None:
        # False everywhere
        mask = np.zeros(data.shape, dtype=bool)

    # Keep only needed columns and make a copy
    # so it is not modified outside of the function
    data = data[:, cols].copy()
    uncert = uncert[:, cols].copy()
    box_weights = box_weights[:, cols].copy()
    mask = mask[:, cols].copy()

    # Check if there are some invalid values
    # in the non masked regions of the 2d inputs
    for map_2d in (data, uncert):
        non_masked = map_2d[~mask]
        if not np.isfinite(non_masked).all():
            message = 'Non masked pixels have invalid values.'
            raise ValueError(message)

    # Apply to weights
    box_weights[mask] = np.nan

    # Extract spectrum (sum over columns)
    spectrum = np.nansum(box_weights * data, axis=0)

    # Extract error variance (sum of variances)
    pix_var = uncert ** 2
    col_var = np.nansum(box_weights * pix_var, axis=0)

    # Things to add to output:
    # - dq
    # - sum of poisson variances?
    # - sum of read noise?
    # - sum of variance falt field?
    # - extract bkgrnd spectrum?

    return cols, spectrum, col_var