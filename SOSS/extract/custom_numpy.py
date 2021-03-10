import numpy as np
# TODO Merge all this into utils.py?


def is_sorted(x, no_dup=True):  # TODO unused, remove?
    """
    Check if x is sorted and has noo duplicates (if no_dup is True).
    Returns True of False.
    """
    if no_dup:
        return (np.diff(x) > 0).all()
    else:
        return (np.diff(x) >= 0).all()


def fill_list(x, fill_value=np.nan, **kwargs):  # TODO unused, remove?
    """
    Fill a list `x` (N, non-constant M)
    to make an array (N, M) with it.
    kwargs are passed to np.ones to initiate
    the output array (so possibility to specify the dtype)
    """
    n1 = len(x)
    n2 = np.max([len(x_i) for x_i in x])

    out = np.ones((n1, n2), **kwargs) * fill_value
    for i, x_i in enumerate(x):
        out[i, :len(x_i)] = x_i

    return out


def first_change(cond, axis=None):  # TODO unused, remove?
    """
    Returns the position before the first change
    in array value, along axis. If no change is found,
    an empty array will be returned.
    """
    # Find first change
    cond = np.diff(cond, axis=axis)
    # Return position
    return np.where(cond)


def vrange(*args, return_where=False, dtype=None):  # TODO remove use of args...
    """
    Create concatenated ranges of integers for multiple start/stop

    usage:
        vrange([starts,] stops, return_where=False, dtype=None)

    Parameters:
        starts (1-D array_like, optional):
            starts for each range, default is 0
        stops (1-D array_like):
            stops for each range (same shape as starts)
        return_where: bool, optional
            return the corresponding indices to be able to
            transform in a 2d array. Default is False.
        dtype: type object, optional
            type of output array

    Returns:
        numpy.ndarray: concatenated ranges
        indices of 2 axis

    For example:

        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """

    if len(args) == 1:
        starts = 0
    elif len(args) == 2:
        starts = args[0]
    else:
        raise TypeError('vrange() takes at most 2 non-kw args')
    stops = args[-1]

    if return_where:
        l = (stops - starts).astype(int)
        ind1 = np.repeat(np.arange(len(l)), l)
        ind2 = _vrange(0, l)
        return _vrange(starts, stops, dtype), (ind1, ind2)
    else:
        return _vrange(starts, stops, dtype)


def _vrange(starts, stops, dtype=None):  # TODO Merge with vrange?
    """
    Taken from a forum.
    Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    if (dtype == int) or (dtype is None):
        stops = np.asarray(stops, dtype=dtype)
        lengths = (stops - starts).astype(int)  # Lengths of each range.
        return np.repeat(stops - lengths.cumsum(), lengths) + np.arange(lengths.sum())
    else:
        return NotImplemented


def arange_2d(*args, dtype=None, return_mask=False):  # TODO remove use of args...
    """
    Equivalent of numpy.arange, but in 2d.

    usage:
        vrange([starts,] stops, dtype=None, return_mask=False)

    Parameters
    ----------
    starts (1-D array_like, optional):
        starts for each range, default is 0
    stops (1-D array_like):
        stops for each range (same shape as starts)
    dtype: type object, optional
        type of output array
    return_mask: bool, optional
        Return a mask where the values are not valid.
        If False, ones are put in these positions (weird).

    Ouput
    -----
    numpy.ndarray: 2d arange
    mask (ooptional)
    """
    if len(args) == 1:
        starts = 0
    elif len(args) == 2:
        starts = args[0]
    else:
        raise TypeError('vrange() takes at most 2 non-kw args')
    stops = args[-1]

    l1 = len(stops)
    l2 = int((stops - starts).max())

    out = np.ones((l1, l2), dtype=dtype)

    values, ind = vrange(starts, stops,
                         dtype=dtype, return_where=True)
    out[ind[0], ind[1]] = values

    if return_mask:
        mask = np.ones((l1, l2), dtype=bool)
        mask[ind[0], ind[1]] = False
        return out, mask
    else:
        return out


def arange_R_cst(x1, x2, res):
    """
    Return an array with constant resolution so
    that x/dx = res = constant
    """
    log_dx = np.log(1.0 + 1.0/res)
    log_x = np.arange(np.log(x1), np.log(x2), log_dx)

    return np.exp(log_x)
