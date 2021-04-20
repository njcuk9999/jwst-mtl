#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TODO Merge all this into utils.py?

# General imports.
import numpy as np


def _vrange(starts, stops, dtype=None):
    """Create concatenated ranges of integers for multiple start/stop values.

    :param starts: start values for each range.
    :param stops: end values for each range.
    :param dtype:

    :type starts: array[]
    :type stops: array[]
    :type dtype:

    :returns: ranges - an array containing the concatenated ranges.
    :rtype: array
    """

    # Check that starts and stops have the same shape.
    if len(starts) != len(stops):
        raise ValueError('starts and stops must have the same length.')

    # Check if the dtype is valid.
    if (dtype is not None) & (dtype != int):
        return NotImplemented

    # Create the array of ranges.
    stops = np.asarray(stops, dtype=dtype)
    lengths = (stops - starts).astype(int)  # Lengths of each range.
    values = np.repeat(stops - lengths.cumsum(), lengths) + np.arange(lengths.sum())

    return values


def vrange(*args, dtype=None):
    """Create concatenated ranges of integers for multiple start/stop values.

    :param starts: start values for each range.
    :param stops: end values for each range.
    :param dtype: type of output array

    :type starts: array[]
    :type stops: array[]
    :type dtype:

    :returns: values, indices - 1D array of concatenated ranges, tuple of
        indices for transforming the values to a 2D array.
    :rtype: Tuple(array, Tuple(array, array))
    """

    # Parse the input args.
    if len(args) == 1:
        starts = 0
    elif len(args) == 2:
        starts = args[0]
    else:
        raise TypeError('vrange() takes at most 2 non-keyword args.')

    stops = args[-1]

    # Compute the 1D array of consecutive ranges.
    values = _vrange(starts, stops, dtype)

    # Compute indices for transforming the values to a 2D array.
    lengths = (stops - starts).astype(int)
    ind1 = np.repeat(np.arange(len(lengths)), lengths)
    ind2 = _vrange(0, lengths)

    return values, (ind1, ind2)


def arange_2d(*args, dtype=None):
    """Create a 2D array containing a series of ranges. The ranges do not have
    to be of equal length.

    :param starts: start values for each range.
    :param stops: end values for each range.
    :param dtype:

    :type starts: array[]
    :type stops: array[]
    :type dtype:

    :returns: out, mask - The 2D array of ranges and a mask indicating valid
        elements.
    :rtype: Tuple(array[int], array[bool])
    """

    # Parse the input args.
    if len(args) == 1:
        starts = 0
    elif len(args) == 2:
        starts = args[0]
    else:
        raise TypeError('arange_2d() takes at most 2 non-keyword args')

    stops = args[-1]

    # Initialize the output array.
    nrows = len(stops)
    ncols = int((stops - starts).max())
    out = np.ones((nrows, ncols), dtype=dtype)
    mask = np.ones((nrows, ncols), dtype='bool')

    # Compute the 1D values and broadcast to 2D.
    values, indices = vrange(starts, stops, dtype=dtype)
    out[indices[0], indices[1]] = values
    mask[indices[0], indices[1]] = False

    return out, mask
