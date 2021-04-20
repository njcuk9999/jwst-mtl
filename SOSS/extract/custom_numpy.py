#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TODO Merge all this into utils.py?

# General imports.
import numpy as np


def _vrange(starts, stops, dtype=None):
    """Create concatenated ranges of integers for multiple start/stop values.

    :param starts: start values for each range.
    :param stops: end values for each range.
    :param dtype: the type of the output values.

    :type starts: int or array[int]
    :type stops: int or array[int]
    :type dtype: str

    :returns: values - 1D array of concatenated ranges.
    :rtype: array[int]
    """

    # Check if the dtype is valid. # TODO not sure what this does?
    if (dtype is not None) & (dtype != int):
        return NotImplemented

    # Create the array of ranges.
    stops = np.asarray(stops, dtype=dtype)
    lengths = (stops - starts).astype(int)  # Lengths of each range.
    values = np.repeat(stops - lengths.cumsum(), lengths) + np.arange(lengths.sum())

    return values


def vrange(starts, stops, dtype=None):
    """Create concatenated ranges of integers for multiple start/stop values.

    :param starts: start values for each range.
    :param stops: end values for each range.
    :param dtype: type of output array

    :type starts: int or array[int]
    :type stops: int or array[int]
    :type dtype: str

    :returns: values, irow, icol - 1D array of concatenated ranges, row and
        column indices for transforming the values to a 2D array.
    :rtype: Tuple(array[int], array[int], array[int])
    """

    # Ensure starts and stops are arrays.
    starts = np.asarray(starts)
    stops = np.asarray(stops)

    # Check input for starts and stops is valid.
    if (starts.shape != stops.shape) & (starts.shape != ()):
        msg = ('Shapes of starts and stops are not compatible, '
               'they must either have the same shape or starts must be scalar.')
        raise ValueError(msg)

    if np.any(stops < starts):
        msg = 'stops must be everywhere greater or equal to starts.'
        raise ValueError(msg)

    # Compute the 1D array of consecutive ranges.
    values = _vrange(starts, stops, dtype)

    # Compute indices for transforming the values to a 2D array.
    lengths = (stops - starts).astype(int)
    irow = np.repeat(np.arange(len(lengths)), lengths)
    icol = _vrange(0, lengths)

    return values, irow, icol


def arange_2d(starts, stops, dtype=None):
    """Create a 2D array containing a series of ranges. The ranges do not have
    to be of equal length.

    :param starts: start values for each range.
    :param stops: end values for each range.
    :param dtype: the type of the output values.

    :type starts: int or array[int]
    :type stops: int or array[int]
    :type dtype: str

    :returns: out, mask - 2D array of ranges and a mask indicating valid
        elements.
    :rtype: Tuple(array[int], array[bool])
    """

    # Ensure starts and stops are arrays.
    starts = np.asarray(starts)
    stops = np.asarray(stops)

    # Check input for starts and stops is valid.
    if (starts.shape != stops.shape) & (starts.shape != ()):
        msg = ('Shapes of starts and stops are not compatible, '
               'they must either have the same shape or starts must be scalar.')
        raise ValueError(msg)

    if np.any(stops < starts):
        msg = 'stops must be everywhere greater or equal to starts.'
        raise ValueError(msg)

    # Initialize the output array.
    nrows = len(stops)
    ncols = int((stops - starts).max())
    out = np.ones((nrows, ncols), dtype=dtype)
    mask = np.ones((nrows, ncols), dtype='bool')

    # Compute the 1D values and broadcast to 2D.
    values, irow, icol = vrange(starts, stops, dtype=dtype)
    out[irow, icol] = values
    mask[irow, icol] = False

    return out, mask
