#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TODO rename to extraction_utils.py?

# General imports.
import numpy as np
from warnings import warn
from scipy.integrate import AccuracyWarning
from scipy.sparse import find, csr_matrix


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


def sparse_k(val, k, n_k):
    """
    Transform a 2D array `val` to a sparse matrix.
    `k` is use for the position in the second axis
    of the matrix. The resulting sparse matrix will
    have the shape : ((len(k), n_k))
    Set k elements to a negative value when not defined
    """

    # Length of axis 0
    n_i = len(k)

    # Get row index
    i_k = np.indices(k.shape)[0]

    # Take only well defined coefficients
    row = i_k[k >= 0]
    col = k[k >= 0]
    data = val[k >= 0]

    mat = csr_matrix((data, (row, col)), shape=(n_i, n_k))

    return mat


def unsparse(matrix, fill_value=np.nan):
    """
    Convert a sparse matrix to a 2D array of values and a 2D array of position.

    Returns
    ------
    out: 2d array
        values of the matrix. The shape of the array is given by:
        (matrix.shape[0], maximum number of defined value in a column).
    col_out: 2d array
        position of the columns. Same shape as `out`.
    """

    col, row, val = find(matrix.T)
    n_row, n_col = matrix.shape

    good_rows, counts = np.unique(row, return_counts=True)

    # Define the new position in columns
    i_col = np.indices((n_row, counts.max()))[1]
    i_col = i_col[good_rows]
    i_col = i_col[i_col < counts[:, None]]

    # Create outputs and assign values
    col_out = np.ones((n_row, counts.max()), dtype=int) * -1
    col_out[row, i_col] = col
    out = np.ones((n_row, counts.max())) * fill_value
    out[row, i_col] = val

    return out, col_out


def get_wave_p_or_m(wave_map):
    # TODO rename function?
    """Compute lambda_plus and lambda_minus of pixel map, given the pixel
    central value.

    :param wave_map: Array of the pixel wavelengths for a given order.

    :type wave_map: array[float]

    :returns: wave_plus, wave_minus - The wavelength edges of each pixel,
        given the central value.
    :rtype: Tuple(array[float], array[float])
    """

    wave_map = wave_map.T  # Simpler to use transpose

    # Iniitialize arrays.
    wave_left = np.zeros_like(wave_map)
    wave_right = np.zeros_like(wave_map)

    # Compute the change in wavelength.
    delta_wave = np.diff(wave_map, axis=0)

    # Compute the wavelength values on the left and right edges of each pixel.
    wave_left[1:] = wave_map[:-1] + delta_wave/2  # TODO check this logic.
    wave_left[0] = wave_map[0] - delta_wave[0]/2
    wave_right[:-1] = wave_map[:-1] + delta_wave/2
    wave_right[-1] = wave_map[-1] + delta_wave[-1]/2

    # The outputs depend on the direction of the spectral axis.
    if (wave_right >= wave_left).all():
        wave_plus, wave_minus = wave_right.T, wave_left.T
    elif (wave_right <= wave_left).all():
        wave_plus, wave_minus = wave_left.T, wave_right.T
    else:
        raise ValueError('Bad pixel values for wavelength.')

    return wave_plus, wave_minus


def oversample_grid(wave_grid, n_os=1):
    """Create an oversampled version of the input 1D wavelength grid.

    :param wave_grid: Wavelength grid to be oversampled.
    :param n_os: Oversampling factor. If it is a scalar, take the same value for each
        interval of the grid. If it is an array, n_os specifies the oversampling
        at each interval of the grid, so len(n_os) = len(wave_grid) - 1.

    :type wave_grid: array[float]
    :type n_os: int or array[int]

    :returns: wave_grid_os - The oversampled wavelength grid.
    :rtype: array[float]
    """

    # Convert n_os to an array.
    n_os = np.asarray(n_os)

    # n_os needs to have the dimension: len(wave_grid) - 1.
    if n_os.ndim == 0:

        # A scalar was given, repeat the value.
        n_os = np.repeat(n_os, len(wave_grid) - 1)

    elif len(n_os) != (len(wave_grid) - 1):
        # An array of incorrect size was given.
        msg = 'n_os must be a scalar or an array of size len(wave_grid) - 1.'
        raise ValueError(msg)

    # Grid intervals.
    delta_wave = np.diff(wave_grid)

    # Initialize the new oversampled wavelength grid.
    wave_grid_os = wave_grid.copy()

    # Iterate over oversampling factors to generate new grid points.
    for i_os in range(1, n_os.max()):

        # Consider only intervals that are not complete yet.
        mask = n_os > i_os

        # Compute the new grid points.
        sub_grid = (wave_grid[:-1][mask] + i_os*delta_wave[mask]/n_os[mask])

        # Add the grid points to the oversampled wavelength grid.
        wave_grid_os = np.concatenate([wave_grid_os, sub_grid])

    # Take only uniqyue values and sort them.
    wave_grid_os = np.unique(wave_grid_os)

    return wave_grid_os


def extrapolate_grid(wave_grid, wave_range, poly_ord):
    """Extrapolate the given 1D wavelength grid to cover a given range of values
    by fitting the derivate with a polynomial of a given order and using it to
    compute subsequent values at both ends of the grid.

    :param wave_grid: Wavelength grid to be extrapolated.
    :param wave_range: Wavelength range the new grid should cover.
    :param poly_ord: Order of the polynomial used to fit the derivative of
        wave_grid.

    :type wave_grid: array[float]
    :type wave_range: list[float]
    :type poly_ord: int

    :returns: wave_grid_ext - The extrapolated 1D wavelength grid.
    :rtype: array[float]
    """

    # Define delta_wave as a function of wavelength by fitting a polynomial.
    delta_wave = np.diff(wave_grid)
    pars = np.polyfit(wave_grid[:-1], delta_wave, poly_ord)
    f_delta = np.poly1d(pars)

    # Extrapolate out-of-bound values on the left-side of the grid.
    grid_left = []
    if wave_range[0] < wave_grid.min():

        # Compute the first extrapolated grid point.
        grid_left = [wave_grid.min() - f_delta(wave_grid.min())]

        # Iterate until the end of wave_range is reached.
        while True:
            next_val = grid_left[-1] - f_delta(grid_left[-1])

            if next_val < wave_range[0]:
                break
            else:
                grid_left.append(next_val)

        # Sort extrapolated vales (and keep only unique).
        grid_left = np.unique(grid_left)

    # Extrapolate out-of-bound values on the right-side of the grid.
    grid_right = []
    if wave_range[-1] > wave_grid.max():

        # Compute the first extrapolated grid point.
        grid_right = [wave_grid.max() + f_delta(wave_grid.max())]

        # Iterate until the end of wave_range is reached.
        while True:
            next_val = grid_right[-1] + f_delta(grid_right[-1])

            if next_val > wave_range[-1]:
                break
            else:
                grid_right.append(next_val)

        # Sort extrapolated vales (and keep only unique).
        grid_right = np.unique(grid_right)

    # Combine the extrapolated sections with the original grid.
    wave_grid_ext = np.concatenate([grid_left, wave_grid, grid_right])

    return wave_grid_ext


def _grid_from_map(wave_map, aperture, out_col=False):
    # TODO is out_col still needed.
    """Define a wavelength grid by taking the wavelength of each column at the
    center of mass of the spatial profile.

    :param wave_map: Array of the pixel wavelengths for a given order.
    :param aperture: Array of the spatial profile for a given order.
    :param out_col:

    :type wave_map: array[float]
    :type aperture: array[float]
    :type out_col: bool

    :returns:
    :rtype:
    """

    # Use only valid columns.
    mask = (aperture > 0).any(axis=0) & (wave_map > 0).any(axis=0)

    # Get central wavelength using PSF as weights.
    num = (aperture * wave_map).sum(axis=0)
    denom = aperture.sum(axis=0)
    center_wv = num[mask]/denom[mask]

    # Make sure the wavelength values are in ascending order.
    sort = np.argsort(center_wv)
    grid = center_wv[sort]

    if out_col:  # TODO I don't like this type of contruction much.
        # Return index of columns if out_col is True.
        icols, = np.where(mask)
        return grid, icols[sort]
    else:
        # Return sorted and unique if out_col is False.
        grid = np.unique(grid)
        return grid


def grid_from_map(wave_map, aperture, wave_range=None, n_os=1, poly_ord=1,
                  out_col=False):
    # TODO is out_col still needed.
    """Define a wavelength grid by taking the central wavelength at each columns
    given by the center of mass of the spatial profile (so one wavelength per
    column). If wave_range is outside of the wave_map, extrapolate with a
    polynomial of order poly_ord.

    :param wave_map: Array of the pixel wavelengths for a given order.
    :param aperture: Array of the spatial profile for a given order.
    :param wave_range: Minimum and maximum boundary of the grid to generate,
        in microns. wave_range must include some wavelenghts of wave_map.
    :param n_os: Oversampling of the grid compare to the pixel sampling. Can be
        specified for each order if a list is given. If a single value is given
        it will be used for all orders.
    :param poly_ord: Order of the polynomial use to extrapolate the grid.
        Default is 1.
    :param out_col: Return columns. TODO It will be forced to False if extrapolation is needed.

    :type wave_map: array[float]
    :type aperture: array[float]
    :type wave_range: List[float]
    :type poly_ord: int
    :type out_col: bool
    :type n_os: int

    :returns:
    :rtype:
    """

    # Different treatement if wave_range is given.
    if wave_range is None:
        out = _grid_from_map(wave_map, aperture, out_col=out_col)
    else:
        # Get an initial estimate of the grid.
        grid, icols = _grid_from_map(wave_map, aperture, out_col=True)

        # Check if extrapolation needed. If so, out_col must be False.
        extrapolate = (wave_range[0] < grid.min()) | (wave_range[1] > grid.max())
        if extrapolate and out_col:
            out_col = False
            msg = ("Cannot extrapolate and return columns. "
                   "Setting out_col = False.")
            warn(msg)

        # Make sure grid is between the range
        mask = (wave_range[0] <= grid) & (grid <= wave_range[-1])

        # Check if grid and wv_range are compatible
        if not mask.any():
            msg = "Invalid wave_map or wv_range. wv_range: {}"
            raise ValueError(msg.format(wave_range))

        grid, icols = grid[mask], icols[mask]

        # Extrapolate values out of the wv_map if needed
        if extrapolate:
            grid = extrapolate_grid(grid, wave_range, poly_ord)

        # Different output depending on `out_col`
        if out_col:
            out = grid, icols
        else:
            out = grid

    # Apply oversampling
    if out_col:
        # Return grid and columns TODO this doesn't seem to do that? Would crash if out was a tuple?
        return [oversample_grid(out_i, n_os=n_os) for out_i in out]
    else:
        # Only the grid
        return oversample_grid(out, n_os=n_os)


def get_soss_grid(wave_maps, apertures, wave_min=0.55, wave_max=3.0, n_os=None):
    # TODO replace wave_min, wave_max with wave_range?
    """Create a wavelength grid specific to NIRISS SOSS mode observations.
    Assumes 2 orders are given, use grid_from_map if only one order is needed.

    Parameters
    ----------
    :param wave_maps: Array containing the pixel wavelengths for order 1 and 2.
    :param apertures: Array containing the spatial profiles for order 1 and 2.
    :param wave_min: Minimum wavelength the output grid should cover.
    :param wave_max: Maximum wavelength the output grid should cover.
    :param n_os: Oversampling of the grid compare to the pixel sampling. Can be
        specified for each order if a list is given. If a single value is given
        it will be used for all orders.

    :type wave_maps: array[float]
    :type apertures: array[float]
    :type wave_min: float
    :type wave_max: float
    :type n_os: int or List[int]

    :returns: wave_grid_soss - A wavelength grid optimized for extracting SOSS
        spectra across order 1 and order 2.
    :rtype: array[float]
    """

    # Check n_os input, default value is 2 for all orders.
    if n_os is None:  # TODO check for integer type?
        n_os = [2, 2]
    elif np.ndim(n_os) == 0:
        n_os = [n_os, n_os]
    elif len(n_os) != 2:
        msg = ("n_os must be an integer or a 2 element list or array of "
               "integers, got {} instead")
        raise ValueError(msg.format(n_os))

    # Generate a wavelength range for each order.
    # Order 1 covers the reddest part of the spectrum,
    # so apply wave_max on order 1 and vice versa for order 2.

    # Take the most restrictive wave_min for order 1
    wave_min_o1 = np.maximum(wave_maps[0].min(), wave_min)

    # Take the most restrictive wave_max for order 2.
    wave_max_o2 = np.minimum(wave_maps[1].max(), wave_max)

    # Now generate range for each orders
    range_list = [[wave_min_o1, wave_max],
                  [wave_min, wave_max_o2]]

    # Use grid_from_map to construct separate oversampled grids for both orders.
    wave_grid_o1 = grid_from_map(wave_maps[0], apertures[0],
                                 wave_range=range_list[0], n_os=n_os[0])
    wave_grid_o2 = grid_from_map(wave_maps[1], apertures[1],
                                 wave_range=range_list[1], n_os=n_os[1])

    # Keep only wavelengths in order 1 that aren't covered by order 2.
    mask = wave_grid_o1 > wave_grid_o2.max()
    wave_grid_o1 = wave_grid_o1[mask]

    # Combine the order 1 and order 2 grids.
    wave_grid_soss = np.concatenate([wave_grid_o1, wave_grid_o2])

    # Sort values (and keep only unique).
    wave_grid_soss = np.unique(wave_grid_soss)

    return wave_grid_soss


def _romberg_diff(b, c, k):
    """Compute the differences for the Romberg quadrature corrections.
    See Forman Acton's "Real Computing Made Real," p 143.

    :param b: R(n-1, m-1) of Rombergs method.
    :param c: R(n, m-1) of Rombergs method.
    :param k: The parameter m of Rombergs method.

    :type b: float or array[float]
    :type c: float or array[float]
    :type k: int

    :returns: R(n, m) of Rombergs method.
    :rtype: float or array[float]
    """

    tmp = 4.0**k
    diff = (tmp * c - b) / (tmp - 1.0)

    return diff


def _difftrap(fct, intervals, numtraps):
    """Perform part of the trapezoidal rule to integrate a function. Assume that
    we had called difftrap with all lower powers-of-2 starting with 1. Calling
    difftrap only returns the summation of the new ordinates. It does not
    multiply by the width of the trapezoids. This must be performed by the
    caller.

    Note: This function is based on scipy.integrate.quadrature. Adapted to work
    with multiple intervals.

    :param fct: Function to be integrated.
    :param intervals: A 2D array of integration intervals of shape (Nx2) or a
        single interval of shape (2,).
    :param numtraps: The number of trapezoids used to integrate the interval.
        numtraps must be a power of 2.

    :type fct: callable
    :type intervals: array[float]
    :type numtraps: int

    :returns: s - The sum of function values at the new trapezoid boundaries
    compared to numtraps = numtraps/2. When numtraps = 1 they are fivided by
    two.
    :rtype: float
    """

    # Convert input intervals to numpy array
    intervals = np.asarray(intervals)

    # If intervals is 1D assume it's a single interval.
    if intervals.ndim == 1:
        intervals = intervals[:, np.newaxis]

    # Check the value of numtraps.
    if numtraps <= 0:  # TODO check it is a power of 2? Or change input to log2(numtraps)?
        raise ValueError("numtraps must be > 0 in difftrap().")

    if numtraps == 1:
        # Return the function evaluations for a single trapezoid.
        # Only points add the edge of the interval need to be halfed.
        ordsum = 0.5*(fct(intervals[0]) + fct(intervals[1]))

    else:
        # Number of new points compared to lower 2**N multiple of trapezoids.
        numtosum = numtraps/2

        # Find coordinates of new points.
        h = (intervals[1] - intervals[0])/numtosum
        lox = intervals[0] + 0.5*h
        points = lox[np.newaxis, :] + h*np.arange(numtosum)[:, np.newaxis]

        # Evalaute and sum the new points.
        ordsum = np.sum(fct(points), axis=0)

    return ordsum


def get_n_nodes(grid, fct, divmax=10, tol=1.48e-4, rtol=1.48e-4):
    """Refine parts of a grid to reach a specified integration precision
    based on Romberg integration of a callable function or method.
    Returns the number of nodes needed in each intervals of
    the input grid to reach the specified tolerance over the integral
    of `fct` (a function of one variable).

    Note: This function is based on scipy.integrate.quadrature.romberg. The
    difference between it and the scipy version is that it is vectorised to deal
    with multiple intervals separately. It also returns the number of nodes
    needed to reached the required precision instead of returning the value of
    the integral.

    :param grid: Grid for integration. Each sections of this grid are treated
        as separate integrals. So if grid has length N; N-1 integrals are
        optimized.
    :param fct: Function to be integrated.
    :param tol: The desired absolute tolerance. Default is 1.48e-4.
    :param rtol: The desired relative tolerance. Default is 1.48e-4.
    :param divmax: Maximum order of extrapolation. Default is 10.

    :type grid: array[float]
    :type fct: callable
    :type tol: float
    :type rtol: float
    :type divmax: int

    :returns: n_grid - Number of nodes needed on each distinct intervals in the
        grid to reach the specified tolerance. If out_res=True also returns
        residual - Estimate of the error in each intervals. Same length as
        n_grid.
    """

    # Initialize some variables.
    n_intervals = len(grid) - 1
    i_bad = np.arange(n_intervals)
    n_grid = np.repeat(-1, n_intervals)
    residual = np.repeat(np.nan, n_intervals)

    # Change the 1D grid into a 2D set of intervals.
    intervals = np.array([grid[:-1], grid[1:]])
    intrange = np.diff(grid)
    err = np.inf

    # First estimate without subdivision.
    numtraps = 1
    ordsum = _difftrap(fct, intervals, numtraps)
    results = intrange * ordsum
    last_row = [results]

    for i_div in range(1, divmax + 1):

        # Increase the number of trapezoids by factors of 2.
        numtraps *= 2

        # Evaluate trpz integration for intervals that are not converged.
        ordsum += _difftrap(fct, intervals[:, i_bad], numtraps)
        row = [intrange[i_bad] * ordsum / numtraps]

        # Compute Romberg for each of the computed sub grids.
        for k in range(i_div):
            romb_k = _romberg_diff(last_row[k], row[k], k + 1)
            row = np.vstack([row, romb_k])

        # Save R(n,n) and R(n-1, n-1) from Romberg method.
        results = row[i_div]
        lastresults = last_row[i_div - 1]

        # Estimate error.
        err = np.abs(results - lastresults)

        # Find intervals that are converged.
        conv = (err < tol) | (err < rtol * np.abs(results))

        # Save number of nodes for these intervals.
        n_grid[i_bad[conv]] = numtraps

        # Save residuals.
        residual[i_bad] = err

        # Stop if all intervals have converged.
        if conv.all():
            break

        # Find intervals not converged.
        i_bad = i_bad[~conv]

        # Save last_row and ordsum for the next iteration for non-converged
        # intervals.
        ordsum = ordsum[~conv]
        last_row = row[:, ~conv]

    else:
        # Warn that convergence is not reached everywhere.
        msg = "divmax {%d} exceeded. Latest difference = {}"
        warn(msg.format(divmax, err.max()), AccuracyWarning)

    # Make sure all values of n_grid where assigned during the process.
    if (n_grid == -1).any():
        msg = "Values where not assigned at grid position: {}"
        raise ValueError(msg.format(np.where(n_grid == -1)))

    return n_grid, residual


def main():

    return


if __name__ == '__main__':
    main()
