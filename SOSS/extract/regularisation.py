#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# General imports.
import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

# Local imports.
from .utils import grid_from_map
from .convolution import get_c_matrix, NyquistKer
from .engine_legacy import WebbKer

# Plotting.
import matplotlib.pyplot as plt


def finite_diff(x):
    """
    Returns the finite difference matrix operator based on x.
    Input:
        x: array-like
    Output:
        sparse matrix. When apply to x `diff_matrix.dot(x)`,
        the result is the same as np.diff(x)
    """
    n_x = len(x)

    # Build matrix
    diff_matrix = diags([-1.], shape=(n_x-1, n_x))
    diff_matrix += diags([1.], 1, shape=(n_x-1, n_x))

    return diff_matrix


def finite_second_d(grid):
    """
    Returns the second derivative operator based on grid
    Inputs:
    -------
    grid: 1d array-like
        grid where the second derivative will be compute.
    Ouputs:
    -------
    second_d: matrix
        Operator to compute the second derivative, so that
        f" = second_d.dot(f), where f is a function
        projected on `grid`.
    """

    # Finite difference operator
    d_matrix = finite_diff(grid)

    # Delta lambda
    d_grid = d_matrix.dot(grid)

    # First derivative operator
    first_d = diags(1./d_grid).dot(d_matrix)

    # Second derivative operator
    second_d = finite_diff(grid[:-1]).dot(first_d)

    # don't forget the delta labda
    second_d = diags(1./d_grid[:-1]).dot(second_d)

    return second_d


def finite_first_d(grid):
    """
    Returns the first derivative operator based on grid
    Inputs:
    -------
    grid: 1d array-like, optional
        grid where the first derivative will be compute.
    Ouputs:
    -------
    first_d: matrix
        Operator to compute the second derivative, so that
        f' = first_d.dot(f), where f is a function
        projected on `grid`.
    """

    # Finite difference operator
    d_matrix = finite_diff(grid)

    # Delta lambda
    d_grid = d_matrix.dot(grid)

    # First derivative operator
    first_d = diags(1./d_grid).dot(d_matrix)

    return first_d


def finite_zeroth_d(grid):
    """
    Gives the zeroth derivative operator on the function
    f(grid), so simply returns the identity matrix... XD
    """
    return identity(len(grid))


def get_nyquist_matrix(grid, integrate=True, n_sampling=2,
                       thresh=1e-5, **kwargs):
    """
    Get the tikhonov regularisation matrix based on
    a Nyquist convolution matrix (convolution with
    a kernel with a resolution given by the sampling
    of a grid). The Tikhonov matrix will be given by
    the difference of the nominal solution and
    the convolved solution.

    Parameters
    ----------
    grid: 1d-array
        Grid to project the kernel
    integrate: bool, optional
        If True, add integration weights to the tikhonov matrix, so
        when the squared norm is computed, the result is equivalent
        to the integral of the integrand squared.
    n_sampling: int, optional
        sampling of the grid. Default is 2, so we assume that
        the grid is Nyquist sampled.
    thresh: float, optional
        Used to define the maximum length of the kernel.
        Truncate when `kernel` < `thresh`
    kwargs:
        `interp1d` kwargs used to get FWHM as a function of the grid.
    """

    # Get nyquist kernel function
    ker = NyquistKer(grid, n_sampling=n_sampling, **kwargs)

    # Build convolution matrix
    conv_matrix = get_c_matrix(ker, grid, thresh=thresh)

    # Build tikhonov matrix
    t_mat = conv_matrix - identity(conv_matrix.shape[0])

    if integrate:
        # The grid may not be evenly spaced, so
        # add an integration weight
        d_grid = np.diff(grid)
        d_grid = np.concatenate([d_grid, [d_grid[-1]]])
        t_mat = diags(np.sqrt(d_grid)).dot(t_mat)

    return t_mat


class TikhoConvMatrix:
    """
    Convolution matrix to be used as
    Tikhonov regularisation matrix.
    This way, the solution of the system can be
    deviated towards solutions closed to a solution
    at a resolution close to `n_os` times the resolution
    given by the `wv_map`.
    """
    def __init__(self, wv_map, psf, n_os=2, thresh=1e-5):
        """
        Parameters
        ----------
        wv_map : (N, M) 2-D arrays
            array of the central wavelength position each pixels
            on the detector. It has to have the same (N, M) as the detector.
        psf : (N, M) 2-D array
            array of the spatial profile on the detector.
            It has to have the same (N, M) as the detector.
        n_os: int, optional
            Dictates the resolution of the convolution kernel.
            The resolution will be `n_os` times the resolution
            of order 2. Default is 2 times (n_os=2).
        thresh: float, optional
            Used to define the maximum length of the kernel.
            Truncate when kernel < `thresh`. Default is 1e-5
        """
        # Save attributes
        self.wv_map = wv_map
        self.psf = psf
        self.n_os = n_os
        self.thresh = thresh

    def __call__(self, grid):
        """
        Return the tikhonov regularisation matrix given
        a grid (to project the convolution kernels).
        grid is a 1d array.
        Returns a sparse matrix.
        """

        # Get needed attributes
        gargs = ['wv_map', 'psf', 'n_os', 'thresh']
        wv_map, psf, n_os, thresh = self.getattrs(*gargs)

        # Generate a fake wv_map to cover all wv_range with a
        # resolution `t_mat_n_os` times the resolution
        # of wv_map (generally order 2).
        wv_range = [grid.min(), grid.max()]
        wv_map = grid_from_map(wv_map, psf, wv_range=wv_range, n_os=n_os)

        # Build convolution matrix
        conv_ord2 = get_c_matrix(WebbKer(wv_map[None, :]),
                                 grid, thresh=thresh)

        # Build tikhonov matrix
        t_mat = conv_ord2 - identity(conv_ord2.shape[0])

        # The grid may not be evenly spaced, so
        # add an integration weight
        d_grid = np.diff(grid)
        d_grid = np.concatenate([d_grid, [d_grid[-1]]])
        t_mat = diags(np.sqrt(d_grid)).dot(t_mat)

        return t_mat

    def getattrs(self, *args):
        """
        Return list of attributes

        Parameters
        ----------
        args: str
            All attributes to return.
        """
        return [getattr(self, arg) for arg in args]


def tikho_solve(a_mat, b_vec, t_mat=None, grid=None,
                verbose=True, factor=1.0, estimate=None, index=None):
    """
    Tikhonov solver to use as a function instead of a class.

    Parameters
    ----------
    a_mat: matrix-like object (2d)
        matrix A in the system to solve A.x = b
    b_vec: vector-like object (1d)
        vector b in the system to solve A.x = b
    t_mat: matrix-like object (2d), optional
        Tikhonov regularisation matrix to be applied on b_vec.
        Default is the default of the Tikhonov class. (Identity matrix)
    grid: array-like 1d, optional
        grid on which b-vec is projected. Used to compute derivative
    verbose: bool
        Print details or not
    factor: float, optional
        multiplicative constant of the regularisation matrix
    estimate: vector-like object (1d)
        Estimate oof the solution of the system.
    index: indexable, optional
        index of the valid row of the b_vec.

    Returns
    ------
    Solution of the system (1d array)
    """
    tikho = Tikhonov(a_mat, b_vec, t_mat=t_mat,
                     grid=grid, verbose=verbose, index=index)

    return tikho.solve(factor=factor, estimate=estimate)


class Tikhonov:
    """
    Tikhonov regularisation to solve the ill-condition problem:
    A.x = b, where A is accidently singular or close to singularity.
    Tikhonov regularisation adds a regularisation term in
    the equation and aim to minimize the equation:
    ||A.x - b||^2 + ||gamma.x||^2
    Where gamma is the Tikhonov regularisation matrix.
    """
    default_mat = {'zeroth': finite_zeroth_d,
                   'first': finite_first_d,
                   'second': finite_second_d}

    def __init__(self, a_mat, b_vec, t_mat=None,
                 grid=None, verbose=True, index=None):
        """
        Parameters
        ----------
        a_mat: matrix-like object (2d)
            matrix A in the system to solve A.x = b
        b_vec: vector-like object (1d)
            vector b in the system to solve A.x = b
        t_mat: matrix-like object (2d), optional
            Tikhonov regularisation matrix to be applied on b_vec.
            Default is the default of the Tikhonov class. (Identity matrix)
        grid: array-like 1d, optional
            grid on which b-vec is projected. Used to compute derivative
        verbose: bool
            Print details or not
        index: indexable, optional
            index of the valid row of the b_vec.
        """

        # b_vec will be passed to default_mat functions
        # if grid not given.
        if grid is None and t_mat is None:
            grid = b_vec

        # If string, search in the default Tikhonov matrix
        if t_mat is None:
            self.type = 'zeroth'
            t_mat = self.default_mat['zeroth'](grid)
        elif callable(t_mat):
            t_mat = t_mat(grid)
            self.type = 'custom'
        else:
            self.type = 'custom'

        # Take all indices by default
        if index is None:
            index = slice(None)

        self.a_mat = a_mat[index, :][:, index]
        self.b_vec = b_vec[index]
        self.t_mat = t_mat[index, :][:, index]
        self.index = index
        self.verbose = verbose
        self.test = None

        return

    def verbose_print(self, *args, **kwargs):
        """Print if verbose is True. Same as `print` function."""

        if self.verbose:
            print(*args, **kwargs)

        return

    def solve(self, factor=1.0, estimate=None):
        """
        Minimize the equation ||A.x - b||^2 + ||gamma.x||^2
        by solving (A_T.A + gamma_T.gamma).x = A_T.b
        gamma is the Tikhonov matrix multiplied by a scale factor

        Parameters
        ----------
        factor: float, optional
            multiplicative constant of the regularisation matrix
        estimate: vector-like object (1d)
            Estimate oof the solution of the system.

        Returns
        ------
        Solution of the system (1d array)
        """
        # Get needed attributes
        a_mat = self.a_mat
        b_vec = self.b_vec
        index = self.index

        # Matrix gamma (with scale factor)
        gamma = factor * self.t_mat

        # Build system
        gamma_2 = (gamma.T).dot(gamma)  # Gamma square
        matrix = a_mat.T.dot(a_mat) + gamma_2
        result = (a_mat.T).dot(b_vec.T)

        # Include solution estimate if given
        if estimate is not None:
            result += gamma_2.dot(estimate[index].T)

        # Solve
        return spsolve(matrix, result)

    def test_factors(self, factors, estimate=None):
        """
        test multiple factors

        Parameters
        ----------
        factors: 1d array-like
            factors to test
        estimate: array like
            estimate of the solution

        Returns
        ------
        dictionnary of test results
        """

        self.verbose_print('Testing factors...')

        # Get relevant attributes
        b_vec = self.b_vec
        a_mat = self.a_mat
        t_mat = self.t_mat

        # Init outputs
        sln, err, reg = [], [], []

        # Test all factors
        for i_fac, factor in enumerate(factors):

            # Save solution
            sln.append(self.solve(factor, estimate))

            # Save error A.x - b
            err.append(a_mat.dot(sln[-1]) - b_vec)

            # Save regulatisation term
            reg.append(t_mat.dot(sln[-1]))

            # Print
            message = '{}/{}'.format(i_fac, len(factors))
            self.verbose_print(message, end='\r')

        # Final print
        message = '{}/{}'.format(i_fac + 1, len(factors))
        self.verbose_print(message)

        # Convert to arrays
        sln = np.array(sln)
        err = np.array(err)
        reg = np.array(reg)

        # Save in a dictionnary
        self.test = {'factors': factors,
                     'solution': sln,
                     'error': err,
                     'reg': reg}

        return self.test

    def _check_plot_inputs(self, fig, ax, label, factors, test):
        """
        Method to manage inputs for plots methods.
        """

        # Use ax or fig if given. Else, init the figure
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots(1, 1, sharex=True)
        elif ax is None:
            ax = fig.subplots(1, 1, sharex=True)

        # Use the type of regularisation as label if None is given
        if label is None:
            label = self.type

        if test is None:

            # Run tests with `factors` if not done already.
            if self.test is None:
                self.test_factors(factors)

            test = self.test

        return fig, ax, label, test

    def error_plot(self, fig=None, ax=None, factors=None,
                   label=None, test=None, test_key=None, y_val=None):
        """
        Plot error as a function of factors

        Parameters
        ----------
        fig: matplotlib figure, optional
            Figure to use for plot
            If not given and ax is None, new figure is initiated
        ax: matplotlib axis, optional
            axis to use for plot. If not given, a new axis is initiated.
        factors: 1d array-like
            factors to test
        label: str, optional
            label too put in legend
        test: dictionnary, optional
            dictionnary of tests (output of Tikhonov.test_factors)
        test_key: str, optional
            which test result to plot. If not specified,
            the euclidian norm of the 'error' key will be used.
        y_val: array-like, optional
            y values to plot. Same length as factors.

        Returns
        ------
        fig, ax
        """

        # Manage method's inputs
        args = (fig, ax, label, factors, test)
        fig, ax, label, test = self._check_plot_inputs(*args)

        # What y value do we plot?
        if y_val is None:

            # Use tests to plot y_val
            if test_key is None:

                # Default is euclidian norm of error.
                # Similar to the chi^2.
                y_val = (test['error']**2).sum(axis=-1)
            else:
                y_val = test[test_key]

        # Plot
        ax.loglog(test['factors'], y_val, label=label)

        # Mark minimum value
        i_min = np.argmin(y_val)
        min_coord = test['factors'][i_min], y_val[i_min]
        ax.scatter(*min_coord, marker="x")
        text = '{:2.1e}'.format(min_coord[0])
        ax.text(*min_coord, text, va="top", ha="center")

        # Show legend
        ax.legend()

        # Labels
        ax.set_xlabel("Scale factor")
        ylabel = r'System error '
        ylabel += r'$\left(||\mathbf{Ax-b}||^2_2\right)$'
        ax.set_ylabel(ylabel)

        return fig, ax

    def l_plot(self, fig=None, ax=None, factors=None, label=None,
               test=None, text_label=True, factor_norm=False):
        """
        make an 'l plot'

        Parameters
        ----------
        fig: matplotlib figure, optional
            Figure to use for plot
            If not given and ax is None, new figure is initiated
        ax: matplotlib axis, optional
            axis to use for plot. If not given, a new axis is initiated.
        factors: 1d array-like
            factors to test
        label: str, optional
            label too put in legend
        test: dictionnary, optional
            dictionnary of tests (output of Tikhonov.test_factors)
        text_label: bool, optional
            Add label of the factor value to each points in the plot.

        Returns
        ------
        fig, ax
        """

        # Manage method's inputs
        args = (fig, ax, label, factors, test)
        fig, ax, label, test = self._check_plot_inputs(*args)

        # Compute euclidian norm of error (||A.x - b||).
        # Similar to the chi^2.
        err_norm = (test['error']**2).sum(axis=-1)

        # Compute norm of regularisation term
        reg_norm = (test['reg']**2).sum(axis=-1)

        # Factors
        if factor_norm:
            reg_norm *= test['factors']**2

        # Plot
        ax.loglog(err_norm, reg_norm, '.:', label=label)

        # Add factor values as text
        if text_label:
            for f, x, y in zip(test['factors'], err_norm, reg_norm):
                plt.text(x, y, "{:2.1e}".format(f), va="center", ha="right")

        # Legend
        ax.legend()

        # Labels
        xlabel = r'$\left(||\mathbf{Ax-b}||^2_2\right)$'
        ylabel = r'$\left(||\mathbf{\Gamma.x}||^2_2\right)$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax
