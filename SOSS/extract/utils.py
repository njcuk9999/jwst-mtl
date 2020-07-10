import numpy as np
from scipy.integrate.quadrature import AccuracyWarning, _romberg_diff
from warnings import warn

def _get_lam_p_or_m(lam):
    '''
    Compute lambda_plus and lambda_minus
    '''
    lam = lam.T  # Simpler to use transpose
    
    # Init
    lam_r = np.zeros_like(lam)
    lam_l = np.zeros_like(lam)
    
    # Def delta lambda
    d_lam = np.diff(lam, axis=0)
    
    # Define lambda left and lambda right of each pixels
    lam_r[:-1] = lam[:-1] + d_lam/2
    lam_r[-1] = lam[-1] + d_lam[-1]/2
    lam_l[1:] = lam[:-1] + d_lam/2  # Same values as lam_r
    lam_l[0] = lam[0] - d_lam[0]/2
    
    if (lam_r >= lam_l).all():
        return lam_r.T, lam_l.T
    elif (lam_r <= lam_l).all():
        return lam_l.T, lam_r.T
    else:
        raise ValueError('Bad pixel values for wavelength')
        
        
def grid_from_map(wv, psf, wv_range=None, poly_ord=1, out_col=False):
    """
    Define wavelength grid by taking the center wavelength
    at each columns where the psf is maximised. If `wv_range` is
    out of the wv map, extrapolate with a polynomial of order `poly_ord`.
    """
    
    if wv_range is None:
        return _grid_from_map(wv, psf, out_col=out_col)
    else:
        # Get grid
        grid = _grid_from_map(wv, psf)
        # Define delta_grid as a function of grid
        # by fitting a polynomial
        d_grid = np.diff(grid)
        f_dgrid = np.polyfit(grid[:-1], d_grid, poly_ord)
        f_dgrid = np.poly1d(f_dgrid)
        
        # Make sure grid is between the range
        grid = grid[
            (wv_range[0] <= grid) & (grid <= wv_range[-1])
        ]
        
        # Extrapolate values out of the wv_map or
        grid_left, grid_right = [], []
        if wv_range[0] < grid.min():
            # Need the grid value to get delta_grid ...
            grid_left = [grid.min() - f_dgrid(grid.min())]
            # ... and iterate to get the next one until
            # the range is reached
            while grid_left[-1] > wv_range[0]:
                grid_left.append(grid_left[-1] - f_dgrid(grid_left[-1]))
            # Need to sort (and unique)
            grid_left = np.unique(grid_left)
        if wv_range[-1] > grid.max():
            # Need the grid value to get delta_grid ...
            grid_right = [grid.max() + f_dgrid(grid.max())]
            # ... and iterate to get the next one until
            # the range is reached
            while grid_right[-1] < wv_range[-1]:
                grid_right.append(grid_right[-1] + f_dgrid(grid_right[-1]))        
                
        return np.concatenate([grid_left, grid, grid_right])
        
def _grid_from_map(wv_map, psf, out_col=False):
    """ 
    Define wavelength grid by taking the center wavelength
    at each columns at the center of mass of
    the spatial profile.
    """
    # Normalisation for each column
    col_sum = psf.sum(axis=0)
    
    # Compute only valid columns
    i_good = (psf > 0).any(axis=0)
    i_good &= (wv_map > 0).any(axis=0)
    
    # Get center wavelength using center of mass
    # with psf as weights
    center_wv = (wv_map * psf)[:, i_good]
    center_wv /= col_sum[i_good]
    center_wv = center_wv.sum(axis=0)
    
    # Return sorted and unique
    out = np.unique(center_wv)
    
    # Return columns if specified
    if out_col:
        return out, i_good
    else:
        return out

def oversample_grid(lam_grid, n_os=1):
    """
    Returns lam_grid evenly oversample at `n_os`.
    
    Parameters
    ----------
    lam_grid: 1D array
        Grid to be oversampled
    n_os: scalar or 1D array, optional
        Oversampling. If it's a scalar, take the same value for each
        intervals of the grid. If it's an array, n_os is then
        specified for each interval of the grid, so
        len(n_os) = len(lam_grid) - 1.
    Returns
    -------
    new_grid: 1D array
        Oversampled grid.
    """
    
    # Convert n_os to array
    n_os = np.array(n_os)
    
    # n_os needs to have the dimension:
    # len(lam_grid) - 1
    if n_os.ndim == 0:
        n_os = np.repeat(n_os, len(lam_grid)-1)

    # Grid intervals
    d_lam = np.diff(lam_grid)
    # Init grid for output
    new_grid = lam_grid.copy()
    # Iterate to generate nodes
    for i_os in range(1, n_os.max()):
        # Compute only nodes that need to be computed
        index = (n_os > i_os)
        # Compute the next node in each grid intervals
        sub_grid = (lam_grid[:-1][index]
                    + i_os * d_lam[index] / n_os[index])
        # Add to ouput grid
        new_grid = np.concatenate([new_grid, sub_grid])
    
    # Return sorted and unique
    return np.unique(new_grid)
    
def uneven_grid(lam_grid, n_os=1, space=None):
    
    if space is None:
        space = 1/n_os
    
    if n_os > 1:
        d_lam = np.diff(lam_grid) * space
        sub_grid = [lam_grid[:-1] + (-1)**(i-1) * ((i+1)//2) * d_lam for i in range(1,n_os)]
        new_grid = np.concatenate([lam_grid, *sub_grid])
        return np.unique(new_grid)
    else:
        return lam_grid

def get_n_nodes(grid, fct, tol=1.48e-4, rtol=1.48e-4, divmax=10, out_res=False):
    """
    *** THIS FUNCTION IS _STRONGLY_ INSPIRED BY SCIPY ***
    See scipy.integrate.quadrature.romberg
    
    Refine parts of a grid to reach a specified integration precision 
    based on Romberg integration of a callable function or method.
    Returns the number of nodes needed in each intervals of
    the input grid to reach the specified tolerance over the integral
    of `fct` (a function of one variable).
    
    Note: The difference between th scipy version is that it is vectorised
    to deal with multiple intervals separately. It also returns the
    number of nodes needed to reached the required precision instead
    of returning the value of the integral.
    Parameters
    ----------
    grid: 1D array-like
        Grid for integration. Each sections of this grid are treated
        as separate integrals. So if grid has length N; N-1 integrals
        are optimized. 
    function : callable
        Function to be integrated.

    Returns
    -------
    n_grid  : 1D array, length = len(grid) - 1
        Number of nodes needed on each distinct intervals in the grid
        to reach the specified tolerance.
    residual : 1D array, optional
        Estimate of the error in each intervals. Same length as n_grid.
    Other Parameters
    ----------------
    tol, rtol : float, optional
        The desired absolute and relative tolerances. Defaults are 1.48e-4.
    divmax : int, optional
        Maximum order of extrapolation. Default is 10.
    out_res: bool, optional
        Return or not the residuals in each intervals. Default is False.
    See Also
    --------
    scipy.integrate.quadrature.romberg
    References
    ----------
    .. [1] 'Romberg's method' https://en.wikipedia.org/wiki/Romberg%27s_method
    """
    n_intervals = len(grid)-1
    i_bad = np.arange(n_intervals)
    n_grid = np.repeat(-1, n_intervals)
    residual = np.repeat(np.nan, n_intervals)
    
    intervals = np.array([grid[:-1], grid[1:]])
    intrange = np.diff(grid)
    err = np.inf
    
    # First estimate without subdivision
    n = 1
    ordsum = difftrap(fct, intervals, n)
    results = intrange * ordsum
    last_row = [results]

    for i_div in range(1, divmax+1):
        
        # Refine number of points
        n *= 2
        
        # Evaluate trpz integration for intervals
        # that are not converged
        ordsum += difftrap(fct, intervals[:,i_bad], n)
        row = [intrange[i_bad] * ordsum / n]
        
        # Compute Romberg for each computed sub grids
        for k in range(i_div):
            romb_k = _romberg_diff(last_row[k], row[k], k+1)
            row = np.vstack([row, romb_k])
            
        # Save R(n,n) and R(n-1,n-1) from Romberg method
        results = row[i_div]
        lastresults = last_row[i_div-1]
        
        # Estimate error
        err = np.abs(results - lastresults)
        
        # Find intervals that are converged
        conv = (err < tol) | (err < rtol * np.abs(results))
        
        # Save number of nodes for these intervals
        n_grid[i_bad[conv]] = n
        
        # Save residuals
        residual[i_bad] = err
        
        # Stop if convergence
        if conv.all():
            # All converged!
            break
            
        # Find intervals not converged
        i_bad = i_bad[~conv]
        
        # Save last_row for the next iteration
        # but keep only non-converged intervals
        last_row = row[:,~conv]
        # Same for ordsum
        ordsum = ordsum[~conv]
        
    else:
        # Warn that convergence is not reached everywhere
        # and print max residual.
        message = "divmax (%d) exceeded. Latest difference = {}"
        warn(message.format(divmax, err.max()), AccuracyWarning)
        
    # Make sure all values of n_grid where assigned during process
    if (n_grid == -1).any():
        raise ValueError("Values where not assigned at grid "
                         +"position: {}".format(np.where(n_grid == -1)))

    # Return
    if out_res:
        return n_grid, residual
    else:
        return n_grid
    
def difftrap(fct, interval, numtraps):
    """
    ** Quasi entirely taken from scipy.integrate.quadrature **
    Adapted to work with multiple intervals.
    
    Desciption taken from scipy:
    Perform part of the trapezoidal rule to integrate a fct.
    Assume that we had called difftrap with all lower powers-of-2
    starting with 1.  Calling difftrap only returns the summation
    of the new ordinates.  It does _not_ multiply by the width
    of the trapezoids.  This must be performed by the caller.
        'fct' is the fct to evaluate (must accept vector arguments).
        'interval' is a sequence with lower and upper limits
                   of integration.
        'numtraps' is the number of trapezoids to use (must be a
                   power-of-2).
    """
    # Convert input intervals to numpy array
    interval = np.array(interval)
    # If 1-d, add dimension so it's 2-d
    # (the algorithm is made for multiple intervals, so 2-d)
    if interval.ndim == 1:
        interval = interval[:,None]
    
    # Quasi-copy of scipy.integrate.quadrature._difftrap code
    if numtraps <= 0:
        raise ValueError("numtraps must be > 0 in difftrap().")
    elif numtraps == 1:
        return 0.5*(fct(interval[0])+fct(interval[1]))
    else:
        numtosum = numtraps/2
        h = (interval[1]-interval[0]).astype(float)/numtosum
        lox = interval[0] + 0.5 * h
        
        points = lox[None,:] + h * np.arange(numtosum)[:,None]
        s = np.sum(fct(points), axis=0)
        return s
