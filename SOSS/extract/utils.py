import numpy as np

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
        
        
def grid_from_map(wv, psf, wv_range=None, poly_ord=1):
    """
    Define wavelength grid by taking the center wavelength
    at each columns where the psf is maximised. If `wv_range` is
    out of the wv map, extrapolate with a polynomial of order `poly_ord`.
    """
    
    if wv_range is None:
        return _grid_from_map(wv, psf)
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
        
def _grid_from_map(wv_map, psf):
    """ 
    Define wavelength grid by taking the center wavelength
    at each columns where the psf is maximised.
    """
    
    # Find max psf to define the grid
    i_grid = np.argmax(psf,axis=0)
    
    # Get value from wv_map
    cols = np.arange(wv_map.shape[-1])
    grid = wv_map[i_grid, cols]
    
    # Keep only valid column
    i_good = (psf > 0).any(axis=0)
    i_good &= (wv_map > 0).any(axis=0)
    
    # Return sorted and unique
    return np.unique(grid[i_good])

def oversample_grid(lam_grid, n_os=1):
    
    if n_os > 1:
        d_lam = np.diff(lam_grid)
        sub_grid = [lam_grid[:-1] + i * d_lam/n_os for i in range(1,n_os)]
        new_grid = np.concatenate([lam_grid, *sub_grid])
        return np.unique(new_grid)
    else:
        return lam_grid
    
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