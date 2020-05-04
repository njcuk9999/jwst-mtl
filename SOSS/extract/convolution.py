import numpy as np
from scipy.sparse import diags, csr_matrix


def get_c_matrix(kernel, grid, bounds=None, i_bounds=None,
                 sparse=True, n_out=None, thresh_out=None, **kwargs):
    """
    Return a convolution matrix
    Can return a sparse matrix (N_k_convolved, N_k)
    or a matrix (N_ker, N_k_convolved).
    N_k is the length of the grid on which the convolution
    will be applied, N_k_convolved is the length of the 
    grid after convolution and N_ker is the maximum length of
    the kernel. If the default sparse matrix option is chosen,
    the convolution can be apply on an array f | f = fct(grid)
    by a simple matrix multiplication: 
    f_convolved = c_matrix.dot(f)
    
    Parameters
    ----------
    kernel: ndarray (1D or 2D), callable
        Convolution kernel. Can be already 2D (N_ker, N_k_convolved),
        giving the kernel for each items of the convolved grid.
        Can be 1D (N_ker), so the kernel is the same. Can be a callable
        with the form f(x, x0) where x0 is the position of the center of 
        the kernel. Must return a 2D array (len(x), len(x0)), so a kernel
        for each element of x0. If kernel is callable, the additional
        kwargs `thresh` and `length` will be used to project the kernel.
    grid: one-d-array:
        The grid on which the convolution will be applied.
        For example, if C is the convolution matrix,
        f_convolved = C.f(grid)
    bounds: 2-elements object
        The bounds of the grid on which the convolution is defined.
        For example, if bounds = (a,b), 
        then grid_convolved = grid[a <= grid <= b].
        It dictates also the dimension of f_convolved
    sparse: bool, optional
        return a sparse matrix (N_k_convolved, N_k) if True.
        return a matrix ((N_ker, N_k_convolved) if False.
    n_out: integer or 2-integer object, optional
        Specify how to deal with the ends of the convolved grid.
        `n_out` points will be used outside from the convolved
        grid. Can be different for each ends if 2-elements are given.
    thresh_out: float, optional
        Specify how to deal with the ends of the convolved grid.
        Points with a kernel value less then `thresh_out` will
        not be used outside from the convolved grid.
    thresh: float, optional
        Only used when `kernel` is callable to define the maximum
        length of the kernel. Truncate when `kernel` < `thresh`
    length: int, optional
        Only used when `kernel` is callable to define the maximum
        length of the kernel.
    """
    
    # Define range where the convolution is defined on the grid.
    # If `i_bounds` is not specified, try with `bounds`.
    if i_bounds is None:
        if bounds is None:
            a, b = 0, len(grid)
        else:
            a = np.min(np.where(grid >= bounds[0])[0])
            b = np.max(np.where(grid <= bounds[1])[0]) + 1
    else:
        a, b = i_bounds
    
    # Generate a 2D kernel if it depending on the input
    if callable(kernel):
        kernel = fct_to_array(kernel, grid, [a,b], **kwargs)
    elif kernel.ndim == 1:
        kernel = to_2d(kernel, grid, [a,b], **kwargs)
        
    # Kernel should now be a 2-D array (N_kernel x N_kc)
    # Apply n_out
    kernel = cut_ker(kernel, n_out, thresh_out)

    # Return sparse or not
    if sparse:
        # Convert to sparse matrix 
        return sparse_c(kernel, len(grid), a)
    else:
        return kernel

def cut_ker(ker, n_out=None, thresh=None):
    
    # Assign kernel length and number of kernels
    n_ker, n_k_c = ker.shape
    
    # Assign half-length of the kernel
    h_len = (n_ker - 1) // 2
    
    # Determine n_out with thresh if not given
    if n_out is None:
        if thresh is None:
            # No cut to apply
            return ker
        else:
            # Find where to cut the kernel according to thresh
            i_left = np.where(ker[:,0] >= thresh)[0][0]
            i_right = np.where(ker[:,-1] >= thresh)[0][-1]
            # Make sure it is on the good wing. Take center if not.
            i_left = np.min([i_left, h_len])
            i_right = np.max([i_right, h_len])
            
    # Else, unpack n_out
    else:
        # Could be a scalar or a 2-elements object)
        try:
            i_left, i_right = n_out
        except TypeError:
            i_left, i_right = n_out, n_out
        # Find the position where to cut the kernel 
        # Make sure it is not out of the kernel grid,
        # so i_left >= 0 and i_right <= len(kernel)
        i_left = np.max([h_len - i_left, 0])
        i_right = np.min([h_len + i_right, n_ker - 1])
        
    # Apply the cut
    for i in range(0,i_left):
        # Add condition in case the kernel is larger
        # than the grid where it's projected.
        if i < n_k_c:
            ker[:i_left-i,i] = 0
    for i in range(i_right + 1 - n_ker,0):
        # Add condition in case the kernel is larger
        # than the grid where it's projected.
        if -i <= n_k_c:
            ker[i_right-n_ker-i:, i] = 0
        
    return ker
        
    
def sparse_c(ker, n_k, i_zero=0):
    '''
    Define the sparse convolution matrix

    Parameters:

    c : ndarray, (N_kernel, N_kc)
    '''
    n_ker, n_k_c = ker.shape

    if n_ker % 2 != 1:
        raise ValueError("length of the convolution kernel should be odd.")
    
    # Assign half-length
    h_len = (n_ker - 1) // 2

    diag_val, offset = [], []
    for i_ker, i in enumerate(range(-h_len, h_len+1)):
        i_k = i_zero + i
        if i_k < 0:
            diag_val.append(ker[i_ker,-i_k:])
        else:
            diag_val.append(ker[i_ker,:])
        offset.append(i_k)

    return diags(diag_val, offset, shape=(n_k_c,n_k), format='csr')
    
def to_2d(kernel, grid, grid_range):
    """Build a 2d kernel array with a constant 1D kernel (input)"""
        
    # Assign range where the convolution is defined on the grid
    a, b = grid_range
        
    # Get length of the convolved axis
    n_k_c = b - a
    
    # Return a 2D array with this length
    return np.tile(kernel, (n_k_c,1)).T
    
        
def fct_to_array(f, grid, grid_range, thresh=1e-8, length=None):
        
    # Assign range where the convolution is defined on the grid
    a, b = grid_range
    
    # Init with a convolution of single length
    out = f(grid, grid)[a:b]
        
    # Add wings
    if length is None:
        # Generate a 2D array of the grid iteratively until
        # thresh is reached everywhere.
        
        # Init parameters
        length = 1
        h_len = 0  # Half length
        
        # Add value on each sides until tresh is reached
        while True:            
            # Already update half-length
            h_len += 1
            
            # Compute next left and right ends of the kernel
            left, right = _get_wings(f, grid, h_len, a, b)
            
            # Check if they are all below threshold.
            if (left<thresh).all() and (right<thresh).all():
                break  # Stop iteration
            else:
                # Update kernel length and add new values
                length += 2
                out = np.vstack([left, out, right])

        return out
    
    elif (length % 2) == 1:  # length needs to be odd
        # Generate a 2D array of the grid iteratively until
        # thresh is reached everywhere.
        
        # Compute number of half-length
        n_h_len = (length - 1) // 2
        
        # Simply iterate to compute needed wings
        for h_len in range(1, n_h_len+1):
            # Compute next left and right ends of the kernel
            left, right = _get_wings(f, grid, h_len, a, b)
            
            # Add new kernel values
            out = np.vstack([left, out, right])

        return out
    
    else:
        raise ValueError("`length` must be odd.")

def _get_wings(f, grid, h_len, a, b):
    """ Compute values of the kernel at grid[+-h_len]"""
    
    # Save length of the non-convolved grid
    n_k = len(grid)
    
    # Get length of the convolved axis
    n_k_c = b - a
    
    # Init values
    left, right = np.zeros(n_k_c), np.zeros(n_k_c)
    
    # Add the left value on the grid
    # Possibility that it falls out of the grid;
    # take first value of the grid if so.
    i_grid = np.max([0,a-h_len])
    # Save the new grid
    grid_new = grid[i_grid:b-h_len]
    # Re-assign `i_grid`
    i_grid = len(grid_new)
    # Compute kernel at the left end
    ker = f(grid_new, grid[b-i_grid:b])
    left[-i_grid:] = ker

    # Add the right value on the grid
    # Possibility that it falls out of the grid;
    # take last value of the grid if so.
    # Same steps as the left end (see above)
    i_grid = np.min([n_k, b + h_len])
    grid_new = grid[a+h_len:i_grid]
    i_grid = len(grid_new)
    ker = f(grid_new, grid[a:a+i_grid])
    right[:i_grid] = ker
    
    return left, right

