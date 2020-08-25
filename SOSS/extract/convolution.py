import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.interpolate import RectBivariateSpline
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm #for better display


def get_c_matrix(kernel, grid, bounds=None, i_bounds=None, norm=True,
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
        # Make sure it is absolute index, not relative
        # So no negative index.
        if i_bounds[1] < 0:
            i_bounds[1] = len(grid) + i_bounds[1]
        a, b = i_bounds
    
    # Generate a 2D kernel depending on the input
    if callable(kernel):
        kernel = fct_to_array(kernel, grid, [a,b], **kwargs)
    elif kernel.ndim == 1:
        kernel = to_2d(kernel, grid, [a,b], **kwargs)
        
    # Kernel should now be a 2-D array (N_kernel x N_kc)
    
    # Normalize if specified
    if norm:
        kernel = kernel / kernel.sum(axis=0)

    # Apply cut for kernel at boundaries
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
    
    # Init with the value at kernel's center
    out = f(grid, grid)[a:b]
        
    # Add wings
    if length is None:
        # Generate a 2D array of the grid iteratively until
        # thresh is reached everywhere.
        
        # Init parameters
        length = 1
        h_len = 0  # Half length
        
        # Add value on each sides until thresh is reached
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
                
        # Weights due to integration (from the convolution)
        weights = trpz_weight(grid, length, out.shape, a, b)

        return out * weights
    
    elif (length % 2) == 1:  # length needs to be odd
        # Generate a 2D array of the grid iteratively until
        # specified length is reached.
        
        # Compute number of half-length
        n_h_len = (length - 1) // 2
        
        # Simply iterate to compute needed wings
        for h_len in range(1, n_h_len+1):
            # Compute next left and right ends of the kernel
            left, right = _get_wings(f, grid, h_len, a, b)
            
            # Add new kernel values
            out = np.vstack([left, out, right])
            
        # Weights due to integration (from the convolution)
        weights = trpz_weight(grid, length, out.shape, a, b)
        
        return out * weights
    
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
    # Re-use dummy variable `i_grid`
    i_grid = len(grid_new)
    # Compute kernel at the left end.
    # `i_grid` accounts for smaller length.
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

def trpz_weight(grid, length, shape, a, b):
    """ Compute weights due to trapeze integration """
    
    # Index of each element on the convolution matrix
    # with respect to the non-convolved grid
    # `i_grid` has the shape (N_k_convolved, kernel_length - 1) 
    i_grid = np.indices(shape)[0] - (length // 2)
    i_grid = np.arange(a, b)[None,:] + i_grid[:-1,:]
    
    # Set values out of grid to -1
    i_bad = (i_grid < 0) | (i_grid >= len(grid)-1) 
    i_grid[i_bad] = -1
    
    # Delta lambda
    d_grid = np.diff(grid)
    # Compute weights from trapezoidal integration
    weight = 1/2 * d_grid[i_grid]
    weight[i_bad] = 0
    
    # Fill output
    out = np.zeros(shape)
    out[:-1] += weight
    out[1:] += weight
    
    return out

class WebbKer():
    
    path = "Ref_files/spectral_kernel_matrix/"
    file_frame = "spectral_kernel_matrix_os_{}_width_{}pixels.fits"
    
    def __init__(self, wv_map, n_os=10, n_pix=21, bounds_error=False, fill_value='extrapolate'):
        
        # Mask where wv_map is equal to 0
        wv_map = np.ma.array(wv_map, mask=(wv_map==0))
        # Force wv_map to have the red wavelengths
        # at the end of the detector
        if np.diff(wv_map,axis=-1).mean() < 0:
            wv_map = np.flip(wv_map, axis=-1)
        
        # Number of columns
        n_col = wv_map.shape[-1]
        
        # Create filename
        file = self.file_frame.format(n_os, n_pix)
        
        # Read file
        hdu = fits.open(self.path + file)
        header = hdu[0].header
        ker, wv_ker = hdu[0].data
        
        # Where is the blue and red end of the kernel
        i_blue, i_red = header['BLUINDEX'], header['REDINDEX']
        # Flip `ker` to put the red part of the kernel at the end 
        if i_blue > i_red:
            ker = np.flip(ker, axis=0)
            
        # Create oversampled pixel position array
        pixels = np.arange(-(n_pix//2), n_pix//2 + 1/n_os, 1/n_os)
        
        # `wv_ker` has only the value of the central wavelength
        # of the kernel at each points because it's a function
        # of the pixels (so depends on wv solution). 
        wv_center = wv_ker[0,:]
        
        # Let's use the wavelength solution to create a mapping
        # First find which kernels that falls on the detector
        wv_min = np.min(wv_map[wv_map > 0])
        wv_max = np.max(wv_map[wv_map > 0])
        i_min = np.searchsorted(wv_center, wv_min)
        i_max = np.searchsorted(wv_center, wv_max) - 1
        
        # FOR LATER ###########
        # Use the next kernels at each extremities to define the
        # boundaries of the interpolation to use in the class
        # RectBivariateSpline (at the end)
        # bbox = [min pixel, max pixel, min wv_center, max wv_center]
        bbox = [None, None,
                wv_center[np.max([i_min-1, 0])],
                wv_center[np.min([i_max+1, len(wv_center)-1])]]
        #######################
        
        # Keep only kernels that falls on the detector
        ker, wv_ker = ker[:,i_min:i_max+1], wv_ker[:,i_min:i_max+1]
        wv_center = np.array(wv_ker[0,:])
        
        # Then find the pixel closest to each kernel center
        # and use the surrounding pixels (columns)
        # to get the wavelength. At the boundaries,
        # wavelenght might not be defined or falls out of
        # the detector, so fit a 1-order polynomial to 
        # extrapolate. The polynomial is also used to interpolate
        # for oversampling.
        i_surround = np.arange(-(n_pix//2), n_pix//2 + 1)
        poly = []
        for i, wv_c in enumerate(wv_center):
            wv = np.ma.masked_all(i_surround.shape)
            # Closest pixel wv
            i_row, i_col = np.unravel_index(
                np.argmin(np.abs(wv_map-wv_c)), wv_map.shape
            )
            # Surrounding columns
            index = i_col + i_surround
            # Make sure it's on the detector
            i_good = (index >= 0) & (index < n_col)
            # Assign wv values
            wv[i_good] = wv_map[i_row, index[i_good]]
            # Fit n=1 polynomial
            poly_i = np.polyfit(i_surround[~wv.mask], wv[~wv.mask],1)
            # Project on os pixel grid
            wv_ker[:,i] = np.poly1d(poly_i)(pixels)
            # Save coeffs
            poly.append(poly_i)
            
        self.n_pix = n_pix
        self.n_os = n_os
        self.wv_ker = wv_ker
        self.ker = ker
        self.pixels = pixels
        self.wv_center = wv_center
        self.poly = np.array(poly)
        self.fill_value = fill_value
        self.bounds_error = bounds_error
        self.f_ker = RectBivariateSpline(pixels, wv_center, ker, bbox=bbox)
        
    def __call__(self, wv, wv_c):
        
        wv_center = self.wv_center
        poly = self.poly
        fill_value = self.fill_value
        bounds_error = self.bounds_error
        n_wv_c = len(wv_center)
        f_ker = self.f_ker
        n_pix = self.n_pix
        
        # #################################
        # First, convert wv value in pixels
        # #################################
        
        # Find corresponding interval
        i_wv_c = np.searchsorted(wv_center, wv_c) - 1
        
        # Deal with values out of bounds
        if bounds_error:
            message = 'Value of wv center out of interpolation range'
            raise ValueError(message)
        elif fill_value=='extrapolate':
            i_wv_c[i_wv_c < 0] = 0
            i_wv_c[i_wv_c >= (n_wv_c - 1)] = n_wv_c - 2
        else:
            return NotImplemented
        
        # Compute coefficients that interpolate along wv_centers
        d_wv_c = wv_center[i_wv_c + 1] - wv_center[i_wv_c]
        a_c = (wv_center[i_wv_c + 1] - wv_c) / d_wv_c
        b_c = (wv_c - wv_center[i_wv_c]) / d_wv_c
        
        # Compute a and b from the equation:
        # pix = a * lambda + b
        a = 1 / (a_c * poly[i_wv_c,0] + b_c * poly[i_wv_c+1,0])
        b = -(a_c * poly[i_wv_c,1] + b_c * poly[i_wv_c+1,1])
        b /= (a_c * poly[i_wv_c,0] + b_c * poly[i_wv_c+1,0])
        
        # Compute pixel values
        pix = a * wv + b
        
        # ######################################
        # Second, compute kernel value on the
        # interpolation grid (pixel x wv_center)
        # ######################################
        
        out = f_ker(pix, wv_c, grid=False)
        # Make sure it's not negative
        out[out < 0] = 0
        # and put values out of pixel range
        # to zero
        out[pix > n_pix//2] = 0
        out[pix < -(n_pix//2)] = 0

        return out
    
    def show(self):
        
        # 2D figure of the kernels
        plt.figure(figsize=(4,4))
        # Log plot, so clip values <= 0
        image = np.clip(self.ker, np.min(self.ker[self.ker > 0]), np.inf)
        # plot
        plt.pcolormesh(self.wv_center, self.pixels,  image, norm=LogNorm())
        # Labels and others
        plt.colorbar(label="Kernel")
        plt.ylabel("Position relative to center [pixel]")
        plt.xlabel("Center wavelength [$\mu m$]")
        plt.tight_layout()

        # 1D figure of all kernels
        plt.figure()
        plt.plot(self.wv_ker, self.ker)
        # Labels and others
        plt.ylabel("Kernel")
        plt.xlabel("Wavelength [$\mu m$]")
        plt.tight_layout()
