import numpy as np
from custom_numpy import is_sorted


class SegmentedLagrangeX():

    def __init__(self, grid, order, extrapolate=False,
                 assume_sorted=False, mode='left', adjust_order=True):
        
        # Specification
        self.extrapolate = extrapolate
        self.sorted = assume_sorted
        self.adjust = adjust_order
        
        # Length of polynomial
        n = order + 1
        self.n = n
        
        # Save grid
        if not is_sorted(grid):
            raise ValueError("`grid` must be sorted and unique.")
        self.grid = grid
        
        # Pad for special treatment of the grid ends
        # Case for mode='left'
        pad = [np.floor((n-2) / 2), np.ceil((n-2) / 2)]
        pad = np.array(pad, dtype=int)
        if mode=='right':  # Flip it if mode = 'right'
            pad = np.flip(pad)
        elif mode!='left':  # Check if valid input
            raise ValueError("`mode` must equal 'left' or 'right'")
        
        # Get index to define each segments
        self.index = self._get_segments_index(len(grid), pad)
        
        # x
        self.x_seg = self._get_segments(grid)
        
    def _get_segments(self, grid, fill_value=np.nan, bad_index=-1):
        
        # Get needed attributes
        index = self.index
        
        # Get values for each segments
        seg = grid[index]
        
        # Change invalid index
        seg[index==bad_index] = fill_value

        # Convert to array and return
        return np.array(seg)
    
    def get_index(self, x):
        
        # Needed attributes
        grid = self.grid
        extrapolate = self.extrapolate
        
        # Where do each points fall on the grid
        index = np.searchsorted(grid, x) - 1
        
        # Special treatment needed for if x_i == grid[0].
        # It will be considered out of bounds otherwise
        # due to how np.searchsorted works (side='left').
        index[x==grid[0]] = 0
        
        # Identify values out of bounds
        out_left = (index == -1)
        out_right = (index == len(grid)-1)
        
        # If values out of bounds and extrapolate is False,
        # raise and error
        if (out_left | out_right).any() and not extrapolate:
            raise ValueError('Values out of interpolation range.')
            
        # Extrapolate with left or rigth segment
        index[out_left] = 0
        index[out_right] = index[out_right] - 1
        
        return index
    
    def get_coeffs(self, x):
        
        # Needed attributes
        x_seg = self.x_seg
        n = self.n
        
        # Check if sorted if assume_sorted is False
        if not self.sorted:
            # If not sorted, sort it!
            if not is_sorted(x):
                x = np.sort(x)
        
        # Which segment of the grid should be used
        # for each value of x
        index = self.get_index(x)
        
        # Assign value
        n_x = len(x)
        coeffs = np.zeros((n, n_x))
        for j in range(n):
            c_j = np.ones(n_x)
            for i in range(n):
                if i!=j:
                    c_ij = (x - x_seg[i,index])
                    c_ij /= (x_seg[j,index] - x_seg[i,index])
                    igood = np.isfinite(c_ij)
                    c_j[igood] *= c_ij[igood]
            coeffs[j] = c_j 
            
        return coeffs
    
    def _get_segments_index(self, ng, pad, fill_value=-1):
        
        # Get needed attributes
        n = self.n
        adjust = self.adjust
        
        # Index of the grid
        i_grid = np.arange(ng)
        
        # Assign values for each segments polynomials
        seg = [i_grid[i:ng+i-n+1] for i in range(n)]
        seg = np.array(seg)
        
        # Special treatment for borders:
        # Use the values from the first segment to build
        # the polynomial for the missing left segments
        for i in range(pad[0]):
            seg = np.hstack([seg[:,0][:,None], seg])
            if adjust: seg[-2*(1+i):,0] = fill_value
            
        # Use the values from the last segment to build
        # the polynomial for the missing right segments
        for i in range(pad[1]):
            seg = np.hstack([seg, seg[:,-1][:,None]])
            if adjust: seg[:2*(i+1),-1] = fill_value

        # Convert to array and return
        return np.array(seg, dtype=int)
        
        
        

class SegmentedLagrange(SegmentedLagrangeX):
    
    def __init__(self, grid, f_grid, order, **kwargs):
        
        # Init the x related part
        super().__init__(grid, order, **kwargs)
        
        # y
        self.y_seg = self._get_segments(f_grid, fill_value=0.)

        
    def __call__(self, x):
        
        # Needed attributes
        x_seg = self.x_seg
        y_seg = self.y_seg
        n = self.n
        
        # Check if sorted if assume_sorted is False
        if not self.sorted:
            # If not sorted, sort it!
            if not is_sorted(x):
                x = np.sort(x)
        
        # Which segment of the grid should be used
        # for each value of x
        index = self._get_index(x)
        
        # Assign value
        y = np.zeros_like(x)
        for j in range(n):
            c_j = np.ones_like(x)
            for i in range(n):
                if i!=j:
                    c_ij = (x - x_seg[i,index])
                    c_ij /= (x_seg[j,index] - x_seg[i,index])
                    igood = np.isfinite(c_ij)
                    c_j[igood] *= c_ij[igood]
            y += y_seg[j,index] * c_j
        
        return y
