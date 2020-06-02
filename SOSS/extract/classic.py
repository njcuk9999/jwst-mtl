import numpy as np

from utils import _get_lam_p_or_m

class OptimalExtract():
    
    def __init__(self, scidata, T_ord, P_ord, lam_ord, lam_grid=None,
                 lam_bounds=None, sig=None, mask=None, thresh=1e-5):
        
        # Use `lam_grid` at the center of the trace if not specified
        if lam_grid is None:
            # Where the trace is defined
            index = np.where((P_ord > 0).any(axis=0))[0]
            # Take the maximum value of the spatial profile
            i_max = np.argmax(P_ord[:,index],axis=0)
            # Assign the wavelength
            lam_grid = lam_ord[i_max, index]
            
        # Save wavelength grid
        self.lam_grid = lam_grid.copy()
        
        # Compute delta lambda for the grid
        self.d_lam = -np.diff(_get_lam_p_or_m(lam_grid), axis=0)[0]
        
        # Basic parameters to save
        self.N_k = len(lam_grid)
        self.thresh = thresh
        
        if sig is None:
            self.sig = np.ones_like(scidata)
        else:
            self.sig = sig.copy()
        
        # Save PSF
        self.P_ord = P_ord.copy()
        
        # Save pixel wavelength
        self.lam_ord = lam_ord.copy()
        
        # Throughput
        # Can be a callable (function) or an array
        # with the same length as lambda grid.
        try:  # First assume it's a function
            self.T_ord = T_ord(self.lam_grid)  # Project on grid
        except TypeError:  # Assume it's an array
            self.T_ord = T_ord.copy() 
        
        # Build global mask
        self.mask = self._get_mask(mask)

        # Assign other trivial attributes
        self.data = scidata.copy()
        # TODO: try setting to np.nan instead?
        self.data[self.mask] = 0 
        
    def _get_mask(self, mask):
            
        # Get needed attributes 
        thresh = self.thresh
        P = self.P_ord
        lam = self.lam_ord
        grid = self.lam_grid
        
        # Mask according to the global troughput (spectral and spatial)
        mask_P = (P < thresh)
        
        # Mask pixels not covered by the wavelength grid
        lam_min, lam_max = grid.min(), grid.max()
        mask_lam = (lam <= lam_min) | (lam >= lam_max)
        
        # Combine all masks including user's defined mask
        if mask is None:
            mask = np.any([mask_P, mask_lam], axis=0)
        else:
            mask = np.any([mask_P, mask_lam, mask], axis=0)
        
        return mask
    
    def extract(self):
        
        # Get needed attributes
        p, t, sig, data, ma, lam, grid =  \
            self.getattrs('P_ord', 'T_ord', 'sig', 'data',
                          'mask', 'lam_ord', 'lam_grid')
        
        # Define delta lambda for each pixels
        d_lam = -np.diff(_get_lam_p_or_m(lam), axis=0)[0]
            
        # Optimal extraction (weighted sum over columns)
        # ------------------
        # Define masked array (easier to sum with the mask)
        # First, compute normalisation factor at each columns
        norm = np.ma.array(p**2/sig**2, mask=ma).sum(axis=0)
        # Second, define numerator
        num = np.ma.array(p*data/sig**2 / (d_lam), mask=ma)
        # Finally compute flux at each columns
        f = (num / norm).sum(axis=0)
        
        # Return flux (divided by throughtput)
        out = (f[~ma.all(axis=0)] / t).data
        
        # Return sorted acoording to lam_grid
        return out[np.argsort(grid)]
    
    def f_th_to_pixel(self, f_th):
        
        # Get needed attributes
        grid, lam, ma =  \
            self.getattrs('lam_grid', 'lam_ord', 'mask')
        
        # Project f_th on the detector valid pixels
        lam_pix = lam[~ma]
        f_k = f_th(lam_pix)

        # Compute extremities of the bins
        # (Assuming grid is the center)
        lam_p, lam_m = _get_lam_p_or_m(grid)

        # Make sur it's sorted
        lam_p, lam_m = np.sort(lam_p), np.sort(lam_m)

        # Special treatment at the end of the bins
        lam_bin = np.concatenate([lam_m, lam_p[-1:]])

        # Compute bins
        f_out = np.histogram(lam_pix, lam_bin, weights=f_k)[0]
        # Normalise (result is the mean in each bins)
        f_out /= np.histogram(lam_pix, lam_bin)[0]

        return f_out
    
    def getattrs(self, *args):
        
        out = [getattr(self, arg) for arg in args]

        if len(out) > 1:
            return out
        else:
            return out[0]