import numpy as np

from utils import get_lam_p_or_m, grid_from_map


class OptimalExtract:

    def __init__(self, scidata, t_ord, p_ord, lam_ord, lam_grid=None,
                 lam_bounds=None, sig=None, mask=None, thresh=1e-5):

        # Use `lam_grid` at the center of the trace if not specified
        if lam_grid is None:
            lam_grid, lam_col = grid_from_map(lam_ord, p_ord, out_col=True)

        # Save wavelength grid
        self.lam_grid = lam_grid.copy()
        self.lam_col = lam_col

        # Compute delta lambda for the grid
        self.d_lam = -np.diff(get_lam_p_or_m(lam_grid), axis=0)[0]

        # Basic parameters to save
        self.N_k = len(lam_grid)
        self.thresh = thresh

        if sig is None:
            self.sig = np.ones_like(scidata)
        else:
            self.sig = sig.copy()

        # Save PSF
        self.p_ord = p_ord.copy()

        # Save pixel wavelength
        self.lam_ord = lam_ord.copy()

        # Throughput
        # Can be a callable (function) or an array
        # with the same length as lambda grid.
        try:  # First assume it's a function
            self.t_ord = t_ord(self.lam_grid)  # Project on grid
        except TypeError:  # Assume it's an array
            self.t_ord = t_ord.copy()

        # Build global mask
        self.mask = self._get_mask(mask)

        # Assign other trivial attributes
        self.data = scidata.copy()
        # TODO: try setting to np.nan instead?
        self.data[self.mask] = 0

    def _get_mask(self, mask):

        # Get needed attributes
        thresh = self.thresh
        p_ord = self.p_ord
        lam = self.lam_ord
        grid = self.lam_grid

        # Mask according to the global troughput (spectral and spatial)
        mask_p = (p_ord < thresh)

        # Mask pixels not covered by the wavelength grid
        lam_min, lam_max = grid.min(), grid.max()
        mask_lam = (lam <= lam_min) | (lam >= lam_max)

        # Combine all masks including user's defined mask
        if mask is None:
            mask = np.any([mask_p, mask_lam], axis=0)
        else:
            mask = np.any([mask_p, mask_lam, mask], axis=0)

        return mask

    def extract(self):

        # Get needed attributes
        psf, sig, data, ma, lam, grid, lam_col =  \
            self.getattrs('p_ord', 'sig', 'data',
                          'mask', 'lam_ord', 'lam_grid', 'lam_col')

        # Define delta lambda for each pixels
        d_lam = -np.diff(get_lam_p_or_m(lam), axis=0)[0]

        # Optimal extraction (weighted sum over columns)
        # ------------------
        # Define masked array (easier to sum with the mask)
        # First, compute normalisation factor at each columns
        norm = np.ma.array(psf**2/sig**2, mask=ma).sum(axis=0)
        # Second, define numerator
        num = np.ma.array(psf*data/sig**2 / (d_lam), mask=ma)
        # Finally compute flux at each columns
        out = (num / norm).sum(axis=0)

        # Return flux where lam_grid is defined
        out = out[lam_col]
        i_good = ~(ma[:, lam_col]).all(axis=0)
        out = out[i_good].data

        # Return sorted according to lam_grid
        i_sort = np.argsort(grid[i_good])
        return grid[i_sort], out[i_sort]

    def f_th_to_pixel(self, f_th):

        # Get needed attributes
        grid, lam, ma =  \
            self.getattrs('lam_grid', 'lam_ord', 'mask')

        # Project f_th on the detector valid pixels
        lam_pix = lam[~ma]
        f_k = f_th(lam_pix)

        # Compute extremities of the bins
        # (Assuming grid is the center)
        lam_p, lam_m = get_lam_p_or_m(grid)

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
