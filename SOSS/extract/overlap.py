import numpy as np

# Local imports
from .custom_numpy import is_sorted, first_change, arange_2d


class _BaseOverlap():
    '''
    Base class for overlaping extraction of the form:
    A f = b
    where A is a matrix and b is an array.
    We want to solve for the array f.
    The elements of f are labelled by 'k'.
    The pixels are labeled by 'i'.
    Every pixel 'i' is covered by a set of 'k' for each order
    of diffraction.
    The classes inheriting from this class should specify the 
    methods get_w which computes the 'k' associated to each pixel 'i'.
    These depends of the type of interpolation used.
    
    Parameters
    ----------
    scidata : (N, M) array_like
        A 2-D array of real values representing the detector image.
    T_list : (N_ord) list or array of functions
        A list or array of the throughput at each order.
        The functions depend on the wavelength
    P_list : (N_ord, N, M) list or array of 2-D arrays
        A list or array of the spatial profile for each order
        on the detector. It has to have the same (N, M) as `scidata`.
    lam_list : (N_ord, N, M) list or array of 2-D arrays
        A list or array of the central wavelength position for each
        order on the detector. 
        It has to have the same (N, M) as `scidata`.
    sig : (N, M) array_like, optional
        Estimate of the error on each pixel. Default is one everywhere.
    mask : (N, M) array_like boolean, optional
        Boolean Mask of the bad pixels on the detector. 
    lam_grid : (N_k) array_like, optional but recommended:
        The grid on which f(lambda) will be projected.
        Default still has to be improved.
    d_lam : float, optional:
        Step to build a default `lam_grid`, but should be replaced
        for the needed resolution.
    tresh : float, optional:
        The pixels where the estimated transmission is less than
        this value will be masked.
    '''

    def __init__(self, scidata, T_list, P_list, lam_list,
                 lam_grid=None, d_lam=5e-4, sig=None, mask=None, tresh=1e-5):
        
        self.n_ord = len(T_list)
        self.tresh = tresh
        
        if sig is None:
            self.sig = np.ones_like(scidata)
        else:
            self.sig = sig.copy()
        
        # Take the lam range of order 1
        # if lam_grid is not given
        # TODO: choose a more adequate default grid
        if lam_grid is None:
            # Mask where the trace is zero to compute the max and min
            ma = P_list[0] < tresh
            # Find range
            lam_p, lam_m = _get_lam_p_or_m(lam_list[0])
            lam_range = (lam_m[~ma].min() - d_lam/4,
                         lam_p[~ma].max()-d_lam/4)
            # Make grid
            lam_grid = np.arange(*lam_range, d_lam)  # in microns
        else:
            if not is_sorted(lam_grid):
                message = "'lam_grid' must be sorted"  \
                        + " and without duplicates"
                raise ValueError(message)
        
        # TODO: add d_lam as an optional argument
        self.d_lam = np.diff(lam_grid)  # Delta lambda
        self.lam_grid = lam_grid.copy()
        
        # Get wavelength at the boundary of each pixel
        # TODO? Could also be an input??
        lam_p, lam_m = [], []
        for lam in lam_list:
            lp, lm = _get_lam_p_or_m(lam)
            lam_p.append(lp), lam_m.append(lm)
        self.lam_p, self.lam_m = lam_p, lam_m
        
        # Throughput
        # TODO? : Could also be array like instead of a function?
        T = []
        for T_n in T_list:
            T.append(T_n(self.lam_grid))
        self.T_list = T
        
        # Define masks
        mask_P = [P < tresh for P in zip(P_list, )]
        mask_lam = [self._get_mask_lam(n) for n in range(self.n_ord)]
        if mask is None:
            mask_ord = np.any([mask_P, mask_lam], axis=0)
        else:
            mask_ord = np.any([mask_P, mask_lam, [mask, mask]], axis=0)
        self.mask_ord = mask_ord
        self.mask = np.any(mask_ord, axis=0)
        
        
        # Find the lowest (L) and highest (H) index 
        # of lam_grid for each pixels and orders
        L, H = [], []
        for n in range(self.n_ord):
            L_n, H_n = self._get_LH(n)            
            L.append(L_n), H.append(H_n)
        self.L, self.H = L, H
        
        # Computes weights for integration
        w, k = [], []
        for n in range(self.n_ord):
            w_n, k_n = self._get_w(n)
            w.append(w_n), k.append(k_n)
        self.w, self.k = w, k
        
        # Get indexing to build the linear system later
        n_wv = len(self.lam_grid)
        j = np.arange(n_wv)
        j_list = []
        p_list = []
        for k_m in self.k:
            j_m, p_m = [], []
            for ij in range(k_m.shape[-1]):
                j_m_ij, p_m_ij = np.where(k_m[:,ij][None,:] == j[:,None])
                j_m.append(j_m_ij), p_m.append(p_m_ij)
            j_m, p_m = np.array(j_m), np.array(p_m)
            j_list.append(j_m), p_list.append(p_m)
        self.j_list, self.p_list = j_list, p_list

        # Assign other trivial attributes
        self.data = scidata.copy()
        self.data[self.mask] = 0
        
        self.P_list = [P.copy() for P in P_list]
        for ma, P in zip(mask_ord, self.P_list):
            P[ma] = 0
        self.lam_list = [lam.copy() for lam in lam_list]
        
    def _get_mask_lam(self, n):
        
        lam_min = self.lam_grid[0]
        lam_max = self.lam_grid[-1]
        lam_p, lam_m = self.getattrs('lam_p', 'lam_m', n=n)
        
        mask = (lam_m < lam_min) | (lam_p > lam_max)
        
        return mask
    
    def extract(self):
        
        I, sig, mask, grid, d_grid  \
            = self.getattrs('data','sig', 'mask',
                            'lam_grid', 'd_lam')
        T, L, H, P, w, k, j, p  \
            = self.getattrs('T_list', 'L', 'H', 'P_list',
                            'w', 'k', 'j_list', 'p_list')
        n_lam = grid.size
        
        I, sig = I[~mask], sig[~mask]
        
        a = []
        for n in range(self.n_ord):
            a_n = []
            P_n = P[n][~mask]
            a_n = P_n[:,None] * T[n][k[n]] * w[n] / sig[:,None]
            a.append(a_n)
        
        # TODO add convolution here or in the loop above
        self.a = a
        b = a

        # Build system
        M, d = _build_system(I/sig, b, k, j, p, n_lam)
        
        return M, d
    
    def inject(self, f):
        
        grid, d_grid, ma  \
            = self.getattrs('lam_grid', 'd_lam', 'mask')
        T, L, H, P, w, k, lam  \
            = self.getattrs('T_list', 'L', 'H', 'P_list',
                            'w', 'k', 'lam_list')
        
        out = np.zeros(ma.shape)
        for n in range(len(T)):
#             ma = mask_ord[n]
            P_n = P[n][~ma]
            lam_n = lam[n][~ma]
            
            k_n, w_n = k[n], w[n]
            
            inj_n = np.nansum(P_n[:,None] * T[n][k_n]
                              * w_n * f(grid[k_n]),
                              axis=-1)
            out[~ma] += np.array(inj_n)
            
        return out
    
    def getattrs(self, *args, n=None):
        
        if n is None:
            out = [getattr(self, arg) for arg in args]
        else:
            out = [getattr(self, arg)[n] for arg in args]

        if len(out) > 1:
            return out
        else:
            return out[0]
    
class TrpzOverlap(_BaseOverlap):
    ''' Version oversampled with trapezoidal integration '''
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs):
             
    def _get_LH(self, n):
        print('Compute LH')
        # Get needed attributes
        grid, mask = self.lam_grid, self.mask
        # ... order dependent attributes
        lam_p, lam_m, mask_ord  \
            = self.getattrs('lam_p', 'lam_m', 'mask_ord', n=n)
        
        # Compute only valid pixels
        lam_p = lam_p[~mask]
        lam_m = lam_m[~mask]
        
        # Find lower (L) index in the pixel
        #
        L = np.ones(lam_m.shape, dtype=int) * -1
        cond = lam_m[:,None] <= grid[None,:]
        ind, L_good = first_change(cond, axis=-1)
        L[ind] = L_good + 1
        # Special treatment when L==grid[0], so cond is all True
        ind = np.where(cond.all(axis=-1))
        L[ind] = 0

        # Find higher (H) index in the pixel
        #
        H = np.ones(lam_p.shape, dtype=int) * -2
        cond = lam_p[:,None] >= grid[None,:]
        ind, H_good = first_change(cond, axis=-1)
        H[ind] = H_good
        # Special treatment when H==grid[-1], so cond all True
        ind = np.where(cond.all(axis=-1))
        H[ind] = len(grid) - 1

        # Set invalid pixels for this order to L=-1 and H=-2
        ma = mask_ord[~mask]
        L[ma], H[ma] = -1, -2
        
        print('Done')

        return L, H

    def _get_w_old(self, n):
        
        # Get needed attributes
        grid, d_grid, mask  \
            = self.getattrs('lam_grid', 'd_lam','mask')
        # ... order dependent attributes
        lam_p, lam_m, L, H, mask_ord  \
            = self.getattrs('lam_p', 'lam_m',
                            'L', 'H', 'mask_ord', n=n)
        
        # Compute only valid pixels
        lam_p = lam_p[~mask]
        lam_m = lam_m[~mask]
        ma = mask_ord[~mask]
        
        # Compute w for each pixel i
        w, k = [], []
        undersampled = False
        if n==1:
            for L_i, ma_i in zip(L, ma):
                print(~ma_i)
        for L_i, H_i, lam_p_i, lam_m_i, ma_i in zip(L, H, lam_p, lam_m, ma):
#             if n==1: print(not ma_i)
            if ~ma_i:
                # Define k_i (index of the wv_grid associated to this pixel)
                if (grid[L_i]-lam_m_i)/d_grid[L_i] <= 1.0e-5:
                    k_first = L_i
                    lam_m_i = grid[L_i]
                else:
                    k_first = L_i - 1
                if (lam_p_i-grid[H_i])/d_grid[H_i-1] <= 1.0e-5:
                    k_last = H_i
                    lam_p_i = grid[H_i]
                else:
                    k_last = H_i + 1
                k_i = np.arange(k_first, k_last + 1).astype(int)
                
                # Compute w_i
                w_i = np.zeros(len(k_i))
                
                # Always true
                w_i[0] = grid[k_i[1]] - lam_m_i
                w_i[-1] = lam_p_i - grid[k_i[-2]]
                
                # Case where pixel limits fall between 2 grid points
                # so UNDERSAMPLED
                if len(k_i) == 2: 
#                     print(k_i)
                    undersampled = True
                    if k_i[0] != L_i:
                        w_i[1] += lam_m_i - grid[k_i[0]]
                    if k_i[-1] != H_i:
                        w_i[-2] += grid[k_i[-1]] - lam_p_i
                    w_i *= (lam_p_i - lam_m_i) / d_grid[k_i[0]]
                    
                elif len(k_i) >= 3:
#                     print(k_i)
                    w_i[1] = grid[k_i[1]] - lam_m_i
                    w_i[-2] += lam_p_i - grid[k_i[-2]]
                    if k_i[0] != L_i:
                        w_i[0] *= (grid[k_i[1]] - lam_m_i) / d_grid[k_i[0]]
                        w_i[1] += (grid[k_i[1]]-lam_m_i) * (lam_m_i-grid[k_i[0]])  \
                                  / d_grid[k_i[0]]
                        
                    if k_i[-1] != H_i:
                        w_i[-1] *= (lam_p_i - grid[k_i[-2]]) / d_grid[k_i[-2]]
                        w_i[-2] += (grid[k_i[-1]]-lam_p_i) * (lam_p_i-grid[k_i[-2]])  \
                                   / d_grid[k_i[-2]]
                    
                if len(k_i) >= 4:
#                     if k_i[0] != L_i:
                    w_i[1] += grid[k_i[2]] - grid[k_i[1]]
#                     if k_i[-1] != H_i:
                    w_i[-2] += grid[k_i[-2]] - grid[k_i[-3]]
                
                if len(k_i) > 4:
#                     print(w_i[2:-2])
#                     print(d_grid[k_i[1]:k_i[-3]+1])
#                     print(d_grid[k_i[2]:k_i[-2]+1])
#                     w_i[2:-2] = (d_grid[L_i+1:H_i] + d_grid[L_i:H_i-1])
                    w_i[2:-2] = (d_grid[k_i[1]:k_i[-3]] + d_grid[k_i[2]:k_i[-2]])

                w.append(w_i/2), k.append(k_i)
            else:
                w.append([]), k.append([])
                
        if undersampled:
            warn('Undersampled')
            
        # Fill w and k so they can be used as arrays
        # w filled with np.nan and k filled with -1
        w, k = fill_list(w), fill_list(k, fill_value=-1, dtype=int)

        return w, k
    
    def _get_w(self, n):
        
        print('Compute weigths and k')
        
        # Get needed attributes
        grid, d_grid, mask  \
            = self.getattrs('lam_grid', 'd_lam','mask')
        # ... order dependent attributes
        lam_p, lam_m, L, H, mask_ord  \
            = self.getattrs('lam_p', 'lam_m',
                            'L', 'H', 'mask_ord', n=n)
        
        # Compute only valid pixels
        lam_p, lam_m = lam_p[~mask], lam_m[~mask]
        ma = mask_ord[~mask]

        # Number of used pixels
        N_i = len(L)
        i = np.arange(N_i)
        
        print('Compute k')
        
        # Define fisrt and last index of lam_grid
        # for each pixel
        k_first, k_last = -1*np.ones(N_i), -1*np.ones(N_i)
        # If lowest value close enough to the exact grid value,
        cond = (grid[L]-lam_m)/d_grid[L] <= 1.0e-8
        # special case (no need for L_i - 1)
        k_first[cond & ~ma] = L[cond & ~ma]
        lam_m[cond & ~ma] = grid[L[cond & ~ma]]
        # else, need L_i - 1
        k_first[~cond & ~ma] = L[~cond & ~ma] - 1
        # Same situation for highest value,
        cond = (lam_p-grid[H])/d_grid[H-1] <= 1.0e-8
        # special case (no need for H_i - 1)
        k_last[cond & ~ma] = H[cond & ~ma]
        lam_p[cond & ~ma] = grid[H[cond & ~ma]]
        # else, need H_i + 1
        k_last[~cond & ~ma] = H[~cond & ~ma] + 1
        
        # Generate array of all k_i. Set to -1 if not valid
        k, bad = arange_2d(k_first, k_last+1, dtype=int, return_mask=True)
        k[bad] = -1
        # Number of valid k per pixel
        N_k = np.sum(~bad, axis=-1)
        
        # Compute array of all w_i. Set to np.nan if not valid
        # Initialize
        w = np.zeros(k.shape, dtype=float)
        ####################
        ####################
        # 4 different cases
        ####################
        ####################
        
        print('compute w')

        # Valid for every cases
        w[:,0] = grid[k[:,1]] - lam_m
        w[i,N_k-1] = lam_p - grid[k[i,N_k-2]]
        
        ##################
        # Case 1, N_k == 2
        ##################
        case = (N_k == 2) & ~ma
        if case.any():
            print('N_k = 2')
            # if k_i[0] != L_i
            cond = case & (k[:,0] != L)
            w[cond,1] += lam_m[cond] - grid[k[cond,0]]
            # if k_i[-1] != H_i
            cond = case & (k[:,1] != H)
            w[cond,0] += grid[k[cond,1]] - lam_p[cond]
            # Finally
            w[case,:] *= ((lam_p[case] - lam_m[case]) / d_grid[k[case,0]])[:,None]

        ##################
        # Case 2, N_k >= 3
        ##################
        case = (N_k >= 3) & ~ma
        if case.any():
            print('N_k = 3')
            N_ki = N_k[case]
            w[case,1] = grid[k[case,1]] - lam_m[case]
            w[case,N_ki-2] += lam_p[case] - grid[k[case,N_ki-2]]
            # if k_i[0] != L_i
            cond = case & (k[:,0] != L)
            w[cond,0] *= (grid[k[cond,1]] - lam_m[cond]) / d_grid[k[cond,0]]
            w[cond,1] += (grid[k[cond,1]]-lam_m[cond]) * (lam_m[cond]-grid[k[cond,0]])  \
                                      / d_grid[k[cond,0]]
            # if k_i[-1] != H_i
            cond = case & (k[i,N_k-1] != H)
            N_ki = N_k[cond]
            w[cond,N_ki-1] *= (lam_p[cond] - grid[k[cond,N_ki-2]]) / d_grid[k[cond,N_ki-2]]
            w[cond,N_ki-2] += (grid[k[cond,N_ki-1]]-lam_p[cond])  \
                              * (lam_p[cond]-grid[k[cond,N_ki-2]])  \
                              / d_grid[k[cond,N_ki-2]]

        ##################
        # Case 3, N_k >= 4
        ##################
        case = (N_k >= 4) & ~ma
        if case.any():
            print('N_k = 4')
            N_ki = N_k[case]
            w[case,1] += grid[k[case,2]] - grid[k[case,1]]
            w[case,N_ki-2] += grid[k[case,N_ki-2]] - grid[k[case,N_ki-3]]            

        ##################
        # Case 4, N_k > 4
        ##################
        case = (N_k > 4) & ~ma
        if case.any():
            print('N_k > 4')
            i_k = np.indices(k.shape)[-1]
            cond = case[:,None] & (2 <= i_k) & (i_k < N_k[:,None]-2)
            ind1, ind2 = np.where(cond)
            w[ind1,ind2] = d_grid[k[ind1,ind2]-1] + d_grid[k[ind1,ind2]]

        
        # Finally, divide w by 2
        w /= 2.
        
        # Make sure invalid values are masked
        w[k<0] = np.nan

        print('Done')
        return w, k
        

def _get_lam_p_or_m(lam):
    '''
    Compute lambda_plus and lambda_minus
    '''
    
    lam_r = np.zeros_like(lam)
    lam_l = np.zeros_like(lam)
    
    # Def delta lambda
    d_lam = np.diff(lam, axis=1)
    
    # Define lambda left and lambda right of each pixels
    lam_r[:,:-1] = lam[:,:-1] + d_lam/2
    lam_r[:,-1] = lam[:,-1] + d_lam[:,-1]/2
    lam_l[:,1:] = lam[:,:-1] + d_lam/2  # Same values as lam_r
    lam_l[:,0] = lam[:,0] - d_lam[:,0]/2
    
    if (lam_r >= lam_l).all():
        return lam_r, lam_l
    elif (lam_r <= lam_l).all():
        return lam_l, lam_r
    else:
        raise ValueError('Bad pixel values for wavelength')



def _build_system_slow(I, b, k, n_lam):

    # Initialize
    d = np.zeros(n_lam)
    M = np.zeros((n_lam,n_lam))

    # Iterate over matrix lines (lam)
    for j in range(n_lam):
#         print(j)
        for k_n, b_n in zip(k, b):
            for i, (k_in, b_in) in enumerate(zip(k_n, b_n)):
#                 p_in = (k_in == j) & (k_in >= 0)
#                 print(k_in)
                p_in = (k_in == j)
                if p_in.any():
#                     print(I[p_in] * b_in[i])
                    d[j] += (I[i] * b_in[p_in])
                
                    for k_m, b_m in zip(k, b):
                        k_im, b_im = k_m[i], b_m[i]
                        p_im = k_im >= 0
                        np.add.at(M, (j, k_im[p_im]), b_in[p_in] * b_im[p_im])

    return M, d


def _build_system(I, b, k, j, p, n_lam):
    
    # Initialize
    d = np.zeros(n_lam)
    M = np.zeros((n_lam,n_lam))
    
    # Transpose k, j and p
    k = [k_n.T for k_n in k]
    b = [b_n.T for b_n in b]

    n_ord = len(k)
    for n in range(n_ord):

        n_ij = len(j[n])
        for ij in range(n_ij):
            np.add.at(d, j[n][ij], (I * b[n][ij])[p[n][ij]])
        
        for m in range(n_ord):
            m_ij = len(j[m])
            for ij1 in range(n_ij):
                for ij2 in range(m_ij):
                    
                    k_ij = k[m][ij2][p[n][ij1]]
                    good = k_ij >= 0 
                    if good.any():
                        np.add.at(M, (j[n][ij1][good], k_ij[good]),
                                  (b[m][ij2][p[n][ij1]] * b[n][ij1][p[n][ij1]])[good])
                    

    return M, d
