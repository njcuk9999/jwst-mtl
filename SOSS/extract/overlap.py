
# General imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import find, issparse, csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

# Local imports
from custom_numpy import arange_2d
from interpolate import SegmentedLagrangeX
from convolution import get_c_matrix, WebbKer
from utils import _get_lam_p_or_m, get_n_nodes, oversample_grid
from throughput import ThroughputSOSS
from regularisation import Tikhonov, tikho_solve


class _BaseOverlap():
    """
    Base class for overlaping extraction of the form:
    (B_T * B) * f = (data/sig)_T * B
    where B is a matrix and f is an array.
    The matrix multiplication B * f is the 2d model of the detector.
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
    p_list : (N_ord, N, M) list or array of 2-D arrays
        A list or array of the spatial profile for each order
        on the detector. It has to have the same (N, M) as `scidata`.
    lam_list : (N_ord, N, M) list or array of 2-D arrays
        A list or array of the central wavelength position for each
        order on the detector.
        It has to have the same (N, M) as `scidata`.
    scidata : (N, M) array_like, optional
        A 2-D array of real values representing the detector image.
    lam_grid : (N_k) array_like, optional
        The grid on which f(lambda) will be projected.
        Default still has to be improved.
    lam_bounds : list or array-like (N_ord, 2), optional
        Boundary wavelengths covered by each orders.
        Default is the wavelength covered by `lam_list`.
    i_bounds : list or array-like (N_ord, 2), optional
        Index of `lam_bounds`on `lam_grid`.
    c_list : array or sparse, optional
        Convolution kernel to be applied on f_k for each orders.
        If array, the shape has to be (N_ker, N_k_c) and it will
        be passed to `convolution.get_c_matrix` function.
        If sparse, the shape has to be (N_k_c, N_k) and it will
        be used directly. N_ker is the length of the effective kernel
        and N_k_c is the length of f_k convolved.
        Default is given by convolution.WebbKer(wv_map, n_os=10, n_pix=21).
    t_list : (N_ord [, N_k]) list or array of callable, optional
        A list of functions or array of the throughput at each order.
        If callable, the function depend on the wavelength.
        If array, projected on `lam_grid`.
        Default is given by `throughput.ThroughputSOSS`.
    c_kwargs : list of N_ord dictionnaries, optional
        Inputs keywords arguments to pass to
        `convolution.get_c_matrix` function.
    sig : (N, M) array_like, optional
        Estimate of the error on each pixel. Default is one everywhere.
    mask : (N, M) array_like boolean, optional
        Boolean Mask of the bad pixels on the detector.
    tresh : float, optional:
        The pixels where the estimated spatial profile is less than
        this value will be masked. Default is 1e-5.
    verbose : bool, optional
        Print steps. Default is False.
    """
    def __init__(self, p_list, lam_list, scidata=None, lam_grid=None,
                 lam_bounds=None, i_bounds=None, c_list=None,
                 c_kwargs=None, t_list=None, sig=None,
                 mask=None, thresh=1e-5, orders=[1, 2], verbose=False):

        ############################
        # Check input
        ############################

        # lam_grid must be specified
        if lam_grid is None:
            raise ValueError("`lam_grid` kwarg must be specified.")

        ###########################
        # Save basic parameters
        ###########################

        # Number of orders
        self.n_ord = len(lam_list)

        # Non-convolved grid length
        self.n_k = len(lam_grid)

        # Shape of the detector used
        # (the wavelength map should have the same shape)
        self.shape = lam_list[0].shape

        # Threshold to build mask
        self.thresh = thresh

        # Verbose option
        self.verbose = verbose

        # Error map of each pixels
        if sig is None:
            # Ones with the detector shape
            self.sig = np.ones(self.shape)
        else:
            self.sig = sig.copy()

        # Save PSF for each orders
        self.update_lists(p_list=p_list)

        # Save pixel wavelength for each orders
        self.lam_list = [lam.copy() for lam in lam_list]

        # Save non-convolved wavelength grid
        self.lam_grid = lam_grid.copy()

        ################################
        # Define throughput
        ################################

        # If None, read throughput from file
        if t_list is None:
            if self.n_ord != len(orders):
                message = 'New implementation:'
                message += ' When extracting or simulating only one order,'
                message += ' you need to specify the `orders` input keyword.'
                raise ValueError(message)
            t_list = [ThroughputSOSS(order=n)
                      for n in orders]

        # Save t_list for each orders
        self.update_lists(t_list=t_list)

        #####################################################
        # Get index of wavelength grid covered by each orders
        #####################################################

        # Assign a first estimate of i_bounds if not specified
        # to be able to compute mask
        if i_bounds is None:
            i_bounds = [[0, len(lam_grid)]
                        for _ in range(self.n_ord)]
        else:
            # Make sure it is absolute index, not relative
            # So no negative index.
            for i_bnds in i_bounds:
                if i_bnds[1] < 0:
                    i_bnds[1] = len(lam_grid) + i_bnds[1]
        self.i_bounds = i_bounds

        ###################################
        # Build detector mask
        ###################################

        # First estimate of a global mask and masks for each orders
        self.mask, self.mask_ord = self._get_masks(mask)

        # Correct i_bounds if it was not specified
        self.i_bounds = self._get_i_bnds(lam_bounds)

        # Re-build global mask and masks for each orders
        self.mask, self.mask_ord = self._get_masks(mask)

        ####################################
        # Build convolution matrix
        ####################################

        # Set convolution to predefined kernels if not given
        # Take maximum oversampling and kernel width available
        # (n_os=10 and n_pix=21)
        if c_list is None:
            c_list = [WebbKer(wv_map, n_os=10, n_pix=21)
                      for wv_map in lam_list]

        # Check c_kwargs input
        if c_kwargs is None:
            c_kwargs = [{} for _ in range(self.n_ord)]
        elif isinstance(c_kwargs, dict):
            c_kwargs = [c_kwargs for _ in range(self.n_ord)]

        # Define convolution sparse matrix
        c = []
        for i, c_n in enumerate(c_list):
            if not issparse(c_n):
                c_n = get_c_matrix(c_n, lam_grid,
                                   i_bounds=self.i_bounds[i],
                                   **c_kwargs[i])
            c.append(c_n)
        self.c_list = c

        #############################
        # Compute weights
        #############################

        # The weights depend on the integration method used solve
        # the integral of the flux over a pixel and are encoded
        # in the class method `_get_w()`.

        w_list, k_list = [], []
        for n in range(self.n_ord):  # For each orders
            w_n, k_n = self.get_w(n)  # Compute weigths
            # Convert to sparse matrix
            # First get the dimension of the convolved grid
            n_kc = np.diff(self.i_bounds[n]).astype(int)[0]
            # Then convert to sparse
            w_n = sparse_k(w_n, k_n, n_kc)
            w_list.append(w_n), k_list.append(k_n)
        self.w_list, self.k_list = w_list, k_list  # Save values

        #########################
        # Save remaining inputs
        #########################

        # Detector image
        if scidata is None:
            # Create a dummy detector image.
            self.data = np.nan * np.ones(lam_list[0].shape)
        else:
            self.data = scidata.copy()
        # Set masked values to zero ... may not be necessary
        # IDEA: try setting to np.nan instead of zero?
        self.data[self.mask] = 0

    def _get_masks(self, mask):

        # Get needed attributes
        thresh, n_ord \
            = self.getattrs('thresh', 'n_ord')
        t_list, p_list, lam_list  \
            = self.getattrs('t_list', 'p_list', 'lam_list')

        # Mask according to the global troughput (spectral and spatial)
        mask_P = [P < thresh for P in p_list]

        # Mask pixels not covered by the wavelength grid
        mask_lam = [self.get_mask_lam(n) for n in range(n_ord)]

        # Apply user's defined mask
        if mask is None:
            mask_ord = np.any([mask_P, mask_lam], axis=0)
        else:
            mask = [mask for n in range(n_ord)]  # For each orders
            mask_ord = np.any([mask_P, mask_lam, mask], axis=0)

        # Mask pixels that are masked at each orders
        global_mask = np.all(mask_ord, axis=0)

        # Mask if mask_P not masked but mask_lam is.
        # This means that an order is contaminated by another
        # order, but the wavelength range does not cover this part
        # of the spectrum. Thus, it cannot be treated correctly.
        global_mask |= (np.any(mask_lam, axis=0)
                        & (~np.array(mask_P)).all(axis=0))

        # Apply this new global mask to each orders
        mask_ord = np.any([mask_lam, global_mask[None, :, :]], axis=0)

        return global_mask, mask_ord

    def _get_i_bnds(self, lam_bounds):
        """
        Define wavelength boundaries for each orders using the order's mask
        """

        # TODO: define this function for only a given order
        lam_grid = self.lam_grid
        i_bounds = self.i_bounds

        # Check if lam_bounds given
        if lam_bounds is None:
            lam_bounds = []
            for i in range(self.n_ord):
                lam = self.lam_list[i][~self.mask_ord[i]]
                lam_bounds.append([lam.min(), lam.max()])

        # What we need is the boundary position
        # on the wavelength grid.
        i_bnds_new = []
        for bounds, i_bnds in zip(lam_bounds, i_bounds):
            a = np.min(np.where(lam_grid >= bounds[0])[0])
            b = np.max(np.where(lam_grid <= bounds[1])[0]) + 1
            # Take the most restrictive bound
            a = np.max([a, i_bnds[0]])
            b = np.min([b, i_bnds[1]])
            # Keep value
            i_bnds_new.append([a, b])

        return i_bnds_new

    def rebuild(self, f, orders=None):
        """
        Build current model of the detector.

        Parameters
        ----------
        f: array-like or callable
            flux as a function of wavelength if callable
            or flux projected on the wavelength grid
        orders: iterable, ooptional
            Order index to model on detector. Default is
            all available orders.
        """

        # If f is callable, project on grid
        if callable(f):
            grid = self.lam_grid

            # Project on wavelength grid
            f = f(grid)

        # Iterate over all orders by default
        if orders is None:
            orders = range(self.n_ord)

        return self._rebuild(f, orders)

    def _rebuild(self, f_k, orders):
        """
        Build current model of the detector.

        Parameters
        ----------
        f_k: array-like
            Flux projected on the wavelength grid
        orders: iterable
            Order index to model on detector.
        """

        # Get needed class attribute
        ma = self.mask

        # Distribute the flux on detector
        out = np.zeros(self.shape)
        for n in orders:
            # Compute `b_n` at each pixels w/o `sig`
            b_n = self.get_b_n(n, sig=False)
            # Add flux to pixels
            out[~ma] += b_n.dot(f_k)

        # nan invalid pixels
        out[ma] = np.nan

        return out

    def update_lists(self, t_list=None, p_list=None, **kwargs):
        """
        Update attributes.
        """
        # Spatial profile
        if p_list is not None:
            self.p_list = [p_n.copy() for p_n in p_list]

        # Throughput
        if t_list is not None:
            # Can be a callable (function) or an array
            # with the same length as lambda grid.
            t = []
            for t_n in t_list:  # For each order
                try:  # First assume it's a function
                    t.append(t_n(self.lam_grid))  # Project on grid
                except TypeError:  # Assume it's an array
                    t.append(t_n)
            self.t_list = t  # Save value

        if kwargs:
            message = ', '.join(kwargs.keys())
            message += ' not supported by'
            message += ' `update_lists` method.'
            raise TypeError(message)

    def get_b_n(self, i_ord, sig=True, quick=False):
        """
        Compute the matrix `b_n = (P/sig).w.T.lambda.c_n` ,
        where `P` is the spatial profile matrix (diag),
        `w` is the integrations weights matrix,
        `T` is the throughput matrix (diag),
        `lambda` is the convolved wavelength grid matrix (diag),
        `c_n` is the convolution kernel.
        The model of the detector at order n (`model_n`)
        is given by the system:
        model_n = b_n.c_n.f ,
        where f is the incoming flux projected on the wavelenght grid.
        Parameters
        ----------
        i_ord : integer
            Label of the order (depending on the initiation of the object).
        sig: bool or (N, M) array_like, optional
            If 2-d array, `sig` is the new error estimation map.
            It is the same shape as `sig` initiation input. If bool,
            wheter to apply sigma or not. The method will return
            b_n/sigma if True or array_like and b_n if False. If True,
            the default object attribute `sig` will be use.
        quick: bool, optional
            If True, only perform one matrix multiplication
            instead of the whole system: (P/sig).(w.T.lambda.c_n)
        """
        #### Input management ######

        # Special treatment for error map
        # Can be bool or array.
        if sig is False:
            # Sigma will have no effect
            sig = np.ones(self.shape)
        else:
            if sig is not True:
                # Sigma must be an array so
                # update object attribute
                self.sig = sig.copy()
            # Take sigma from object
            sig = self.sig

        # Get needed attributes ...
        attrs = ('lam_grid', 'mask')
        lam, mask = self.getattrs(*attrs)

        # ... order dependent attributes
        attrs = ('t_list', 'p_list', 'c_list', 'w_list', 'i_bounds')
        t_n, p_n, c_n, w_n, i_bnds = self.getattrs(*attrs, n=i_ord)

        # Keep only valid pixels (P and sig are still 2-D)
        # And apply direcly 1/sig here (quicker)
        p_n = p_n[~mask] / sig[~mask]

        ### Compute b_n ###

        # Quick mode if only `p_n` or `sig` has changed
        if quick:
            # Get pre-computed (right) part of the equation
            right = self.w_t_lam_c[i_ord]
            # Apply new p_n
            b_n = diags(p_n).dot(right)
        else:
            # First (T * lam) for the convolve axis (n_k_c)
            product = (t_n * lam)[slice(*i_bnds)]
            # then convolution
            product = diags(product).dot(c_n)
            # then weights
            product = w_n.dot(product)
            # Save this product for quick mode
            self.save_w_t_lam_c(i_ord, product)
            # Then spatial profile
            b_n = diags(p_n).dot(product)

        return b_n

    def save_w_t_lam_c(self, order, product):

        # Get needed attributes
        n_ord = self.n_ord

        # Check if attribute exists
        try:
            self.w_t_lam_c
        except AttributeError:
            # Init w_t_lam_c.
            self.w_t_lam_c = [[] for i_ord in range(n_ord)]

        # Assign value
        self.w_t_lam_c[order] = product.copy()

    def build_sys(self, data=None, sig=True, **kwargs):
        """
        Build linear system arising from the logL maximisation.
        TIPS: To be quicker, only specify the psf (`p_list`) in kwargs.
              There will be only one matrix multiplication:
              (P/sig).(w.T.lambda.c_n).
        Parameters
        ----------
        data : (N, M) array_like, optional
            A 2-D array of real values representing the detector image.
            Default is the object attribute `data`.
        sig: bool or (N, M) array_like, optional
            Estimate of the error on each pixel.
            If 2-d array, `sig` is the new error estimation map.
            It is the same shape as `sig` initiation input. If bool,
            wheter to apply sigma or not. The method will return
            b_n/sigma if True or array_like and b_n if False. If True,
            the default object attribute `sig` will be use.
        t_list : (N_ord [, N_k]) list or array of functions, optional
            A list or array of the throughput at each order.
            The functions depend on the wavelength
            Default is the object attribute `t_list`
        p_list : (N_ord, N, M) list or array of 2-D arrays, optional
            A list or array of the spatial profile for each order
            on the detector. It has to have the same (N, M) as `scidata`.
            Default is the object attribute `p_list`
        """
        ##### Input management ######

        # Use data from object as default
        if data is None: data = self.data

        # Take mask from object
        mask = self.mask

        # Get some dimensions infos
        n_k, n_ord = self.n_k, self.n_ord

        # Update p_list and t_list if given.
        self.update_lists(**kwargs)

        # Check if inputs are suited for quick mode;
        # Quick mode if `t_list` is not specified.
        quick = ('t_list' not in kwargs)
        quick &= hasattr(self, 'w_t_lam_c')  # Pre-computed
        if quick:
            self.v_print('Quick mode is on!')

        ####### Calculations ########

        # Build matrix B
        # Initiate with empty matrix
        n_i = (~mask).sum()  # n good pixels
        b_matrix = csr_matrix((n_i, n_k))
        # Sum over orders
        for i_ord in range(n_ord):
            # Get sparse b_n
            b_matrix += self.get_b_n(i_ord, sig=sig, quick=quick)

        # Build system
        # Fisrt get `sig` which have been update`
        # when calling `get_b_n`
        sig = self.sig
        # Take only valid pixels and apply `sig`on data
        data = data[~mask] / sig[~mask]
        # (B_T * B) * f = (data/sig)_T * B
        # (matrix ) * f = result
        matrix = b_matrix.T.dot(b_matrix)
        result = csr_matrix(data.T).dot(b_matrix)

        return matrix, result.toarray().squeeze()

    def extract(self, tikhonov=False, tikho_kwargs=None, **kwargs):
        """
        Build linear system arising from the logL maximisation.
        All parameters are passed to `build_sys` method.
        Parameters
        ----------
        tikhonov : bool, optional
            Wheter to use tikhonov extraction
            (see regularisation.tikho_solve function).
            Default is False.
        tikho_kwargs : dictionnary or None, optional
            Arguments passed to `tikho_solve`.
        data : (N, M) array_like, optional
            A 2-D array of real values representing the detector image.
            Default is the object attribute `data`.
        t_list : (N_ord [, N_k]) list or array of functions, optional
            A list or array of the throughput at each order.
            The functions depend on the wavelength
            Default is the object attribute `t_list`
        p_list : (N_ord, N, M) list or array of 2-D arrays, optional
            A list or array of the spatial profile for each order
            on the detector. It has to have the same (N, M) as `scidata`.
            Default is the object attribute `p_list`
        sig : (N, M) array_like, optional
            Estimate of the error on each pixel`
            Same shape as `scidata`.
            Default is the object attribute `sig`.
        """
        # Build the system to solve
        matrix, result = self.build_sys(**kwargs)

        # Get index of `lam_grid` convered by the pixel.
        # `lam_grid` may cover more then the pixels.
        i_grid = self.get_i_grid(result)

        # Init f_k with nan
        f_k = np.ones(result.shape[-1]) * np.nan

        # Solve with the specified solver.
        # Only solve for valid range `i_grid` (on the detector).
        # It will be a singular matrix otherwise.
        if tikhonov:
            if tikho_kwargs is None:
                tikho_kwargs = {'index': i_grid}
            else:
                default_kwargs = {'grid': self.lam_grid,
                                  'index': i_grid}
                tikho_kwargs = {**default_kwargs, **tikho_kwargs}
            f_k[i_grid] = self._solve_tikho(matrix, result, **tikho_kwargs)
        else:
            f_k[i_grid] = self._solve(matrix, result, index=i_grid)

        return f_k

    def _solve(self, matrix, result, index=slice(None)):
        """
        Simply pass `matrix` and `result`
        to `scipy.spsolve` and apply index.
        """
        return spsolve(matrix[index, :][:, index], result[index])

    def _solve_tikho(self, matrix, result, index=slice(None), **kwargs):
        """Solve system using Tikhonov regularisation"""
        # Note that the indexing is applied inside the function
        return tikho_solve(matrix, result, index=index, **kwargs)

    def get_i_grid(self, d):
        """ Return the index of the grid that are well defined, so d != 0 """
        try:
            self.i_grid
        except AttributeError:
            i_grid = np.nonzero(d)[0]
            self.i_grid = i_grid

        return self.i_grid

    def get_logl(self, f_k=None):
        """
        Return the log likelihood compute on each pixels
        """
        data = self.data
        sig = self.sig

        if f_k is None:
            f_k = self.extract()

        model = self.rebuild(f_k)

        return - np.nansum((model-data)**2/sig**2)

    def get_adapt_grid(self, f_k=None, n_max=3, **kwargs):
        """
        Return an irregular grid needed to reach a
        given precision when integrating over each pixels.

        Parameters (all optional)
        ----------
        f_k: 1D array-like
            Input flux in the integral to be optimized.
            f_k is the projection of the flux on self.lam_grid
        n_max: int (n_max > 0)
            Maximum number of nodes in each intervals of self.lam_grid.
            Needs to be greater then zero.

        kwargs (arguments passed to the function get_n_nodes)
        ------
        tol, rtol : float, optional
            The desired absolute and relative tolerances. Defaults are 1.48e-4.
        divmax : int, optional
            Maximum order of extrapolation. Default is 10.

        Returns
        -------
        os_grid  : 1D array
            Oversampled grid which minimizes the integration error based on
            Romberg's method
        See Also
        --------
        utils.get_n_nodes
        scipy.integrate.quadrature.romberg
        References
        ----------
        [1] 'Romberg's method' https://en.wikipedia.org/wiki/Romberg%27s_method

        """
        # Generate f_k if not given
        if f_k is None:
            f_k = self.extract()

        # Init output oversampled grid
        os_grid = []

        # Iterate starting with the last order
        for i_ord in range(self.n_ord - 1, -1, -1):

            # Grid covered by this order
            grid_ord = self.lam_grid_c(i_ord)

            # Estimate the flux at this order
            f_k_c = self.c_list[i_ord].dot(f_k)
            # Interpolate with a cubic spline
            fct = interp1d(grid_ord, f_k_c, kind='cubic')

            # Find number of nodes to reach the precision
            n_oversample = get_n_nodes(grid_ord, fct, **kwargs)

            # Make sure n_oversample is not greater than
            # user's define `n_max`
            n_oversample = np.clip(n_oversample, 0, n_max)

            # Generate oversampled grid
            grid_ord = oversample_grid(grid_ord, n_os=n_oversample)

            # Keep only wavelength that are not already
            # covered by os_grid.
            if os_grid:
                # Under or above os_grid
                index = (grid_ord < np.min(os_grid))
                index |= (grid_ord > np.max(os_grid))
            else:
                index = slice(None)

            # Keep these values
            os_grid.append(grid_ord[index])

        # Convert os_grid to 1D array
        os_grid = np.concatenate(os_grid)

        # Return sorted and unique
        return np.unique(os_grid)

    def get_tikho_tests(self, factors, tikho=None, estimate=None, **kwargs):

        # Build the system to solve
        matrix, result = self.build_sys()

        # Get valid grid index
        i_grid = self.get_i_grid(result)

        if tikho is None:
            if kwargs:
                default_kwargs = {'grid': self.lam_grid,
                                  'index': i_grid}
                kwargs = {**default_kwargs, **kwargs}
                tikho = Tikhonov(matrix, result, **kwargs)
                self.tikho = tikho
            else:
                tikho = self.tikho

        # Test all factors
        tests = tikho.test_factors(factors, estimate)
        # Generate logl using solutions for each factors
        logl_list = []
        for sln in tests['solution']:
            # Init f_k with nan, so it has the adequate shape
            f_k = np.ones(result.shape[-1]) * np.nan
            f_k[i_grid] = sln  # Assign valid values
            logl_list.append(self.get_logl(f_k))  # log_l
        # Save in tikho's tests
        tikho.test['-logl'] = -1 * np.array(logl_list)

        return tikho.test

    def plot_tikho_factors(self):

        tikho = self.tikho

        # Init figure
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        # logl plot
        tikho.error_plot(ax=ax[0], test_key='-logl')
        # Error plot
        tikho.error_plot(ax=ax[1])
        # Labels
        ax[0].set_ylabel(r'$\log{L}$ on detector')
        # Other details
        fig.tight_layout()

        return fig, ax

    @staticmethod
    def _check_plot_inputs(fig, ax):
        """
        Method to manage inputs for plots methods.
        """
        # Use ax or fig if given. Else, init the figure
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots(1, 1, sharex=True)
        elif ax is None:
            ax = fig.subplots(1, 1, sharex=True)

        return fig, ax

    def plot_sln(self, f_k, fig=None, ax=None, i_ord=0,
                 ylabel='Flux', xlabel=r'Wavelength [$\mu$m]', **kwargs):

        # Manage method's inputs
        args = (fig, ax)
        fig, ax = self._check_plot_inputs(*args)

        # Set values to plot
        x = self.lam_grid_c(i_ord)
        y = self.c_list[i_ord].dot(f_k)

        # Plot
        ax.plot(x, y, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax

    def plot_err(self, f_k, f_th_ord, fig=None, ax=None,
                 i_ord=0, error='relative', ylabel='Error',
                 xlabel=r'Wavelength [$\mu$m]', **kwargs):

        # Manage method's inputs
        args = (fig, ax)
        fig, ax = self._check_plot_inputs(*args)

        # Set values to plot
        x = self.lam_grid_c(i_ord)
        f_k_c = self.c_list[i_ord].dot(f_k)

        if error == 'relative':
            y = (f_k_c - f_th_ord) / f_th_ord
        elif error == 'absolute':
            y = f_k_c - f_th_ord
        elif error == 'to_noise':
            y = (f_k_c - f_th_ord) / np.sqrt(f_th_ord)
        else:
            raise ValueError('`error` argument is not valid.')

        # Add info to ylabel
        ylabel += ' ({})'.format(error)

        # Plot
        ax.plot(x, y, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax

    def get_w(self, *args):
        """Dummy method to be able to init this class"""

        return None, None

    def get_mask_lam(self, n):
        """ Mask according to wavelength grid """
        lam = self.lam_list[n]
        a, b = self.i_bounds[n]
        lam_min = self.lam_grid[a]
        lam_max = self.lam_grid[b-1]

        mask = (lam <= lam_min) | (lam >= lam_max)

        return mask

    def lam_grid_c(self, order):
        """
        Return lam_grid for the convolved flux at a given order.
        """
        index = slice(*self.i_bounds[order])
        return self.lam_grid[index]

    def getattrs(self, *args, n=None):

        if n is None:
            out = [getattr(self, arg) for arg in args]
        else:
            out = [getattr(self, arg)[n] for arg in args]

        if len(out) > 1:
            return out
        else:
            return out[0]

    def v_print(self, *args, **kwargs):

        if self.verbose:
            print(*args, **kwargs)


class LagrangeOverlap(_BaseOverlap):

    def __init__(self, *args, lagrange_ord=1, lam_grid=None, **kwargs):

        # Attribute specific to the interpolation method
        self.lagrange_ord = lagrange_ord

        # TODO: Set a default lam_grid

        super().__init__(*args, lam_grid=lam_grid, **kwargs)

    def get_mask_lam(self, n):

        lam, i_bnds   \
            = self.getattrs('lam_list', 'i_bounds', n=n)
        lam_min = self.lam_grid[i_bnds[0]]
        lam_max = self.lam_grid[i_bnds[1]-1]

        mask = (lam <= lam_min) | (lam >= lam_max)

        return mask

    def get_w(self, n):

        # Get needed attributes
        order = self.lagrange_ord

        # Get needed attributes
        grid, mask, order \
            = self.getattrs('lam_grid', 'mask', 'lagrange_ord')
        # ... diffraction-order dependent attributes
        lam, mask_ord, i_bnds  \
            = self.getattrs('lam_list', 'mask_ord', 'i_bounds', n=n)

        # Use the convolved grid (depends on the order)
        grid = grid[i_bnds[0]:i_bnds[1]]

        # Compute delta lamda of each pixel
        # delta_lambda = lambda_plus - lambda_minus
        d_lam = - np.diff(_get_lam_p_or_m(lam), axis=0).squeeze()

        # Compute only for valid pixels
        lam, d_lam = lam[~mask], d_lam[~mask]
        ma = mask_ord[~mask]

        # Use a pre-defined interpolator
        interp = SegmentedLagrangeX(grid, order)

        # Get w and k
        # Init w and k
        n_i = (~mask).sum()  # Number of good pixels
        w_n = np.ones((order+1, n_i)) * np.nan
        k_n = np.ones((order+1, n_i), dtype=int) * -1
        # Compute values in grid range
        w_n[:,~ma] = interp.get_coeffs(lam[~ma])
        i_segment = interp.get_index(lam[~ma])
        k_n[:,~ma] = interp.index[:,i_segment]

        # Include delta lambda in the weights
        w_n[:,~ma] = w_n[:,~ma] * d_lam[~ma]

        return w_n.T, k_n.T


class TrpzOverlap(_BaseOverlap):
    """
    Version of overlaping extraction with oversampled trapezoidal integration
    overlaping extraction solve the equation of the form:
    (B_T * B) * f = (data/sig)_T * B
    where B is a matrix and f is an array.
    The matrix multiplication B * f is the 2d model of the detector.
    We want to solve for the array f.
    The elements of f are labelled by 'k'.
    The pixels are labeled by 'i'.
    Every pixel 'i' is covered by a set of 'k' for each order
    of diffraction.

    Parameters
    ----------
    p_list : (N_ord, N, M) list or array of 2-D arrays
        A list or array of the spatial profile for each order
        on the detector. It has to have the same (N, M) as `scidata`.
    lam_list : (N_ord, N, M) list or array of 2-D arrays
        A list or array of the central wavelength position for each
        order on the detector.
        It has to have the same (N, M) as `scidata`.
    scidata : (N, M) array_like, optional
        A 2-D array of real values representing the detector image.
    lam_grid : (N_k) array_like, optional
        The grid on which f(lambda) will be projected.
        Default still has to be improved.
    lam_bounds : list or array-like (N_ord, 2), optional
        Boundary wavelengths covered by each orders.
        Default is the wavelength covered by `lam_list`.
    i_bounds : list or array-like (N_ord, 2), optional
        Index of `lam_bounds`on `lam_grid`.
    c_list : array or sparse, optional
        Convolution kernel to be applied on f_k for each orders.
        If array, the shape has to be (N_ker, N_k_c) and it will
        be passed to `convolution.get_c_matrix` function.
        If sparse, the shape has to be (N_k_c, N_k) and it will
        be used directly. N_ker is the length of the effective kernel
        and N_k_c is the length of f_k convolved.
        Default is given by convolution.WebbKer(wv_map, n_os=10, n_pix=21).
    t_list : (N_ord [, N_k]) list or array of callable, optional
        A list of functions or array of the throughput at each order.
        If callable, the function depend on the wavelength.
        If array, projected on `lam_grid`.
        Default is given by `throughput.ThroughputSOSS`.
    c_kwargs : list of N_ord dictionnaries, optional
        Inputs keywords arguments to pass to
        `convolution.get_c_matrix` function.
    sig : (N, M) array_like, optional
        Estimate of the error on each pixel. Default is one everywhere.
    mask : (N, M) array_like boolean, optional
        Boolean Mask of the bad pixels on the detector.
    tresh : float, optional:
        The pixels where the estimated spatial profile is less than
        this value will be masked. Default is 1e-5.
    verbose : bool, optional
        Print steps. Default is False.
    """

    def __init__(self, p_list, lam_list, **kwargs):

        # Get wavelength at the boundary of each pixel
        # TODO? Could also be an input??
        lam_p, lam_m = [], []
        for lam in lam_list:  # For each order
            lp, lm = _get_lam_p_or_m(lam)  # Lambda plus or minus
            lam_p.append(lp), lam_m.append(lm)
        self.lam_p, self.lam_m = lam_p, lam_m  # Save values

        # Init upper class
        super().__init__(p_list, lam_list, **kwargs)

    def _get_LH(self, grid, n):
        """
        Find the lowest (L) and highest (H) index
        of lam_grid for each pixels and orders.
        """
        self.v_print('Compute LH')

        # Get needed attributes
        mask = self.mask
        # ... order dependent attributes
        lam_p, lam_m, mask_ord  \
            = self.getattrs('lam_p', 'lam_m', 'mask_ord', n=n)

        # Compute only for valid pixels
        lam_p = lam_p[~mask]
        lam_m = lam_m[~mask]

        # Find lower (L) index in the pixel
        #
        L = np.searchsorted(grid, lam_m, side='right')

        # Find higher (H) index in the pixel
        #
        H = np.searchsorted(grid, lam_p) - 1

        # Set invalid pixels for this order to L=-1 and H=-2
        ma = mask_ord[~mask]
        L[ma], H[ma] = -1, -2

        self.v_print('Done')

        return L, H

    def get_mask_lam(self, n):

        lam_p, lam_m, i_bnds   \
            = self.getattrs('lam_p', 'lam_m', 'i_bounds', n=n)
        lam_min = self.lam_grid[i_bnds[0]]
        lam_max = self.lam_grid[i_bnds[1]-1]

        mask = (lam_m < lam_min) | (lam_p > lam_max)

        return mask

    def get_w(self, n):

        self.v_print('Compute weigths and k')

        # Get needed attributes
        grid, mask  \
            = self.getattrs('lam_grid', 'mask')
        # ... order dependent attributes
        lam_p, lam_m, mask_ord, i_bnds  \
            = self.getattrs('lam_p', 'lam_m',
                            'mask_ord', 'i_bounds', n=n)

        # Use the convolved grid (depends on the order)
        grid = grid[i_bnds[0]:i_bnds[1]]

        # Compute the wavelength coverage of the grid
        d_grid = np.diff(grid)

        # Get LH
        L, H = self._get_LH(grid, n)  # Get indexes

        # Compute only valid pixels
        lam_p, lam_m = lam_p[~mask], lam_m[~mask]
        ma = mask_ord[~mask]

        # Number of used pixels
        n_i = len(L)
        i = np.arange(n_i)

        self.v_print('Compute k')

        # Define fisrt and last index of lam_grid
        # for each pixel
        k_first, k_last = -1*np.ones(n_i), -1*np.ones(n_i)

        # If lowest value close enough to the exact grid value,
        # NOTE: Could be approximately equal to the exact grid
        # value. It would look like that.
        # >>> L_dgrid = L
        # >>> L_dgrid[L_dgrid==len(d_grid)] = len(d_grid) - 1
        # >>> cond = (grid[L]-lam_m)/d_grid[L_dgrid] <= 1.0e-8
        # But let's stick with the exactly equal
        cond = (grid[L] == lam_m)
        # special case (no need for L_i - 1)
        k_first[cond & ~ma] = L[cond & ~ma]
        lam_m[cond & ~ma] = grid[L[cond & ~ma]]
        # else, need L_i - 1
        k_first[~cond & ~ma] = L[~cond & ~ma] - 1
        # Same situation for highest value. If we follow the note
        # above (~=), the code could look like
        # >>> cond = (lam_p-grid[H])/d_grid[H-1] <= 1.0e-8
        # But let's stick with the exactly equal
        cond = (lam_p == grid[H])
        # special case (no need for H_i - 1)
        k_last[cond & ~ma] = H[cond & ~ma]
        lam_p[cond & ~ma] = grid[H[cond & ~ma]]
        # else, need H_i + 1
        k_last[~cond & ~ma] = H[~cond & ~ma] + 1

        # Generate array of all k_i. Set to -1 if not valid
        k_n, bad = arange_2d(k_first, k_last+1, dtype=int, return_mask=True)
        k_n[bad] = -1
        # Number of valid k per pixel
        n_k = np.sum(~bad, axis=-1)

        # Compute array of all w_i. Set to np.nan if not valid
        # Initialize
        w_n = np.zeros(k_n.shape, dtype=float)
        ####################
        ####################
        # 4 different cases
        ####################
        ####################

        self.v_print('compute w')

        # Valid for every cases
        w_n[:,0] = grid[k_n[:,1]] - lam_m
        w_n[i,n_k-1] = lam_p - grid[k_n[i,n_k-2]]

        ##################
        # Case 1, n_k == 2
        ##################
        case = (n_k == 2) & ~ma
        if case.any():
            self.v_print('n_k = 2')
            # if k_i[0] != L_i
            cond = case & (k_n[:,0] != L)
            w_n[cond,1] += lam_m[cond] - grid[k_n[cond,0]]
            # if k_i[-1] != H_i
            cond = case & (k_n[:,1] != H)
            w_n[cond,0] += grid[k_n[cond,1]] - lam_p[cond]
            # Finally
            w_n[case,:] *= ((lam_p[case] - lam_m[case])
                          / d_grid[k_n[case,0]])[:,None]

        ##################
        # Case 2, n_k >= 3
        ##################
        case = (n_k >= 3) & ~ma
        if case.any():
            self.v_print('n_k = 3')
            n_ki = n_k[case]
            w_n[case,1] = grid[k_n[case,1]] - lam_m[case]
            w_n[case,n_ki-2] += lam_p[case] - grid[k_n[case,n_ki-2]]
            # if k_i[0] != L_i
            cond = case & (k_n[:,0] != L)
            w_n[cond,0] *= (grid[k_n[cond,1]] - lam_m[cond]) / d_grid[k_n[cond,0]]
            w_n[cond,1] += ((grid[k_n[cond,1]] - lam_m[cond])
                          * (lam_m[cond] - grid[k_n[cond,0]])
                          / d_grid[k_n[cond,0]])
            # if k_i[-1] != H_i
            cond = case & (k_n[i,n_k-1] != H)
            n_ki = n_k[cond]
            w_n[cond,n_ki-1] *= ((lam_p[cond] - grid[k_n[cond,n_ki-2]])
                               / d_grid[k_n[cond,n_ki-2]])
            w_n[cond,n_ki-2] += ((grid[k_n[cond,n_ki-1]] - lam_p[cond])
                               * (lam_p[cond] - grid[k_n[cond,n_ki-2]])
                               / d_grid[k_n[cond,n_ki-2]])

        ##################
        # Case 3, n_k >= 4
        ##################
        case = (n_k >= 4) & ~ma
        if case.any():
            self.v_print('n_k = 4')
            n_ki = n_k[case]
            w_n[case,1] += grid[k_n[case,2]] - grid[k_n[case,1]]
            w_n[case,n_ki-2] += grid[k_n[case,n_ki-2]] - grid[k_n[case,n_ki-3]]

        ##################
        # Case 4, n_k > 4
        ##################
        case = (n_k > 4) & ~ma
        if case.any():
            self.v_print('n_k > 4')
            i_k = np.indices(k_n.shape)[-1]
            cond = case[:,None] & (2 <= i_k) & (i_k < n_k[:,None]-2)
            ind1, ind2 = np.where(cond)
            w_n[ind1,ind2] = d_grid[k_n[ind1,ind2]-1] + d_grid[k_n[ind1,ind2]]

        # Finally, divide w_n by 2
        w_n /= 2.

        # Make sure invalid values are masked
        w_n[k_n < 0] = np.nan

        self.v_print('Done')
        return w_n, k_n


def sparse_k(val, k, n_k):
    '''
    Transform a 2D array `val` to a sparse matrix.
    `k` is use for the position in the second axis
    of the matrix. The resulting sparse matrix will
    have the shape : ((len(k), n_k))
    Set k elements to a negative value when not defined
    '''
    # Length of axis 0
    n_i = len(k)

    # Get row index
    i_k = np.indices(k.shape)[0]

    # Take only well defined coefficients
    row = i_k[k >= 0]
    col = k[k >= 0]
    data = val[k >= 0]

    return csr_matrix((data, (row, col)), shape=(n_i, n_k))


def unsparse(matrix, fill_value=np.nan):

    col, row, val = find(matrix.T)
    n_row, n_col = matrix.shape

    good_rows, counts = np.unique(row, return_counts=True)

    # Define the new position in columns
    i_col = np.indices((n_row, counts.max()))[1]
    i_col = i_col[good_rows]
    i_col = i_col[i_col < counts[:,None]]

    # Create outputs and assign values
    col_out = np.ones((n_row, counts.max()), dtype=int) * -1
    col_out[row, i_col] = col
    out = np.ones((n_row, counts.max())) * fill_value
    out[row, i_col] = val

    return out, col_out
