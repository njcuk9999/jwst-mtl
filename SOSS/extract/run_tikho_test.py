import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from overlap import TrpzOverlap
from utils import get_soss_grid, grid_from_map, oversample_grid
from convolution import WebbKer, get_c_matrix
from scipy.sparse import identity
import matplotlib.pyplot as plt

default_file_root = 'tikho_test'
default_file_ext = '.n_os_{}.c_thresh_{:1.0e}.tikho_os_{}'
default_path = ''
default_ref_files = {'wv_1': 'Ref_files/wavelengths_m1.fits',
                     'wv_2': 'Ref_files/wavelengths_m2.fits',
                     'P_1': 'Ref_files/spat_profile_m1.fits',
                     'P_2': 'Ref_files/spat_profile_m2.fits'}


class TikhoTest:

    file_root = default_file_root
    file_ext = default_file_ext
    path = default_path

    def __init__(self, n_os, c_thresh, t_mat_n_os,
                 results=None, ref_files=None):

        if results is None:
            results = {}

        if ref_files is None:
            ref_files = default_ref_files

        # Set object attributes
        self.results = results
        self.n_os = n_os
        self.c_thresh = c_thresh
        self.t_mat_n_os = t_mat_n_os
        self.ref_files = ref_files

    @classmethod
    def read(cls, n_os, c_thresh, t_mat_n_os,
             file_root=None, file_ext=None, path=None):

        # Default kwargs
        if file_root is None:
            file_root = cls.file_root

        if file_ext is None:
            file_ext = cls.file_ext

        if path is None:
            path = cls.path

        # Read file
        file_name = path + file_root
        file_name += file_ext.format(n_os, c_thresh, t_mat_n_os)
        results = np.load(file_name + '.npz')

        # Convert to dictionnary and return instanciate
        return cls(n_os, c_thresh, t_mat_n_os, results=dict(results))

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

    def error_plot(self, fig=None, ax=None, factors=None,
                   results=None, key=None, y_val=None, label=None):

        # Manage method's inputs
        args = (fig, ax)
        fig, ax = self._check_plot_inputs(*args)

        if results is None:
            results = self.results

        # What y value do we plot?
        if y_val is None:
            # Use tests to plot y_val
            if key is None:
                # Default is euclidian norm of error.
                # Similar to the chi^2.
                y_val = (results['error']**2).sum(axis=-1)
            else:
                y_val = results[key]

        # Plot
        ax.loglog(results['factors'], y_val, label=label)

        # Mark minimum value
        i_min = np.argmin(y_val)
        min_coord = results['factors'][i_min], y_val[i_min]
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

    def l_plot(self, fig=None, ax=None, label=None,
               results=None, text_label=True):

        # Manage method's inputs
        args = (fig, ax)
        fig, ax = self._check_plot_inputs(*args)

        if results is None:
            results = self.results

        # Compute euclidian norm of error (||A.x - b||).
        # Similar to the chi^2.
        err_norm = (results['error']**2).sum(axis=-1)

        # Compute norm of regularisation term
        reg_norm = (results['reg']**2).sum(axis=-1)

        # Plot
        ax.loglog(err_norm, reg_norm, '.:', label=label)

        # Add factor values as text
        if text_label:
            for f, x, y in zip(results['factors'], err_norm, reg_norm):
                plt.text(x, y, "{:2.1e}".format(f), va="center", ha="right")

        # Legend
        ax.legend()

        # Labels
        xlabel = r'$\left(||\mathbf{Ax-b}||^2_2\right)$'
        ylabel = r'$\left(||\mathbf{\Gamma.x}||^2_2\right)$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax

    def plot_sln(self, factor=None, index=None, fig=None, ax=None,
                 results=None, i_ord=0, f_th=False, **kwargs):

        # Manage method's inputs
        args = (fig, ax)
        fig, ax = self._check_plot_inputs(*args)

        # Manage other inputs
        if results is None:
            results = self.results

        if index is None:
            if factor is None:
                message = 'At least `factor` or `index` must be specified.'
                raise ValueError(message)
            else:
                index = np.argmin(np.abs(results['factors'] - factor))

        # Set some variables for plot
        f_k = results['solution'][index]

        # Get extraction object
        extra = self.get_extraction()

        # Plot f_th if specified
        if f_th:
            f_th = results['f_k_th_{}'.format(i_ord + 1)]
            ax.plot(extra.lam_grid_c(i_ord), f_th, label='Injected')

        # Plot solution using this extraction
        fig, ax = extra.plot_sln(f_k, fig=fig, ax=ax, i_ord=i_ord, **kwargs)

        return

    def plot_sln_err(self, factor=None, index=None, fig=None, ax=None,
                     results=None, i_ord=0, error='relative', **kwargs):

        # Manage method's inputs
        args = (fig, ax)
        fig, ax = self._check_plot_inputs(*args)

        # Manage other inputs
        if results is None:
            results = self.results

        if index is None:
            if factor is None:
                message = 'At least `factor` or `index` must be specified.'
                raise ValueError(message)
            else:
                index = np.argmin(np.abs(results['factors'] - factor))

        # Set some variables for plot
        f_k = results['solution'][index]
        f_th = results['f_k_th_{}'.format(i_ord + 1)]

        # Get extraction object
        extra = self.get_extraction()

        # Plot solution using this extraction
        fig, ax = extra.plot_err(f_k, f_th, fig=fig, ax=ax,
                                 i_ord=i_ord, error=error, **kwargs)

        return fig, ax

    def plot_extract_err(self, fig=None, ax=None, results=None,
                         i_ord=0, **kwargs):

        # Manage method's inputs
        args = (fig, ax)
        fig, ax = self._check_plot_inputs(*args)

        if results is None:
            results = self.results

        # What to plot
        x = results['factors']
        y = self.get_extract_err(i_ord=i_ord, results=results)

        # Plot
        ax.loglog(x, y, **kwargs)

        # Mark minimum value
        i_min = np.argmin(y)
        min_coord = x[i_min], y[i_min]
        ax.scatter(*min_coord, marker="x")
        text = '{:2.1e}'.format(min_coord[0])
        ax.text(*min_coord, text, va="top", ha="center")

        # Labels
        ax.set_xlabel("Scale factor")
        ylabel = r'1D extraction error '
        ylabel += r'$\left(||\mathbf{\tilde{f}_k-\tilde{f}_{th}}||^2_2\right)$'
        ax.set_ylabel(ylabel)

        return fig, ax

    def get_extract_err(self, i_ord=0, results=None):

        # Manage inputs
        if results is None:
            results = self.results

        # Set some variables for plot
        f_k_list = results['solution']
        f_th = results['f_k_th_{}'.format(i_ord + 1)]

        # Get extraction object
        extra = self.get_extraction()

        chi2 = []
        for f_k in f_k_list:
            f_k_c = extra.c_list[i_ord].dot(f_k)
            chi2.append(np.sum((f_k_c - f_th)**2))

        return np.array(chi2)

    def load_ref_files(self):

        ref_files = self.ref_files

        out = {}
        for key in ref_files.keys():
            # Read file
            data = fits.open(ref_files[key])[0].data.squeeze()
            # Convert to float (fits precision is 1e-8)
            out[key] = data.astype(float)

        return out

    def get_extraction(self):

        try:
            self.extra
        except AttributeError:
            extra = self.init_extraction()
            self.extra = extra

        return self.extra

    def init_extraction(self, order_list=None):

        if order_list is None:
            order_list = [1, 2]

        # Load reference files
        ref_files = self.load_ref_files()
        try:
            p_list = [ref_files['P_{}'.format(order)]
                      for order in order_list]
            wv_list = [ref_files['wv_{}'.format(order)]
                       for order in order_list]
        except KeyError:
            message = ('`KeyError` was raised while loading ref files.'
                       + 'You must manualy run `init_extraction` method.')
            raise ValueError(message)

        # Init extraction object
        c_kwargs = {'thresh': self.c_thresh}
        grid = self.results['grid']
        extra = TrpzOverlap(p_list, wv_list,
                            lam_grid=grid,
                            c_kwargs=c_kwargs)

        return extra


def run_tikho_tests(p_list, lam_list, scidata, f_th_c,
                    n_os_list, c_thresh_list, t_mat_n_os_list,
                    factors=None, file_root=None, file_ext=None, path=None):

    # Unpack some lists
    P1, P2 = p_list
    wv_1, wv_2 = lam_list

    # Default kwargs
    if factors is None:
        factors = 10.**(-1*np.arange(10, 25, 0.3))

    if file_root is None:
        file_root = 'tikho_test'

    if file_ext is None:
        file_ext = '.n_os_{}.c_thresh_{:1.0e}.tikho_os_{}'

    if path is None:
        path = ''

    # Message to print
    status = 'n_os={}, c_thresh={:1.0e}, t_mat_n_os={}'

    # Iterate on grid oversampling
    for n_os in n_os_list:
        # Generate grid
        lam_grid = get_soss_grid([P1, P2], [wv_1, wv_2], n_os=n_os)

        # Iterate on convolution kernel wings threshold
        for c_thresh in c_thresh_list:
            # Init extraction object
            extra = TrpzOverlap([P1, P2], [wv_1, wv_2], scidata=scidata,
                                lam_grid=lam_grid, thresh=1e-5,
                                c_kwargs={'thresh': c_thresh})
            # Project injected spectrum on grid
            f_k_th = {'f_k_th_1': f_th_c[0](extra.lam_grid_c(0)),
                      'f_k_th_2': f_th_c[1](extra.lam_grid_c(1))}

            # Save values that do not need to be recomputed
            wv_range = [extra.lam_grid.min(), extra.lam_grid.max()]
            # Iterate on resolution of the tikhonov matrix
            for t_mat_n_os in t_mat_n_os_list:
                # Print status
                print(status.format(n_os, c_thresh, t_mat_n_os))

                # Generate a fake wv_map to cover all wv_range with a
                # resolution `t_mat_n_os` times the resolution of order 2.
                wv_map = grid_from_map(wv_2, P2, wv_range=wv_range)
                wv_map = oversample_grid(wv_map, n_os=t_mat_n_os)
                # Build convolution matrix
                conv_ord2 = get_c_matrix(WebbKer(wv_map[None, :]),
                                         extra.lam_grid, thresh=1e-5)
                # Build tikhonov matrix
                t_mat = conv_ord2 - identity(conv_ord2.shape[0])

                # Test factors
                test_conv = extra.get_tikho_tests(factors, t_mat=t_mat)

                # Save results
                file_name = path + file_root
                file_name += file_ext.format(n_os, c_thresh, t_mat_n_os)
                to_save = {**test_conv, **f_k_th, 'grid': extra.lam_grid}
                np.savez(file_name, **to_save)

if __name__ == '__main__':

    # Read relevant files
    path = "Ref_files/"
    wv_1 = fits.open(path + "wavelengths_m1.fits")[0].data
    wv_2 = fits.open(path + "wavelengths_m2.fits")[0].data
    T1 = fits.open(path + "trace_profile_m1.fits")[0].data.squeeze()
    T2 = fits.open(path + "trace_profile_m2.fits")[0].data.squeeze()

    # Convert to float (fits precision is 1e-8)
    wv_1 = wv_1.astype(float)
    wv_2 = wv_2.astype(float)
    T1 = T1.astype(float)
    T2 = T2.astype(float)

    # Normalised spatial profile
    P1 = T1 / T1.sum(axis=0)
    P1[np.isnan(P1)] = 0.
    P2 = T2 / T2.sum(axis=0)
    P2[np.isnan(P2)] = 0.

    # Generate grid for simulations
    lam_simu = get_soss_grid([P1, P2], [wv_1, wv_2], n_os=15)

    # Choose a small threshold for the spatial profile cut
    # (less than for a normal extraction)
    simu = TrpzOverlap([P1, P2], [wv_1, wv_2], lam_grid=lam_simu,
                       thresh=1e-8, c_kwargs={'thresh': 0.00001})

    # Sinus
    # Generate flux to inject
    flux = 1 + 0.5 * np.sin(lam_simu*500)
    # Multiplication by a fudge factor to get
    # a realistic number of counts on the detector
    flux *= 1e9

    # Inject
    scidata = simu.rebuild(flux)

    # Compute injected convolved flux for each orders
    f_th_c = [interp1d(simu.lam_grid_c(i_ord),
                       simu.c_list[i_ord].dot(flux),
                       kind='cubic', fill_value="extrapolate")
              for i_ord in range(2)]

    # Run tests
    file_root = 'sinus_clean'
    file_ext = '.n_os_{}.c_thresh_{:1.0e}.tikho_os_{}'
    path = ''

    # Parameters to test
    n_os_list = [1, 2, 4, 6, 8]
    c_thresh_list = [1e-3, 5e-4, 1e-4]
    t_mat_n_os_list = [1, 2, 3, 4]

    run_tikho_tests([P1, P2], [wv_1, wv_2], scidata, f_th_c,
                    n_os_list, c_thresh_list, t_mat_n_os_list,
                    file_root=file_root, file_ext=file_ext, path=path)

    print('Done')
