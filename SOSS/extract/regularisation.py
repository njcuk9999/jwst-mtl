import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve


def finite_diff(x):
    """
    Returns the finite difference matrix operator based on x.
    Input:
        x: array-like
    Output: 
        sparse matrix. When apply to x `diff_matrix.dot(x)`,
        the result is the same as np.diff(x)
    """
    
    n_x = len(x)
    
    # Build matrix
    diff_matrix = diags([-1.], shape=(n_x-1, n_x))
    diff_matrix += diags([1.], 1, shape=(n_x-1, n_x))
    
    return diff_matrix

def finite_second_d(grid):
    """
    Returns the second derivative operator based on grid
    Inputs:
    -------
    grid: 1d array-like
        grid where the second derivative will be compute.
    Ouputs:
    -------
    second_d: matrix
        Operator to compute the second derivative, so that
        f" = second_d.dot(f), where f is a function
        projected on `grid`.
    """
    # Finite difference operator
    d_matrix = finite_diff(grid)

    # Delta lambda
    d_grid = d_matrix.dot(grid)

    # First derivative operator
    first_d = diags(1./d_grid).dot(d_matrix)

    # Second derivative operator
    second_d = finite_diff(grid[:-1]).dot(first_d)
    # don't forget the delta labda
    second_d = diags(1./d_grid[:-1]).dot(second_d)
    
    return second_d

def finite_first_d(grid):
    """
    Returns the first derivative operator based on grid
    Inputs:
    -------
    grid: 1d array-like, optional
        grid where the first derivative will be compute.
    Ouputs:
    -------
    first_d: matrix
        Operator to compute the second derivative, so that
        f' = first_d.dot(f), where f is a function
        projected on `grid`.
    """    
    # Finite difference operator
    d_matrix = finite_diff(grid)

    # Delta lambda
    d_grid = d_matrix.dot(grid)

    # First derivative operator
    first_d = diags(1./d_grid).dot(d_matrix)
    
    return first_d

def finite_zeroth_d(grid):
    """
    Gives the zeroth derivative operator on the function
    f(grid), so simply returns the identity matrix... XD
    """
    return identity(len(grid))

def tikho_solve(a_mat, b_vec, t_mat=None, grid=None,
                verbose=True, factor=1.0, estimate=None, index=None):
    """
    Tikhonov solver to use as a function instead of a class.
    """
    
    tikho = Tikhonov(a_mat, b_vec, t_mat=t_mat,
                     grid=grid, verbose=verbose, index=index)

    return tikho.solve(factor=factor, estimate=estimate)

    
class Tikhonov:
    """
    Tikhonov regularisation to solve the ill-condition problem:
    A.x = b, where A is accidently singular or close to singularity.
    Tikhonov regularisation adds a regularisation term in 
    the equation and aim to minimize the equation:
    ||A.x - b||^2 + ||gamma.x||^2
    Where gamma is the Tikhonov regularisation matrix.
    """
    default_mat = {'zeroth':finite_zeroth_d,
                   'first':finite_first_d,
                   'second':finite_second_d}
    
    def __init__(self, a_mat, b_vec, t_mat=None, grid=None, verbose=True, index=None):
        
        # Take the identity matrix as default (zeroth derivative)
        if t_mat is None:
            t_mat = 'zeroth'
            
        # b_vec will be passed to default_mat functions
        # if grid not given.
        if grid is None and t_mat is 'zeroth':
            grid = b_vec
        
        # If string, search in the default Tikhonov matrix
        if isinstance(t_mat, str):
            self.type = t_mat
            t_mat = self.default_mat[t_mat](grid)
        else:
            self.type = 'custom'
            
        # Take all indices by default
        if index is None:
            index = slice(None)

        self.a_mat = a_mat[index,:][:,index]
        self.b_vec = b_vec[index]
        self.t_mat = t_mat[index,:][:,index]
        self.verbose = verbose
        
    def solve(self, factor=1.0, estimate=None):
        """
        Minimize the equation ||A.x - b||^2 + ||gamma.x||^2
        by solving (A_T.A + gamma_T.gamma).x = A_T.b
        gamma is the Tikhonov matrix multiplied by a scale factor
        """
        # Get needed attributes
        a_mat = self.a_mat
        b_vec = self.b_vec
        t_mat = self.t_mat
        
        # Matrix gamma (with scale factor)
        gamma = factor * self.t_mat
        
        # Build system
        gamma_2 = (gamma.T).dot(gamma)  # Gamma square
        matrix = a_mat.T.dot(a_mat) + gamma_2
        result = (a_mat.T).dot(b_vec.T)
        # Include solution estimate if given
        if estimate is not None:
            result += gamma_2.dot(estimate.T)
        
        # Solve
        return spsolve(matrix, result)
    
    def test_factors(self, factors, estimate=None):
        
        self.v_print('Testing factors...')
        
        # Get relevant attributes
        b_vec = self.b_vec
        a_mat = self.a_mat
        t_mat = self.t_mat
        
        # Init outputs
        sln, err, reg = [], [], []
        # Test all factors
        for i_fac, factor in enumerate(factors):
            # Save solution
            sln.append(self.solve(factor, estimate))
            # Save error A.x - b
            err.append(a_mat.dot(sln[-1]) - b_vec)
            # Save regulatisation term
            reg.append(t_mat.dot(sln[-1]))
            # Print
            message = '{}/{}'.format(i_fac, len(factors))
            self.v_print(message, end='\r')
        # Final print
        self.v_print('{}/{}'.format(i_fac + 1, i_fac + 1))
        # Convert to arrays
        sln = np.array(sln)
        err = np.array(err)
        reg = np.array(reg)

        # Save in a dictionnary
        self.test = {'factors': factors,
                     'solution': sln,
                     'error': err,
                     'reg': reg}
        
        return self.test
        
    def _check_plot_inputs(self, fig, ax, label, factors, test):
        """
        Method to manage inputs for plots methods.
        """
        # Use ax or fig if given. Else, init the figure
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots(1, 1, sharex=True)
        elif ax is None:
            ax = fig.subplots(1, 1, sharex=True)
            
        # Use the type of regularisation as label if None is given
        if label is None:
            label = self.type
           
        if test is None:
            # Run tests with `factors` if not done already
            try:
                self.test
            except AttributeError:
                self.test_factors(factors)
            finally:
                test = self.test
            
        return fig, ax, label, test
    
    def error_plot(self, fig=None, ax=None, factors=None,
                   label=None, test=None, test_key=None, y_val=None):
        
        # Manage method's inputs
        args = (fig, ax, label, factors, test)
        fig, ax, label, test = self._check_plot_inputs(*args)
        
        # What y value do we plot?
        if y_val is None:
            # Use tests to plot y_val
            if test_key is None:
                # Default is euclidian norm of error.
                # Similar to the chi^2.
                y_val = (test['error']**2).sum(axis=-1)
            else:
                y_val = test[test_key]
        
        # Plot
        ax.loglog(test['factors'], y_val, label=label)
        
        # Mark minimum value
        i_min = np.argmin(y_val)
        min_coord = test['factors'][i_min], y_val[i_min]
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
        
    def l_plot(self, fig=None, ax=None, factors=None, label=None, test=None, text_label=True):
        
        # Manage method's inputs
        args = (fig, ax, label, factors, test)
        fig, ax, label, test = self._check_plot_inputs(*args)
    
        # Compute euclidian norm of error (||A.x - b||).
        # Similar to the chi^2.
        err_norm = (test['error']**2).sum(axis=-1)
        
        # Compute norm of regularisation term
        reg_norm = (test['reg']**2).sum(axis=-1)

        # Plot
        ax.loglog(err_norm, reg_norm, '.:', label=label)
        
        # Add factor values as text
        if text_label:
            for f, x, y in zip(test['factors'], err_norm, reg_norm):
                plt.text(x, y, "{:2.1e}".format(f), va="center", ha="right")
        
        # Legend
        ax.legend()
        
        # Labels
        xlabel = r'$\left(||\mathbf{Ax-b}||^2_2\right)$'
        ylabel = r'$\left(||\mathbf{\Gamma.x}||^2_2\right)$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax
    
    def v_print(self, *args, **kwargs):

        if self.verbose:
            print(*args, **kwargs)
