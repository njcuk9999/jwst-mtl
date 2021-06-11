import matplotlib.pyplot as plt
import numpy as np

def resample_models(dw, starmodel_wv, starmodel_flux, ld_coeff, planetmodel_wv, planetmodel_rprs, pars, tracePars,
                    simuPars, gridtype = 'planet', wavelength_start = None, wavelength_end = None,
                    resolving_power = None, dispersion = None) :
    """Resamples star and planet model onto common grid.

    Usage:
    bin_starmodel_wv,bin_starmodel_flux,bin_ld_coeff,bin_planetmodel_wv,bin_planetmodel_rprs\
      =resample_models(dw,starmodel_wv,starmodel_flux,ld_coeff,\
      planetmodel_wv,planetmodel_rprs,pars,tracePars)

      Inputs:
        dw - wavelength spacing.  This should be calculated using get_dw
        starmodel_wv - stellar model wavelength array
        starmodel_flux - stellar model flux array
        ld_coeff - non-linear limb-darkening coefficients array
        planetmodel_wv - planet model wavelength array
        planetmodel_rprs - planet model Rp/R* array
        pars - model parameters
        tracePars - trace solution

      Output:
        bin_starmodel_wv - binned star wavelength array
        bin_starmodel_flux - binned star model array
        bin_ld_coeff - binned limb-darkening array
        bin_planetmodel_wv - binned planet wavelength array (should be same size as bin_starmodel_wv)
        bin_planetmodel_rprs - binned Rp/R* array
    """
    # TODO: Remove the following once dw becomes an option after testing
    if dispersion is None: dispersion = dw
    # Define the grid over which to resample models
    if gridtype == 'planet':
        # Resample on the same grid as the planet model grid
        x_grid = np.copy(planetmodel_wv)
        dx_grid = np.zeros_like(x_grid)
        dx_grid[1:] = np.abs(x_grid[1:] - x_grid[0:-1])
        dx_grid[0] = dx_grid[1]
        # Resample star model and limb darkening matrix. But leave the planet model unchanged.
        bin_starmodel_flux = bin_array(starmodel_wv, starmodel_flux, x_grid, dx_grid)
        bin_starmodel_wv = np.copy(x_grid)
        bin_planetmodel_wv = np.copy(x_grid)
        bin_planetmodel_rprs = np.copy(planetmodel_rprs)
        bin_ld_coeff = bin_limb_darkening(starmodel_wv, ld_coeff, x_grid, dx_grid)
    elif gridtype == 'constant_dispersion':
        # Resample on a constant dispersion grid
        # wavelength_start, wavelength_end = 5000, 55000
        # TODO: Check that wavelength_start/end are passed as input
        nsample = 1 + int((wavelength_end - wavelength_start) / dispersion)
        x_grid = np.arange(nsample) * dispersion + wavelength_start
        dx_grid = np.ones_like(x_grid) * dispersion
        # Resample the star model, the planet model as well as the LD coefficients.
        bin_starmodel_flux = bin_array(starmodel_wv, starmodel_flux, x_grid, dx_grid)
        bin_starmodel_wv = np.copy(x_grid)
        bin_planetmodel_wv = np.copy(x_grid)
        bin_planetmodel_rprs = bin_array(planetmodel_wv, planetmodel_rprs, x_grid, dx_grid)
        bin_ld_coeff = bin_limb_darkening(starmodel_wv, ld_coeff, x_grid, dx_grid)
    elif gridtype == 'constant_R':
        # Resample on a constant resolving power grid
        # TODO: Check that resolving+power is passed as input
        # x_grid, dx_grid = constantR_samples(wavelength_start, wavelength_end, resolving_power=resolving_power)
        x_grid = np.copy(planetmodel_wv)    # !!!
        dx_grid = np.zeros_like(x_grid)    # !!!
        dx_grid[1:] = np.abs(x_grid[1:] - x_grid[0:-1])   # !!!
        dx_grid[0] = dx_grid[1]   # !!!
        # Resample the star model, the planet model as well as the LD coefficients.
        bin_starmodel_flux = bin_array_conv(starmodel_wv, starmodel_flux, x_grid, dx_grid)
        bin_starmodel_wv = np.copy(planetmodel_wv)
        bin_planetmodel_wv = np.copy(planetmodel_wv)
        bin_planetmodel_rprs = np.copy(planetmodel_rprs)
        bin_ld_coeff = bin_limb_darkening(starmodel_wv, ld_coeff, x_grid, dx_grid)
    else:
        print('Possible resample_models gridtype are: planet, constant_dispersion or constant_R.')
        sys.exit()

    """
    # To get constant dispersion
    star_grid = np.copy(starmodel_wv)
    dstar_grid = np.zeros_like(starmodel_wv)
    dstar_grid[1:] = np.abs(star_grid[1:] - star_grid[0:-1])
    dstar_grid[0] = dstar_grid[1]
    new_dstar = np.min(dstar_grid)  # Plus petit pas
    new_star_grid = np.arange(np.min(starmodel_wv),np.max(starmodel_wv)+new_dstar, new_dstar)  #New array de wv
    new_star_flux = np.interp(new_star_grid, star_grid, starmodel_flux)  #New flux interpolated with new_dwv
    # The points of starmodel_flux are now equally spaced in order to do the convolution.
    """

    # Resample on the same grid as the planet model grid


    plt.figure()
    plt.plot(starmodel_wv, starmodel_flux, color='b')
    plt.plot(bin_starmodel_wv, bin_starmodel_flux, ls='--', color='r')
    plt.show()
    return bin_starmodel_wv, bin_starmodel_flux, bin_ld_coeff, bin_planetmodel_wv, bin_planetmodel_rprs

def bin_array_conv(starmodel_wv, starmodel_flux, planetmodel_wv, dx_grid):
    # For R constant (both for the planet and the star)
    R_s = np.mean(starmodel_wv[:-1] / np.diff(starmodel_wv))
    R_p = np.mean(planetmodel_wv[:-1] / np.diff(planetmodel_wv))
    fwhm = R_s / R_p    # FWHM
    sig = fwhm / np.sqrt(8*np.log(2))  # Convert FWHM to sigma for gaussian
    length = int(5 * fwhm)   # Length of gaussian array
    if length%2 == 0:
        length += 1
    elif length%2 == 1:
        None

    x_gauss = np.mgrid[-length//2 : length//2+1]   # x grid for gaussian
    gauss = np.exp(-x_gauss ** 2 / 2 / sig ** 2)   # Gaussian
    gauss = gauss / np.sum(gauss)   # Normalization of gaussian

    conv_star_flux = np.convolve(starmodel_flux, gauss, mode='same')

    return np.interp(x_grid, starmodel_wv, conv_star_flux)

def bin_limb_darkening(x, ld_coeff, x_grid, dx_grid):
    '''
    Bin down each parameter of the limb darkening coefficient
    on the new x_grid using the function bin_array()
    :param x:
    :param ld_coeff:
    :param x_grid:
    :param dx_grid:
    :return:
    '''
    # Shape of the ld_coeff input: (200001,4)
    # Shape of the output bin_ld_coeff: (n,4)
    bin_ld_coeff = np.zeros((len(x_grid), 4))
    for i in range(4):
        param_i = ld_coeff[:,i]
        bin_param_i = bin_array(x, param_i, x_grid, dx_grid)
        bin_ld_coeff[:,i] = np.copy(bin_param_i)
    return bin_ld_coeff

def bin_array(x, fx, x_grid, dx_grid):
    '''
    :param x:
    :param fx: represents the array values at each x sample
    :param x_grid: represents the sample center of the binned array
    :param dx_grid: represents the sample width of the bined array
    :return:
    '''
    # Arrays of the starting and ending x_grid values for each grid sample
    xgrid_left = x_grid - dx_grid/2
    xgrid_right = x_grid + dx_grid/2
    # Arrays of resampled fx at the position of the left and right x_grid
    fx_resampleft = np.interp(xgrid_left, x, fx)
    fx_resampright = np.interp(xgrid_right, x, fx)
    # Initialize the binned flux output array
    fx_grid = np.zeros_like(x_grid)
    # Sum up each original bin (fx * dx)
    grid_nsample = len(x_grid)
    for i in range(grid_nsample):
        print(i, grid_nsample)
        sum = 0.0
        # Original sample points fully included in new grid current bin
        ind = np.where((x >= xgrid_left[i]) & (x <= xgrid_right[i]))[0]
        # Integral for those full original samples
        for j in range(len(ind)-1):
            sum = sum + (fx[ind[j+1]] + fx[ind[j]])/2 * (x[ind[j+1]] - x[ind[j]])
        # Partial sample on the left
        sum = sum + (fx[ind[0]]+fx_resampleft[i])/2 * (x[ind[0]] - xgrid_left[i])
        # Partial sample on the right
        sum = sum + (fx[ind[-1]]+fx_resampright[i])/2 * (xgrid_right[i] - x[ind[-1]])
        #To get fx_grid, need to divide sum by the width of that binned grid
        fx_grid[i] = sum / dx_grid[i]
    return fx_grid