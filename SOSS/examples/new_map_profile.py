# Creates a map profile from clear_000000.fits image

import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from astropy.modeling import models, fitting
from scipy.interpolate import interp2d, interp1d
from scipy.optimize import least_squares

# Matplotlib defaults
plt.rc('figure', figsize=(12,7))
plt.rc('font', size=14)
plt.rc('lines', lw=2)

WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/'

sys.path.insert(0, "/genesis/jwst/jwst-ref-soss/fortran_lib/")

# Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_kim.txt')
pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              # Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars)   # Read in parameter file

"""
###############################
# CHOOSE OVERSAMPLE  !!!
simuPars.noversample = 11
os = simuPars.noversample
###############################


clear_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_000000.fits".format(os))
clear = np.empty(shape=(3, 256, 2048), dtype=float)
map_clear = np.empty_like(clear, dtype=float)

padd = 10   # Because of x_padding and y_padding

for i in range(len(clear_00[0].data)):
    if os == 1:
        clear[i] = clear_00[0].data[i, padd:-padd, padd:-padd]
    else:
        clear_i = soss.rebin(clear_00[0].data[i], os, flux_method='sum')
        clear[i] = clear_i[padd:-padd, padd:-padd]
    sum_col = np.sum(clear[i], axis=0)
    map_clear[i] = clear[i] / sum_col

map_clear[1, :, 1790:] = 0  # Problem with end of order 2 trace

# Save map_profile
#hdu = fits.PrimaryHDU(map_clear)
#hdu.writeto(WORKING_DIR + "new_map_profile_clear_{}.fits".format(os), overwrite=True)
"""

#####################################################################
# New wavelength map
with fits.open(WORKING_DIR + "with_peaks/oversampling_1/peaks_wl.fits") as hdulist:
    all_peaks = hdulist[0].data

with fits.open(WORKING_DIR + "tmp/with_peaks/oversampling_1/clear_000000.fits") as hdulist:
    comb2D = hdulist[0].data
    xp = hdulist[0].header['xpadding']
    yp = hdulist[0].header['ypadding']
    comb2D = comb2D[:, yp:-yp, xp:-xp]

# Now we have the positions, we need to interpolate wavelength values over all pixels of the detector
wave_map2D = np.zeros_like(comb2D, dtype=float)

for order in [0, 1, 2]:
    if order == 0:
        x_max = 2048
        w_range = [0.8437, 2.833]   # Order 1
        j_min, j_max = 0, 153
        flux_threshold = 150
    elif order == 1:
        x_max = 1754
        w_range = [0.6, 1.423]   # Order 2
        j_min, j_max = 0, 255
        flux_threshold = 10
    elif order == 2:
        x_max = 1150
        w_range = [0.6, 0.956]   # Order 3
        j_min, j_max = 125, 255
        flux_threshold = 250
    n_rows = j_max - j_min +1

    # First collapse the image along y to get a first estimate of peak positions, and identify peaks
    v = comb2D[order, :, :x_max].sum(axis=0)

    # Find peaks
    pk_ind, _ = find_peaks(v, distance=27, prominence=(1000, None))

    # Identify peaks
    pk_wave = np.flip(all_peaks[(all_peaks > w_range[0]) & (all_peaks < w_range[1])])

    if pk_ind.size != pk_wave.size:
        print('Problem in peak identification!!!')
        print('Do something !!!')
        break

    # For Gaussian fitting
    modFitter = fitting.LevMarLSQFitter()   # Model fitter
    model = models.Gaussian1D(stddev=1., bounds={'stddev': (0.5, 2.)}) + models.Linear1D(0, 0)
    x_arr = np.arange(comb2D.shape[2])

    # To store measured positions
    x_meas = np.zeros(pk_ind.size * n_rows, dtype=float)
    y_meas = np.zeros(pk_ind.size * n_rows, dtype=float)
    z_meas = np.zeros(pk_ind.size * n_rows, dtype=float)
    count = 0

    for j in range(j_min, j_max+1):   # Loop through each row of image
        print('\r...doing row {}'.format(j), end='', flush=True)
        # For this row, loop through each peak, try to fit a gaussian to find its center
        for i, w in zip(pk_ind, pk_wave):
            if comb2D[order, j, i] < flux_threshold:   # flux too low, not a valid peak
                continue
            tmp_y = comb2D[order, j, i-5: i+6]   # subarray to use for fitting   # i-3: i+4
            tmp_x = x_arr[i-5: i+6]   # subarray to use for ftting   # i-3: i+4
            model.amplitude_0 = tmp_y.max()
            model.mean_0.value = tmp_x[tmp_y.argmax()]
            model.mean_0.bounds = (i-2.5, i+2.5)
            fitted_model = modFitter(model, tmp_x, tmp_y, maxiter=200)
            if modFitter.fit_info['ierr'] in [1, 2, 3, 4] and fitted_model.amplitude_0.value > flux_threshold:
                # fit successful, add this position to the list of measured positions
                x_meas[count] = fitted_model.mean_0.value
                y_meas[count] = j
                z_meas[count] = w
                count += 1

    # Keep only good measurements
    x_meas = x_meas[:count]
    y_meas = y_meas[:count]
    z_meas = z_meas[:count]

    if False:
        # Interpolation in 2D doesn't work very well!
        interp_func = interp2d(x_meas, y_meas, z_meas, kind='linear') #, fill_value=0)
        x = np.mgrid[:comb2D.shape[2]]
        y = np.mgrid[:comb2D.shape[1]]
        wave_map2D[order, :, :] = interp_func(x, y)
    if True:
        # Do the interpolation in 1D, horizontally
        x = np.mgrid[:comb2D.shape[2]]

        for k in range(len(pk_wave)):
            ind, = (z_meas == pk_wave[k]).nonzero()

            def fit_resFunc(x, y, coeff):
                p = np.poly1d(coeff)
                return y - (p(x))

            p0, p1 = robust_polyfit(fit_resFunc, x_meas[ind], y_meas[ind], [-1000, 1000])
            new_p = np.poly1d([p0, p1])
            new_y_meas = new_p(x_meas[ind])

        #for j in range(j_min, j_max+1):   # Loop through each row of image
        ### T'ES RENDU LÀ
            #ind, = (y_meas == j).nonzero()
            #if ind.size < 5:
             #   continue
            interp_func = interp1d(x_meas[ind], z_meas[ind], kind='linear', bounds_error=False, fill_value='extrapolate')
            wave_map2D[order, j, :] = interp_func(x)

    # We could try to fit a polynomial as well instead of interpolating ???

    # wave_maps2D[order, :, :] *= (comb2D[order, :, :] > flux_threshold)

    plt.figure()
    plt.plot(pk_ind, v[pk_ind], '.')
    plt.show()

    plt.figure()
    plt.plot(x_meas, y_meas, '+', markersize=3)
    plt.vlines(pk_ind, j_min, j_max, lw=1, color='r', label='pk_ind')
    plt.legend()
    plt.show()

    plt.figure()
    plt.imshow(wave_map2D[order, :, :], origin='lower', aspect='auto')

    plt.figure()
    plt.imshow(np.log10(comb2D[order, :, :]), origin='lower', cmap='gray', aspect='auto', vmin=2, vmax=6)
    plt.plot(x_meas, y_meas, 'r+', markersize=3)
    plt.xlim(440, 470)
    plt.ylim(0, 155)
    plt.show()

# Save wave_map2D
hdu = fits.PrimaryHDU(wave_map2D)
hdu.writeto(WORKING_DIR + "with_peaks/oversampling_1/wave_map2D.fits".format(os), overwrite=True)

"""
# For verification of what going on for a specific peak fit
order = 1
j = 134
i = pk_ind[44]
v = comb2D[order, j, :]
tmp_y = v[i-3: i+4]
tmp_x = x_arr[i-3: i+4]
line = models.Linear1D()
model = models.Gaussian1D(tmp_y.max(), tmp_x[tmp_y.argmax()], 1, bounds={'mean':(i-3, i+3), 'stddev':(0.5, 2.)})
g_fitted = modFitter(model+line, tmp_x, tmp_y, maxiter=200)

plt.figure()
plt.plot(tmp_x, tmp_y)
plt.plot(tmp_x, g_fitted(tmp_x))
print(g_fitted.mean_0)
print(g_fitted.amplitude_0)
print(g_fitted.parameters)
print(modFitter.fit_info)
"""