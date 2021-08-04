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
import box_kim
import SOSS.trace.tracepol as tp

# Matplotlib defaults
plt.rc('figure', figsize=(12,7))
plt.rc('font', size=14)
plt.rc('lines', lw=2)

def fit_resFunc(coeff, y, x):
    p = np.poly1d(coeff)
    return (p(y)) - x

WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/'

sys.path.insert(0, "/genesis/jwst/jwst-ref-soss/fortran_lib/")

# Read in all paths used to locate reference files and directories
config_paths_filename = os.path.join(WORKING_DIR, 'jwst-mtl_configpath_kim.txt')
pathPars = soss.paths()
soss.readpaths(config_paths_filename, pathPars)

# Create and read the simulation parameters
simuPars = spgen.ModelPars()              # Set up default parameters
simuPars = spgen.read_pars(pathPars.simulationparamfile, simuPars)   # Read in parameter file

# Position of trace for box extraction (TEMPORARY VERSION)
x, y, w = box_kim.readtrace(os=1)

###############################
# CHOOSE OVERSAMPLE  !!!
simuPars.noversample = 4
os = simuPars.noversample

###############################
clear_00 = fits.open('/genesis/jwst/userland-soss/loic_review/timeseries_20210730_normalizedPSFs/clear_trace_000000.fits')
#clear_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_000000.fits".format(os))
clear = np.empty(shape=(3, 256, 2048), dtype=float)
map_clear = np.empty_like(clear, dtype=float)

#padd = 10
padd = 100   # Because of x_padding and y_padding

for i in range(len(clear_00[0].data)):
    if os == 1:
        clear[i] = clear_00[0].data[i, padd:-padd, padd:-padd]
    else:
        clear_i = soss.rebin(clear_00[0].data[i], os, flux_method='sum')
        clear[i] = clear_i[padd:-padd, padd:-padd]
    map_clear[i] = box_kim.normalize_map(clear[i])

map_clear[1, :, 1790:] = 0  # Problem with end of order 2 trace

# Save map_profile
hdu = fits.PrimaryHDU(map_clear)
#hdu.writeto(WORKING_DIR + "new_map_profile_clear_{}.fits".format(os), overwrite=True)
hdu.writeto(WORKING_DIR + "new_map_profile_ref_clear_{}.fits".format(os), overwrite=True)


"""
#####################################################################
# New wavelength map
with fits.open(WORKING_DIR + "with_peaks/peaks_wl.fits") as hdulist:
    all_peaks = hdulist[0].data

with fits.open(WORKING_DIR + "tmp/with_peaks/clear_000000.fits") as hdulist:
    comb2D = hdulist[0].data
    xp = hdulist[0].header['xpadding']
    yp = hdulist[0].header['ypadding']
    comb2D = comb2D[:, yp:-yp, xp:-xp]

wavelength, order1_tilt = np.loadtxt('/genesis/jwst/userland-soss/loic_review/SOSS_wavelength_dependent_tilt_extrapolated.txt',
                                     unpack=True, usecols=(0, 1))   # [um], [degrees]

# Now we have the positions, we need to interpolate wavelength values over all pixels of the detector
wave_map2D = np.zeros_like(comb2D, dtype=float)
tilt_list = []
pk_wave_list = []

for order in [0, 1, 2]:
    if order == 0:
        x_max = 2048
        w_range = [0.8437, 2.833]   # Order 1
        j_min, j_max = 0, 153
        flux_threshold = 600
    elif order == 1:
        x_max = 1754
        w_range = [0.6, 1.423]   # Order 2
        j_min, j_max = 0, 255
        flux_threshold = 20
    elif order == 2:
        x_max = 1150
        w_range = [0.6, 0.956]   # Order 3
        j_min, j_max = 125, 255
        flux_threshold = 260
    n_rows = j_max - j_min +1

    # First collapse the image along y to get a first estimate of peak positions, and identify peaks
    v = comb2D[order, :, :x_max].sum(axis=0)

    # Find peaks
    pk_ind, _ = find_peaks(v, distance=18, prominence=(1000, None))

    # Identify peaks
    pk_wave = np.flip(all_peaks[(all_peaks > w_range[0]) & (all_peaks < w_range[1])])
    pk_wave_list.append(pk_wave)

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
            tmp_y = comb2D[order, j, i-4: i+5]   # subarray to use for fitting   # i-3: i+4
            tmp_x = x_arr[i-4: i+5]   # subarray to use for ftting   # i-3: i+4
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

    fig1, ax1 = plt.subplots(1, 1)
    plt.ylim(0, 255)
    if False:
        # Interpolation in 2D doesn't work very well!
        interp_func = interp2d(x_meas, y_meas, z_meas, kind='linear') #, fill_value=0)
        x = np.mgrid[:comb2D.shape[2]]
        y = np.mgrid[:comb2D.shape[1]]
        wave_map2D[order, :, :] = interp_func(x, y)
    if True:
        # Do the interpolation in 1D
        x = np.mgrid[:comb2D.shape[2]]

        new_y_meas = np.zeros(shape=(256, len(pk_wave)), dtype=float)
        new_x_meas = np.zeros(shape=(256, len(pk_wave)), dtype=float)    #zeros_like(x_meas)
        new_z_meas = np.zeros(shape=(256, len(pk_wave)), dtype=float)   #np.zeros_like(z_meas, dtype=float)

        tilt_list_ord = []

        for k in range(len(pk_wave)):   # Loop through each peak
            ind_z, = (z_meas == pk_wave[k]).nonzero()

            new_z_meas[:, k] = pk_wave[k]
            #y_min = y[pk_ind[k]] - 11
            #y_max = y[pk_ind[k]] + 11
            #y_poly = np.array(y_meas[ind_z])
            #x_poly = np.array(x_meas[ind_z])
            #y_poly = y_poly[((y_poly >= y_min) & (y_poly <= y_max))]
            #x_poly = x_poly[np.where((y_poly >= y_min) & (y_poly <= y_max))[0]]
            p1, p0 = box_kim.robust_polyfit(fit_resFunc, y_meas[ind_z], x_meas[ind_z], [-0.005, pk_ind[k]+2])   # -50, 60000
            #p1, p0 = box_kim.robust_polyfit(fit_resFunc, y_poly, x_poly, [-0.005, pk_ind[k] + 2])
            tilt = np.arctan(p1)
            tilt_list_ord.append(np.degrees(tilt))
            poly_fit = np.poly1d([p1, p0])
            fit_x_meas = poly_fit(y_meas[ind_z])   # y_poly
            #fit_y_meas = poly_fit(x_meas[ind_z])
            y_range = np.arange(256)
            new_x_meas[:, k] = poly_fit(y_range)
            #new_x_meas[:, k] = (y_range - p0) / p1
            new_y_meas[:, k] = y_range

            if k == 5:
                fig2, ax2 = plt.subplots(1, 1)
                ax2.plot(x_meas[ind_z], y_meas[ind_z], '+', markersize=3, label='peaks meas.')
                #ax2.plot(fit_x_meas, y_poly, color='r', label='fit')
                ax2.plot(fit_x_meas, y_meas[ind_z], color='r', label='fit')
                #ax2.plot(new_x_meas[:, k], y_range, '--', color='g', label='interp fit')
            #ax1.plot(x_poly, y_poly, '+', markersize=4, color='Blue')
        #ax1.plot(x_poly, y_poly, '+', markersize=4, color='Blue', label='meas')
        ax1.plot(x_meas, y_meas, '+', markersize=4, color='Blue', label='meas')
        ax1.plot(new_x_meas, new_y_meas, '+', markersize=2, color='HotPink')
        ax1.legend(), ax2.legend()

        for j in y_range:   # Loop through each row of image
        #for j in range(j_min, j_max+1):   # Loop through each row of image
            #ind_y, = (new_y_meas == j).nonzero()
            #if ind_y.size < 5:
             #   continue
            interp_func = interp1d(new_x_meas[j, :], new_z_meas[j, :], kind='linear', bounds_error=False,
                                   fill_value='extrapolate')
            wave_map2D[order, j, :] = interp_func(x)

    tilt_list.append(tilt_list_ord)

    # We could try to fit a polynomial as well instead of interpolating ???

    # wave_maps2D[order, :, :] *= (comb2D[order, :, :] > flux_threshold)

    plt.figure()
    #plt.plot(pk_ind, v[pk_ind], '.')
    plt.plot(pk_ind, pk_wave, '.')
    plt.title('Wavelengths of peaks')
    plt.ylabel(r"Wavelength [$\mu m$]")

    plt.figure()
    plt.plot(x_meas, y_meas, '+', color='MediumOrchid', markersize=3)
    plt.title('Positions of peaks')

    plt.figure()
    plt.imshow(wave_map2D[order], origin='lower', aspect='auto')
    plt.colorbar(label=r'Wavelength [$\mu m$]')

    plt.figure()
    plt.imshow(np.log10(comb2D[order, :, :]), origin='lower', cmap='gray', aspect='auto', vmin=2, vmax=6)
    plt.plot(x_meas, y_meas, 'r+', markersize=3)
    plt.xlim(400, 470)
    plt.ylim(0, 155)
plt.show()

# Save wave_map2D
hdu = fits.PrimaryHDU(wave_map2D)
hdu.writeto(WORKING_DIR + "with_peaks/wave_map2D.fits", overwrite=True)

plt.figure()
plt.plot(pk_wave_list[0], tilt_list[0], '.', color='Blue', label='tilts from fit')
plt.plot(wavelength, order1_tilt, lw=1, color='r', label='SOSS data')
plt.xlim(0.75, 2.80)
plt.ylabel('Tilt [degrees]'), plt.xlabel(r"Wavelength [$\mu m$]")
plt.title('Order 1 tilts')
plt.legend()
plt.savefig(WORKING_DIR + 'with_peaks/order1_tilts.png', overwrite=True)
plt.show()

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