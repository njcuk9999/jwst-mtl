# cv3_calibration is a collection of functions that pertain to the wavelength calibration.

import sys
sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/SOSS/')
sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/SOSS/trace/')
sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/SOSS/specgen/')
sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/SOSS/detector/')
sys.path.insert(0, '/genesis/jwst/jwst-ref-soss/fortran_lib/')
import numpy as np
import matplotlib.pyplot as plt
import tracepol as tp
import get_uncontam_centroids_edgetrig as centroid
from astropy.io import fits
from scipy import interpolate
from astropy.table import Table
from astropy.io import ascii
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import os.path


def get_tracepars():

    # Measured positions of the laser light sources at CV3 (measured in the native pixel coordinates)
    spatpix = 256 - np.array([210,218,218,190])
    specpix = 2048 - np.array([239,499,745,1625])
    w = np.array([1.06262,1.30838,1.54676,2.410])   # microns

    # Fit the specpix vs wavelength
    param_spec = np.polyfit(w, specpix, 2)

    return param_spec


def specpix_to_wavelength(specpix):

    pars = get_tracepars()

    # Generate a dense realization
    w = np.linspace(0.5,3.0,5001)
    x = np.polyval(pars, w)

    # Fit the other way around (w vs x)
    param = np.polyfit(x, w, 11)
    wavelength = np.polyval(param, specpix)

    return wavelength




def wavelength_to_pixels(wavelength=None):

    # Measured positions of the laser light sources at CV3
    spatpix = 256 - np.array([210,218,218,190])
    specpix = 2048 - np.array([239,499,745,1625])
    w = np.array([1.06262,1.30838,1.54676,2.410])

    if wavelength is None:
        return specpix, spatpix

    # Fit the specpix vs wavelength
    param_spec = np.polyfit(w, specpix, 2)
    # w and x are the model values for those parameters
    x = np.polyval(param_spec, w)
    # wfit and xfit are for displaying the fit
    wfit = np.linspace(0.6,2.8,23)
    xfit = np.polyval(param_spec, wfit)
    #print(param_spec)

    # Fit the spatpix vs specpix
    param_spat = np.polyfit(specpix, spatpix, 2)
    y = np.polyval(param_spat, x)
    # yfit is for display purpose
    yfit = np.polyval(param_spat, xfit)
    #print(param_spat)

    if False:
        plt.figure(figsize=(10,3))
        #plt.scatter(wave, specpix, label='Positions of the lasers')
        plt.plot(wfit, xfit-xfit, label='Fit - order 2')
        plt.scatter(w, x-specpix, label='residuals')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,3))
        plt.scatter(x, y, label='Position of the lasers')
        plt.plot(xfit, yfit, label='Fit')
        plt.legend()
        plt.show()

    # Use fits for the input wavelengths
    spectralpixel = np.polyval(param_spec, wavelength)
    spatialpixel = np.polyval(param_spat, spectralpixel)

    return spectralpixel, spatialpixel


def write_measurements(filename=None):
    # Measures the Order 1 trace position using the edge trigger algorithm
    # as well as calls the optics table trace positions and write those on
    # disk. That will then be used by get_calibration to fit the best
    # transformation model to bring both into agreement.

    wavelength = np.linspace(0.9,2.8,2*19+1)

    # Call tracepol, disabling the default rotation, back to original Optics Model
    param = tp.get_tracepars(filename='/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv', disable_rotation=True)
    x_om, y_om, mask_om = tp.wavelength_to_pix(wavelength, param, m=1, frame='dms', subarray='SUBSTRIP256', oversample=1)

    # Call get_edge_centroids() on CV3 deep stack
    a = fits.open('/genesis/jwst/userland-soss/loic_review/stack_256_ng3_DMS.fits')
    im = a[0].data
    x_edg, y_edg = centroid.get_edge_centroids(im, return_what='edgecomb_xy', polynomial_order=10, verbose=False)

    # Calibrate EDGE in wavelength from CV3
    w_edg = specpix_to_wavelength(x_edg)
    f_w2x = interpolate.interp1d(w_edg, x_edg)
    x_edg_calib = f_w2x(wavelength)
    f_w2y = interpolate.interp1d(w_edg, y_edg)
    y_edg_calib = f_w2y(wavelength)

    data = Table([wavelength, x_om, y_om, x_edg_calib, y_edg_calib], \
                 names=['wavelength', 'x_model', 'y_model', 'x_cv3', 'y_cv3'])
    if filename is None: filename = '/genesis/jwst/userland-soss/loic_review/trace_positions.txt'
    ascii.write(data, filename, overwrite=True)

    return filename



def generate_calibration():

    filename = '/genesis/jwst/userland-soss/loic_review/trace_positions.txt'
    if os.path.exists(filename) == False: tmp = write_measurements(filename=filename)
    a = ascii.read(filename)

    x_mod = np.array(a['x_model'])
    y_mod = np.array(a['y_model'])
    x_obs = np.array(a['x_cv3'])
    y_obs = np.array(a['y_cv3'])

    def fmodel(param):

        theta = param[0]
        x0 = param[1]
        y0 = param[2]
        offsetx = 0  # param[3] degenerate with x0 and y0 - Do the maths - no need
        offsety = 0  # param[4] degenerate with x0 y0 - Do the maths - no need

        angle = np.deg2rad(theta)
        dx, dy = x_mod - x0, y_mod - y0
        x_rot = offsetx + np.cos(angle) * dx - np.sin(angle) * dy + x0
        y_rot = offsety + np.sin(angle) * dx + np.cos(angle) * dy + y0

        return np.array([x_rot, y_rot])

    xy_obs = np.array([x_obs, y_obs])

    def f2minimize(param):
        return (xy_obs - fmodel(param)).flatten()

    # Informed guess for origin is the CLEAR sweet spot: in DMS coords: x,y=(2048-100),(256-850)=1948,-596
    param_guess = [-1.39, 1948, -596]# 0,0]
    res2 = least_squares(f2minimize, param_guess, ftol=1e-12)
    #bounds=([-np.inf,-np.inf,-np.inf,-0.0001,-0.0001],[np.inf,np.inf,np.inf,0,0])) - no need Do the maths

    print(res2)
    print('cost = {:}'.format(res2.cost))
    print('Best fit parameters (in DMS coordinates):')
    print('theta = {:15.10f}'.format(res2.x[0]))
    print('origin_x = {:15.10f}'.format(res2.x[1]))
    print('origin_y = {:15.10f}'.format(res2.x[2]))
    #print('offset_x = {:15.10f}'.format(res2.x[3]))
    #print('offset_y = {:15.10f}'.format(res2.x[4]))
    print()
    print('Best fit parameters (in native (ds9) coordinates):')
    print('theta = {:15.10f}'.format(-res2.x[0]))
    print('origin_x = {:15.10f}'.format(256-res2.x[2]))
    print('origin_y = {:15.10f}'.format(2048-res2.x[1]))
    #print('offset_x = {:15.10f}'.format(-res2.x[4]))
    #print('offset_y = {:15.10f}'.format(-res2.x[3]))
    print()
    print('Once converted to native (aka ds9) pixel coordinates used by tracepol.py,')
    print('this becomes:')
    print('get_tracepars(filename=None, origin=np.array([256 - origin_y, 2048 - origin_x]),')
    print('              angle=-theta,')
    print('              disable_rotation=False):')

    return res2.x

# RUN!

a = generate_calibration()

