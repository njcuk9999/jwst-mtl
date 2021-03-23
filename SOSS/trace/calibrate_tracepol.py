import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
from scipy import interpolate
from scipy.optimize import least_squares

import sys
sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/')

from SOSS.trace.contaminated_centroids import get_soss_centroids
import SOSS.trace.tracepol as tp

def get_CV3_tracepars(order=1):

    '''
    Return the polynomial fit of the spectral pixel --- wavelength
    relation, based on the few observed LED sources at Cryo Vacuum
    Campaign (CV3).
    :param order:
    :return: Set of polynomial coefficients for the spectral pixels
    versus the wavelength (in microns).
    '''

    if order == 1:
        # Measured positions of the laser light sources at CV3
        # (measured in the native pixel coordinates).
        spatpix = 256 - np.array([210,218,218,190])
        specpix = 2048 - np.array([239,499,745,1625])
        w = np.array([1.06262,1.30838,1.54676,2.410])   # microns
        # Fit the specpix vs wavelength
        param_spec = np.polyfit(w, specpix, 2)
        if False:
            wfit = np.linspace(0.8,2.9,200)
            xfit = np.polyval(param_spec, wfit)
            plt.figure(figsize=(10,5))
            plt.scatter(w, specpix)
            plt.plot(wfit, xfit)
            plt.show()

    if order == 2:
        # Measured positions of the laser light sources at CV3
        # (measured in the native pixel coordinates).
        spatpix = 256 - np.array([60,161,161])
        specpix = 2048 - np.array([395,1308,1823])
        w = np.array([0.6412, 1.06262,1.30838])   # microns
        # Fit the specpix vs wavelength
        param_spec = np.polyfit(w, specpix, 1)
        if False:
            wfit = np.linspace(0.6,1.4,200)
            xfit = np.polyval(param_spec, wfit)
            plt.figure(figsize=(10,5))
            plt.scatter(w, specpix)
            plt.plot(wfit, xfit)
            plt.show()

    if order == 3:
        # Measured positions of the laser light sources at CV3
        # (measured in the native pixel coordinates).
        ### WARNING - only one LED was actually observed:
        # lambda = 0.6412 at position 256-30 and 2048-1040.
        # Once we had the optics model calibrated, we used
        # it to determine that the trace covers from
        # 0.60378 to 0.95575 microns between specpix=1 and
        # specpix=1137. The slope is therefore:
        # d(lambda)/dx = -3.095602463e-4
        dldx = -3.095602463e-4
        # We can anchor from the one LED point
        xintercept = (2048-1040) - 0.6412/dldx
        # Fit the specpix vs wavelength, order 1
        param_spec = np.array([1.0/dldx, xintercept])

        if False:
            wfit = np.linspace(0.6,0.9,200)
            xfit = np.polyval(param_spec, wfit)
            plt.figure(figsize=(10,5))
            plt.scatter(w, specpix)
            plt.plot(wfit, xfit)
            plt.show()

    return param_spec


def CV3_wavelength_to_specpix(wavelength=None, order=1):

    '''
    Return the spectral pixel positions for the input wavelengths
    supplied, based on the CV3 solution.
    :param order: 1, 2, or 3
    :param wavelength: in microns, the wavelength array for which
    positions are requested.
    :return:
    '''

    # Get the CV3 specpix vs wavelength trace fit parameters as well
    # as the spatpix vs. specpix fit parameters.
    param_spec = get_CV3_tracepars(order=order)

    # wfit and xfit are for displaying the fit
    # Compute the spectral pixel and spatial pixel positions based on
    # the input wavelength.
    spectralpixel = np.polyval(param_spec, wavelength)

    return spectralpixel


def CV3_specpix_to_wavelength(specpix, order=1):

    # Get the CV3 specpix vs wavelength trace fit parameters
    param_spec = get_CV3_tracepars(order=order)

    # Generate a dense realization
    w = np.linspace(0.5,3.0,5001)
    x = np.polyval(param_spec, w)

    # Fit the other way around (w vs x)
    param = np.polyfit(x, w, 11)
    wavelength = np.polyval(param, specpix)

    return wavelength


def apply_calibration(param, x, y):
    '''
    The rotation+offset transformation when one wants to apply the
    best fit parameters.
    :param param: A length=3 array: angle, origin in x, origin in y
    :param x: x position of input
    :param y: y position of input
    :return: x_rot, y_rot: x,y positions after rotation
    '''

    theta = param[0]
    x0 = param[1]
    y0 = param[2]
    offsetx = 0  # param[3] degenerate with x0 and y0 - Do the maths - no need
    offsety = 0  # param[4] degenerate with x0 y0 - Do the maths - no need

    angle = np.deg2rad(theta)
    dx, dy = x - x0, y - y0
    x_rot = offsetx + np.cos(angle) * dx - np.sin(angle) * dy + x0
    y_rot = offsety + np.sin(angle) * dx + np.cos(angle) * dy + y0

    return x_rot, y_rot


def calibrate_tracepol():
    '''
    Calibrate the tracepol default rotation+offsets based on the CV3
    deep stack and the contaminated_centroid function.
    :return:
    '''

    # Optics model reference file
    optmodel = '/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv'

    # Read the CV3 deep stack and bad pixel mask
    bad = fits.getdata('/genesis/jwst/userland-soss/loic_review/badpix_DMS.fits')
    im = fits.getdata('/genesis/jwst/userland-soss/loic_review/stack_256_ng3_DMS.fits')

    # im is the dataframe, bad the bad pixel map
    badpix = np.zeros_like(bad, dtype='bool')
    badpix[~np.isfinite(bad)] = True

    # Measure the trace centroid position for the deep stack image.
    cv3 = get_soss_centroids(im, subarray='SUBSTRIP256', apex_order1=None,
                                    badpix=badpix, verbose=False)
    x_o1 = cv3['order 1']['X centroid']
    y_o1 = cv3['order 1']['Y centroid']
    pars_o1 = cv3['order 1']['poly coefs']
    x_o2 = cv3['order 2']['X centroid']
    y_o2 = cv3['order 2']['Y centroid']
    pars_o2 = cv3['order 2']['poly coefs']
    x_o3 = cv3['order 3']['X centroid']
    y_o3 = cv3['order 3']['Y centroid']
    pars_o3 = cv3['order 3']['poly coefs']

    # Wavelengths at which the measured traces and the
    # optics model traces are going to be compared for the fit.
    wavelength_o1 = np.linspace(0.9, 2.8, 50)
    wavelength_o2 = np.linspace(0.6, 1.4, 50)
    wavelength_o3 = np.linspace(0.6, 0.95, 50)

    # Calibrate in wavelength the measured traces and make a
    # realization of positions at a few selected wavelengths.
    # ORDER 1
    w_o1 = CV3_specpix_to_wavelength(x_o1, order=1)
    # Generate a transformation wavelength --> specpix
    # based on the measured x positions and calibrated wavelengths
    f_w2x = interpolate.interp1d(w_o1, x_o1)
    # Apply it for the few selected wavelengths for later fit
    x_obs_o1 = f_w2x(wavelength_o1)
    # Generate a transformation wavelength --> spatpix
    # based on the measured y positions and calibrated wavelengths
    f_w2y = interpolate.interp1d(w_o1, y_o1)
    # Apply the wavelength --> spatpix relation to a few points
    y_obs_o1 = f_w2y(wavelength_o1)

    # ORDER 2
    w_o2 = CV3_specpix_to_wavelength(x_o2, order=2)
    # Generate a transformation wavelength --> specpix
    # based on the measured x positions and calibrated wavelengths
    f_w2x = interpolate.interp1d(w_o2, x_o2)
    # Apply it for the few selected wavelengths for later fit
    x_obs_o2 = f_w2x(wavelength_o2)
    # Generate a transformation wavelength --> spatpix
    # based on the measured y positions and calibrated wavelengths
    f_w2y = interpolate.interp1d(w_o2, y_o2)
    # Apply the wavelength --> spatpix relation to a few points
    y_obs_o2 = f_w2y(wavelength_o2)

    # ORDER 3
    w_o3 = CV3_specpix_to_wavelength(x_o3, order=3)
    # Generate a transformation wavelength --> specpix
    f_w2x = interpolate.interp1d(w_o3, x_o3)
    # Apply it for the few selected wavelengths for later fit
    x_obs_o3 = f_w2x(wavelength_o3)
    # Generate a transformation wavelength --> spatpix
    # based on the measured y positions and calibrated wavelengths
    f_w2y = interpolate.interp1d(w_o3, y_o3)
    # Apply the wavelength --> spatpix relation to a few points
    y_obs_o3 = f_w2y(wavelength_o3)

    # For order 3, what does the model say about the wavelength
    # calibration x_fit_o3 vs w_fit_o3? Oh! But these 3 arrays
    # are badly calibrated. Do from the model instead.
    #print('Order 3 x, y, wavelength')
    #for i in range(np.size(w_o3)):
    #    print(x_o3[i], y_o3[i], w_o3[i])


    # Call tracepol's optics model then compute rotation offsets by
    # minimizing deviations to either only order 1 or all orders
    # simultaneously.
    # Call tracepol, disabling the default rotation, back to original
    # Optics Model. x/y_mod_N are realization of the model at a few
    # wavelengths.
    param = tp.get_tracepars(filename=optmodel, disable_rotation=False)

    x_mod_o1, y_mod_o1, mask_mod_o1 = tp.wavelength_to_pix(wavelength_o1,
                                               param, m=1,
                                               frame='dms',
                                               subarray='SUBSTRIP256',
                                               oversample=1)
    x_mod_o2, y_mod_o2, mask_mod_o2 = tp.wavelength_to_pix(wavelength_o2,
                                               param, m=2,
                                               frame='dms',
                                               subarray='SUBSTRIP256',
                                               oversample=1)
    x_mod_o3, y_mod_o3, mask_mod_o3 = tp.wavelength_to_pix(wavelength_o3,
                                               param, m=3,
                                               frame='dms',
                                               subarray='SUBSTRIP256',
                                               oversample=1)

    if False:
        # Check if it all makes sense
        plt.figure(figsize=(10,3))
        plt.scatter(x_mod_o1, y_mod_o1)
        plt.scatter(x_mod_o2, y_mod_o2)
        plt.scatter(x_mod_o3, y_mod_o3)
        plt.scatter(x_obs_o1, y_obs_o1)
        plt.scatter(x_obs_o2, y_obs_o2)
        plt.scatter(x_obs_o3, y_obs_o3)
        plt.show()

        plt.figure(figsize=(10,10))
        plt.plot(wavelength_o1, x_obs_o1)
        plt.plot(wavelength_o1, x_mod_o1)
        plt.show()

    # What orders should be used for fitting for rotation?
    if True:
        # Package the Orders 1 and 2 model points and observation points
        x_mod = np.concatenate((x_mod_o1, x_mod_o2), axis=None)
        y_mod = np.concatenate((y_mod_o2, y_mod_o2), axis=None)
        x_obs = np.concatenate((x_obs_o1, x_obs_o2), axis=None)
        y_obs = np.concatenate((y_obs_o1, y_obs_o2), axis=None)
        xy_obs = np.array([x_obs, y_obs])
    else:
        # Package the Orders 1 ONLY model points and observation points
        x_mod = np.copy(x_mod_o1)
        y_mod = np.copy(y_mod_o2)
        x_obs = np.copy(x_obs_o1)
        y_obs = np.copy(y_obs_o1)
        xy_obs = np.array([x_obs, y_obs])

    # These 2 functions need to be there in the code because they use
    # variables declared outside.
    def fmodel(param):
        # That is the transformation matrix coded to perform the fit
        # Note that the x_mod and y_mod are outside variables that
        # need to be declared before for this to work.

        theta = param[0]
        x0 = param[1]
        y0 = param[2]

        angle = np.deg2rad(theta)
        dx, dy = x_mod - x0, y_mod - y0 #x_mod and y_mod are global variable
        x_rot = np.cos(angle) * dx - np.sin(angle) * dy + x0
        y_rot = np.sin(angle) * dx + np.cos(angle) * dy + y0

        return np.array([x_rot, y_rot])

    def f2minimize(param):
        # Minimize the difference between observations and model points
        return (xy_obs - fmodel(param)).flatten()

    # Informed guess for origin is the CLEAR sweet spot: in DMS coords: x,y=(2048-100),(256-850)=1948,-596
    param_guess = [-1.3868425075, 1577.9020186702, -1109.1909267381]
    res2 = least_squares(f2minimize, param_guess, ftol=1e-12)
    #bounds=([-np.inf,-np.inf,-np.inf,-0.0001,-0.0001],[np.inf,np.inf,np.inf,0,0])) - no need Do the maths
    param_bestfit = res2.x

    # OVERRIDE !!!
    # Best fit obtained with prior version of this script (cv3_calibration)
    param_bestfit = [-1.3868425075, 1577.9020186702, -1109.1909267381]
    res2.x = param_bestfit
    # TODO: Find the bug that makes the fit be wrong.

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
    print('get_tracepars(filename=None, origin=np.array([{:}, {:}]),'.format(256-res2.x[2], 2048-res2.x[1]))
    print('              angle={:},'.format(-res2.x[0]))
    print('              disable_rotation=False):')

    # Check that the rotated points overplot the observations
    x_fit_o1, y_fit_o1 = apply_calibration(param_bestfit, x_mod_o1, y_mod_o1)
    x_fit_o2, y_fit_o2 = apply_calibration(param_bestfit, x_mod_o2, y_mod_o2)
    x_fit_o3, y_fit_o3 = apply_calibration(param_bestfit, x_mod_o3, y_mod_o3)




    please_check = True

    if please_check:

        # Figure to show the positions for all 3 orders
        plt.figure(figsize=(10, 10))
        plt.xlim((0, 2048))
        plt.ylim((0, 256))
        plt.imshow(np.log10(im), vmin=0.7, vmax=3, origin='lower', aspect='auto')

        plt.plot(x_o1, y_o1, color='orange', label='Order 1 - CV3')
        plt.plot(x_mod_o1, y_mod_o1, linestyle='dashed', color='orange', label='Order 1 - Model')
        plt.scatter(x_fit_o1, y_fit_o1, color='orange', label='Order 1 - Model rotated')

        if x_o2 is not None:
            plt.plot(x_o2, y_o2, color='black', label='Order 2 - CV3')
            plt.plot(x_mod_o2, y_mod_o2, linestyle='dashed', color='black', label='Order 2 - Model')
            plt.scatter(x_fit_o2, y_fit_o2, color='black', label='Order 2 - Model rotated')

        if x_o3 is not None:
            plt.plot(x_o3, y_o3, color='red', label='Order 3 -CV3')
            plt.plot(x_mod_o3, y_mod_o3, linestyle='dashed', color='red', label='Order 3 - Model')
            plt.scatter(x_fit_o3, y_fit_o3, color='red', label='Order 3 - Model rotated')

        plt.legend()
        plt.show()

    # TODO: Add a measurement of the quality of the fit. i.e. the pixel
    # position rms deviations between rotated optics model and
    # observations.



#run
calibrate_tracepol()