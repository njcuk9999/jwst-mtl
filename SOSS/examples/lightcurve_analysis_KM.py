import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from astropy.io import fits

import utils_KM as utils

import juliet

# Extracted spectra directory
spec_dir = '/home/kmorel/ongenesis/jwst-user-soss/master/WASP_96/pipeline_outputs_directory/Stage3/'
spec_filename = spec_dir + 'WASP-96_box_spectra_fullres.fits'

# Integrations of ingress and egress.
baseline_ints = [107, -70]


with fits.open(spec_filename) as hdulist:
    # Order 1
    print('Data = ', hdulist[3].header['EXTNAME'])
    spec_ord1 = hdulist[3].data   # Extracted flux [time, wl]
    err_ord1 = hdulist[4].data    # Error
    # Convert data from fits files to float (fits precision is 1e-8)
    spec_ord1 = spec_ord1.astype('float64', copy=False)
    err_ord1 = err_ord1.astype('float64', copy=False)

    # Order 2
    print('Data = ', hdulist[7].header['EXTNAME'])
    spec_ord2= hdulist[7].data   # Extracted flux [time, wl]
    err_ord2 = hdulist[8].data    # Error
    # Convert data from fits files to float (fits precision is 1e-8)
    spec_ord2 = spec_ord2.astype('float64', copy=False)
    err_ord2 = err_ord2.astype('float64', copy=False)

    # Time
    # print('Data = ', hdulist[-1].header['EXTNAME'])
    # time = hdulist[-1].data  # Time
    # print(np.shape(time))
    # # Convert data from fits files to float (fits precision is 1e-8)
    # time = time.astype('float64', copy=False)

# White light curve, order 1
wlc = np.sum(spec_ord1, axis=1)
# Normalization
wlc_norm = utils.normalization(wlc, baseline_ints, 'transit')

plt.figure()
plt.plot(time, wlc_norm)
# plt.savefig(spec_dir + 'wlc_norm')


#------------- Transit fit -----------------
# Put data arrays into dictionaries so we can fit it with juliet
times, fluxes_ord1, fluxes_err_ord1 = {},{},{}
times['SOSS'], fluxes_ord1['SOSS'], fluxes_err_ord1['SOSS'] = time, spec_ord1, err_ord1

priors = {}

# # Name of the parameters to be fit:
# params = ['P_p1','t0_p1','r1_p1','r2_p1','q1_TESS','q2_TESS','a_p1','ecc_p1','omega_p1',\
#               'rho', 'mdilution_TESS', 'mflux_TESS', 'sigma_w_TESS']
#
# # Distributions:
# dists = ['fixed','normal','uniform','uniform','uniform','uniform','fixed','fixed','fixed',\
#                  'loguniform', 'fixed', 'normal', 'loguniform']
#
# # Hyperparameters
# hyperps = [3.4252602, [1358.4,0.1], [0.119,1], [0.,1.], [0., 1.], [0., 1.], 8.84, 0.0, 90.,\
#                    [100., 10000.], 1.0, [0.,0.1], [0.1, 1000.]]
# P_p1 OK, ecc_p1 OK, a_p1 OK
#
# # Populate the priors dictionary:
# for param, dist, hyperp in zip(params, dists, hyperps):
#     priors[param] = {}
#     priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp