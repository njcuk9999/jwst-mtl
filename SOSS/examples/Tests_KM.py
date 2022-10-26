import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import sys
import os

import utils_KM as utils

plt.rc('figure', figsize=(11,8))

# Extracted spectra directory
spec_dir = '/home/kmorel/ongenesis/jwst-user-soss/master/WASP_96/pipeline_outputs_directory/Stage3/'
spec_filename = spec_dir + 'WASP-96_box_spectra_fullres.fits'

# Integrations of ingress and egress.
baseline_ints = [107, -70]
# Integrations of in-transit
transit_ints = [130, 187]

occultation_type = 'transit'

with fits.open(spec_filename) as hdulist:
    # Order 1
    print('Data = ', hdulist[3].header['EXTNAME'])
    spec_ord1 = hdulist[3].data   # Extracted flux [time, wl]
    err_ord1 = hdulist[4].data    # Error
    # Convert data from fits files to float (fits precision is 1e-8)
    spec_ord1 = spec_ord1.astype('float64', copy=False)
    err_ord1 = err_ord1.astype('float64', copy=False)

    print('Data = ', hdulist[1].header['EXTNAME'])
    wl_low_ord1 = hdulist[1].data   # Order 1 wavelength bin lower limits
    print('Data = ', hdulist[2].header['EXTNAME'])
    wl_up_ord1 = hdulist[2].data  # Order 1 wavelength bin upper limits
    # Convert data from fits files to float (fits precision is 1e-8)
    wl_low_ord1 = wl_low_ord1.astype('float64', copy=False)
    wl_up_ord1 = wl_up_ord1.astype('float64', copy=False)
    wl_ord1 = np.zeros(shape=(2, wl_low_ord1.shape[0], wl_low_ord1.shape[1]), dtype=float)
    wl_ord1[0] = wl_low_ord1
    wl_ord1[1] = wl_up_ord1
    wl_ord1 = np.mean(wl_ord1, axis=0)

    # Order 2
    print('Data = ', hdulist[7].header['EXTNAME'])
    spec_ord2= hdulist[7].data   # Extracted flux [time, wl]
    err_ord2 = hdulist[8].data    # Error
    # Convert data from fits files to float (fits precision is 1e-8)
    spec_ord2 = spec_ord2.astype('float64', copy=False)
    err_ord2 = err_ord2.astype('float64', copy=False)

    print('Data = ', hdulist[5].header['EXTNAME'])
    wl_low_ord2 = hdulist[5].data   # Order 2 wavelength bin lower limits
    print('Data = ', hdulist[6].header['EXTNAME'])
    wl_up_ord2 = hdulist[6].data  # Order 2 wavelength bin upper limits
    # Convert data from fits files to float (fits precision is 1e-8)
    wl_low_ord2 = wl_low_ord2.astype('float64', copy=False)
    wl_up_ord2 = wl_up_ord2.astype('float64', copy=False)
    wl_ord2 = np.zeros(shape=(2, wl_low_ord2.shape[0], wl_low_ord2.shape[1]), dtype=float)
    wl_ord2[0] = wl_low_ord2
    wl_ord2[1] = wl_up_ord2
    wl_ord2 = np.mean(wl_ord2, axis=0)

#Normalization
spec_ord1_norm = utils.normalization(spec_ord1, baseline_ints, occultation_type)
spec_ord2_norm = utils.normalization(spec_ord2, baseline_ints, occultation_type)

plt.figure()
l = 1000
plt.scatter(np.arange(len(spec_ord1_norm[:,l])), spec_ord1_norm[:,l], s=3, label=r'$\lambda$={}'.format(wl_ord1[0,l]),
            color='b')
plt.xlabel('Time')
plt.legend()
plt.savefig(spec_dir + 'spec_ord1_norm')

plt.figure()
l=1000
plt.scatter(np.arange(len(spec_ord2_norm[:,l])), spec_ord2_norm[:,l], s=3, label=r'$\lambda$={}'.format(wl_ord2[0,l]),
            color='b')
plt.xlabel('Time')
plt.legend()
plt.savefig(spec_dir + 'spec_ord2_norm')

# White light curve, order 1
wlc_ord1 = np.sum(spec_ord1, axis=1)
err_wlc_ord1 = np.sum(err_ord1, axis=1)
# Normalization
wlc_norm_ord1 = utils.normalization(wlc_ord1, baseline_ints, 'transit')

plt.figure()
plt.errorbar(np.arange(len(wlc_ord1)), wlc_ord1, yerr=err_wlc_ord1, color='b')
# plt.scatter(np.arange(len(wlc_ord1)), wlc_ord1, s=3, color='k')
plt.xlabel('Time')
plt.title("Whitelight curve")
plt.savefig(spec_dir + 'wlc_norm_ord1_test')

# Dispersion
out_frames = utils.format_out_frames(baseline_ints, occultation_type)
spec_ord1_out_frames = spec_ord1[out_frames]
dispersion_ord1 = np.std(spec_ord1_out_frames, axis=0)

plt.figure()
plt.scatter(np.mean(wl_ord1,axis=0), dispersion_ord1)
plt.xlabel(r"Wavelength [$\mu m$]")
plt.ylabel('Dispersion')
plt.savefig(spec_dir + 'dispersion_ord1')

# Transit curve
transit_curve = utils.transit_depth(spec_ord1_norm, baseline_ints, transit_ints, occultation_type)

plt.figure()
plt.scatter(np.mean(wl_ord1,axis=0), transit_curve*1e6, s=3, color='b')
plt.xlabel(r"Wavelength [$\mu m$]")
plt.ylabel(r'($R_p)/R_s)^2$ [ppm]')
plt.title('Transit spectrum')
plt.savefig(spec_dir + 'transit_curve')