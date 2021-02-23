import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt

import sys
sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/')

from SOSS.extract import soss_read_refs
from SOSS.dms import soss_centroids as cen
from SOSS.trace.contaminated_centroids import get_soss_centroids
import SOSS.trace.tracepol as tp

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

    # Example for the call
    cv3 = get_soss_centroids(im, subarray='SUBSTRIP256', apex_order1=None,
                                    badpix=badpix, verbose=False)
    x_o1 = cv3['order 1']['X centroid']
    y_o1 = cv3['order 1']['Y centroid']
    w_o1 = cv3['order 1']['trace widths']
    x_o2 = cv3['order 2']['X centroid']
    y_o2 = cv3['order 2']['Y centroid']
    w_o2 = cv3['order 2']['trace widths']
    x_o3 = cv3['order 3']['X centroid']
    y_o3 = cv3['order 3']['Y centroid']
    w_o3 = cv3['order 3']['trace widths']



    # Call tracepol's optics model then compute rotation offsets by
    # minimizing deviations to either only order 1 or all orders
    # simultaneously.
    # Call tracepol, disabling the default rotation, back to original
    # Optics Model.
    param = tp.get_tracepars(filename=optmodel, disable_rotation=True)

    wavelength = np.linspace(0.9, 2.8, 39)

    x_om_o1, y_om_o1, mask_om_o1 = tp.wavelength_to_pix(wavelength,
                                               param, m=1,
                                               frame='dms',
                                               subarray='SUBSTRIP256',
                                               oversample=1)
    x_om_o2, y_om_o2, mask_om_o2 = tp.wavelength_to_pix(wavelength,
                                               param, m=2,
                                               frame='dms',
                                               subarray='SUBSTRIP256',
                                               oversample=1)
    x_om_o3, y_om_o3, mask_om_o3 = tp.wavelength_to_pix(wavelength,
                                               param, m=3,
                                               frame='dms',
                                               subarray='SUBSTRIP256',
                                               oversample=1)



    please_check = True

    if please_check:

        # Figure to show the positions for all 3 orders
        plt.figure(figsize=(10, 10))
        plt.ylim((0, 256))
        plt.imshow(np.log10(im), vmin=0.7, vmax=3, origin='lower', aspect='auto')

        plt.plot(x_o1, y_o1, color='orange', label='Order 1')
        plt.plot(x_o1, y_o1 - w_o1 / 2, color='orange')
        plt.plot(x_o1, y_o1 + w_o1 / 2, color='orange')
        plt.plot(x_om_o1, y_om_o1, color='orange')

        if x_o2 is not None:
            plt.plot(x_o2, y_o2, color='black', label='Order 2')
            plt.plot(x_o2, y_o2 - w_o2 / 2, color='black')
            plt.plot(x_o2, y_o2 + w_o2 / 2, color='black')
            plt.plot(x_om_o2, y_om_o2, color='black')

        if x_o3 is not None:
            plt.plot(x_o3, y_o3, color='red', label='Order 3')
            plt.plot(x_o3, y_o3 - w_o3 / 2, color='red')
            plt.plot(x_o3, y_o3 + w_o3 / 2, color='red')
            plt.plot(x_om_o3, y_om_o3, color='red')

        plt.legend()
        plt.show()

#run
calibrate_tracepol()