#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-06

@author: cook
"""
import matplotlib.pyplot as plt

from soss_tfit.core import core
from soss_tfit.science import general



# =============================================================================
# Define variables
# =============================================================================
CONFIG_FILE = '/data/jwst-soss/bin/jwst-mtl-soss/SOSS/soss_tfit/inputs/example.yaml'

# =============================================================================
# Define functions
# =============================================================================


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Step 1: load parameter file into our parameter class
    # -------------------------------------------------------------------------
    print('Loading parameters from yaml file')
    params = core.load_params(CONFIG_FILE)

    # -------------------------------------------------------------------------
    # Step 2: load data
    # -------------------------------------------------------------------------
    # load data model
    print('Loading input data: {0}'.format(params['INSPECTRUM']))
    data = general.Inputdata(params, filename=params['INSPECTRUM'],
                             verbose=True)

    # -------------------------------------------------------------------------
    # create time array (in future should be loaded from data model)
    print('Computing time vector for data')
    data.compute_time()

    # -------------------------------------------------------------------------
    # remove wavelengths corresponding to null flux
    print('Removing null flux values')
    data.remove_null_flux()

    # -------------------------------------------------------------------------
    # apply spectral binning
    print('Applying spectral binning')
    data.apply_spectral_binning(params)

    # -------------------------------------------------------------------------
    # apply normalization (normalize by mean out-of-transit)
    print('Normalizing by out-of-transit flux for each wavelength')
    data.normalize_by_out_of_transit_flux(params)

    # -------------------------------------------------------------------------
    # check that all looks good
    order, ibin = 2, 0
    time_arr = data.spec[order]['TIME'][:, ibin]
    flux_arr = data.spec[order]['FLUX'][:, ibin]
    eflux_arr = data.spec[order]['FLUX_ERROR'][:, ibin]
    fig, frame = plt.subplots(ncols=1, nrows=1)
    frame.errorbar(time_arr, flux_arr, yerr=eflux_arr, ls='None', marker='o')
    frame.set(xlabel='Time [days]', ylabel='Flux',
              title=f'Order {order} wave bin: {ibin}')

    # -------------------------------------------------------------------------
    # remove a few lambdas from order 2 (S/N) too low
    order = 2
    imin = 1   # david uses 4 for bins = [30, 15]
    # loop around each quantity
    for quantity in data.spec[order]:
        data.spec[order][quantity] = data.spec[order][quantity][:, imin:]
    # update nwave
    data.n_wav[order] = data.spec[order]['WAVELENGTH'].shape[1]

    # -------------------------------------------------------------------------
    # create the photospectra
    data.photospectra()

    # -------------------------------------------------------------------------
    # Step 3: fit the multi-spectrum model (trial run)
    # -------------------------------------------------------------------------
    print('done')

    # -------------------------------------------------------------------------
    # Step 4: fit the multi-spectrum model (full run)
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # Step 5: save results
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # Step 6: plot spectrum
    # -------------------------------------------------------------------------



# =============================================================================
# End of code
# =============================================================================
