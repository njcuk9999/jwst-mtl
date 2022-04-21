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
from soss_tfit.science import mcmc
from soss_tfit.science import plot


# =============================================================================
# Define variables
# =============================================================================
CONFIG_FILE = ('/data/jwst-soss/bin/jwst-mtl-soss/SOSS/soss_tfit/recipes/'
               'example_neiltest.yaml')

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
    data = general.InputData(params, filename=params['INSPECTRUM'],
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
    # TODO: Move to David only
    # check that all looks good
    order, ibin = 2, 0
    time_arr = data.spec[order]['TIME'][:, ibin]
    flux_arr = data.spec[order]['FLUX'][:, ibin]
    eflux_arr = data.spec[order]['FLUX_ERROR'][:, ibin]
    fig, frame = plt.subplots(ncols=1, nrows=1)
    frame.errorbar(time_arr, flux_arr, yerr=eflux_arr, ls='None', marker='o')
    frame.set(xlabel='Time [days]', ylabel='Flux',
              title=f'Order {order} wave bin: {ibin}')
    plt.show()
    plt.close()

    # -------------------------------------------------------------------------
    # TODO: Move to David only
    # remove a few lambdas from order 2 (S/N) too low
    order = 2
    imin = 1   # david uses 4 for bins = [30, 15]
    # loop around each quantity
    for quantity in data.spec[order]:
        data.spec[order][quantity] = data.spec[order][quantity][:, imin:]
    # update nwave
    data.n_wav[order] = data.spec[order]['WAVELENGTH'].shape[1]

    # -------------------------------------------------------------------------
    # create the photospectra dictionary (data.phot)
    data.photospectra()

    # update zeropoint parameters using data
    params = data.update_zeropoint(params)

    # Show a plot of the data. Each colour is a different wavelength.
    plot.plot_flux(data)

    # TODO: Move to David only
    # Fill in a few necessary parameters
    # (Overwrites default parameters that were given in def)

    # [g/cm³]
    params['RHO_STAR'].value = 2.48
    # boundaries for valid models, if needed.
    params['RHO_STAR'].prior = dict(minimum=2.0, maximum=3.0)
    # limb dark. coeff1
    params['LD3'].value = 0.2
    # limb dark. coeff2
    params['LD4'].value = 0.15
    # zero point to zero
    params['ZEROPOINT'].value = 0.0
    params['ZEROPOINT'].prior = dict(minimum=-0.005, maximum=0.005)
    # period priors
    params['PLANET1']['PERIOD'].value = 1.7497798
    params['PLANET1']['PERIOD'].ftype = 'fixed'
    # t0: center of transit time
    params['PLANET1']['T0'].value = 0.093
    params['PLANET1']['T0'].prior = dict(minimum=0.093 - 0.03,
                                         maximum=0.093 + 0.03)
    # impact parameter
    params['PLANET1']['B'].value = 0.6
    params['PLANET1']['B'].prior = dict(minimum=0.4, maximum=0.8)
    # Rp/Rs
    params['PLANET1']['RPRS'].value = 0.1645
    params['PLANET1']['RPRS'].prior = dict(minimum=0.1, maximum=0.2)

    # -------------------------------------------------------------------------
    # Step 3: set up the parameters
    #         All data manipulation + parameter changes should be done
    #         before this point
    # -------------------------------------------------------------------------
    # get starting parameters for transit fit
    tfit = mcmc.setup_params_mcmc(params, data)

    # TODO: David adjustment
    # added by DL, trying to adjust the beta factors get better acceptance
    # rates right away
    bkwargs = dict()
    bkwargs['p'] = 0.03
    bkwargs['q1'] = 0.2
    bkwargs['q1'] = 0.2
    bkwargs['q2'] = 0.2
    bkwargs['ZPT'] = 1.0e-3
    bkwargs['EP1'] = 2.0e-4
    bkwargs['BB1'] = 2.0e-3
    bkwargs['RD1'] = 3.0e-3
    bkwargs['DSC'] = 0.002

    # assign beta values
    tfit.assign_beta(beta_kwargs=bkwargs)

    # Check that all is good
    plot.plot_transit_fit(tfit, bandpass=3)

    # -------------------------------------------------------------------------
    # Step 3: Calculate rescaling of beta to improve acceptance rates
    # -------------------------------------------------------------------------
    # Calculate rescaling of beta to improve acceptance rates
    corscale = mcmc.beta_rescale(params, tfit, mcmc.mhg_mcmc, mcmc.lnprob)

    # -------------------------------------------------------------------------
    # Step 4: fit the multi-spectrum model (trial run)
    # -------------------------------------------------------------------------
    # run the mcmc in trial mode
    sampler1 = mcmc.Sampler(params, tfit, mode='trial')
    sampler1.run_mcmc(corscale, mcmc.lnprob, mcmc.mhg_mcmc)
    # print result
    sampler1.posterior_print()

    # -------------------------------------------------------------------------
    # Step 5: fit the multi-spectrum model (full run)
    # -------------------------------------------------------------------------
    sampler2 = mcmc.Sampler(params, tfit, mode='full')
    sampler2.run_mcmc(corscale, mcmc.lnprob, mcmc.mhg_mcmc,
                      trial=sampler1)
    # print result
    sampler2.posterior_print()
    # plot a specific chain
    plot.plot_chain(sampler2.chain, chain_num=-1)
    # update tfit
    tfit_final = mcmc.update_x0_p0_from_chain(tfit, sampler2.chain, -1)
    # plot transit
    plot.plot_transit_fit(tfit, 5)
    # plot the chains
    plot.plot_chains(sampler2.chain, 0, tfit.xnames)

    # -------------------------------------------------------------------------
    # Step 6: save results
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Step 7: plot spectrum
    # -------------------------------------------------------------------------


# =============================================================================
# End of code
# =============================================================================
