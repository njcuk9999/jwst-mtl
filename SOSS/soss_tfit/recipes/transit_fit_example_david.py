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

# printer
cprint = core.base_classes.Printer()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

    # deal with command line arguments (currently just loads config file)
    # config_file = core.get_args()

    # config file (set manually)
    config_file = ('/data/jwst-soss/bin/jwst-mtl-soss/SOSS/soss_tfit/recipes/'
                   'example_david_change_in_code.yaml')

    # -------------------------------------------------------------------------
    # Step 1: load parameter file into our parameter class
    # -------------------------------------------------------------------------
    cprint('Loading parameters from yaml file', level='info')
    params = core.load_params(config_file)

    # -------------------------------------------------------------------------
    # Step 2: load data
    # -------------------------------------------------------------------------
    # load data model
    cprint('Loading input data: {0}'.format(params['INSPECTRUM']), level='info')
    data = general.InputData(params, filename=params['INSPECTRUM'],
                             verbose=True)

    # -------------------------------------------------------------------------
    # create time array (in future should be loaded from data model)
    cprint('Computing time vector for data')
    data.compute_time()

    # -------------------------------------------------------------------------
    # remove wavelengths corresponding to null flux
    cprint('Removing null flux values')
    data.remove_null_flux()

    # -------------------------------------------------------------------------
    # apply spectral binning
    cprint('Applying spectral binning')
    data.apply_spectral_binning(params)

    # -------------------------------------------------------------------------
    # apply normalization (normalize by mean out-of-transit)
    cprint('Normalizing by out-of-transit flux for each wavelength')
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
    plt.show()
    plt.close()

    # -------------------------------------------------------------------------
    # TODO: remove
    # # remove a few lambdas from order 2 (S/N) too low
    # order = 2
    # imin = 1   # david uses 4 for bins = [30, 15]
    # # loop around each quantity
    # for quantity in data.spec[order]:
    #     data.spec[order][quantity] = data.spec[order][quantity][:, imin:]
    # # update nwave
    # data.n_wav[order] = data.spec[order]['WAVELENGTH'].shape[1]

    # remove any bins user wishes to remove
    cprint('Remove bins from order')
    data.remove_bins(params)

    # -------------------------------------------------------------------------
    # create the photospectra dictionary (data.phot)
    data.photospectra()

    # update zeropoint parameters using data
    if params['UPDATE_ZEROPOINT']:
        params = data.update_zeropoint(params)

    # Show a plot of the data. Each colour is a different wavelength.
    plot.plot_flux(params, data)

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

    # added by DL, trying to adjust the beta factors get better acceptance
    # rates right away
    bkwargs = dict()
    bkwargs['p'] = 0.03
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
    plot.plot_transit_fit(params, tfit, bandpass=3)

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
    # add data to sampler2 (for dump and plotting)
    sampler2.data = data
    # print result
    sampler2.posterior_print()
    # plot a specific chain
    plot.plot_chain(params, sampler2.chain, chain_num=-1)
    # -------------------------------------------------------------------------
    # update tfit
    tfit_final = mcmc.update_x0_p0_from_chain(tfit, sampler2.wchains[0], -1)
    # add tfit to sampler
    sampler2.tfit = tfit_final
    # -------------------------------------------------------------------------
    # plot transit
    plot.plot_transit_fit(params, tfit_final)
    # plot the chains
    plot.plot_chains(params, sampler2.chain, 0, tfit_final.xnames)
    # quick check of posterior
    plot.plot_hist(params, tfit_final, sampler2.chain[::10], param_num=None)

    # -------------------------------------------------------------------------
    # Step 6: generate statistics
    # -------------------------------------------------------------------------
    # compile results
    cprint('Compiling results [mode, percentiles]')
    result_table = sampler2.results(start_chain=10)

    # print modes
    cprint('Result [mode]:')
    sampler2.print_results('mode')

    cprint('Results [percentile]:')
    sampler2.print_results('percentile')

    # print modes for parameter RD1
    cprint('Result for RD [mode]:')
    sampler2.print_results('mode', key='RD1')

    # print modes for parameter RD1
    cprint('Results for RD [percentile]:')
    sampler2.print_results('percentile', key='RD1')

    # -------------------------------------------------------------------------
    # Step 7: save results
    # -------------------------------------------------------------------------
    # save the results to a fits file
    cprint('Saving results to fits file')
    sampler2.save_results()

    # save the chains (and rejects) to a fits file for later
    cprint('Saving chains to fits file')
    sampler2.save_chains()

    # dump the sampler class to disk so it can be loaded later
    #   this can be BIG - may want to disable for large runs
    cprint('Dumping sampler to pickle file')
    sampler2.dump()

    # -------------------------------------------------------------------------
    # Step 8: plot spectrum - using sampler only so we can load from file
    # -------------------------------------------------------------------------
    # load model (and bin it) if available
    binmodel, fullmodel = general.load_model(sampler2.params, sampler2.data)
    # plot the spectrum
    plot.plot_spectrum(sampler2.params, sampler2.data, sampler2.results_table,
                       model=binmodel, fullmodel=fullmodel, pkind='mode')
    plot.plot_spectrum(sampler2.params, sampler2.data, sampler2.results_table,
                       model=binmodel, fullmodel=fullmodel, pkind='percentile')


# =============================================================================
# End of code
# =============================================================================
