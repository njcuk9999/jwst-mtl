#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-06

@author: cook
"""
from soss_tfit.core import core
from soss_tfit.science import general
from soss_tfit.science import mcmc
from soss_tfit.science import plot


# =============================================================================
# Define variables
# =============================================================================
# config file (could be an argument)
CONFIG_FILE = ('/data/jwst-soss/bin/jwst-mtl-soss/SOSS/soss_tfit/recipes/'
               'example_neiltest.yaml')
# printer
cprint = core.base_classes.Printer()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

    # deal with command line arguments (currently just loads config file)
    config_file = core.get_args()

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

    # -------------------------------------------------------------------------
    # Step 3: set up the parameters
    #         All data manipulation + parameter changes should be done
    #         before this point
    # -------------------------------------------------------------------------
    # get starting parameters for transit fit
    tfit = mcmc.setup_params_mcmc(params, data)

    # assign beta values
    tfit.assign_beta()

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
    # print result
    sampler2.posterior_print()
    # plot a specific chain
    plot.plot_chain(params, sampler2.chain, chain_num=-1)
    # -------------------------------------------------------------------------
    # update tfit
    tfit_final = mcmc.update_x0_p0_from_chain(tfit, sampler2.wchains[0], -1)
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
    cprint('Dumping sampler to pickle file')
    sampler2.dump(data=data)

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
