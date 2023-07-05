#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-27

@author: cook
"""
from soss_tfit.core import core
from soss_tfit.science import mcmc
from soss_tfit.science import general
from soss_tfit.science import plot

# =============================================================================
# Define variables
# =============================================================================
SAMPLER_PICKLE = ('/data/jwst-soss/data/jwst-mtl-user/wasp52b/outputs/'
                  'david_bins_30_15/wasp52b_sampler.pickle')
# printer
cprint = core.base_classes.Printer()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # load sampler
    sampler = mcmc.Sampler.load(SAMPLER_PICKLE)

    # -------------------------------------------------------------------------
    # plot transit
    plot.plot_transit_fit(sampler.params, sampler.tfit)
    # plot the chains
    plot.plot_chains(sampler.params, sampler.chain, 0, sampler.tfit.xnames)
    # quick check of posterior
    plot.plot_hist(sampler.params, sampler.tfit, sampler.chain[::10],
                   param_num=None)

    # -------------------------------------------------------------------------
    # Step 6: generate statistics
    # -------------------------------------------------------------------------

    # print modes
    cprint('Result [mode]:')
    sampler.print_results('mode')

    cprint('Results [percentile]:')
    sampler.print_results('percentile')

    # print modes for parameter RD1
    cprint('Result for RD [mode]:')
    sampler.print_results('mode', key='RD1')

    # print modes for parameter RD1
    cprint('Results for RD [percentile]:')
    sampler.print_results('percentile', key='RD1')

    # -------------------------------------------------------------------------
    # Step 8: plot spectrum - using sampler only so we can load from file
    # -------------------------------------------------------------------------
    # load model (and bin it) if available
    binmodel, fullmodel = general.load_model(sampler.params, sampler.data)
    # plot the spectrum
    plot.plot_spectrum(sampler.params, sampler.data, sampler.results_table,
                       model=binmodel, fullmodel=fullmodel, pkind='mode')
    plot.plot_spectrum(sampler.params, sampler.data, sampler.results_table,
                       model=binmodel, fullmodel=fullmodel, pkind='percentile')


# =============================================================================
# End of code
# =============================================================================
