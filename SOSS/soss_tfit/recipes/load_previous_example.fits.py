#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-27

@author: cook
"""
from soss_tfit.science import mcmc
from soss_tfit.science import general
from soss_tfit.science import plot

# =============================================================================
# Define variables
# =============================================================================
SAMPLER_PICKLE = ('/data/jwst-soss/data/jwst-mtl-user/wasp52b/outputs/'
                  'results_sampler.pickle')

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # load sampler
    sampler = mcmc.Sampler.load(SAMPLER_PICKLE)

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
