##runs the beta rescale step and the trial mcmc run
##requires one argument: TransitFit object pickle filename

from soss_tfit.science import mcmc
import sys
import os

#load the TransitFit object from pickle file
pfile=sys.argv[1]
tfit=mcmc.TransitFit.load(pfile)

    
# -------------------------------------------------------------------------
# Step 3: Calculate rescaling of beta to improve acceptance rates
# -------------------------------------------------------------------------
# Calculate rescaling of beta to improve acceptance rates
corscale = mcmc.beta_rescale_nwalker(tfit, mcmc.mhg_mcmc, mcmc.lnprob)
#apply correction
tfit.beta*=corscale
#dump tfit in pickle file
tfit.dump()
