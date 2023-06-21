##runs the full mcmc
##requires one argument: trial sampler object pickle filename

from soss_tfit.core import core
from soss_tfit.science import mcmc
import sys
import os

#load the TransitFit object from pickle file
pfile=sys.argv[1]
sampler1=mcmc.Sampler.load(pfile)
tfit=sampler1.tfit

os.environ["OMP_NUM_THREADS"] = str(tfit.params['N_CHAIN_THREADS'])
os.environ["KMP_INIT_AT_FORK"]= "FALSE"

if __name__=="__main__": #this is important for multiprocessing
    
    # -------------------------------------------------------------------------
    # Step 5: fit the multi-spectrum model (full run)
    # -------------------------------------------------------------------------
    sampler2 = mcmc.Sampler(tfit, mode='full')
    sampler2.run_mcmc(mcmc.lnprob, mcmc.mhg_mcmc,trial=sampler1)

    #update x0 & p0 in tfit from last iteration in chain
#     mcmc.update_x0_p0_from_chain(sampler2.tfit, sampler2.wchains[0], -1)
    #update x0 & p0 in tfit from highest likelihood value of chain
    i=sampler2.lls.argmax()
    mcmc.update_x0_p0_from_chain(sampler2.tfit, sampler2.chain, i)
    # -------------------------------------------------------------------------

    #dump sampler in pickle file
    sampler2.dump()