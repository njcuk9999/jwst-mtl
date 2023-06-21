##runs the beta rescale step and the trial mcmc run
##requires one argument: TransitFit object pickle filename

from soss_tfit.science import mcmc
import sys
import os

#load the TransitFit object from pickle file
pfile=sys.argv[1]
tfit=mcmc.TransitFit.load(pfile)

os.environ["OMP_NUM_THREADS"] = str(tfit.params['N_CHAIN_THREADS'])
os.environ["KMP_INIT_AT_FORK"]= "FALSE"

if __name__=="__main__": #this is important for multiprocessing
    
    # -------------------------------------------------------------------------
    # Step 3: Calculate rescaling of beta to improve acceptance rates
    # -------------------------------------------------------------------------
    # Calculate rescaling of beta to improve acceptance rates
    corscale = mcmc.beta_rescale_nwalker(tfit, mcmc.mhg_mcmc, mcmc.lnprob)
    #apply correction
    tfit.beta*=corscale
    #dump tfit after beta update
    tfit.dump()

    # -------------------------------------------------------------------------
    # Step 4: fit the multi-spectrum model (trial run)
    # -------------------------------------------------------------------------
    # run the mcmc in trial mode
    sampler1 = mcmc.Sampler(tfit, mode='trial')
    sampler1.run_mcmc(mcmc.lnprob, mcmc.mhg_mcmc)
    #update x0 & p0 in tfit from last iteration in chain
#     mcmc.update_x0_p0_from_chain(sampler1.tfit, sampler1.wchains[0], -1)
    #update x0 & p0 in tfit from highest likelihood value of chain
    i=sampler1.lls.argmax()
    mcmc.update_x0_p0_from_chain(sampler1.tfit, sampler1.chain, i)

    #dump sampler in pickle file
    sampler1.dump()

