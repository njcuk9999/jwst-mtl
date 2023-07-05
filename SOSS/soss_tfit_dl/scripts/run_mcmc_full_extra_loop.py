##runs the full mcmc
##requires one argument: sampler object pickle filename
##2nd argument optional: number of additional loops to run

from soss_tfit.science import mcmc
import sys
import os

#load the TransitFit object from pickle file
pfile=sys.argv[1]
sampler2=mcmc.Sampler.load(pfile)
tfit=sampler2.tfit

os.environ["OMP_NUM_THREADS"] = str(tfit.params['N_CHAIN_THREADS'])
os.environ["KMP_INIT_AT_FORK"]= "FALSE"

n_repeat=1 if len(sys.argv)<3 else int(sys.argv[2])

if __name__=="__main__": #this is important for multiprocessing
    
    for n in range(n_repeat):
        sampler2.single_loop(mcmc.lnprob, mcmc.mhg_mcmc,in_sampler=sampler2)

        #update x0 & p0 in tfit from last iteration in chain
    #     mcmc.update_x0_p0_from_chain(sampler2.tfit, sampler2.wchains[0], -1)
        #update x0 & p0 in tfit from highest likelihood value of chain
        i=sampler2.lls.argmax()
        mcmc.update_x0_p0_from_chain(sampler2.tfit, sampler2.chain, i)
        # -------------------------------------------------------------------------

        #dump sampler in pickle file
        #we do it after each loop to have access to results
        sampler2.dump()