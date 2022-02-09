#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ["OMP_NUM_THREADS"] = "24" 

import sys
sys.path.insert(0, "/genesis/jwst/jwst-ref-soss/fortran_lib/transitfit/") #pre-compiled Fortran library for python.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Import module
import transitfit5 as tf #import transitfit5 modules

import tfit5 as tfit5 #Fortran-low-level modules

import utils_python.transitfit_modelling as tfmod

import utils_python.transitfit_mcmcfitting as tfmc

#from utils_python.transitfit_utils import read_soss_spectra
from utils_python.transitfit_utils import *

import math

from scipy.stats import binned_statistic #for binned data.

from tqdm.notebook import trange

import corner

#Modules for timing (not needed for production)
import time


# Read the extracted spectra produced by the DMS extract-1d
photospectra = [] # contains a series of phot classes, wavelength by wavelength

path_userland = '/genesis/jwst/userland-soss/loic_review/'

raw_wavelength, raw_flux, raw_flux_err = read_soss_spectra(os.path.join(path_userland,'GTO/wasp52b/extracted_spectra_alone.fits'))

# Select order 1 data only
m = 0

# Bin the data to lower resolution
nbins=4 #Let's bin the data to speed up the code.
wavelength, flux, flux_err = bin_lightcurves(raw_wavelength, raw_flux, raw_flux_err, nbins)

exptime = 76.916

norder, ninteg, nwave = np.shape(wavelength)
#for p in range(len(nwave):
for sampl in range(nbins):

    phot = tfmod.phot_class()
    phot.wavelength = wavelength[m, :, sampl]
    phot.time = exptime/(24*3600) * np.arange(ninteg)
    # center the transit on zero
    phot.time = phot.time - (np.max(phot.time) - np.min(phot.time))/2
    phot.flux = flux[m, :, sampl]
    phot.ferr = flux_err[m, :, sampl]
    phot.itime = ninteg

    # Normalize flux and errors
    norma = np.percentile(phot.flux,75)
    phot.flux = phot.flux / norma
    phot.ferr = phot.ferr / norma

    photospectra.append(phot)

# In[12]:


#Show a plot of the data.  Each colour is a different wavelength.

matplotlib.rcParams.update({'font.size': 30}) #adjust font
matplotlib.rcParams['axes.linewidth'] = 2.0
fig=plt.figure(figsize=(16,14)) #adjust size of figure
ax = plt.axes()
ax.tick_params(direction='in', which='major', bottom=True, top=True, left=True, right=True, length=10,width=2)
ax.tick_params(direction='in', which='minor', bottom=True, top=True, left=True, right=True, length=4,width=2)

ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_xlabel('Time (days)')
ax.set_ylabel('Relative Flux')


for p in photospectra:
    ax.plot(p.time,p.flux)#/np.median(p.flux))
    
#fig.show()
plt.savefig('/genesis/jwst/userland-soss/loic_review/tf_lightcurves.png')

# ## Let's fit the multi-spectrum model

# In[14]:


#number of planets to include
nplanet=1

#Set up default parameters 
tpars = tfmod.sptransit_model_parameters([photospectra, nplanet])

#Fill in a few necessary parameters  
tpars.rhostar[0]=np.array([2.48])
tpars.rhostar[3]=np.array([0.1,10.0]) #boundaries for valid models, if needed.    

tpars.period[0][0]=np.array([1.7498])
tpars.period[0][2]='fit'

tpars.t0[0][0]=np.array([0.0])
tpars.t0[0][3]=np.array([-0.01, 0.01])

tpars.rprs[0][0]=np.ones(len(photospectra))*0.17
tpars.rprs[0][3]=np.array([0.15,0.19])

#Set search scale for zero-point (normalization)
#fmin=np.min(photospectra[0].flux)
#fmax=np.max(photospectra[0].flux)
#for p in photospectra:
#    fmin=np.min([fmin,np.min(p.flux)])
#    fmax=np.max([fmax,np.max(p.flux)])
#tpars.zeropoint[3]=np.array([fmin,fmax])
tpars.zeropoint[3]=np.array([0.98,1.02])

#Simple labels to identify parameters.
clabels=['p','c1','c2','q1','q2','DIL','VOF','ZPT','EP','PE','BB','RD','EC','ES','KRV','TED','ELL','ALB','DSC','ASC','LSC']

# DSC : Error Scale (tpars.error_scale)
# ASC : tpars.amplitude_scale
# LSC : tpars.length_scale


# In[15]:


def lnprob(x):
    ''' ln prob model
    Nested sampling and many MCMC routines use global variables.  Thus, only the fitted parameters are passed.
    Fitted parameters are contained in the input 'x' array.
    '''
    
    logl=1.0e0 #initialize log-likelihood to some value.
    #check validity of array
    badlpr=-np.inf #if outside bounds, then mark poor likelihood.
    
    nwav=sol.shape[1] #number of bandpasses
    npars=sol.shape[0] #number of model parameters
    
    solnew = tfmod.update_sol(tpars, x, sol) #contains sol but updated with values from 'x'
    
    #check validity of array
    #logl = tfmod.checksolution(tpars, solnew, badlpr)
    logl = tfmod.checksolution(tpars, solnew, badlpr, sol)
    
    if logl>badlpr:
    
        for i in range(nwav):
            dscale=solnew[npars-3][i] #photometric scale
            ascale=solnew[npars-2][i] #photometric scale
            lscale=solnew[npars-1][i] #length scale for GP

            #check dscale, ascale and lscale hyper parameters (they must be positive)
            if (dscale <= 0.0)&(tpars.error_scale[2]=='fit'):
                logl=badlpr
            if (ascale <= 0.0)&(tpars.amplitude_scale[2]=='fit'):
                logl=badlpr
            if (lscale <= 0.0)&(tpars.length_scale[2]=='fit'):
                logl=badlpr

            if (tpars.amplitude_scale[2]=='fit')|(tpars.length_scale[2]=='fit'):
                modeltype=1 #GP model
            else:
                modeltype=0 #uncorrelated noise model

            npt=len(photospectra[i].time) #number of data points

            sol1=np.array([s[i] for s in solnew])

            #zpt=np.copy(sol1[7])
            sol1[7]-=1.0 

            if logl>badlpr: #check that we have a valid model to use
                #Retrieve transit model using sol3 array 
                ans = tf.transitmodel(sol1, photospectra[i].time, itime=photospectra[i].itime,
                                      ntt=tpars.ntt, tobs=tpars.tobs, omc=tpars.omc)
                #ans = ans*zpt #put in zero-point

                if np.isnan(np.sum(ans))==False: #check for NaNs -- we don't want these.

                    if modeltype==0: #non-correlated noise-model
                        logl+=-0.5*(sum(np.log(photospectra[i].ferr*photospectra[i].ferr*dscale*dscale))+sum((photospectra[i].flux-ans)*(photospectra[i].flux-ans)/(photospectra[i].ferr*photospectra[i].ferr*dscale*dscale)))
                else:
                    logl=badlpr

            #plt.plot(photospectra[i].time,photospectra[i].flux)
            #plt.plot(photospectra[i].time,ans)
            #plt.show()
            #Add Priors here...
    
            
    return logl


# In[16]:


nwalkers=3 #number of walkers for MCMC
nsteps1 = 10000 #total length of chain will be nwalkers*nsteps
nsteps2 = 200000 #nstep1 is to check that MCMC is okay, nstep2 is the real work.
nsteps_inc = 100000
burninf=0.5 #burn-in for evalulating convergence
niter_cor=5000
burnin_cor=1000
nthin=101
nloopmax=5
converge_crit=1.02 #Convergence criteria
buf_converge_crit=1.2 #Convergence criteria for buffer
itermax=5 #maximum iterations allowed.

sol = tfmod.get_all_parameters(tpars, photospectra) #Creates internal array used to create transit model.
x = tfmod.get_fitted_parameters(tpars)
tran_par_names = tfmod.get_names(clabels, tpars, sol)
beta = np.random.rand(len(x))*1.0e-5


# In[17]:
print('\n\nGoing into betarescale...')

corscale = tfmc.betarescale(x, beta, niter_cor, burnin_cor, tfmc.mhgmcmc, lnprob, imax=7)


# In[18]:
print('\n\nLets fit the multi-spectrum model (Going into MCMC loop)...')

nloop=0
nsteps=np.copy(nsteps1)
mcmcloop=True
while mcmcloop==True:

    nloop+=1 #count number of loops

    hchain1,haccept1=tfmc.genchain(x,nsteps,beta*corscale,tfmc.mhgmcmc,lnprob)
    hchain2,haccept2=tfmc.genchain(x,nsteps,beta*corscale,tfmc.mhgmcmc,lnprob)
    hchain3,haccept3=tfmc.genchain(x,nsteps,beta*corscale,tfmc.mhgmcmc,lnprob)

    if nloop==1:
        chain1=np.copy(hchain1)
        chain2=np.copy(hchain2)
        chain3=np.copy(hchain3)
        accept1=np.copy(haccept1)
        accept2=np.copy(haccept2)
        accept3=np.copy(haccept3)
    else:
        chain1=np.concatenate((chain1,hchain1))
        chain2=np.concatenate((chain2,hchain2))
        chain3=np.concatenate((chain3,hchain3))
        accept1=np.concatenate((accept1,haccept1))
        accept2=np.concatenate((accept2,haccept2))
        accept3=np.concatenate((accept3,haccept3))

    burnin=int(chain1.shape[0]*burninf)
    tfmc.calcacrate(accept1,burnin)

    grtest=tfmc.gelmanrubin(chain1,chain2,chain3,burnin=burnin,npt=len(phot.time))
    print('Gelman-Rubin Convergence:')
    print('parameter  Rc')
    for i in range(0,len(chain1[1,:])):
        print('%8s %3s %.4f' %(str(i),tran_par_names[i],grtest[i]))
    if int(np.sum(grtest[grtest<buf_converge_crit]/grtest[grtest<buf_converge_crit]))==len(grtest):
        mcmcloop=False
    else:
        mcmcloop=True
        nsteps+=nsteps1

    #runtest=np.array(tf.checkperT0(chain1,burninf,TPnthin,sol,serr))
    #print('runtest:',runtest)
    #if int(np.sum(runtest[runtest<1.0]/runtest[runtest<1.0]))!=4.0:
    #    mcmcloop=False #run-away

    if nloop>=nloopmax: #break if too many loops
        mcmcloop=False

    #print("---- %s seconds ----" % (time.time() - start_time))


# In[19]:

print('\n\nGoing in 2nd MCMC loop...')
mcmcloop=True
nloop=0
nsteps=np.copy(nsteps2)
while mcmcloop==True:

    nloop+=1 #count number of loops

    burnin=int(chain1.shape[0]*burninf)
    buffer=np.concatenate((chain1[burnin:],chain2[burnin:],chain3[burnin:])) #create buffer for deMCMC
    x1=np.copy(chain1[chain1.shape[0]-1,:])
    x2=np.copy(chain1[chain1.shape[0]-1,:])
    x3=np.copy(chain1[chain1.shape[0]-1,:])
    corbeta=0.3
    burnin=int(chain1.shape[0]*burninf)
    chain1,accept1=tfmc.genchain(x1,nsteps,beta*corscale,tfmc.demhmcmc,lnprob,buffer=buffer,corbeta=corbeta,progress=True)
    chain2,accept2=tfmc.genchain(x2,nsteps,beta*corscale,tfmc.demhmcmc,lnprob,buffer=buffer,corbeta=corbeta,progress=True)
    chain3,accept3=tfmc.genchain(x3,nsteps,beta*corscale,tfmc.demhmcmc,lnprob,buffer=buffer,corbeta=corbeta,progress=True)

    burnin=int(chain1.shape[0]*burninf)
    grtest=tfmc.gelmanrubin(chain1,chain2,chain3,burnin=burnin,npt=len(phot.time))
    print('Gelman-Rubin Convergence:')
    print('parameter  Rc')
    for i in range(0,len(chain1[1,:])):
        print('%8s %3s %.4f' %(str(i),tran_par_names[i],grtest[i]))

    if int(np.sum(grtest[grtest<converge_crit]/grtest[grtest<converge_crit]))==len(grtest):
        mcmcloop=False
    else:
        mcmcloop=True

    burnin=int(chain1.shape[0]*burninf)
    chain=np.concatenate((chain1[burnin:,],chain2[burnin:,],chain3[burnin:,]))
    accept=np.concatenate((accept1[burnin:,],accept2[burnin:,],accept3[burnin:,]))
    burnin=int(chain.shape[0]*burninf)
    tfmc.calcacrate(accept,burnin)

    nsteps+=nsteps_inc #make longer chain to help with convergence

    ##check for run-away Chain.
    #runtest=np.array(tf.checkperT0(chain1,burninf,nthin,sol,serr))
    #print('runtest:',runtest)
    #if int(np.sum(runtest[runtest<1.0]/runtest[runtest<1.0]))!=4.0:
    #    mcmcloop=False #run-away

    if nloop>=nloopmax: #break if too many loops
        mcmcloop=False

    #print("---- %s seconds ----" % (time.time() - start_time))

#print("done %s seconds ---" % (time.time() - start_time))


# In[31]:


matplotlib.rcParams.update({'font.size': 12}) #adjust font
tfmc.plotchains(chain,0,tran_par_names, '/genesis/jwst/userland-soss/loic_review/tf_mcmcchains.png')


# In[32]:


npars=len(chain[1,:])
mm=np.zeros(npars)
for i in range(0,npars):
    mode,x_eval,kde1=tf.modekdestimate(chain[::10,i],0)
    mm[i]=mode
    perc1 = tf.intperc(mode,x_eval,kde1)#,perc=0.997)
    print('%s = %.8f +%.8f -%.8f (1 Sigma)' %(tran_par_names[i],mode,np.abs(perc1[1]-mode),np.abs(mode-perc1[0])))


# In[27]:


npars=len(chain[1,:])
mm=np.zeros(npars)



nthin=10
chain_thin=chain[::nthin,:]

rprs_model=[]
rprs_model_ep=[]
rprs_model_em=[]
for i in range(npars):
    mode,x_eval,kde1=tf.modekdestimate(chain_thin[:,i],0)
    mm[i]=mode
    perc1 = tf.intperc(mode,x_eval,kde1)#,perc=0.997)
    rprs_model.append(mode)
    rprs_model_ep.append( np.abs(perc1[1]-mode))
    rprs_model_em.append(-np.abs(mode-perc1[0]))
    print('%s = %.8f +%.8f -%.8f (1 Sigma)' %(tran_par_names[i],mode,np.abs(perc1[1]-mode),np.abs(mode-perc1[0])))

# All parameters (not only Rp/Rs) fitted
rprs_model=np.array(rprs_model)       # Fitted parameters
rprs_model_ep=np.array(rprs_model_ep) # Error bar (positive)
rprs_model_em=np.array(rprs_model_em) # Error bar (negative)

    


# In[30]:


matplotlib.rcParams.update({'font.size': 20}) #adjust font
matplotlib.rcParams['axes.linewidth'] = 2.0
fig=plt.figure(figsize=(16,14)) #adjust size of figure
ax = plt.axes()
ax.tick_params(direction='in', which='major', bottom=True, top=True, left=True, right=True, length=10,               width=2)
ax.tick_params(direction='in', which='minor', bottom=True, top=True, left=True, right=True, length=4,               width=2)

ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_xlabel('Wavelength ($\mu$m)')
ax.set_ylabel(' $R_{p}/R_{\star}$')


#rprs_binned=np.array([0.08425564673418401, 0.08391695345274379, 0.08375053190856288, 0.08364267099710652, \
#                     0.08348632677920928, 0.083973083683014,   0.08342468358830134, 0.08336638722554368, \
#                     0.08380774401419508, 0.08355463168731934, 0.0832839245500194,  0.08329519340547527, \
#                     0.08345741726828987, 0.08418017555669018, 0.08437886537394856])

#get wavelength and transit depth info
if False: # We don't have the model like Jason did
    psg_model.wavelength=np.array([float(d[0]) for d in model_data]) #wavelength in microns
    psg_model.Rp=np.array([d[1]*1000 for d in model_data]) #this is in m and is contribution of atmosphere
    psg_model.Rp_err=np.array([d[2] for d in model_data])

    ax.plot(psg_model.wavelength,(psg_model.Rp+Rp_m)/Rs_m,c='lightblue',zorder=0)
#ax.plot(tpars.wavelength,rprs_binned,zorder=1,c='blue',lw=2)

nlba, _ = np.shape(tpars.wavelength)
lba = []
for i in range(nlba):
    lba.append(tpars.wavelength[i][0])
lba = np.array(lba)

rprs = []
rprs_em = []
rprs_ep = []
for i in range(npars):
    if tran_par_names[i] == 'RD':
        rprs.append(rprs_model[i])
        rprs_em.append(rprs_model_em[i])
        rprs_ep.append(rprs_model_ep[i])

rprs = np.array(rprs)
rprs_em = np.array(rprs_em)
rprs_ep = np.array(rprs_ep)


ax.plot(lba, rprs)
ax.errorbar(lba, rprs, np.array([-rprs_em,rprs_ep]), fmt='o', lw=3, c='black', zorder=2)

plt.savefig('/genesis/jwst/userland-soss/loic_review/tf_spectrum.png')


# In[ ]:


#Corner is SLOW
figure = corner.corner(chain, quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, labels=tran_par_names)
figure.savefig('/genesis/jwst/userland-soss/loic_review/tf_cornerplot.png')
