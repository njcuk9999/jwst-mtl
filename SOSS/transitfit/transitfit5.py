import os
##avoid MPI since we are pooling
#os.environ["MKL_NUM_THREADS"] = "1" 
#os.environ["NUMEXPR_NUM_THREADS"] = "1" 
#os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np 
from numpy import zeros
from numpy import ones
import tfit5
import fittransitmodel as ftf
import matplotlib  #ploting
#matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import math   #used for floor command
from scipy import stats #For Kernel Density Estimation
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import ScalarFormatter
import subprocess
import scipy.linalg as lin

#constants
mearth=5.972e24 #kg
rearth=6.371e6 #meters
mjup=317.828 #Earth masses
msun=1.9891e30 #kg
rsun=6.96265e8 #m
lsun=3.8e26 #w
G=6.67e-11
pc=3.086e+16 #pc in meters
deg2rad=0.0174533 #converting deg to radians
cs=299792458 #speed of light
secinyear=3.154e+7 #seconds in a year

def checkperT0(samples,burninfrac,nthin,sol,serr):
    
    #get indices of Period and T0 from chain
    nparsol=len(sol)
    j=-1
    iT0=-1
    iPer=-1
    for i in range(nparsol):
        if np.abs(serr[i])>1.0e-10:
            j+=1
            if i==8:
                iT0=j
            if i==9:
                iPer=j
                
    #print('ii',iT0,iPer)
    
    nburnin=len(samples)*burninfrac
    burnin=int(nburnin/nthin) #correct burnin using nthin.

    chain=np.array(samples)
    chain=chain[::nthin,:] #Thin out chain.
    #print('Thin size: ',len(chain))
    #print('Burnin: ',burnin)

    if burnin > 0:
        burnin_bak=np.copy(burnin)
    else:
        burnin=np.copy(burnin_bak)
    sigcut=4
    niter=3
    for k in range(niter):
        npars=chain.shape[1]
        test=np.copy(chain[burnin:,:])
        for i in range(npars):
            nch=test.shape[0]
            #print(nch)
            mean=np.mean(test[:,i])
            std=np.std(test[:,i])
            #print(mean,std)
            test2=[]
            for j in range(nch):
                #print(test[j,i], np.abs(test[j,i]-mean),std*sigcut)
                #input()
                if np.abs(test[j,i]-mean) < sigcut*std:
                    test2.append(test[j,:])
            test=np.array(test2)
        nch=test.shape[0]
        #print("nchains:",nch)
        chain=np.copy(test)
        burnin=0
    
    
    if iT0>-1:
        mode,x_eval,kde1=modekdestimate(chain[:,iT0],burnin)
        perc1 = intperc(mode,x_eval,kde1)
        t0_ep=np.abs(perc1[1]-mode)
        t0_em=np.abs(mode-perc1[0])
    else:
        t0_ep=0.0
        t0_em=0.0
    
    if iPer>-1:
        mode,x_eval,kde1=modekdestimate(chain[:,iPer],burnin)
        perc1 = intperc(mode,x_eval,kde1)
        per_ep=np.abs(perc1[1]-mode)
        per_em=np.abs(mode-perc1[0])
    else:
        per_ep=0.0
        per_em=0.0
        
    return t0_ep,t0_em,per_ep,per_em

def gelmanrubin(chain,burninfrac,npt):
    "Estimating PSRF"
    M=chain.shape[1]      #number of chains
    burnin=int(chain.shape[2]*burninfrac)
    N=chain.shape[2]-burnin #assuming all chains have the same size.
    npars=chain.shape[0] #number of parameters
    pmean=np.zeros(shape=(M,npars)) #allocate array to hold mean calculations 
    pvar=np.zeros(shape=(M,npars))  #allocate array to hold variance calculations

    
    for i in range(0,M):
        for j in range(0,npars):
            pmean[i,j]=np.mean(chain[j,i,burnin:]) #Generate means for each parameter in each chain
            pvar[i,j]=np.var(chain[j,i,burnin:])   #Generate variance for each parameter in each chain
    
    posteriormean=np.zeros(npars) #allocate array for posterior means
    for j in range(0,npars):
        posteriormean[j]=np.mean(pmean[:,j]) #calculate posterior mean for each parameter
        
    #Calculate between chains variance
    B=np.zeros(npars)
    for j in range(0,npars):
        for i in range(0,M):
            B[j]+=np.power((pmean[i,j]-posteriormean[j]),2)
    B=B*N/(M-1.0)    
    
    #Calculate within chain variance
    W=np.zeros(npars)
    for j in range(0,npars):
        for i in range(0,M):
            W[j]+=pvar[i,j]
    W=W/M 
    
    
    #Calculate the pooled variance
    V=(N-1)*W/N + (M+1)*B/(M*N)
    
    dof=npt-1 #degrees of freedom 
    Rc=np.sqrt((dof+3.0)/(dof+1.0)*V/W) #PSRF from Brooks and Gelman (1997)
    
    #Calculate Ru
    #qa=0.95
    #ru=np.sqrt((dof+3.0)/(dof+1.0)*((N-1.0)/N*W+(M+1.0)/M*qa))
    
    return Rc;

def transitplot_wchains_v2(phot,sol,serr,chain,burnin,modeltype=[0,0,0,0],nplanetplot=1, \
        itime=-1, ntt=0, tobs=0, omc=0, dtype=0, koi_id=None, savefig='null'):
    """Plot the transit Model"""
    
    plt.figure(figsize=(12,10)) #adjust size of figure
    matplotlib.rcParams.update({'font.size': 22}) #adjust font
    
    nplanet=int((len(sol)-8)/10) #number of planets
    
    if type(itime) is int :
        if itime < 0 :
            itime=np.ones(len(phot.time))*0.020434
        else:
            itime=np.ones(len(phot.time))*float(itime)

    if type(ntt) is int :
        nttin=  np.zeros(nplanet, dtype="int32") #number of TTVs measured
        tobsin= np.zeros(shape=(nplanet,len(phot.time))) #time stamps of TTV measurements (days)
        omcin=  np.zeros(shape=(nplanet,len(phot.time))) #TTV measurements (O-C) (days)
    else:
        nttin=ntt
        tobsin=tobs
        omcin=omc
    
    
    #get standard-deviation of data
    ans_all = transitmodel(sol, phot.time, itime=phot.itime, ntt=ntt, tobs=tobs, omc=omc)
    std=np.std(phot.flux-ans_all)
    tdepth=np.min(ans_all)
    
    nc=8+10*(nplanetplot-1)
    sol2=np.copy(sol)
    sol2[nc+3]=0.0 #rdrs
    ans2 = transitmodel(sol2, phot.time, itime=phot.itime, ntt=ntt, tobs=tobs, omc=omc)
    
    epo=sol[nc+0] #time of center of transit
    per=sol[nc+1] #orbital period
    zpt=sol[7] #photometric zero-point
    tdur=tfit5.transitdur(sol,1)/3600.0 #transit duration in hours
    
    ph1=epo/per-math.floor(epo/per) #calculate phases
    phase=[]
    #tcor=tfit5.lininterp(tobsin,omcin,nplanetplot,nttin,epo)
    #print(tcor,nttin,tobsin[1,1],omcin[1,1])
    for x in phot.time:
        if nttin[nplanetplot-1] > 0:
            tcor=tfit5.lininterp(tobsin,omcin,nplanetplot,nttin,x)
        else:
            tcor=0.0
        t=x-tcor
        ph=(t/per-math.floor(t/per)-ph1)*per*24.0 #phase in hours offset to zero.
        phase.append(ph)
    phase = np.array(phase) #convert from list to array
    
    plt.scatter(phase,phot.flux-ans2+1,c="blue", s=100.0, alpha=0.35, edgecolors="none")
    
    #remove other planets for plotting
    sol2=np.copy(sol)
    for i in range(1,nplanet+1):
        if i!=nplanetplot:
            nc=8+10*(i-1)
            sol2[nc+3]=0.0 #rdrs 
    

    xmin=np.min(phase)/24.0
    xmax=np.max(phase)/24.0
    nmcsamp=np.max((1000,len(phot.time)))
    #mctime=np.linspace(epo-2*tdur/24.0,epo+2*tdur/24.0,nmcsamp)
    mctime=np.linspace(epo+xmin,epo+xmax,nmcsamp)
    mcitime=np.ones(len(mctime))*np.median(itime)
    nmcmc=chain.shape[0]
    ans = transitmodel(sol2, mctime, itime=mcitime)
    #print(24*(mctime-epo))
    plt.plot(24*(mctime-epo),ans,color='red', lw=3.0)


    npars=chain.shape[1]
    tmin=np.min(phot.time)
    tmax=np.max(phot.time)

    #pre-compute sample regions
    n1=np.int((tmin-epo)/per)
    n2=np.int((tmax-epo)/per)
    nsamp=(n2-n1+1)
    diffs_temp=[]
    #print("nmcsamp",nmcsamp)
    for i in range(n1,n2+1):
        epo1=epo+i*per
        #mctime_temp=np.linspace(epo1-2*tdur/24.0,epo1+2*tdur/24.0,nmcsamp)
        mctime_temp=np.linspace(epo1+xmin,epo1+xmax,nmcsamp)
        phot_temp=phot_class()
        phot_temp.time=np.copy(mctime_temp)
        phot_temp.flux=mctime*0.0
        phot_temp.ferr=mctime*0.0
        diffs_temp.append(calcdiffs(phot,phot_temp))

    #phot2,diffs2 are for making Kernels 
    phot2=phot_class()
    phot2.time=np.copy(mctime)
    phot2.flux=mctime*0.0
    phot2.ferr=mctime*0.0
    diffs2=calcdiffs(phot,phot2)
    diffs=calcdiffs(phot,phot)
    for ii in range(100):
        nchain=int(np.random.rand()*(nmcmc-burnin)+burnin)
        
        if modeltype[3]==1:
            dscale=chain[nchain][npars-3]
            ascale=chain[nchain][npars-2]
            lscale=chain[nchain][npars-1]
            #print(dscale,ascale,lscale)
        
        sol2=np.copy(sol)
        j=-1
        for i in range(npars):
            if np.abs(serr[i]) > 1.0e-30:
                j=j+1
                #if i==0:
                #    sol2[i]=np.copy(np.exp(chain[nchain,j])) #lnp -> p
                #else:
                #    sol2[i]=np.copy(chain[nchain,j])
                sol2[i]=np.copy(chain[nchain,j])

        for i in range(1,nplanet+1):
            if i!=nplanetplot:
                nc=8+10*(i-1)
                sol2[nc+3]=0.0 #rdrs

        zpt2=sol2[7] #photometric zero-point
        
        tmodel = transitmodel(sol2, phot.time, itime=phot.itime, ntt=ntt, tobs=tobs, omc=omc)
        tmodel2 = transitmodel(sol2, mctime, itime=mcitime)
        
        if modeltype[3]==1:
            kernel=makekernelm32(phot,phot,dscale,ascale,lscale,diffs)
            factor, low = lin.cho_factor(kernel)
            alpha=lin.cho_solve((factor, low), phot.flux-tmodel)

            samp=np.zeros(len(mctime))
            j=-1
            for i in range(n1,n2+1):
                j=j+1
                epo1=epo+i*per
                #mctime_temp=np.linspace(epo1-2*tdur/24.0,epo1+2*tdur/24.0,1000)
                mctime_temp=np.linspace(epo1+xmin,epo1+xmax,nmcsamp)
                phot_temp=phot_class()
                phot_temp.time=np.copy(mctime_temp)
                phot_temp.flux=mctime*0.0
                phot_temp.ferr=mctime*0.0
                #print(len(phot.time),len(phot_temp.time))       
                kernel2=makekernelm32(phot,phot_temp,0.0,ascale,lscale,diffs_temp[j])
                samp+=np.matmul(kernel2.T,alpha)
            samp=samp/nsamp
            plt.plot(24*(mctime-epo),tmodel2+samp-zpt2,color='seagreen',alpha=0.1,lw=3.0)
            plt.plot(24*(mctime-epo),tmodel2,color='orange',alpha=0.1,lw=3.0)
        else:
            plt.plot(24*(mctime-epo),tmodel2,color='seagreen',alpha=0.1,lw=3.0)
    
    #plt.xlim(-2*tdur,)
    #plt.ylim(tdepth-4*std,1+4*std)

    plt.xlabel('Phase (hours)') #x-label
    plt.ylabel('Relative Flux') #y-label
    x1,x2,y1,y2 = plt.axis()    #get range of plot
    #x1=-2*tdur
    #x2=2*tdur
    x1=xmin*24.0
    x2=xmax*24.0
    y1=tdepth-4*std
    y2=1+4*std
    plt.axis((x1,x2,y1,y2)) #readjust range

    if koi_id!=None:
        axis=plt.axis()
        pstr=str(koi_id)
        plt.text(axis[1]-0.20*(axis[1]-axis[0]),axis[3]-0.1*(axis[3]-axis[2]),pstr)

    if savefig!='null':
        plt.savefig(savefig)
    plt.show()  #show the plot
    
    plt.show()


def makekernelm32(phot1,phot2,dscale,ascale,lscale,diffs):
    """Make Kernel for Matern 3/2 model
    """
    
    sqrtthree=np.sqrt(3)
    dscale2=np.power(dscale,2)
    ascale2=np.power(ascale,2)
    npt1=len(phot1.time)
    npt2=len(phot2.time)
    #print('n:',npt1,npt2)
    kernel=np.zeros((npt1,npt2))
    for i in range(npt1):
        kernel[i,:]=ascale2*(1+sqrtthree*diffs[i,:]/lscale)*\
          np.exp(-sqrtthree*diffs[i,:]/lscale)
        kernel[i,i]+=phot1.ferr[i]*phot2.ferr[i]*dscale2
            
    return kernel

def calcdiffs(phot1,phot2):
    
    npt1=len(phot1.time)
    npt2=len(phot2.time)
    diffs=np.zeros((npt1,npt2))
    for i in range(npt1):
        for j in range(npt2):
            diffs[i,j]=np.abs(phot1.time[i]-phot2.time[j])
            
    return diffs


def setup_priors(sol,nhyper):
    """Setup default priors for the transit model.
    Note: rmin and rmax contain parameter boundaries for valid models."""
    
    priors=[None]*(len(sol)+int(nhyper))
    
    npar_sys=8 #number of system-wide parameters (e.g., limb-darkening)
    npar_pln=10 #planet specific parameters (e.g., r/R*)
    
    npl=int(len(sol)) #number of parameters
    for i in range(npl):
        if (np.mod(i-npar_sys,npar_pln)==0) & (i>=npar_sys):
            priors[i]=[sol[i],10.0,10.0] #generous prior on EPO
        if (np.mod(i-npar_sys,npar_pln)==1) & (i>=npar_sys):
            priors[i]=[sol[i],10.0,10.0] #generous prior on PER

    
    return priors


def write_n1dat(koi_id,sol,serr,modeltype,datadir,filedir):

    koi_dir=datadir+'koi'+str(int(koi_id))+'.n/'
    npl=int(100*(0.001+koi_id-np.floor(koi_id))) #planet number
    nfile = koi_dir+'n'+str(npl)+'.dat'

    npars=0
    titles=[]
    output=subprocess.check_output(["cat",nfile]).decode('ascii')
    for row in output.split('\n'):
        nline=row.split()
        if len(nline) == 5:
            npars+=1
            titles.append(nline[0])
    
    #create n1.dat file
    nfilenew='n_'
    if modeltype[0]==1:
        nfilenew+='ld_'
    if modeltype[1]==1:
        nfilenew+='rh_'
    if modeltype[2]==1:
        nfilenew+='ec_'
    if modeltype[3]==1:
        nfilenew+='gpm32_'
    nfilenew+=str(npl)+'.dat' #MCMC results are stored in an HDF5 file
    
    
    f=open(filedir+nfilenew,'w')
    for i in range(npars):
        f.write('%s %.10e %.10e %.10e %.10e\n' %(titles[i],sol[i],0.0,serr[i],0.0))
    f.close()

def addeccn(serr):

    npar_sys=8 #number of system-wide parameters (e.g., limb-darkening)
    npar_pln=10 #planet specific parameters (e.g., r/R*)
    
    npl=int(len(serr)) #number of parameters
    for i in range(npl):
        if (np.mod(i-npar_sys,npar_pln)==4) & (i>=npar_sys):
            serr[i]=0.003
        if (np.mod(i-npar_sys,npar_pln)==5) & (i>=npar_sys):
            serr[i]=0.003
            
    return serr

def updateT0(sol,phot):
    
    npar_sys=8 #number of system-wide parameters (e.g., limb-darkening)
    npar_pln=10 #planet specific parameters (e.g., r/R*)
    
    npars=len(sol)
    npl=int((npars-npar_sys)/npar_pln)
    
    for i in range(npl):
        t0=sol[npar_sys+i*npar_pln+0]
        per=sol[npar_sys+i*npar_pln+1]
        
        if per>0.0:

            if len(phot.time)>0:
                tmin=np.min(phot.time)
                tmax=np.max(phot.time)

                tmid=(tmin+tmax)/2.0

                #print(tmid)
                tcorr=int((tmid-t0)/per)*per
            else:
                tcorr=0

            sol[npar_sys+i*npar_pln+0]+=tcorr


    return sol;

def get_limb_q1q2(teff,logg,feh,bindir):
    
    output=subprocess.check_output([bindir+"claretquad.sh",\
                                    str(teff),str(logg),str(feh)]).decode('ascii')
    for row in output.split('\n'):
        nline=row.split()
        if len(nline) == 2:
            #titles.append(nline[0])
            u1=np.float(nline[0])
            u2=np.float(nline[1])
            
            q1=np.power(u1+u2,2)
            q1=max(min(1.0,q1),0.0) # 0 <=q1<=1
            q2=u1/(2*(u1+u2))
            q2=max(min(1.0,q2),0.0) # 0 <=q2<=1
            
    return q1,q2


def limbprior(koi_id,koicat,datadir):
    """Get Limb-darkening priors
    """
    
    idx=get_idx(koi_id,koicat) #get Index for parameters from the KOI catalogue

    teff=koicat.teff[idx]
    logg=koicat.logg[idx]
    feh=koicat.feh[idx]
    q1,q2=get_limb_q1q2(teff,logg,feh,datadir)
    
    nsamp=100
    q1samp=[]
    q2samp=[]
    for i in range(nsamp):
        teff1=teff+np.random.normal()*koicat.teff_e[idx]
        pchoose=np.random.rand()
        if pchoose>=0.5:
            logg1=logg+np.random.normal()*koicat.logg_ep[idx]
        else:
            logg1=logg+np.random.normal()*koicat.logg_em[idx]
        feh1=feh+np.random.normal()*koicat.feh_e[idx]
        q1_1,q2_1=get_limb_q1q2(teff1,logg1,feh1,datadir)
        q1samp.append(q1_1)
        q2samp.append(q2_1)
        #print(q1_1,q2_1,teff1,logg1,feh1)
        
    q1samp=np.array(q1samp)
    q2samp=np.array(q2samp)
    q1sig=np.std(q1samp)
    q2sig=np.std(q2samp)
        
    
    return q1,q2,q1sig,q2sig



def updatelimb(sol,serr):

    q1=np.power(sol[1]+sol[2],2)
    q1=max(min(1.0,q1),0.0) # 0 <=q1<=1
    q2=sol[1]/(2*(sol[1]+sol[2]))
    q2=max(min(1.0,q2),0.0) # 0 <=q2<=1
    sol[1]=0.0
    sol[2]=0.0
    sol[3]=q1
    sol[4]=q2
    serr[1]=0.0
    serr[2]=0.0
    serr[3]=0.003
    serr[4]=0.003

    return sol,serr

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def transitplot_wchains(time,flux,sol,serr,chain,burnin,nplanetplot=1, \
        itime=-1, ntt=0, tobs=0, omc=0, dtype=0, koi_id=None, savefig='null'):
    "plot the transit model"
    nplanet=int((len(sol)-8)/10) #number of planets

    #deal with input vars and translating to FORTRAN friendly.

    if type(itime) is int :
        if itime < 0 :
            itime=np.ones(len(time))*0.020434
        else:
            itime=np.ones(len(time))*float(itime)

    if type(ntt) is int :
        nttin=  np.zeros(nplanet, dtype="int32") #number of TTVs measured
        tobsin= np.zeros(shape=(nplanet,len(time))) #time stamps of TTV measurements (days)
        omcin=  np.zeros(shape=(nplanet,len(time))) #TTV measurements (O-C) (days)
    else:
        nttin=ntt
        tobsin=tobs
        omcin=omc

    if type(dtype) is int :
        dtypein=np.ones(len(time), dtype="int32")*int(dtype) #contains data type, 0-photometry,1=RV data

    #remove other planets for plotting
    sol2=np.copy(sol)
    for i in range(1,nplanet+1):
        if i!=nplanetplot:
            nc=8+10*(i-1)
            sol2[nc+3]=0.0 #rdrs
    tmodel= np.zeros(len(time)) #contains the transit model
    tfit5.transitmodel(nplanet,sol2,time,itime,nttin,tobsin,omcin,tmodel,dtypein)
    stdev=np.std(flux-tmodel)
    #print(stdev)

    #make a model with only the other transits to subtract
    nc=8+10*(nplanetplot-1)
    sol2=np.copy(sol)
    sol2[nc+3]=0.0 #rdrs
    tmodel2= np.zeros(len(time)) #contains the transit model
    tfit5.transitmodel(nplanet,sol2,time,itime,nttin,tobsin,omcin,tmodel2,dtypein)

    epo=sol[nc+0] #time of center of transit
    per=sol[nc+1] #orbital period
    zpt=sol[7] #photometric zero-point

    tdur=tfit5.transitdur(sol,1)/3600.0 #transit duration in hours

    ph1=epo/per-math.floor(epo/per) #calculate phases
    phase=[]
    tcor=tfit5.lininterp(tobsin,omcin,nplanetplot,nttin,epo)
    #print(tcor,nttin,tobsin[1,1],omcin[1,1])
    for x in time:
        if nttin[nplanetplot-1] > 0:
            tcor=tfit5.lininterp(tobsin,omcin,nplanetplot,nttin,x)
        else:
            tcor=0.0
        t=x-tcor
        ph=(t/per-math.floor(t/per)-ph1)*per*24.0 #phase in hours offset to zero.
        phase.append(ph)

    phase = np.array(phase) #convert from list to array
    phase[phase>0.5*per*24.0]+=-1*per*24.0
    phase[phase<-0.5*per*24.0]+=1*per*24.0

    phasesort=np.copy(phase)
    fluxsort=np.copy(tmodel)
    p=np.ones(len(phase), dtype="int32") #allocate array for output. FORTRAN needs int32
    tfit5.rqsort(phase,p)
    i1=0
    i2=len(phase)-1
    for i in range(0,len(phase)):
        phasesort[i]=phase[p[i]-1]
        fluxsort[i]=tmodel[p[i]-1]
        if phasesort[i] < -tdur:
            i1=i
        if phasesort[i] <  tdur:
            i2=i

    fplot=flux-tmodel2+1.0

    plt.figure(figsize=(12,10)) #adjust size of figure
    matplotlib.rcParams.update({'font.size': 22}) #adjust font
    plt.scatter(phase, fplot, c="blue", s=100.0, alpha=0.35, edgecolors="none") #scatter plot
    plt.plot(phasesort, fluxsort, c="red", lw=3.0)


    #Add MCMC
    mctime=np.linspace(epo-2*tdur/24.0,epo+2*tdur/24.0,1000)
    mcitime=np.ones(len(mctime))*np.median(itime)
    nmcmc=chain.shape[0]

    for ii in range(100):
        nchain=int(np.random.rand()*(nmcmc-burnin)+burnin)
        #remove other planets for plotting
        sol2=np.copy(sol)
        npars=len(sol)
        j=-1
        for i in range(npars):
            if np.abs(serr[i]) > 1.0e-30:
                j=j+1
                #if i==0:
                #    sol2[i]=np.copy(np.exp(chain[nchain,j])) #lnp -> p
                #else:
                #    sol2[i]=np.copy(chain[nchain,j])
                sol2[i]=np.copy(chain[nchain,j])

        for i in range(1,nplanet+1):
            if i!=nplanetplot:
                nc=8+10*(i-1)
                sol2[nc+3]=0.0 #rdrs
        tmodel2 = np.zeros(len(mctime)) #contains the transit model
        dtypein2=np.ones(len(mctime), dtype="int32")*int(dtype)
        tfit5.transitmodel(nplanet,sol2,mctime,mcitime,nttin,tobsin,omcin,tmodel2,dtypein2)
        plt.plot(24*(mctime-epo),tmodel2,color='seagreen',alpha=0.1)


    plt.xlabel('Phase (hours)') #x-label
    plt.ylabel('Relative Flux') #y-label
    x1,x2,y1,y2 = plt.axis()    #get range of plot
    ymin=min(fluxsort[i1:i2])
    ymax=max(fluxsort[i1:i2])
    y1=ymin-0.1*(ymax-ymin)-2.0*stdev
    y2=ymax+0.1*(ymax-ymin)+2.0*stdev
    plt.axis((-2.0*tdur,2.0*tdur,y1,y2)) #readjust range

    if koi_id!=None:
        axis=plt.axis()
        pstr=str(koi_id)
        plt.text(axis[1]-0.20*(axis[1]-axis[0]),axis[3]-0.1*(axis[3]-axis[2]),pstr)

    if savefig!='null':
        plt.savefig(savefig)
    plt.show()  #show the plot

    return;

def get_dtime(phot):
    
    mintime=np.min(phot.time)
    maxtime=np.max(phot.time)
    
    obslen=maxtime-mintime
    
    dt=[]
    sortidx=np.argsort(phot.time) #Sort by KOI Number.
    for i in sortidx[1:]:
        dt.append(phot.time[i]-phot.time[i-1])
    dt=np.array(dt)
    
    mediandt=np.median(dt)
    
    return obslen,mediandt

def setup_gibbs():
    #setting up walkers.  First array gives amount of shuffle to add to good solution.  Second is min bounds on
    #variable, third is max bounds on variable.
    #               rho.     c1     c2.     q1    q2.    dil   vof   zpt.     epo.   per    b    rdr     ecw    esw
    #              krv     ted    ell    alb    DSC,     a       l
    rsuf=np.array([ 0.003 ,3.0e-4,3.0e-4,3.0e-4,3.0e-4,0.001, 0.1,   1.0e-6, 1.0e-6,1.0e-8, 0.01,1.0e-6, 0.001, 0.001,\
                    0.1,    0.1,  0.1,   0.1,   0.5,    1.0,    1.0])
    rmin=np.array([ 1.0e-5,-10.0 ,-10.0 ,0.0000,0.0000,0.000,-1.0e6,-1.0   ,-1.0e+5,1.0e-6, 0.0 ,0.0   ,-1.000,-1.000,\
                   -1.0e6,-1.0e6,-1.0e6,-1.0e6, 0.0   , 0.0,    0.0])
    rmax=np.array([ 1.0e3 , 10.0 , 10.0 ,1.0000,1.0000,1.000, 1.0e6, 1.0   , 1.0e+5,1.0e+6,10.0 ,2.0   , 1.000, 1.000,\
                    1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e10, 1.0e10, 1.0e10 ])

    #DSC - scale for photometry errors
    #l - length scale for GPs

    return rsuf,rmin,rmax;

def setup_walkers(sol,serr,phot,rsuf,rmin,rmax,nwalkers,koi_id,koicat,modeltype):

    npar_sys=8 #number of system-wide parameters (e.g., limb-darkening)
    npar_pln=10 #planet specific parameters (e.g., r/R*)
    eps=np.finfo(float).eps #A small number
    favg=np.median(phot.ferr) #average error 
    
    idx=get_idx(koi_id,koicat) #get Index for parameters from the KOI catalogue

    nbodies=int((len(sol)-npar_sys)/npar_sys)
    npl=npar_sys+nbodies*npar_pln
    x=[]
    xerr=[]
    for i in range(npl): #npl only includes transit model parameters.
        if np.abs(serr[i]) > 1.0e-30:
            #adding different parameterizations
            if i==0:
                errm=(koicat.rhostarmep[idx]-koicat.rhostarmem[idx])/2.0
                serr[i] = np.max([serr[i],errm])
                #x.append(np.log(np.abs(sol[i])))
                x.append(sol[i])
            elif (np.mod(i-npar_sys,npar_pln)==0) & (i>=npar_sys): #T0
                serr[i] = np.max([serr[i],koicat.t0err[idx]])
                x.append(sol[i])
            elif (np.mod(i-npar_sys,npar_pln)==1) & (i>=npar_sys): #per
                serr[i] = np.max([serr[i],koicat.pererr[idx]])
                x.append(sol[i])
            elif (np.mod(i-npar_sys,npar_pln)==2) & (i>=npar_sys): #b
                errm=(koicat.bep[idx]-koicat.bem[idx])/2.0
                serr[i] = np.max([serr[i],errm])  
                x.append(sol[i])
            elif (np.mod(i-npar_sys,npar_pln)==3) & (i>=npar_sys): #r/r*
                errm=(koicat.rdrsep[idx]-koicat.rdrsem[idx])/2.0
                serr[i] = np.max([serr[i],errm])
                #x.append(np.log(np.abs(sol[i])))
                x.append(sol[i])
            else:
                x.append(sol[i])

            if serr[i]<0:
                if i<npar_sys:
                    print(i,rsuf[i],rmin[i],rmax[i])
                    xerr.append(rsuf[i])
                if i>=npar_sys:
                    j=i-int((i-npar_sys+1)/npar_pln)
                    print(j)
                    xerr.append(rsuf[j])
            else:
                xerr.append(serr[i])
        #print(sol[i],serr[i])
    x.append(1.0) #error scale
    xerr.append(0.1)
    if modeltype[3]==1:
        x.append(favg) #ascale
        xerr.append(0.1)
        x.append(1.0) #lscale
        xerr.append(0.1)
    x=np.array(x)
    xerr=np.array(xerr)

    ndim = len(x)

    p0=np.zeros((nwalkers,len(x)))
    k=-1
    for j in range(npar_sys):
        if np.abs(serr[j]) > 1.0e-30:
            k=k+1
            for ii in range(nwalkers):
                maxiter=10
                iter=0
                p0[ii,k]=1.0e30
                while p0[ii,k] >= rmax[j]+eps or p0[ii,k] <= rmin[j]-eps:
                    iter=iter+1
                    if iter < maxiter:
                        p0[ii,k]=x[k]+np.random.normal(scale=xerr[k])
                    else:
                        p0[ii,k]=np.min([np.max([x[k]+np.random.normal(scale=xerr[k]),rmin[j]]),\
                            rmax[j]])   
    for i in range(nbodies):
        for j in range(npar_sys):
            n=npar_sys+i*npar_pln+j
            if np.abs(serr[n]) > 1.0e-30:
                k=k+1
                for ii in range(nwalkers):
                    maxiter=10
                    iter=0
                    p0[ii,k]=1.0e30
                    while p0[ii,k] >= rmax[j+npar_sys]-eps or p0[ii,k] <= rmin[j+npar_sys]+eps:
                        iter=iter+1
                        if iter < maxiter:
                            p0[ii,k]=x[k]+np.random.normal(scale=xerr[k])
                        else:
                            p0[ii,k]=np.min([np.max([x[k]+\
                                                     np.random.normal(scale=xerr[k]),\
                                                     rmin[j+npar_sys]]),rmax[j+npar_sys]])
                            #print(ii,k,p0[ii,k],x[k],rmin[j+npar_sys],rmax[j+npar_sys])
    
    if modeltype[3]==0:
        p0[:,[len(x)-1]]=[np.random.rand(1)+0.5 for ii in range(nwalkers)] #deal with dscale
    elif modeltype[3]==1:
        p0[:,[len(x)-3]]=[np.random.rand(1)+0.5 for ii in range(nwalkers)] #deal with dscale
        p0[:,[len(x)-2]]=[(np.random.rand(1)+0.5)*favg for ii in range(nwalkers)] #deal with ascale
        obslen,mediandt=get_dtime(phot)
        tdur=tfit5.transitdur(sol,1)/86400.0 #Transit duration in days.
        p0[:,[len(x)-1]]=[np.random.rand(1)*(2*tdur-2*mediandt)+2*mediandt \
          for ii in range(nwalkers)] #deal with lscale
    p0[0,:]=np.copy(x) #make our good solution a walker
    
    return x,p0,ndim

def get_data_tpars(koi_id,koicat,datadir,raw=0,tdurcut=2.0):
    
    idx=get_idx(koi_id,koicat) #get Index for parameters from the KOI catalogue
    koi_dir=datadir+'koi'+str(int(koi_id))+'.n/'
    
    npl=int(100*(0.001+koi_id-np.floor(koi_id))) #planet number
    
    phot,photflag=getphot(koi_dir,koi_id,koicat,idx,npl,raw=raw,tdurcut=tdurcut)
    
    nfile = koi_dir+'n'+str(npl)+'.dat'
    #nfile = koi_dir+'n0.dat'
    if os.path.isfile(nfile):
        sol,serr=readsol(nfile) #read in a transit solution
        solflag=1

        #update errors from previous MCMC
        npars=len(sol)
        for i in range(npars):
            if np.abs(serr[i])>1.0e-10:
                if i==0:
                    err1=(koicat.rhostar_ep[idx]-koicat.rhostar_em[idx])/2
                    if err1>0:
                        serr[i]=np.copy(err1)
                elif i==8:
                    err1=koicat.t0err[idx]
                    if err1>0:
                        serr[i]=np.copy(err1)
                elif i==9:
                    err1=koicat.pererr[idx]
                    if err1>0:
                        serr[i]=np.copy(err1)
                elif i==10:
                    err1=(koicat.bep[idx]-koicat.bem[idx])/2
                    if err1>0:
                        serr[i]=np.copy(err1)
                elif i==11:
                    err1=(koicat.rdrsep[idx]-koicat.rdrsem[idx])/2
                    if err1>0:
                        serr[i]=np.copy(err1)

    else:
        sol=[]
        serr=[]
        solflag=0
        
    ttfile=[]
    ttfile.append(koi_dir+'koi'+str(int(koi_id)).zfill(4)+'.0'+str(npl)+'.tt')
    if os.path.isfile(ttfile[0]):
        ntt,tobs,omc = readtt(ttfile) #read in TTVs
        ttflag=1
    else:
        ntt=0; tobs=0; omc=0;
        ttflag=0

    #fix zero point
    sol[7]=0
    ans = transitmodel(sol, phot.time, itime=phot.itime, ntt=ntt, tobs=tobs, omc=omc)
    sol[7]=np.median(phot.flux-ans)
    serr[7]=np.std(phot.flux-ans)/np.sqrt(len(phot.time))

    #add errors for photflag and solflag

    return phot,photflag,sol,serr,solflag,ntt,tobs,omc,ttflag

def get_data_sol(koi_id,koicat,datadir,filedir,modeltype,raw=1):
    
    idx=get_idx(koi_id,koicat) #get Index for parameters from the KOI catalogue
    koi_dir=datadir+'koi'+str(int(koi_id))+'.n/'

    chainfile=filedir+'chain_'
    if modeltype[0]==1:
        chainfile+='ld_'
    if modeltype[1]==1:
        chainfile+='rh_'
    if modeltype[2]==1:
        chainfile+='ec_'
    if modeltype[3]==1:
        chainfile+='gpm32_'
    chainfile+=str(koi_id)+'.h5' #MCMC results are stored in an HDF5 file

    npl=int(100*(0.001+koi_id-np.floor(koi_id))) #planet number

    phot,photflag=getphot(koi_dir,koi_id,koicat,idx,npl,raw=raw)
    #print('got phot..')

    nfile = koi_dir+'n'+str(npl)+'.dat'
    #nfile = koi_dir+'n0.dat'
    if os.path.isfile(nfile):
        sol,serr=readsol(nfile) #read in a transit solution
        solflag=1

        #update errors from previous MCMC
        npars=len(sol)
        for i in range(npars):
            if np.abs(serr[i])>1.0e-10:
                if i==0:
                    err1=(koicat.rhostar_ep[idx]-koicat.rhostar_em[idx])/2
                    if err1>0:
                        serr[i]=np.copy(err1)
                elif i==8:
                    err1=koicat.t0err[idx]
                    if err1>0:
                        serr[i]=np.copy(err1)
                elif i==9:
                    err1=koicat.pererr[idx]
                    if err1>0:
                        serr[i]=np.copy(err1)
                elif i==10:
                    err1=(koicat.bep[idx]-koicat.bem[idx])/2
                    if err1>0:
                        serr[i]=np.copy(err1)
                elif i==11:
                    err1=(koicat.rdrsep[idx]-koicat.rdrsem[idx])/2
                    if err1>0:
                        serr[i]=np.copy(err1)

    else:
        sol=[]
        serr=[]
        solflag=0

    ttfile=[]
    ttfile.append(koi_dir+'koi'+str(int(koi_id)).zfill(4)+'.0'+str(npl)+'.tt')
    if os.path.isfile(ttfile[0]):
        ntt,tobs,omc = readtt(ttfile) #read in TTVs
        ttflag=1
    else:
        ntt=0; tobs=0; omc=0;
        ttflag=0
    
    #fix zero point
    sol[7]=0
    ans = transitmodel(sol, phot.time, itime=phot.itime, ntt=ntt, tobs=tobs, omc=omc)
    sol[7]=np.median(phot.flux-ans)
    serr[7]=np.std(phot.flux-ans)/np.sqrt(len(phot.time))

    #add errors for photflag and solflag

    return phot,photflag,sol,serr,solflag,ntt,tobs,omc,ttflag,chainfile

def getphot(koi_dir,koi_id,koicat,idx,npl,tdurcut=2,raw=1):
    
    npar_sys=8 #number of system-wide parameters (e.g., limb-darkening)
    npar_pln=10 #planet specific parameters (e.g., r/R*)
    
    #read in n0.dat data
    nfile0 = koi_dir+'n0.dat'
    kid=koicat.kid[idx] #Kepler-ID
    if raw == 1:
        photfile0 = koi_dir+'klc'+str(kid).zfill(8)+'.dat'
    else:
        photfile0 = koi_dir+'klc'+str(kid).zfill(8)+'.dc.dat'
    #print(nfile0,photfile0)
    #print(nfile0,os.path.isfile(nfile0))
    #print(photfile0,os.path.isfile(photfile0))
    if os.path.isfile(nfile0) & os.path.isfile(photfile0):
        sol0,serr0=readsol(nfile0) #read in a transit solution
        phot0=phot_class()
        phot0=readphotometry(photfile0,phot0) # Read in photometry
        
        phot0.time = phot0.time-54900+0.5 #correction for Kepler Data
        
        #get number of planets
        nplanet=int((len(sol0)-npar_sys)/npar_pln) #number of planets
        
        #read in TTVs if present
        tfiles=[]
        ntfiles=0
        for i in range(1,nplanet+1):
            ttfile = koi_dir+'koi'+str(int(koi_id)).zfill(4)+'.0'+str(i)+'.tt'
            if os.path.isfile(ttfile):
                #print(ttfile)
                tfiles.append(ttfile)
                ntfiles+=1
            else:
                tfiles.append('null')
        if ntfiles>0:
            #print(tfiles)
            ntt,tobs,omc = readtt(tfiles) #read in TTVs
            ttflag=1
        else:
            ntt=0; tobs=0; omc=0;
            ttflag=0
            
        #remove other planets
        nc=8+10*(npl-1)
        sol2=np.copy(sol0)
        sol2[7]=0.0 #zpt -- keep offset.
        sol2[nc+3]=0.0 #rdrs
        ans2 = transitmodel(sol2, phot0.time, itime=phot0.itime, ntt=ntt, tobs=tobs, omc=omc)
        
        #keep only data near transit
        epo=sol0[nc+0] #time of center of transit
        per=sol0[nc+1] #orbital period
        zpt=sol0[7] #photometric zero-point
        tdur=tfit5.transitdur(sol0,npl)/86400.0/per #transit duration in days
        
        phase=(phot0.time-epo)/per-np.floor((phot0.time-epo)/per)
        phase[phase<-0.5]+=1.0
        phase[phase>0.5]-=1.0
        
        phot=phot_class()
        if tdurcut>0:
            phot.time=phot0.time[(phase>-tdurcut*tdur)&(phase<tdurcut*tdur)]
            phot.flux=phot0.flux[(phase>-tdurcut*tdur)&(phase<tdurcut*tdur)]\
                -ans2[(phase>-tdurcut*tdur)&(phase<tdurcut*tdur)]+1.0
            phot.ferr=phot0.ferr[(phase>-tdurcut*tdur)&(phase<tdurcut*tdur)]
            phot.itime=phot0.itime[(phase>-tdurcut*tdur)&(phase<tdurcut*tdur)]
        else:
            phot.time=np.copy(phot0.time)
            phot.flux=np.copy(phot0.flux)
            phot.ferr=np.copy(phot0.ferr)
            phot.itime=np.copy(phot0.itime)

        #undo Kepler-offset
        phot.time = phot.time+54900-0.5
        
        photflag=1
        
        #plt.plot(phot.time,phot.flux)
        #plt.show()
        
    else: #if n0.dat is missing or klcXX.dc.dat, fall back to getting tremove.1.dat data
        photfile=koi_dir+'tremove.'+str(npl)+'.dat'
        if os.path.isfile(photfile):
            phot=phot_()
            phot =readphotometry(photfile,phot) # Read in photometry
            photflag=1
        else:
            phot=[]
            photflag=0
            
    return phot,photflag;

def histplots(chain,sol,serr,burnin,nbin,label,colour):

    npar_sys=8 #number of system-wide parameters (e.g., limb-darkening)
    npar_pln=10 #planet specific parameters (e.g., r/R*)

    nbodies=int((len(sol)-npar_sys)/npar_pln)
    npl=npar_sys+nbodies*npar_pln

    matplotlib.rcParams.update({'font.size': 10}) #adjust font
    plt.figure(1, figsize=(20, 8*nbodies))
    nullfmt = NullFormatter()

    n1=int(nbodies+1) #added 1
    n2=int(npar_pln)

    jpar=-1
    for j in range(npar_sys):
        if np.abs(serr[j]) > 1.0e-30:
            jpar=jpar+1
            npanel=int(jpar+1)
            axScatter=plt.subplot(n1, n2, npanel)

            minx=np.min(chain[burnin:,jpar]) #range of parameter
            maxx=np.max(chain[burnin:,jpar])

            x_eval = np.linspace(minx, maxx, num=100) #make a uniform sample across the parameter range

            kde1 = stats.gaussian_kde(chain[burnin:,jpar],0.3) #Kernel Density Estimate

            #plot the histogram
            plt.hist(chain[burnin:,jpar],nbin,histtype='stepfilled', density=True, facecolor=colour[jpar],\
                     alpha=0.6)

            #overlay the KDE
            plt.plot(x_eval, kde1(x_eval), 'k-', lw=3)

            plt.xlabel(label[jpar])
            #plt.ylabel('Probability Density')
            axScatter.yaxis.set_major_formatter(nullfmt)

    #Dscale parameter
    npanel=npanel+1
    axScatter=plt.subplot(n1, n2, npanel)
    k=chain.shape[1]-1
    minx=np.min(chain[burnin:,k]) #range of parameter
    maxx=np.max(chain[burnin:,k])
    x_eval = np.linspace(minx, maxx, num=100) #make a uniform sample across the parameter range
    kde1 = stats.gaussian_kde(chain[burnin:,k],0.3) #Kernel Density Estimate
    #plot the histogram
    plt.hist(chain[burnin:,k],nbin,histtype='stepfilled', density=True, facecolor=colour[k], alpha=0.6)
    #overlay the KDE
    plt.plot(x_eval, kde1(x_eval), 'k-', lw=3)
    plt.xlabel(label[k])
    #plt.ylabel('Probability Density')
    axScatter.yaxis.set_major_formatter(nullfmt)

    for i in range(nbodies):
        for j in range(npar_sys):
            n=npar_sys+i*npar_pln+j
            #print(n,j+7)
            if np.abs(serr[n]) > 1.0e-30:
#                print(i*n1+j,j)
                jpar=jpar+1
                npanel=int((i+1)*n2+j+1)
                axScatter=plt.subplot(n1, n2, npanel)

                minx=np.min(chain[burnin:,jpar]) #range of parameter
                maxx=np.max(chain[burnin:,jpar])

                x_eval = np.linspace(minx, maxx, num=100) #make a uniform sample across the parameter range

                kde1 = stats.gaussian_kde(chain[burnin:,jpar],0.3) #Kernel Density Estimate

                #plot the histogram
                plt.hist(chain[burnin:,jpar],nbin,histtype='stepfilled', density=True, facecolor=colour[jpar],\
                     alpha=0.6)

                #overlay the KDE
                plt.plot(x_eval, kde1(x_eval), 'k-', lw=3)

                plt.xlabel(label[jpar])
                #plt.ylabel('Probability Density')
                axScatter.yaxis.set_major_formatter(nullfmt)

    plt.show()

def plotchains(chain,label,colour,burnin,psize=0.1):
    nullfmt = NullFormatter()
    matplotlib.rcParams.update({'font.size': 10}) #adjust font
    plt.figure(figsize=(12,37)) #adjust size of figure

    x=np.arange(burnin+1,len(chain)+1,1)
    npar=len(chain[0,:])
    for i in range(0,npar):
        axScatter=plt.subplot(npar, 1, i+1)
        plt.scatter(x,chain[burnin:,i],c=colour[i],s=psize,alpha=0.1)  #plot parameter a
        plt.ylabel(label[i])                   #y-label

        x1,x2,y1,y2 = plt.axis()
        y1=np.min(chain[burnin:,i])
        y2=np.max(chain[burnin:,i])
        plt.axis((x1,x2,y1,y2))


        if i < npar-1:
            axScatter.xaxis.set_major_formatter(nullfmt)

    plt.xlabel('Iteration')           #x-label

    plt.show()

def modekdestimate(chain,burnin):
    'Estimate Mode with KDE and return KDE'
    #range of data
    minx=np.min(chain[burnin:])
    maxx=np.max(chain[burnin:])
    x_eval = np.linspace(minx, maxx, num=1000)
    kde1 = stats.gaussian_kde(chain[burnin:])#,0.3)
    modeval=[]
    modekde=0
    for x in x_eval:
        if kde1(x) > modekde:
            modekde=kde1(x)
            modeval=x
    return modeval,x_eval,kde1 ;

def intperc(x,x_eval,kde1,perc=0.6827):
    'find error bounds'
    idx = (np.abs(x_eval-x)).argmin()
    kdea=np.array(kde1(x_eval))

    n=len(x_eval)

    #print(x,idx)

    i1=1
    i2=1
    intval=0.0

    j1=np.copy(idx)
    j2=np.copy(idx)
    j1old=np.copy(j1)
    j2old=np.copy(j2)
    while intval < perc:
        j1test=np.max((0,idx-i1-1))
        j2test=np.min((n-1,idx+i2+1))
        if kdea[j1test] > kdea[j2test]:
            if j1test>0:
                j1=np.copy(j1test)
                i1=i1+1
            else:
                j1=np.copy(j1test)
                j2=np.copy(j2test)
                i2=i2+1
            #print('case1')
        else:
            if j2test<n-1:
                j2=np.copy(j2test)
                i2=i2+1
            else:
                j2=np.copy(j2test)
                j1=np.copy(j1test)
                i1=i1+1
            #print('case2')

        intval=np.trapz(kdea[j1:j2],x_eval[j1:j2])
        #print(j1,j2,intval,kdea[j1test],kdea[j2test])

        #make sure we can break from loop
        if (j1 == 0) and (j2 == n-1):  #break we reach boundaries of array
            #print('break1')
            intval=1.0
        if (j1 == j1old) and (j2 == j2old): #break if stuck in loop.
            #print('break2')
            intval=1.0

        #Update old values to check we are making progress.
        j1old=np.copy(j1)
        j2old=np.copy(j2)

    #print(x_eval[j1],x_eval[j2])
    return x_eval[j1],x_eval[j2];

class phot_class:
    def __init__(self):
        self.time=[]  #initialize arrays
        self.flux=[]
        self.ferr=[]
        self.itime=[]

class koicat_class:
    def __init__(self):
        #IDs
        self.kid=[]  #Kepler ID [0]
        self.koi=[]  #KOI [1]
        #model parameters
        self.rhostarm=[] #rhostar model [31]
        self.rhostarmep=[] #+error in rhostar [32]
        self.rhostarmem=[] #-error in rhostar [33]
        self.t0=[] #model T0 [4]
        self.t0err=[] #error in T0 [5]
        self.per=[] #period [2]
        self.pererr=[] #error in period [3]
        self.b=[] #impact parameter [9]
        self.bep=[] #+error in b [10]
        self.bem=[] #-error in b [11]
        self.rdrs=[] #model r/R* [6]
        self.rdrsep=[] #+error in r/R* [7]
        self.rdrsem=[] #-error in r/R* [8]
        #stellar parameters
        self.rstar=[] #stellar radius [39]
        self.rstar_ep=[] #stellar radius +err [40]
        self.rstar_em=[] #stellar radius -err [41]
        self.teff=[] #Teff [37]
        self.teff_e=[] #Teff error [38]
        self.rhostar=[] #rhostar [34]
        self.rhostar_ep=[] #rhostar +err [35]
        self.rhostar_em=[] #rhostar -err [36]
        self.logg=[] #stellar radius [45]
        self.logg_ep=[] #stellar radius +err [46]
        self.logg_em=[] #stellar radius -err [47]
        self.feh=[] #metallicity [48]
        self.feh_e=[] #metallicity error [49]
        #disposition
        self.statusflag=[]

def readarchcat(koipropsfile,koicat):
    """Read in KOI Catalogue based on Architecture Work
    Usage: koicat=readarchcat(koipropsfile)
    """
    icount=0

    f = open(koipropsfile, 'r')
    for line in f:
        line = line.strip()
        columns = line.split(',') #break into columns
        if(columns[0][0] != '#'):
            if(icount > 0):  #skip header info
                koicat.kid.append(int(columns[0]))
                koicat.koi.append(float(columns[1]))
                koicat.rhostarm.append(float(columns[13]))
                koicat.rhostarmep.append(float(columns[14]))
                koicat.rhostarmem.append(float(columns[15]))
                koicat.t0.append(float(columns[5]))
                koicat.t0err.append(float(columns[6]))
                koicat.per.append(float(columns[3]))
                koicat.pererr.append(float(columns[4]))
                koicat.b.append(float(columns[10]))
                koicat.bep.append(float(columns[11]))
                koicat.bem.append(float(columns[12]))
                koicat.rdrs.append(float(columns[7]))
                koicat.rdrsep.append(float(columns[8]))
                koicat.rdrsem.append(float(columns[9]))
                koicat.rstar.append(float(columns[52]))
                koicat.rstar_ep.append(float(columns[53]))
                koicat.rstar_em.append(float(columns[54]))
                koicat.teff.append(float(columns[50]))
                koicat.teff_e.append(float(columns[51]))
                koicat.rhostar.append(float(columns[47]))
                koicat.rhostar_ep.append(float(columns[48]))
                koicat.rhostar_em.append(float(columns[49]))
                #if np.abs(float(columns[38]))> 1.0e-10:
                #    koicat.rhostar_em.append(float(columns[38]))
                #else:
                #    koicat.rhostar_em.append(float(-0.01))
                koicat.logg.append(float(columns[58]))
                koicat.logg_ep.append(float(columns[59]))
                koicat.logg_em.append(float(columns[60]))
                koicat.feh.append(float(columns[61]))
                koicat.feh_e.append(float(columns[62]))
                koicat.statusflag.append(columns[63])
            icount+=1
    f.close()

    return koicat;

def get_idx(koi_id,koicat):
    """Given a KOI number, get the corresponding index from the KOI catalogue
    Usage: idx=get_idx(koi_id,koicat)
      koi_id - KOI number (e.g., 1.01)
      koicat - catalogue read using readarchcat()
    """
    try:
        idx=koicat.koi.index(koi_id) #get index if KOI exists
    except ValueError: #index is not found, we append KOI to the list.
        print("not found: ",koi25)
        idx=-1

    return idx;

def readtt(files):
    "reading in TT files"
    nmax=0 #we will first scan through the files to determine what size of array we need.
    for filename in files:
        #print('f:',filename)
        if filename == 'null':
            i=0
        else:
            f = open(filename,'r')
            i=0
            for line in f:
                i+=1
            f.close()
        
        nmax=max(i,nmax) #save the largest number.
        
    npl=len(files) #number of planets to read in
    ntt =zeros(npl, dtype="int32") #allocate array for number of TTVs measured
    tobs=zeros(shape=(npl,nmax)) #allocate array for timestamps
    omc =zeros(shape=(npl,nmax)) #allocate array for O-C
    
    i=-1 #counter for files scanned 
    for filename in files: #scan through all files from input
        i+=1
        if filename == 'null':
            ntt[i]=0
        else:
            f = open(filename,'r')
            j=-1 #counter for valid O-C read 
            for line in f:
                line = line.strip() #get rid of line breaks
                columns = line.split()
                if float(columns[2]) > 0.0 :
                    j+=1
                    tobs[i,j]=float(columns[0])
                    omc[i,j]=float(columns[1])
            ntt[i]=j+1
    return ntt, tobs, omc;
    

def transitplot(time,flux,sol,nplanetplot=1, itime=-1, ntt=0, tobs=0, omc=0, dtype=0):
    "plot the transit model"
    nplanet=int((len(sol)-8)/10) #number of planets

    #deal with input vars and translating to FORTRAN friendly.
    
    if type(itime) is int :
        if itime < 0 :
            itime=ones(len(time))*0.020434
        else:
            itime=ones(len(time))*float(itime)
    
    if type(ntt) is int :
        nttin=  zeros(nplanet, dtype="int32") #number of TTVs measured 
        tobsin= zeros(shape=(nplanet,len(time))) #time stamps of TTV measurements (days)
        omcin=  zeros(shape=(nplanet,len(time))) #TTV measurements (O-C) (days)
    else:
        nttin=ntt
        tobsin=tobs
        omcin=omc
 
    if type(dtype) is int :
        dtypein=ones(len(time), dtype="int32")*int(dtype) #contains data type, 0-photometry,1=RV data
    
    #remove other planets for plotting
    sol2=np.copy(sol)
    for i in range(1,nplanet+1):
        if i!=nplanetplot:
            nc=8+10*(i-1)
            sol2[nc+3]=0.0 #rdrs
    tmodel= zeros(len(time)) #contains the transit model
    tfit5.transitmodel(nplanet,sol2,time,itime,nttin,tobsin,omcin,tmodel,dtypein)
    stdev=np.std(flux-tmodel)
    #print(stdev)
    
    #make a model with only the other transits to subtract
    nc=8+10*(nplanetplot-1)
    sol2=np.copy(sol)
    sol2[7]=0.0
    sol2[nc+3]=0.0 #rdrs
    tmodel2= zeros(len(time)) #contains the transit model
    tfit5.transitmodel(nplanet,sol2,time,itime,nttin,tobsin,omcin,tmodel2,dtypein)
    
    epo=sol[nc+0] #time of center of transit
    per=sol[nc+1] #orbital period
    zpt=sol[7] #photometric zero-point
    
    tdur=tfit5.transitdur(sol,1)/3600.0 #transit duration in hours
    #check for NaNs and INFs
    if np.isnan(tdur) or np.isinf(tdur):
        tdur=2.0
    #if tdur is tiny, default to 2-hours
    if tdur<0.01:
        tdur=2.0 
    
    ph1=epo/per-math.floor(epo/per) #calculate phases
    phase=[]
    tcor=tfit5.lininterp(tobsin,omcin,nplanetplot,nttin,epo)
    #print(tcor,nttin,tobsin[1,1],omcin[1,1])
    for x in time:
        if nttin[nplanetplot-1] > 0:
            tcor=tfit5.lininterp(tobsin,omcin,nplanetplot,nttin,x)
        else:
            tcor=0.0
        t=x-tcor
        ph=(t/per-math.floor(t/per)-ph1)*per*24.0 #phase in hours offset to zero.
        phase.append(ph)
        
    phase = np.array(phase) #convert from list to array
    
    phasesort=np.copy(phase)
    fluxsort=np.copy(tmodel)
    p=ones(len(phase), dtype="int32") #allocate array for output. FORTRAN needs int32 
    tfit5.rqsort(phase,p)    
    i1=0
    i2=len(phase)-1
    for i in range(0,len(phase)):
        phasesort[i]=phase[p[i]-1]
        fluxsort[i]=tmodel[p[i]-1]
        if phasesort[i] < -tdur:
            i1=i
        if phasesort[i] <  tdur:
            i2=i
    
    fplot=flux-tmodel2+1.0
    
    plt.figure(figsize=(12,10)) #adjust size of figure
    matplotlib.rcParams.update({'font.size': 22}) #adjust font
    plt.scatter(phase, fplot, c="blue", s=100.0, alpha=0.35, edgecolors="none") #scatter plot
    plt.plot(phasesort, fluxsort, c="red", lw=3.0)
    plt.xlabel('Phase (hours)') #x-label
    plt.ylabel('Relative Flux') #y-label
    x1,x2,y1,y2 = plt.axis()    #get range of plot
    ymin=min(fluxsort[i1:i2])
    ymax=max(fluxsort[i1:i2])
    y1=ymin-0.1*(ymax-ymin)-2.0*stdev
    y2=ymax+0.1*(ymax-ymin)+2.0*stdev
    #Make sure y1!=y2
    if np.abs(y2-y1)<1.0e-10:
        y1=min(flux)
        y2=max(flux)
    plt.axis((-tdur,tdur,y1,y2)) #readjust range
    plt.show()  #show the plot
        
    return;
        

def transitmodel (sol,time, itime=-1, ntt=0, tobs=0, omc=0, dtype=0 ): 
    "read in transitmodel solution"  
    nplanet=int((len(sol)-8)/10) #number of planets
    
    if type(itime) is int :
        if itime < 0 :
            itime=ones(len(time))*0.020434
        else:
            itime=ones(len(time))*float(itime)
    
    if type(ntt) is int :
        nttin=  zeros(nplanet, dtype="int32") #number of TTVs measured 
        tobsin= zeros(shape=(nplanet,len(time))) #time stamps of TTV measurements (days)
        omcin=  zeros(shape=(nplanet,len(time))) #TTV measurements (O-C) (days)
    else:
        nttin=ntt
        tobsin=tobs
        omcin=omc
    
    if type(dtype) is int :
        dtypein=ones(len(time), dtype="int32")*int(dtype) #contains data type, 0-photometry,1=RV data

    tmodel= zeros(len(time)) #contains the transit model
    tfit5.transitmodel(nplanet,sol,time,itime,nttin,tobsin,omcin,tmodel,dtypein)
    return tmodel;

def readphotometry (filename,phot):
    "reading in Kepler photometry"

    f = open(filename, 'r')
    for line in f:
        line = line.strip() #get rid of the \n at the end of the line
        columns = line.split() #break into columns
        phot.time.append(float(columns[0])) #correct for file zero-points to get BJD-2454900
        phot.flux.append(float(columns[1])+1.0) #photometry
        phot.ferr.append(float(columns[2])) #photometric uncertainty 
        if len(columns)>=4:
            phot.itime.append(float(columns[3])) #itime will be in minutes
        else:
            phot.itime.append(float(0.0204340))
    f.close()
    phot.time = np.array(phot.time)
    phot.flux = np.array(phot.flux)
    phot.ferr = np.array(phot.ferr)
    phot.itime = np.array(phot.itime)
    return phot;

def readsol (filename):
    "read in transitmodel solution"    
    nplanetmax=9 #maximum number of planets that an n0.dat file can handle
    nplanet=0 #count number of planets found in the solution
    solin=zeros(nplanetmax*10+8) #allocate array to hold parameters. init to zero.
    serrin=zeros(nplanetmax*10+8)
    f = open(filename, 'r')
    for line in f:
        line = line.strip() #get rid of the \n at the end of the line
        columns = line.split() #break into columns
        if columns[0][0:3]=='RHO':
            solin[0]=columns[1]
            serrin[0]=columns[3]
        elif columns[0][0:3]=='NL1':
            solin[1]=columns[1]
            serrin[1]=columns[3]
        elif columns[0][0:3]=='NL2':
            solin[2]=columns[1]
            serrin[2]=columns[3]
        elif columns[0][0:3]=='NL3':
            solin[3]=columns[1]
            serrin[3]=columns[3]
        elif columns[0][0:3]=='NL4':
            solin[4]=columns[1]
            serrin[4]=columns[3]
        elif columns[0][0:3]=='DIL':
            solin[5]=columns[1]
            serrin[5]=columns[3]
        elif columns[0][0:3]=='VOF':
            solin[6]=columns[1]
            serrin[6]=columns[3]
        elif columns[0][0:3]=='ZPT':
            solin[7]=columns[1]
            serrin[7]=columns[3]
        elif columns[0][0:2]=='EP':
            np1=float(columns[0][2])
            if np1>nplanet:
                nplanet=np1
            j=int(10*(np1-1)+8+0)
            solin[j]=columns[1]
            serrin[j]=columns[3]
        elif columns[0][0:2]=='PE':
            np1=float(columns[0][2])
            if np1>nplanet:
                nplanet=np1
            j=int(10*(np1-1)+8+1)
            solin[j]=columns[1]
            serrin[j]=columns[3]
        elif columns[0][0:2]=='BB':
            np1=float(columns[0][2])
            if np1>nplanet:
                nplanet=np1
            j=int(10*(np1-1)+8+2)
            solin[j]=columns[1] 
            serrin[j]=columns[3]
        elif columns[0][0:2]=='RD':
            np1=float(columns[0][2])
            if np1>nplanet:
                nplanet=np1
            j=int(10*(np1-1)+8+3)
            solin[j]=np.abs(np.float(columns[1]))
            serrin[j]=columns[3]
        elif columns[0][0:2]=='EC':
            np1=float(columns[0][2])
            if np1>nplanet:
                nplanet=np1
            j=int(10*(np1-1)+8+4)
            solin[j]=columns[1]
            serrin[j]=columns[3]
        elif columns[0][0:2]=='ES':
            np1=float(columns[0][2])
            if np1>nplanet:
                nplanet=np1
            j=int(10*(np1-1)+8+5)
            solin[j]=columns[1]
            serrin[j]=columns[3]
        elif columns[0][0:2]=='KR':
            np1=float(columns[0][2])
            if np1>nplanet:
                nplanet=np1
            j=int(10*(np1-1)+8+6)
            solin[j]=columns[1]
            serrin[j]=columns[3]
        elif columns[0][0:2]=='TE':
            np1=float(columns[0][2])
            if np1>nplanet:
                nplanet=np1
            j=int(10*(np1-1)+8+7)
            solin[j]=columns[1]
            serrin[j]=columns[3]
        elif columns[0][0:2]=='EL':
            np1=float(columns[0][2])
            if np1>nplanet:
                nplanet=np1
            j=int(10*(np1-1)+8+8)
            solin[j]=columns[1]
            serrin[j]=columns[3]
        elif columns[0][0:2]=='AL':
            np1=float(columns[0][2])
            if np1>nplanet:
                nplanet=np1
            j=int(10*(np1-1)+8+9)
            solin[j]=columns[1]
            serrin[j]=columns[3]
    f.close()
    #print(nplanet)
    sol=solin[0:int(nplanet*10+8)]
    serr=serrin[0:int(nplanet*10+8)]
    return sol, serr;

def fittrmodel(time,flux,ferr,sol,serr,itime=-1,ntt=0,tobs=0,omc=0,dtype=0):
    nfit=108 
    npt=len(time)
    solin=zeros(nfit)
    solin[0:len(sol)]=sol
    serrin=zeros(shape=(nfit,2))
    serrin[0:len(serr),1]=serr
    nplanet=int((len(sol)-8)/10) #number of planets
    
    if type(itime) is int :
        if itime < 0 :
            itime=ones(len(time))*0.020434
        else:
            itime=ones(len(time))*float(itime)

    if type(ntt) is int :
        nttin=  zeros(nplanet, dtype="int32") #number of TTVs measured 
        tobsin= zeros(shape=(nplanet,len(time))) #time stamps of TTV measurements (days)
        omcin=  zeros(shape=(nplanet,len(time))) #TTV measurements (O-C) (days)
    else:
        nttin=ntt
        tobsin=tobs
        omcin=omc
            
    if type(dtype) is int :
        dtypein=ones(len(time), dtype="int32")*int(dtype) #contains data type, 0-photometry,1=RV data
        
    nfrho=1 #disable mean stellar density 
    rhoi=np.float(0.0)
    rhoierr=zeros(9) #contains stellar density prior (currently disabled)
    
    #fittransitmodel3(nfit,sol,serr,nplanet,npt,at,am,ae,ait,dtype,ntt,tobs,omc,nfrho,rhoi,rhoierr)
    ftf.fittransitmodel3(nfit,solin,serrin,nplanet,npt,time,flux,ferr,itime,dtypein,nttin,tobsin,\
        omcin,nfrho,rhoi,rhoierr)
    
    solout=zeros(len(sol))
    solout=solin[0:len(sol)]
    
    return solout;

def bindata(time,data,tbin):
    bin_time=[]
    bin_flux=[]
    npt=len(time)
    tmin=np.min(time)
    tmax=np.max(time)
    bin=[int((t-tmin)/tbin) for t in time]
    bin=np.array(bin)
    #nc=0
    for b in range(np.max(bin)+1):
        npt=len(bin[bin==b])
        #nc=nc+npt
        if npt>1:
            #print(npt)
            bint1=np.average(time[bin==b])
            binf1=np.average(data[bin==b])
            bin_time.append(bint1)
            bin_flux.append(binf1)
    bin_time=np.array(bin_time)
    bin_flux=np.array(bin_flux)

    #print(nc)

    return bin_time,bin_flux;
