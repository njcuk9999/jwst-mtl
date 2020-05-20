#!/usr/bin/env python
# Code to calculate NRM exposure parameters
# Deepashri Thatte and Anand Sivaramakrishnan
# Included by Johannes Sahlmann into first github package
# Double-checked & corrected 24 Nov 2015 for apt values of NINT, NGROUP calculation
# Generation of WebbPSF files was removed.
# Code expects fov 255 and fov 31 detector pixel files to support calculation.  Oversample 3 used initially.
# Updated in May 2016 to accommodate faint magnitudes for which total e- in central pixel are less than sat_e
# Updated in June 2016 to accommodate more limiting cases.

""" 
    Using Kevin Volk's zero points, based on V band photometry.  Replace with K=9??
    F277W 26.14
    F380M 23.75
    F430M 23.32
    F480M 23.19
    Target magnitude is 6.33, total  number of detected photons is 1e+10


anand@imac24:16  
"""
"""
nint_        Exact value of NINT corresponding to exact (float) value of NGROUPS
nint         Updated value of NINT corresponding to integer value of NGROUPS to account for loss of last partial group per integration 
nint_ceil    Integer value of NINT that completes last or only partial integration ----------> THIS IS NINT FOR APT USE
ngroups_     exact (float) value of NGROUPS
ngroups      integer value of NGROUPS (lost partial group gets collected in additional integrations represented by nint) ---------> THIS IS NGROUPS FOR APT USE
"""

import os, sys
import numpy as np
from astropy.io import fits
import pyami.simcode.utils as U   

# Instrument/detector 'constants' hardcoded here...
#TFRAME = 0.0745 # seconds Alex Fullerton
MAXGROUPS = 800
MAXINTS = 10000
# SAT_E = 35000.0  # Kevin Volk now comes in as a parameter.

# SRC = "A0V"
# f277w MAG_T = 6.35  for <35ke/TFRAME in cp pixctr   DTfull 11.74  Approx and old.  Numbers in 'live' code are more recent
# f380m MAG_T = 3.42  for <35ke/TFRAME in cp pixctr   DTfull  8.78
# f430m MAG_T = 2.76  for <35ke/TFRAME in cp pixctr   DTfull  8.15
# f480m MAG_T = 2.42  for <35ke/TFRAME in cp pixctr   DTfull  7.80
# MAG_T = 9.0
# TOT_E = 1.0e10 # Greeenbaum et al. ApJ, Ireland MNRAS  contrast (1e4) ~ sqrt(totalphotons)/10


def timestring(min_):
    """
    input: floating point minutes
    output: sensible time string"""

    if min_ < 0.01:
        return "%.4f minutes = %.3f seconds"%(min_, min_*60.0)
    elif 0.01 <= min_  and  min_ < 1.0:
       return "%.3f minutes = %.2f seconds"%(min_, min_*60.0)
    elif 1.0 <= min_  and  min_ < 60.0:
       return "%.3f minutes = %.3f hours"%(min_, min_/60.0)
    elif min_ >= 60.0:
       return "%.0f minutes = %.1f hours"%(min_, min_/60.0)
    else:
       return "invalid time value"



def apt_values(ngroups_, nint_):
    """
    Corrected May 26 2015 Thatte: 
    report="\n\t\tCR %.1e NINT %.1f NGROUPS_ceil %d NGROUPS_float %f NINT_new %f" %
    (cr,nint_,np.ceil(ngroups_),ngroups_,(nint_+(ngroups_-int(ngroups_))*nint_/int(ngroups_)))

    Compensates for loss of fractional part of float ngroup_

    Sum up how many of the fractional parts of the group are lost over nint_ integrations, 
    then add these back to the original number of integrations (nint_).  Returns integer values for APT use

    This function returns ngroups_ for the ngroups_apt if nint_ is 1 - eg faint source that needs only 1 integration.
    This case is treated by calling this function with ceil(ngroups_) .  Also in this case nint_apt is 1.

    """
    if ngroups_ > MAXGROUPS:
        # ngroups_ > MAXGROUPS when:
        #  i. an integration does not reach saturation but may have more than MAXGROUPS (faint object, shallow expoure)
        #  ii. Observation with more than one integration with at least one integration reaching saturation 
        #      requiring more than MAXGROUPS number of groups (faint object, deep exposure).
        ngroups_apt = MAXGROUPS
        nint_apt = nint_ + (ngroups_ - MAXGROUPS) * nint_ / MAXGROUPS
        nint_apt_ceil = np.ceil(nint_apt) 
    else:
        if nint_ == 1 and isinstance(nint_,int): 
            # isinstance(nint_,int) is used to avoid conflict when nint_ = cptot / sat_e is 1.0 (which is float)
            # nint_ = 1 for observation that does not reach saturation and ngroups_ <= MAXGROUPS                    
            ngroups_apt = np.ceil(ngroups_)
            nint_apt = nint_ 
            nint_apt_ceil = np.ceil(nint_apt)
        else:
            # Sum up how many of the fractional parts of the group are lost over nint_ integrations,
            # then add these back to the original number of integrations (nint_).  Returns integer values for APT use
            ngroups_apt = int(ngroups_)
            nint_apt = nint_ + (ngroups_ - int(ngroups_)) * nint_ / int(ngroups_)
            nint_apt_ceil = np.ceil(nint_apt)   

    return ngroups_apt, nint_apt, nint_apt_ceil


def generatePSF(filt=None, fov=None, osample=5, cr=None, tot_e=None, sat_e=None, SRC=None, return_params=0, DATADIR = "./pyami/etc/NIRISSami_apt_calcPSF/"):

    if os.access(DATADIR+'%s_%d_%s_det.fits'%(filt,fov,SRC), os.F_OK) == True:

        resfov = fits.open(DATADIR+'%s_%d_%s_det.fits'%(filt,fov,SRC))
        resbig = fits.open(DATADIR+'%s_%d_%s_det.fits'%(filt,255,SRC))
        print("Opening psf file ", DATADIR+'%s_%d_%s_det.fits'%(filt,fov,SRC))
        print("Opening psf file ", DATADIR+'%s_%d_%s_det.fits'%(filt,255,SRC))
        nrmfov, hdrfov = (resfov[0].data, resfov[0].header)
        nrmbig, hdrbig = (resbig[0].data, resbig[0].header)
        readfromdisk = True
    else:
        sys.exit("Missing PSF %s directory or PSF files for spectral type/filter %s/%s" % (DATADIR, SRC, filt))

    # add to hdr of small nrm psf
    
    fract_31 = nrmfov.sum()/nrmbig.sum() # fraction of flux in small nrm array compared to "almost infinite" nrm array
    cpf = nrmfov.max() # CPF  compared to CLEAR aperture large psf array total being 1
    nrmtot = nrmbig.sum() # double check NRM throughput is about 15%, clear psf total 1
    cpfnrm = cpf / nrmtot # Central pixel fraction of NRM array, same as nrmbig.max()/nrmbig.sum()

    # total desired electrons in CLEAR given our NRM total electron needs.
    # 1/(nrmtot=0.15)(frac_31=0.85) = Factor of about 8 for F430M
    tot_e_full = tot_e / (nrmtot*fract_31)

    # total number of electrons in central pixel of NRM array given total number in CLEAR psf
    cptot = cpf * tot_e_full  #same as 1e6*cpfnrm/fract_31
    print("cptot", cptot)

    # NINT = number of ramps (not STScI wording)
    if cptot < sat_e:
        # Faint object, total photon count reached before saturation and need less photons. eg F480M 14 1e6
        
        nint_ = 1
        cp_e_per_frame = U.tframe * (cpf*cr)
        ngroups_ = cptot / cp_e_per_frame
        ngroups, nint, nint_ceil = apt_values(ngroups_, nint_)

    else:  
        #Saturation reached in an integration before reaching the total number of required photons. eg F480M 8 1e10   
        #  approx., assuming cptot >> sat_e
        nint_ = cptot / sat_e
        # given CR, cr*cpf in central pixel: need this many frames to reach saturation
        # NISRAPID NGROUPS = NFRAMES:
        Tsat = sat_e/(cr*cpf)
        ngroups_ =  Tsat/U.tframe # also  sat_e/(U.tframe * cr * cpf)
        ngroups, nint, nint_ceil = apt_values(ngroups_, nint_)
                
    # to reach sat_e at given countrate in CP,  cr in CP is (cpf*cr)
    cp_e_per_frame = U.tframe * (cpf*cr)
    totnframes = cptot / cp_e_per_frame



    hdrfov["fluxfrac"] = (fract_31, "frac of WP power in this array")
    hdrfov["cpf"] = (cpf, "nrm peak wrt CLEAR psf total of 1")
    hdrfov["cpfnrm"] = (cpfnrm, "nrm peak wrt Inf. nrm psf total")
    hdrfov["nrmtot"] = (nrmtot, "WP nrm psf total is normed to mask thruput")
    hdrfov["tote"] = (tot_e, "total number of electrons required")
    hdrfov["totefull"] = (tot_e_full, "total number of electrons if CLEAR used")
    hdrfov["cptot"] = (cptot, "total in central pixel for TOT_E electrons")
    hdrfov["sate"] = (sat_e, "saturation limit in electrons")
    if cptot < sat_e:
        hdrfov["Tsat"] = ('', "cptot < sat_e, saturation not reached")
    else:
        hdrfov["Tsat"] = (Tsat, "time for 1 integration, reaches saturation in CP")
    hdrfov["NINT"] = (nint_, "number of integrations required to reach TOT_E")
    hdrfov["NGROUPS"] = (np.ceil(ngroups_), "number of integrations required to reach TOT_E")
    hdrfov["TFRAME"] = (U.tframe, "single frame read time")
    hdrfov["totnfram"] = (totnframes, "total number of frames of data read")
    hdrfov["GAIN"] = (1.5, "electrons/ADU - not used here")


    if cptot < sat_e:
        Tsatstr = ""
    else:
        Tsatstr = "%5.1e"%Tsat

    report = "\n\t%s  Saturation %.1e e: \n\t\t%.1f%% of nrmflux in 31x31,  " % (filt, sat_e, 100.0*fract_31) + \
    "%.2f%% peak frac wrt fullpsf.sum,  " % (100.0*cpf) + \
    "%.2f%% peak frac wrt nrmpsf.sum,  " % (100.0*cpfnrm) + \
    "%.2f%% nrmtot in 255x255  " % (100.0*nrmbig.sum()) + \
    "\n\t\tCRclearp %.1e  cpfCR %.3e  cptot %.1e  cp_e/frame %.1e  Tsat %s, " %( cr, cpf*cr, cptot, cp_e_per_frame, Tsatstr) +\
    "\n\t\tNINT %f NINT(updated to account for lost partial group per integration or to account for NGROUPS exceeding MAXGROUPS) %f,"  %( nint_, nint) +\
    "\n\t\tNGROUPS %f totnframes %d,"  %( ngroups_,totnframes) +\
    "\n\t===================================================" +\
    "\n\t     FOR APT INPUT: NGROUPS = %.0f NINT = %.0f     " % ( ngroups, nint_ceil ) + \
    "\n\t===================================================" +\
    "\n\tdatacollect %s" % timestring(nint_ * ngroups_ * U.tframe / 60.0)

    if np.isinf(nint_ceil):
        #Saturation reaches in less than 1 group. eg F480M 2.41 1e10
        report = report + "\n\tBRIGHTNESS LIMIT EXCEEDED: some pixels may saturate: \n\t\tFOR APT INPUT NGROUPS = 1 NINT = 1"

    if nint_ceil > MAXINTS and np.isfinite(nint_ceil): 
        report = report + "\n\t\tMAXIMUM NUMBER OF INTEGRATIONS (%d) EXCEEDED."%MAXINTS


    #params gets used in the simulations
    if return_params:
        if np.isinf(nint_ceil):
            params = [1,1,1]
        else:
            params = [np.int(ngroups), nint, np.int(nint_ceil)]
        return report,params
    else:
        return report
        


"""
def driver(tot_e, sat_e, mag_t):  # programmers can base a multifilter script on this driver
    reports = []
    reports.append(generatePSF(filt=F277W, fov=31, osample=3, cr=cr_from_mag(mag_t, ZP[F277W]), tot_e=tot_e, sat_e=sat_e))
    reports.append(generatePSF(filt=F380M, fov=31, osample=3, cr=cr_from_mag(mag_t, ZP[F380M]), tot_e=tot_e, sat_e=sat_e))
    reports.append(generatePSF(filt=F430M, fov=31, osample=3, cr=cr_from_mag(mag_t, ZP[F430M]), tot_e=tot_e, sat_e=sat_e))
    reports.append(generatePSF(filt=F480M, fov=31, osample=3, cr=cr_from_mag(mag_t, ZP[F480M]), tot_e=tot_e, sat_e=sat_e))
    return reports
""" 

def cr_from_mag(M, zp=None):
    " Flux of M magnitude star in counts/sec given a zero point zp is the zero point when using CLEARP "
    return pow(10,-(M-zp)/2.5)
