import numpy as np

TMODEL_FUNC = []
# model stellar parameters in correct order
TPS_ORDERED = []
# model star spot fit parameters (one per spot in this order)
TPSP_ORDERED = []
# model planet fit parameters (one per planet in this order)
TPP_ORDERED = []
# model hyper parameters in correct order
TPH_ORDERED = []
# model additionnal parameters
TP_KWARGS = []

#####JASON ROWE original model#####
from soss_tfit.utils import tfit5

def tmodel_jr(tfit, phot_it):
    
    model=np.empty(tfit.n_int)
    tfit5.transitmodel(tfit.n_planets, tfit.p0[:, phot_it], tfit.time[phot_it],
                       tfit.itime[phot_it], tfit.tt_n, tfit.tt_tobs,
                       tfit.tt_omc, model, tfit.tmodel_dtype, tfit.pkwargs['NINTG'])

    #add the trends
    #trends are always the last parameters
    for n in range(tfit.n_trends):
        i=-tfit.n_trends+n
        model+=tfit.p0[i,phot_it]*tfit.trends_vec[n][phot_it]
      
    return model

TMODEL_FUNC.append(tmodel_jr)
TPS_ORDERED.append(['RHO_STAR', 'LD1', 'LD2', 'LD3', 'LD4',
                    'DILUTION', None,'ZEROPOINT'])
TPSP_ORDERED.append([])
TPP_ORDERED.append(['T0', 'PERIOD', 'B', 'RPRS', 'SQRT_E_COSW',
                    'SQRT_E_SINW',None, 'ECLIPSE_DEPTH', 'ELLIPSOIDAL',
                    'PHASECURVE'])
TPH_ORDERED.append(['ERROR_SCALE', 'AMPLITUDE_SCALE', 'LENGTH_SCALE'])
TP_KWARGS.append(['NTT', 'T_OBS', 'OMC', 'NINTG'])

#####JASON ROWE original model, 
#####but specifying stellar Teff to get LD coeff from models#####

#should first build a grid of q1&q2 values vs Teff (&logg?), fixed [M/H]
#build at the wavelengths of data
#the grid should be in the tfit object

def tmodel_jr_model_ld(tfit, phot_it):
    
    #interpolate the q1 & q2 at desired Teff & logg
    pt=(tfit.p0[1,phot_it],tfit.p0[2,phot_it]) #(teff,logg)
    q1=tfit.ld_func[phot_it][0](pt)
    q2=tfit.ld_func[phot_it][1](pt)
    
    #build the proper p0 vector, as needed by tfit5.transitmodel
    p0=tfit.p0[:, phot_it].copy()
    p0[1]=0. #must set LD1 to 0
    p0[2]=0. #must set LD2 to 0
    p0[3]=q1 #LD3 to q1
    p0[4]=q2 #LD4 to q2
        
    model=np.empty(tfit.n_int)
    tfit5.transitmodel(tfit.n_planets, p0, tfit.time[phot_it],
                       tfit.itime[phot_it], tfit.tt_n, tfit.tt_tobs,
                       tfit.tt_omc, model, tfit.tmodel_dtype, tfit.pkwargs['NINTG'])

    #add the trends
    #trends are always the last parameters
    for n in range(tfit.n_trends):
        i=-tfit.n_trends+n
        model+=tfit.p0[i,phot_it]*tfit.trends_vec[n][phot_it]
      
    return model

TMODEL_FUNC.append(tmodel_jr_model_ld)
TPS_ORDERED.append(['RHO_STAR', 'TEFF', 'LOGG', None, None, 'DILUTION', None, 'ZEROPOINT'])
TPSP_ORDERED.append([])
TPP_ORDERED.append(['T0', 'PERIOD', 'B', 'RPRS', 'SQRT_E_COSW',
                    'SQRT_E_SINW',None, 'ECLIPSE_DEPTH', 'ELLIPSOIDAL',
                    'PHASECURVE'])
TPH_ORDERED.append(['ERROR_SCALE', 'AMPLITUDE_SCALE', 'LENGTH_SCALE'])
TP_KWARGS.append(['NTT', 'T_OBS', 'OMC', 'NINTG'])



######################################################################
######SPOTROD model, LD sampling###########
######################################################################
from ctypes import c_void_p, c_double, c_int, cdll
# from numpy.ctypeslib import ndpointer

#load the compiled C library
#must be compiled with: icc -fPIC -shared -O3 spotrod_tfit.c -o spotrod_tfit.so
import os
spotrod = cdll.LoadLibrary(os.path.dirname(__file__)+"/../utils/spotrod_tfit.so")
transitmodel_c = spotrod.transitmodel

def tmodel_spotrod(tfit,phot_it,n_r=1000):   
    n_s=tfit.n_spots #number of spots
    t= tfit.time[phot_it].copy() #must put a copy here! so would be better if it was a list of arrays
    q1 = tfit.p0[1,phot_it]
    q2 = tfit.p0[2,phot_it]
    i=3+4*np.arange(n_s)
    spotx=   tfit.p0[i,phot_it]
    spoty=   tfit.p0[i+1,phot_it]
    spotrad= tfit.p0[i+2,phot_it]
    spotcont=tfit.p0[i+3,phot_it]
    t0,period,b,rprs,sma,ecw,esw=tfit.p0[3+4*n_s:10+4*n_s,phot_it]
        
    model=np.empty(tfit.n_int)
    transitmodel_c(
        c_void_p(t.ctypes.data),c_int(t.size), c_double(q1), c_double(q2), 
        c_void_p(spotx.ctypes.data), c_void_p(spoty.ctypes.data),
        c_void_p(spotrad.ctypes.data), c_void_p(spotcont.ctypes.data),
        c_int(n_s), c_int(n_r),
        c_double(t0),c_double(period), c_double(b), c_double(sma), c_double(rprs),
        c_double(ecw), c_double(esw), c_void_p(model.ctypes.data)
        )

    #add the zeropoint
    model+=tfit.p0[0,phot_it]
    
    #add the trends
    #trends are always the last parameters
    for n in range(tfit.n_trends):
        i=-tfit.n_trends+n
        model+=tfit.p0[i,phot_it]*tfit.trends_vec[n][phot_it]
        
    return model
    
TMODEL_FUNC.append(tmodel_spotrod)
TPS_ORDERED.append(['ZEROPOINT','Q1', 'Q2'])
TPSP_ORDERED.append(['SPOTX', 'SPOTY', 'SPOTR', 'SPOTC'])
TPP_ORDERED.append(['T0', 'PERIOD', 'B', 'RPRS', 'SMA', 'SQRT_E_COSW','SQRT_E_SINW'])
TPH_ORDERED.append(['ERROR_SCALE'])
TP_KWARGS.append([])

######################################################################
######SPOTROD model, Teff & logg sampling###########
######################################################################

def tmodel_spotrod2(tfit,phot_it,n_r=1000): 
    n_s=tfit.n_spots #number of spots
    t= tfit.time[phot_it].copy() #must put a copy here! so would be better if it was a list of arrays

    #interpolate the q1 & q2 at desired Teff & logg
#     pt=(tfit.p0[1,phot_it],tfit.p0[2,phot_it]) #(teff,logg)
#     q1=tfit.ld_func[phot_it][0](pt) #for RegularGridInterpolator
#     q2=tfit.ld_func[phot_it][1](pt)
    q1=tfit.ld_func[phot_it][0](tfit.p0[1,phot_it],tfit.p0[2,phot_it])
    q2=tfit.ld_func[phot_it][1](tfit.p0[1,phot_it],tfit.p0[2,phot_it])


    i=3+4*np.arange(n_s)
    spotx=   tfit.p0[i,phot_it]
    spoty=   tfit.p0[i+1,phot_it]
    spotrad= tfit.p0[i+2,phot_it]
    spotcont=tfit.p0[i+3,phot_it]
    t0,period,b,rprs,sma,ecw,esw=tfit.p0[3+4*n_s:10+4*n_s,phot_it]
        
    model=np.empty(tfit.n_int)
    transitmodel_c(
        c_void_p(t.ctypes.data),c_int(t.size), c_double(q1), c_double(q2), 
        c_void_p(spotx.ctypes.data), c_void_p(spoty.ctypes.data),
        c_void_p(spotrad.ctypes.data), c_void_p(spotcont.ctypes.data),
        c_int(n_s), c_int(n_r),
        c_double(t0),c_double(period), c_double(b), c_double(sma), c_double(rprs),
        c_double(ecw), c_double(esw), c_void_p(model.ctypes.data)
        )

    #add the zeropoint
    model+=tfit.p0[0,phot_it]
    
    #add the trends
    #trends are always the last parameters
    for n in range(tfit.n_trends):
        i=-tfit.n_trends+n
        model+=tfit.p0[i,phot_it]*tfit.trends_vec[n][phot_it]
        
    return model
    
TMODEL_FUNC.append(tmodel_spotrod2)
TPS_ORDERED.append(['ZEROPOINT','TEFF', 'LOGG'])
TPSP_ORDERED.append(['SPOTX', 'SPOTY', 'SPOTR', 'SPOTC'])
TPP_ORDERED.append(['T0', 'PERIOD', 'B', 'RPRS', 'SMA', 'SQRT_E_COSW','SQRT_E_SINW'])
TPH_ORDERED.append(['ERROR_SCALE'])
TP_KWARGS.append([])

