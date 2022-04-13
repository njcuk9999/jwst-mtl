#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-06

@author: cook
"""
import numpy as np
from typing import Any, Dict, List

from soss_tfit.core import base
from soss_tfit.core import base_classes
from soss_tfit.science import general

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'science.general.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get parameter dictionary
ParamDict = base_classes.ParamDict
# get FitParam class
FitParam = base_classes.FitParam
# get data class
InputData = general.InputData
# transit fit (fortran code) stellar parameters in correct order
TPS_ORDERED = ['RHO_STAR', 'LD1', 'LD2', 'LD3', 'LD4', 'DILUTION', None,
           'ZEROPOINT']
# transit fit (fortran code) planet fit parameters (one per planet in this
#    order)
TPP_ORDERED = ['T0', 'PERIOD', 'B', 'RPRS', 'SQRT_E_COSW', 'SQRT_E_SINW',
               None, 'ECLIPSE_DEPTH', 'ELLIPSOIDAL', 'PHASECURVE']

# transit fit hyper parameters
TPH_ORDERED = ['ERROR_SCALE', 'AMPLITUDE_SCALE', 'LENGTH_SCALE']
# transit fit additional arguments
TP_KWARGS = ['NTT', 'T_OBS', 'OMC']


# =============================================================================
# Define fit class passed to mcmc code
# =============================================================================
class TransitFit:
    # not sure about these - can't be pickled
    loglikelihood: None
    mcmcfunc: None
    # the number of integrations we have
    n_int: int
    # the number of parameters we have
    n_param: int
    # the number of photometric bandpasses we have
    n_phot: int
    # numpy array [n_param, n_phot] the initial value of each parameter
    p0: np.ndarray
    # numpy array [n_param, n_phot] the current value of each parameter
    ptmp: np.ndarray
    # numpy array [n_param] whether we are fitting each parameter [Bool]
    fmask: np.ndarray
    # numpy array [n_param] whether chromatic fit used for each parameter [Bool]
    wmask: np.ndarray
    # numpy array [n_param]
    beta: np.ndarray
    # priors [n_param]
    prior: Dict[str, Any]
    # name of each parameter
    pnames: List[str]
    # additional arguments passed to mcmc
    pkwargs: Dict[str, Any]
    # the data:
    #     [0][WAVELENGTH][n_phot, n_int]
    phot: List[Dict[str, np.ndarray]]
    # the wavelength array [n_phot, n_int]
    wavelength: np.ndarray
    # the time array [n_phot, n_int]
    time: np.ndarray
    # the integration time array [n_phot, n_int]
    itime: np.ndarray
    # the flux array [n_phot, n_int]
    flux: np.ndarray
    # the flux error array [n_phot, n_int]
    fluxerr: np.ndarray
    # the order array [n_phot, n_int]
    orders: np.ndarray

    def __getstate__(self) -> dict:
        """
        For when we have to pickle the class
        :return:
        """
        # set state to __dict__
        state = dict(self.__dict__)
        # return dictionary state
        return state

    def __setstate__(self, state: dict):
        """
        For when we have to unpickle the class

        :param state: dictionary from pickle
        :return:
        """
        # update dict with state
        self.__dict__.update(state)

    def __init__(self):
        pass

    def copy(self):
        """
        Copy class - deep copying values which need copying
        (p0, ptmp)
        
        :return: 
        """
        pass

    def update(self):
        """
        Update parameters which need changing after a change
        (ptmp)
        
        :return: 
        """
        pass

    def view(self, parameter):
        """
        View a parameter (i.e. p0, ptmp, fmask, wmasl, beta, prior)
        with its associated

        :param parameter: anything with length (on first axis) equal to
                          n_param

        examples:
            >> tfit.view(tfit.beta[:, 0]) # view the 0th bandpass beta values
            >> tfit.view(tfit.p0[:, 0])   # view the 0th bandpass p0 values
            >> tfit.view(tfit.prior)      # view the prior values

        :return: None, prints to stdout
        """
        # make sure we can view this parameter
        if len(parameter) != self.n_param:
            emsg = (f'TransitFit View Error: Cannot view parameter lenght '
                    f'(on first axis) must be: {self.n_param}')
            base_classes.TransitFitExcept(emsg)
        # loop around and print info
        for it in range(len(self.pnames)):
            print(f'{self.pnames[it]:14s}: {parameter[it]}')

    def assign_beta(self, beta_kwargs: Dict[str, Any]):
        """
        Assign beta from either beta_kwargs[key] = value
        or as a random number: np.random.rand(size) * 1.0e-5

        note the value can either be a float or a np.ndarray of length the
           number of band passes

        :param beta_kwargs: dictionary, key value pairs to change beta values
                            for a specific parameter
        :return:
        """
        # beta factor
        bfactor = 1.0e-5

        # start of with all values at zero (all fixed parameters have beta=0)
        self.beta = np.zeros_like(self.p0)

        # loop through and find fitted parameters
        for it in range(self.n_param):
            # override beta values
            if self.pnames[it] in beta_kwargs:
                # set beta value
                self.beta[it] = beta_kwargs[self.pnames[it]]
                # skip step
                continue
            # if we have a chromatic fitted parameter set each bandpass
            #   to a random number
            if self.fmask[it] and self.wmask[it]:
                self.beta[it] = np.random.rand(self.n_phot) * bfactor
            # if we have a bolometic fitted parameter set all band passes
            #   to the same random number
            elif self.fmask[it]:
                self.beta[it] = np.random.rand() * bfactor


class Sampler:
    def __init__(self, params: ParamDict, tfit: TransitFit, mode='full'):
        self.params = params
        self.tfit = tfit
        self.mode = mode

    # TODO: fill out
    def run_mcmc(self):
        pass


# =============================================================================
# Define functions
# =============================================================================
def get_starting_params(params: ParamDict, data: InputData) -> TransitFit:
    """
    Using params and data load everything in to the Transit Fit data class

    :param params: ParamDict, parameter dictionary of constants
    :param data: InputData, input data class

    :return: transit fit data class
    """
    # set function name
    func_name = __NAME__ + '.get_starting_params()'
    # get number of wavelength bandpasses
    n_phot = data.n_phot
    # get the number of planets to add
    n_planets = params['NPLANETS']
    # get the number of parameters we are adding
    n_param = len(TPS_ORDERED) + len(TPP_ORDERED) + len(TPH_ORDERED)
    # -------------------------------------------------------------------------
    # set up storage
    transit_p0 = np.zeros([n_param, n_phot])
    transit_fmask = np.zeros(n_param, dtype=bool)
    transit_wmask = np.zeros(n_param, dtype=bool)
    transit_prior = []
    transit_pnames = []
    # -------------------------------------------------------------------------
    # get the transit fit stellar params
    # -------------------------------------------------------------------------
    pnum = 0
    # loop around stellar parameters
    for key in TPS_ORDERED:
        p0, fmask, wmask, prior, pname = __assign_pvalue(key, params, n_phot,
                                                         func_name)
        # add values to storage
        transit_p0[pnum] = p0
        transit_fmask[pnum] = fmask
        transit_wmask[pnum] = wmask
        transit_prior.append(prior)
        transit_pnames.append(pname)
        # add to the parameter number
        pnum += 1
    # -------------------------------------------------------------------------
    # get the transit fit planet params
    # -------------------------------------------------------------------------
    # loop around planets
    for nplanet in range(1, n_planets + 1):
        # need the planet key
        pkey = f'PLANET{nplanet}'
        # check that we have this planet
        if pkey not in params:
            emsg = (f'ParamError: planet {nplanet} not found. '
                    f'Please add or adjust "NPLANETS" parameter.')
            raise base_classes.TransitFitExcept(emsg)
        # get planet dictionary
        planetdict = params[pkey]
        # loop around planetN parameters
        for key in TPP_ORDERED:
            p0, fmask, wmask, prior, pname = __assign_pvalue(key, planetdict,
                                                             n_phot, func_name)
            # add values to storage
            transit_p0[pnum] = p0
            transit_fmask[pnum] = fmask
            transit_wmask[pnum] = wmask
            transit_prior.append(prior)
            transit_pnames.append(f'{pname}{nplanet}')
            # add to the parameter number
            pnum += 1
    # -------------------------------------------------------------------------
    # get the transit fit hyper params
    # -------------------------------------------------------------------------
    # loop around hyper parameters
    for key in TPH_ORDERED:
        p0, fmask, wmask, prior, pname = __assign_pvalue(key, params, n_phot,
                                                         func_name)
        # hyper parameters are set to zero if not in fit mode
        if not fmask:
            transit_p0[pnum] = np.zeros(n_phot)
        else:
            transit_p0[pnum] = p0
        # add other values to storage
        transit_fmask[pnum] = fmask
        transit_wmask[pnum] = wmask
        transit_prior.append(prior)
        transit_pnames.append(pname)
        # add to the parameter number
        pnum += 1
    # -------------------------------------------------------------------------
    # create a class for the fit
    # -------------------------------------------------------------------------
    tfit = TransitFit()
    # add the data to the transit fit
    tfit.wavelength = data.phot['WAVELENGTH']
    tfit.time = data.phot['TIME']
    tfit.itime = data.phot['ITIME']
    tfit.flux = data.phot['FLUX']
    tfit.fluxerr = data.phot['FLUX_ERROR']
    tfit.orders = data.phot['ORDERS']
    # add the parameters
    tfit.p0 = transit_p0
    tfit.ptmp = np.array(transit_p0)
    tfit.fmask = transit_fmask
    tfit.wmask = transit_wmask
    tfit.prior = transit_prior
    tfit.pnames = np.array(transit_pnames)
    # set the number of params
    tfit.n_param = len(transit_p0)
    # set the number of photometric bandpasses
    tfit.n_phot = int(data.n_phot)
    # set the number of integrations
    tfit.n_int = int(data.n_int)
    # -------------------------------------------------------------------------
    # set additional arguments
    # -------------------------------------------------------------------------
    tfit.pkwargs = dict()
    # loop through transit param additional keyword arguments
    for key in TP_KWARGS:
        # if this key exists in params set it
        if key in params:
            tfit.pkwargs[key] = params[key]
    # -------------------------------------------------------------------------
    # return the transit fit class
    # -------------------------------------------------------------------------
    return tfit


# TODO: Fill out
def lnprob(tfit: TransitFit) -> float:
    pass


# TODO: Fill out
def mhg_mcmc(tfit: TransitFit):
    pass


# TODO: Fill out
def beta_rescale(params: ParamDict, tfit: TransitFit) -> TransitFit:
    return tfit


# TODO: Fill out
def run_mcmc(params: ParamDict, tfit: TransitFit, mode='full'):
    pass


# =============================================================================
# Define worker functions
# =============================================================================
def __assign_pvalue(key, pdict, n_phot, func_name):
    # deal with key = None
    if key is None:
        p0 = np.zeros(n_phot)
        fmask = False
        wmask = False
        prior = None
        pnames = 'Undefined'
        return p0, fmask, wmask, prior, pnames
    # deal with key not in params (bad)
    if key not in pdict:
        emsg = f'ParamError: key {key} must be set. func={func_name}'
        raise base_classes.TransitFitExcept(emsg)
    # get the value for this key
    fitparam = pdict[key]
    # only consider FitParams classes
    if not isinstance(fitparam, FitParam):
        emsg = (f'ParamError: key {key} must be a FitParam. '
                f'\n\tGot {key}={fitparam}'
                f'\n\tfunc={func_name}')
        raise base_classes.TransitFitExcept(emsg)
    # get the properties from the fit param class
    p0, fmask, wmask, prior, pnames = fitparam.get_value(n_phot)

    return p0, fmask, wmask, prior, pnames


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
