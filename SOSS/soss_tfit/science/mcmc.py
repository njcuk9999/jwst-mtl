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
tpsorder = ['RHO_STAR', 'LD1', 'LD2', 'LD3', 'LD4', 'DILUTION', None,
           'ZEROPOINT']
# transit fit (fortran code) planet fit parameters (one per planet in this
#    order)
tpporder = ['TO', 'PERIOD', 'B', 'RPRS', 'SQRT_E_COSW', 'SQRT_E_SINW',
            None, 'ECLIPSE_DEPTH', 'ELLIPSOIDAL', 'PHASECURVE']

# transit fit hyper parameters
tphorder = ['ERROR_SCALE', 'AMPLITUDE_SCALE', 'LENGTH_SCALE']


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
    # numpy array [n_param, n_phot] whether we are fitting each parameter
    #             [True/False]
    pmask: np.ndarray
    # name of each parameter
    pnames: List[str]
    # additional arguments passed to mcmc
    pkwargs: Dict[str, Any]
    # the data:
    #     [0][WAVELENGTH][n_phot, n_int]
    phot: List[Dict[str, np.ndarray]]
    # the time array [n_phot, n_int]
    time: np.ndarray
    # the integration time array [n_phot, n_int]
    itime: np.ndarray
    # the flux array [n_phot, n_int]
    flux: np.ndarray
    # the flux error array [n_phot, n_int]
    fluxerr: np.ndarray

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
    n_param = len(tpsorder) + len(tpporder) + len(tphorder)
    # -------------------------------------------------------------------------
    # set up storage
    transit_p0 = np.zeros([n_param, n_phot])
    transit_pmask = np.zeros([n_param, n_phot], dtype=bool)
    transit_pnames = []
    # -------------------------------------------------------------------------
    # get the transit fit stellar params
    # -------------------------------------------------------------------------
    pnum = 0
    # loop around stellar parameters
    for key in tpsorder:
        # deal with key = None
        if key is None:
            transit_p0[pnum] = np.zeros(n_phot)
            transit_pmask[pnum] = np.zeros(n_phot, dtype=bool)
            transit_pnames.append('Undefined')
            continue
        # deal with key not in params (bad)
        if key not in params:
            emsg = f'ParamError: key {key} must be set. func={func_name}'
            raise base_classes.TransitFitExcept(emsg)
        # get the value for this key
        fitparam = params[key]
        # only consider FitParams classes
        if not isinstance(fitparam, FitParam):
            continue
        # get the properties from the fit param class
        p0, pmask, pname = fitparam.get_value(n_phot)
        # add values to storage
        transit_p0[pnum] = p0
        transit_pmask[pnum] = pmask
        transit_pnames.append(pname)
        # add to the parameter number
        pnum += 1
    # -------------------------------------------------------------------------
    # get the transit fit planet params
    # -------------------------------------------------------------------------
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
        # loop around stellar parameters
        for key in tpporder:
            # deal with key = None
            if key is None:
                transit_p0[pnum] = np.zeros(n_phot)
                transit_pmask[pnum] = np.zeros(n_phot, dtype=bool)
                transit_pnames.append('Undefined')
                continue
            # deal with key not in params (bad)
            if key not in planetdict:
                emsg = f'ParamError: key {key} must be set. func={func_name}'
                raise base_classes.TransitFitExcept(emsg)
            # get the value for this key
            fitparam = planetdict[key]
            # only consider FitParams classes
            if not isinstance(fitparam, FitParam):
                continue
            # get the properties from the fit param class
            p0, pmask, pname = fitparam.get_value(n_phot)
            # add values to storage
            transit_p0[pnum] = p0
            transit_pmask[pnum] = pmask
            transit_pnames.append(f'P{nplanet}_{pname}')
            # add to the parameter number
            pnum += 1
    # -------------------------------------------------------------------------
    # get the transit fit hyper params
    # -------------------------------------------------------------------------
    # loop around stellar parameters
    for key in tpsorder:
        # deal with key = None
        if key is None:
            transit_p0[pnum] = np.zeros(n_phot)
            transit_pmask[pnum] = np.zeros(n_phot, dtype=bool)
            transit_pnames.append('Undefined')
            continue
        # deal with key not in params (bad)
        if key not in params:
            emsg = f'ParamError: key {key} must be set. func={func_name}'
            raise base_classes.TransitFitExcept(emsg)
        # get the value for this key
        fitparam = params[key]
        # only consider FitParams classes
        if not isinstance(fitparam, FitParam):
            continue
        # get the properties from the fit param class
        p0, pmask, pname = fitparam.get_value(n_phot)
        # add values to storage
        transit_p0[pnum] = p0
        transit_pmask[pnum] = pmask
        transit_pnames.append(pname)
        # add to the parameter number
        pnum += 1
    # -------------------------------------------------------------------------
    # create a class for the fit
    tfit = TransitFit()
    # add the data to the transit fit
    tfit.time = data.phot['TIME']
    tfit.itime = data.phot['ITIME']
    tfit.flux = data.phot['FLUX']
    tfit.fluxerr = data.phot['FLUXERR']
    # add the parameters
    tfit.p0 = transit_p0
    tfit.ptmp = np.array(transit_p0)
    tfit.pmask = transit_pmask
    tfit.pnames = transit_pnames
    # -------------------------------------------------------------------------
    # return the transit fit class
    # -------------------------------------------------------------------------
    return tfit


def run_mcmc():
    pass

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
