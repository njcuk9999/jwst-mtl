#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-06

@author: cook
"""
import numpy as np
from typing import Any, Dict, List, Optional

import transitfit5 as transit_fit

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
# Out of bounds value
BADLPR = -np.inf
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

# These are the attributes of TransitFit that have the same length as x0
X_ATTRIBUTES = ['x0', 'xtmp', 'beta', 'x0pos']


# =============================================================================
# Define fit class passed to mcmc code
# =============================================================================
class TransitFit:
    """
    One could just set all the correct values manually in this and
    expect it to work later
    """
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
    # priors [n_param]
    prior: Dict[str, Any]
    # name of each parameter
    pnames: List[str]
    # additional arguments passed to mcmc
    pkwargs: Dict[str, Any]
    # the position in the flattened x array [n_param, n_phot]
    p0pos: np.ndarray
    # -------------------------------------------------------------------------
    # the data:
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # parameters that must have shape [n_x]
    # -------------------------------------------------------------------------
    # the initial fitted parameters
    x0: np.ndarray
    # the fitted params flattened [n_x]
    xtmp: np.ndarray
    # name of the fitted params flattened [n_x]
    xnames: np.ndarray
    # length of fitted params flattened [n_x]
    n_x: int
    # numpy array [n_param]
    beta: np.ndarray
    # the mapping from x0 onto p0 (or xtmp onto ptmp) [n_x, 2] each element
    #   is the tuple poisition in p0 or ptmp
    x0pos: np.ndarray
    # -------------------------------------------------------------------------
    # will be filled out by the mcmc
    # -------------------------------------------------------------------------
    # the current position chosen by the sampler
    n_tmp: int = 0
    # the previous loglikelihood value
    llx: float
    # the rejection [rejected, parameter number]   where rejected = 0 for
    #   accepted and rejected = 1 for rejected
    ac: List[int]

    def __init__(self):
        pass

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

    def copy(self):
        """
        Copy class - deep copying values which need copying
        (p0, ptmp)

        :return:
        """
        pass

    def get_fitted_params(self):
        """
        Get the fitted parameters in a flattened way
        Here chromatic values add self.n_phot values and bolometric values add
        a single value

        :return: None, updates self.xtmp and self.xnames
        """
        # set up storage
        x0 = []
        xnames = []
        # position of x in p
        p0pos = np.zeros_like(self.p0)
        x0pos = []
        # counter for x0 position
        count_x = 0
        # loop around all parameters
        for it, name in enumerate(self.pnames):
            # if we are not fitting this parameter skip
            if not self.fmask[it]:
                p0pos[it] = np.full(self.n_phot, -1)
                continue
            # if we have a wavelength dependence add all terms (one for each
            #   bandpass)
            if self.wmask[it]:
                x0 += list(self.p0[it])
                xnames += [self.pnames[it]] * self.n_phot
                # add to the positions so we can recover p0 from x0
                p0pos[it] = np.arange(count_x, count_x + self.n_phot)
                # update the x positions
                x0pos += list(zip(np.full(self.n_phot, it), np.arange(self.n_phot)))
                # update x0 position
                count_x += self.n_phot
            # else add the first term (all band passes should have the same
            #   value)
            else:
                x0 += [self.p0[it][0]]
                xnames += [self.pnames[it]]
                # add to the positions so we can recover p0 from x0
                p0pos[it] = np.full(self.n_phot, count_x)
                # update x0 position
                count_x += 1
                # update the x positions
                x0pos += [(it, 0)]
        # set values
        self.x0 = np.array(x0)
        self.xtmp = np.array(x0)
        self.xnames = np.array(xnames)
        self.n_x = len(x0)
        self.p0pos = np.array(p0pos, dtype=int)
        self.x0pos = np.array(x0pos)

    def update_ptmp_from_xtmp(self):
        """
        Take a set of fitted parameters (xtmp) and project back into the
        full solution (ptmp)

        :return:
        """
        # loop around all parameters
        for it in range(self.n_x):
            # find all places where it is valid
            mask = it == self.p0pos
            # update ptmp with xtmp value
            self.ptmp[mask] = self.xtmp[it]

    def update_xtmp_from_ptmp(self):
        """
        Take a full solution (ptmp) and project back into the set of fitted
        parameters (xtmp)

        :return:
        """
        for it in range(self.n_x):
            # push the value into x
            self.xtmp[it] = self.ptmp[tuple(self.x0pos)]

    def generate_gibbs_sample(self):
        # choose random parameter to vary
        param_it = np.random.randint(0, self.n_x + 1)
        # update the position choosen by the sampler
        self.n_tmp = param_it
        # update the choosen value by a random number drawn from a gaussian
        #   with centre = 0 and fwhm = beta
        self.xtmp[param_it] += np.random.normal(0.0, self.beta[param_it])
        # update full solution
        self.update_ptmp_from_xtmp()

    def generate_demcmc_sample(self, buffer, corbeta: float):
        # get the length of the buffer
        nbuffer = len(buffer[:, 0])
        # update the position choosen by the sampler
        self.n_tmp = -1
        # get two random numbers
        int1, int2 = np.random.randint(0, nbuffer + 1, size=2)
        # calculate the vector jump
        vector_jump = buffer[int1, :] - buffer[int2, :]
        # apply the vector jump to xtmp
        self.xtmp += vector_jump * corbeta
        # update full solution
        self.update_ptmp_from_xtmp()

    def get(self, param_name: str, attribute_name: str):
        # get attribute
        attribute = self.__get_attribute(attribute_name)
        # deal with fitted parameters
        if attribute_name in X_ATTRIBUTES:
            return dict(zip(self.xnames, attribute))[param_name]
        else:
            return dict(zip(self.pnames, attribute))[param_name]

    def view(self, attribute_name: str,
             attribute: Optional[Any] = None):
        """
        View a parameter (i.e. p0, ptmp, fmask, wmasl, beta, prior)
        with its associated

        :param attribute_name: if parameter is None this must be an attribute of
                     TransitFit, if parameter is defined can any string
                     describing the variable name

        :param attribute: can be unset, but if parameter_name if not in
                          TransitFit must be a vector of length = self.n_param

        examples:
            >> tfit.view('beta', beta[:, 0]) # view the 0th bandpass beta values
            >> tfit.view('p0', tfit.p0[:, 0])  # view the 0th bandpass p0 values
            >> tfit.view('prior')      # view the prior values
            >> tfit.view('fmask')     # view which values are fitted
            >> tfit.view('xtmp')      # view the flattened fitted parameters

        :return: None, prints to stdout
        """
        # get attribute
        attribute = self.__get_attribute(attribute_name, attribute)
        # deal with fitted parameters
        if attribute_name in X_ATTRIBUTES:
            names = self.xnames
            length = self.n_x
        else:
            names = self.pnames
            length = self.n_param

        # make sure we can view this parameter
        if len(attribute) != length:
            emsg = (f'TransitFit View Error: Cannot view parameter length '
                    f'(on first axis) must be: {length}')
            base_classes.TransitFitExcept(emsg)
        # loop around and print info
        for it in range(length):
            print(f'{names[it]:14s}: {attribute[it]}')

    def __get_attribute(self, attribute_name: str,
                        attribute: Optional[Any] = None) -> Any:
        # report error
        if not hasattr(self, attribute_name) and attribute is None:
            emsg = ('TransitFit View Error: Parameter name not set and '
                    '"parameter" not set. Please define one')
            base_classes.TransitFitExcept(emsg)
        elif attribute is None:
            attribute = getattr(self, attribute_name)

        return attribute

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
        self.beta = np.zeros(self.n_x)
        # loop through and find fitted parameters
        for it in range(self.n_x):
            # override beta values
            if self.xnames[it] in beta_kwargs:
                # set beta value
                self.beta[it] = beta_kwargs[self.xnames[it]]
                # skip step
                continue
            # else set beta as a random number (multiplied by the beta factor)
            else:
                self.beta[it] = np.random.rand() * bfactor


# =============================================================================
# Define prior functions
#     Must have arguments value, **kwargs
#     Must return True if passed, False otherwise
# =============================================================================
def tophat_prior(value: float, minimum: float, maximum: float) -> bool:
    """
    Simple tophat proir
    :param value:
    :param minimum:
    :param maximum:
    :return:
    """
    # test whether condition is less than the minimum of the prior
    if value < minimum:
        return False
    # test whether condition is greater than the maximum of the prior
    if value > maximum:
        return False
    # if we get to here we have passed this priors conditions
    return True


# =============================================================================
# Define functions
# =============================================================================
def setup_params_mcmc(params: ParamDict, data: InputData) -> TransitFit:
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
    # update fitted parameters
    # -------------------------------------------------------------------------
    # set xtmp and xnames
    tfit.get_fitted_params()
    # start of beta
    tfit.beta = np.zeros_like(tfit.xtmp)
    # -------------------------------------------------------------------------
    # return the transit fit class
    # -------------------------------------------------------------------------
    return tfit


def lnpriors(tfit) -> float:
    """
    Log prior function - basically runs the prior function for
    each parameter and each bandpass

    If any fail returns immediately

    :param tfit: the transit fit class

        tfit must have attributes:
            n_phot: int, the number of band passes
            n_param: int, the number of parameters (fitted and fixed)
            ptmp: np.ndarray - the trial solution [n_param, n_phot]
            pnames: np.ndarray [n_param, n_phot] {STRING} - parameter names
            fmask: np.ndarray [n_param] {BOOL} - whether parameter is to be fit

    :return: float, either 1 (if good) or -np.inf (if bad)
    """
    # set initial value of loglikelihood
    logl = 1.0
    # the number of band passes
    n_phot = tfit.n_phot
    # the number of parameters
    n_param = tfit.n_param
    # name of parameters
    names = tfit.pnames
    # whether parameters are fitted or fixed
    fmask = tfit.fmask
    # trial solution
    sol = tfit.ptmp
    # get priors
    priors = tfit.get(names, 'prior')
    # -------------------------------------------------------------------------
    # loop around all parameters and test priors
    for param_it in range(n_param):
        # skip fixed terms (we don't care about the priors for these parameters)
        if not fmask[param_it]:
            continue
        # get this parameters priors
        prior = priors[param_it]
        # loop around band passes
        for phot_it in range(n_phot):
            # get prior function - default prior is a tophat function
            func = prior.get('func', tophat_prior)
            # function returns True if pass prior condition
            if not func(sol[:, phot_it], **prior):
                return BADLPR
    # if we have got to here we return the good loglikelihood (all priors have
    #    passed)
    return logl


def lnprob(tfit: TransitFit) -> float:
    """
    The loglikelihood function

    :param tfit: the transit fit class

        tfit must have attributes:
            time: np.ndarray
            itime: np.ndarray
            flux: np.ndarray
            fluxerr: np.ndarray
            n_phot: int, the number of band passes
            n_param: int, the number of parameters (fitted and fixed)
            ntt:
            tobs:
            omc:
            ptmp: np.ndarray - the trial solution [n_param, n_phot]
            pnames: np.ndarray [n_param, n_phot] {STRING} - parameter names

    :return:
    """
    # -------------------------------------------------------------------------
    # lets get the values out of tfit (shallow copy)
    # -------------------------------------------------------------------------
    # the data
    time = tfit.time
    itime = tfit.itime
    flux = tfit.flux
    fluxerr = tfit.fluxerr
    # -------------------------------------------------------------------------
    # other parameters
    n_phot = tfit.n_phot
    ntt = tfit.pkwargs['NTT']
    tobs = tfit.pkwargs['T_OBS']
    omc = tfit.pkwargs['OMC']
    # trial solution
    sol = tfit.ptmp
    # -------------------------------------------------------------------------
    # the hyper parameters
    # photometric error scale (DSC) for this bandpass
    dscale = tfit.get('DSC', 'ptmp')
    # GP Kernel Amplitude (ASC) for this bandpass
    ascale = tfit.get('ASC', 'ptmp')
    # GP length scale (LSC) for this bandpass
    lscale = tfit.get('LSC', 'ptmp')
    # deal with the fitting of hyper parameters
    fit_error_scale = tfit.get('DSC', 'fmask')
    fit_amplitude_scale = tfit.get('DSC', 'fmask')
    fit_length_scale = tfit.get('DSC', 'fmask')
    # -------------------------------------------------------------------------
    # Select model type: 1 = GP model, 2 = uncorrelated noise model
    model_type = int(fit_amplitude_scale | fit_length_scale)

    # -------------------------------------------------------------------------
    # Step 1: Prior calculation
    # -------------------------------------------------------------------------
    # check priors
    logl = lnpriors(tfit)
    # if out of limits return here
    if not (logl > BADLPR):
        return BADLPR
    # -------------------------------------------------------------------------
    # Step 2: Calculate loglikelihood per band pass
    # -------------------------------------------------------------------------
    # QUESTION: Can we parallelize this?
    # QUESTION: If one phot_it is found to be inf, we can skip rest?
    # loop around band passes
    for phot_it in range(n_phot):
        # check dscale, ascale and lscale hyper parameters
        #    (they must be positive)
        if (dscale[phot_it] <= 0.0) and fit_error_scale:
            return BADLPR
        if (ascale[phot_it] <= 0.0) and fit_amplitude_scale:
            return BADLPR
        if (lscale[phot_it] <= 0.0) and fit_length_scale:
            return BADLPR
        # get transit for current parameters
        tkwargs = dict(sol=sol[:, phot_it], time=time[phot_it],
                       itime=itime[phot_it], ntt=ntt, tobs=tobs, omc=omc)
        # get and plot the model
        model = transit_fit.transitmodel(**tkwargs)
        # check for NaNs -- we don't want these.
        if np.isfinite(model.sum()):
            # Question: What about the GP model, currently it does not update
            #           logl
            # non-correlated noise-model
            if model_type == 0:
                sum1 = np.sum(np.log(fluxerr[phot_it] ** 2 * dscale ** 2))
                sqrdiff = (flux[phot_it] - model)**2
                sum2 = np.sum(sqrdiff / (fluxerr**2 * dscale[phot_it]**2))
                logl += -0.5 * (sum1 + sum2)
        # else we return our bad log likelihood
        else:
            return BADLPR
    # if we have got to here we return the good loglikelihood (sum of each
    #   bandpass)
    return logl


def mhg_mcmc(tfit: TransitFit, loglikelihood: Any, buffer, corbeta):
    """
    A Metropolis-Hastings MCMC with Gibbs sampler

    :param tfit: the Transit fit class
    :param loglikelihood: The loglikelihodd function here
                          loglikelihood must have a single argument of type
                          Transit fit class
    :param buffer:
    :param corbeta:

    :return:
    """
    # we do not use bugger and corbeta here (used for dmhg_mcmc)
    _ = buffer, corbeta
    # -------------------------------------------------------------------------
    # Step 1: Generate trial state
    # -------------------------------------------------------------------------
    # Generate trial state with Gibbs sampler
    tfit.generate_gibbs_sample()
    # -------------------------------------------------------------------------
    # Step 2: Compute log(p(x'|d))=log(p(x'))+log(p(d|x'))
    # -------------------------------------------------------------------------
    if loglikelihood is None:
        llxt = tfit.loglikelihood(tfit)
    else:
        llxt = loglikelihood(tfit)
    # -------------------------------------------------------------------------
    # Step 3 Compute the acceptance probability
    # -------------------------------------------------------------------------
    alpha = min(np.exp(llxt - tfit.llx), 1.0)
    # -------------------------------------------------------------------------
    # Step 4 Accept or reject trial
    # -------------------------------------------------------------------------
    # generate a random number
    test = np.random.rand()
    # if test is less than our acceptance level accept new trial
    if test <= alpha:
        tfit.x0 = np.array(tfit.xtmp)
        tfit.llx = float(llxt)
        tfit.ac = [0, tfit.n_tmp]
    # else we reject and start from previous point
    else:
        tfit.xtmp = tfit.x0
        tfit.ac = [1, tfit.n_tmp]
    # return tfit instance
    return tfit


def de_mhg_mcmc(tfit: TransitFit, loglikelihood: Any, buffer, corbeta):
    """
    A Metropolis-Hastings MCMC with Gibbs sampler

    :param tfit: the Transit fit class
    :param loglikelihood: The loglikelihodd function here
                          loglikelihood must have a single argument of type
                          Transit fit class
    :param buffer:
    :param corbeta:

    :return:
    """
    # draw a random number to decide which sampler to use
    rsamp = np.random.rand()
    # -------------------------------------------------------------------------
    # Step 1: Generate trial state
    # -------------------------------------------------------------------------
    # if rsamp is less than 0.5 use a Gibbs sampler
    if rsamp < 0.5:
        # Generate trial state with Gibbs sampler
        tfit.generate_gibbs_sample()
    # else we use our deMCMC sampler
    else:
        tfit.generate_demcmc_sample(buffer, corbeta)
    # -------------------------------------------------------------------------
    # Step 2: Compute log(p(x'|d))=log(p(x'))+log(p(d|x'))
    # -------------------------------------------------------------------------
    llxt = loglikelihood(tfit)
    # -------------------------------------------------------------------------
    # Step 3 Compute the acceptance probability
    # -------------------------------------------------------------------------
    alpha = min(np.exp(llxt - tfit.llx), 1.0)
    # -------------------------------------------------------------------------
    # Step 4 Accept or reject trial
    # -------------------------------------------------------------------------
    # generate a random number
    test = np.random.rand()
    # if test is less than our acceptance level accept new trial
    if test <= alpha:
        tfit.x0 = np.array(tfit.xtmp)
        tfit.llx = float(llxt)
        tfit.ac = [0, tfit.n_tmp]
    # else we reject and start from previous point
    else:
        tfit.xtmp = tfit.x0
        tfit.ac = [1, tfit.n_tmp]
    # return tfit instance
    return tfit


# TODO: Fill out
def beta_rescale(params: ParamDict, tfit: TransitFit) -> TransitFit:

    # get alow, ahigh define the acceptance rate range we want
    alow = params['BETA_ALOW']
    ahigh = params['BETA_AHIGH']
    # parameter controling how fast corscale changes - from Gregory 2011.
    delta = params['BETA_DELTA']

    return tfit


# =============================================================================
# MCMC Sampler
# =============================================================================
class Sampler:
    def __init__(self, params: ParamDict, tfit: TransitFit, mode='full'):
        self.params = params
        self.tfit = tfit
        self.mode = mode

    # TODO: fill out
    def run_mcmc(self):
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
